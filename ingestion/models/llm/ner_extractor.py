import os
import dspy
import spacy
from typing import List, Optional
from collections import Counter
from config.settings import settings
from domain import MentionEntity, EntityLabels
from ingestion.models.base import IExtractor
from shared.utils import extract_exact_sentence, _locate_mention, NON_LINKABLE_TYPES


model_id = "openai/" + settings.LLM_MODEL_NAME

lm = dspy.LM(
    model=model_id,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    max_tokens=4096,
    temperature=0.0,
    top_p=1.0,
    n=1,
    stop=None
)
dspy.configure(lm=lm)


_LABEL_LIST = ", ".join(EntityLabels.__members__.keys())
_LABEL_DEFS = "\n".join(
    f"      {name} = {name.lower().replace('_', ' ')}"  # or pull from a descriptions dict
    for name in EntityLabels.__members__.keys()
)


class NameEntityRecognition(dspy.Signature):
    __doc__ = f"""You are an expert Named Entity Recognition system.

    Read the provided tokenized text and extract ALL named entities.
    For each entity, provide:
    - name: the exact entity name as it appears in the input
    - label: one of [{_LABEL_LIST}]
    - confidence: your confidence from 0.0 to 1.0

    Entity label definitions:
    {_LABEL_DEFS}

    IMPORTANT: Return ONLY a valid JSON object. Do not write explanations,
    do not show your work, and do not wrap the output in markdown code blocks.
    """

    tokens: str = dspy.InputField(desc="text of strings")
    entities: List[MentionEntity] = dspy.OutputField(
        desc="List of extracted entities with name, label, and confidence."
    )


class NERExtractor(dspy.Module):
    def __init__(self, use_cot: bool = False, nlp: Optional[spacy.Language] = None):
        super().__init__()
        if nlp is None:
            try:
                nlp = spacy.load(os.path.join(settings.SPACY_MODEL_PATH, settings.SPACY_MODEL_NAME))
            except:
                nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        self.extractor = dspy.ChainOfThought(NameEntityRecognition) if use_cot else dspy.Predict(NameEntityRecognition)

    def forward(self, sentence: str, doc_id:str, chunk_id:str) -> List[MentionEntity]:
        result = self.extractor(tokens=sentence)
        parsed_entities = []
        used_spans = set()  # already-assigned (start, end) to handle duplicates

        for ent in result.entities:
            ent_label = ent.label.upper().strip()
            ent_label = EntityLabels.find_best_match(ent_label)
            if ent_label is None:
                ent_label = EntityLabels.MISC

            conf = max(0.0, min(1.0, float(ent.confidence)))
            # Robustly locate the entity, regardless of LLM return order
            start, end = _locate_mention(sentence, ent.name, used_spans)

            # Only extract a mention sentence if we actually found the span
            if start != -1:
                use_nlp = ent_label not in NON_LINKABLE_TYPES
                mention = extract_exact_sentence(
                    doc_text=sentence,
                    start_char=start,
                    end_char=end,
                    use_nlp=use_nlp,
                    nlp=self.nlp,
                )
            else:
                mention = ""  # hallucinated / not found in text

            parsed_entities.append(
                MentionEntity(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    name=ent.name,
                    label=ent_label,
                    start=start,
                    end=end,
                    mention_sentence=mention,
                    confidence=round(conf, 3)))

        return parsed_entities


class NERWithConfidence(dspy.Module):
    """
    Runs NERExtractor multiple times and keeps only entities that a
    sufficient fraction of passes agree on, using agreement rate as
    the confidence score.
    """

    def __init__(self, n_passes: int = 3, agreement_threshold: float = 0.6, nlp:spacy=None):
        super().__init__()
        if nlp is None:
            try:
                nlp = spacy.load(os.path.join(settings.SPACY_MODEL_PATH, settings.SPACY_MODEL_NAME))
            except:
                nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        self.extractor = NERExtractor(use_cot=False, nlp=self.nlp)
        self.n_passes = n_passes
        self.agreement_threshold = agreement_threshold

    def _entity_key(self, ent: MentionEntity) -> str:
        # NOTE: ent is a pydantic Entity, not a dict — must use attribute
        # access. Item access (ent['text']) raises TypeError.
        return f"{ent.name.lower().strip()}::{ent.label}"

    def forward(self, sentence: str, doc_id:str, chunk_id:str):
        all_extractions: List[MentionEntity] = []

        # 1. Run multiple passes
        for _ in range(self.n_passes):
            try:
                result = self.extractor.forward(sentence=sentence, doc_id=doc_id, chunk_id=chunk_id)
                all_extractions.extend(result)
            except Exception:
                continue

        if not all_extractions:
            return {"sentence": sentence, "entities": []}

        # 2. Vote by (text, label) — do this AFTER all passes complete
        votes = Counter(self._entity_key(e) for e in all_extractions)

        final_entities: List[MentionEntity] = []
        seen = set()

        for ent in all_extractions:
            key = self._entity_key(ent)
            if key in seen:
                continue

            agreement = votes[key] / self.n_passes
            if agreement >= self.agreement_threshold:
                seen.add(key)
                # `confidence` IS a declared field on Entity, so attribute
                # assignment (not item assignment) works with
                # validate_assignment=True. Vote counts are returned
                # alongside instead of stashed on the model, since "votes"
                # isn't part of the Entity schema.
                ent.confidence = round(agreement, 3)
                final_entities.append(ent)

        return {
            "sentence": sentence,
            "entities": final_entities,
            "votes": {k: v for k, v in votes.items() if v / self.n_passes >= self.agreement_threshold},
        }


class DSPyNERExtractor(IExtractor):
    """Adapter: your existing NERExtractor becomes INERExtractor-compliant."""
    def __init__(self, use_cot:bool=False, dspy_extractor=None):
        if dspy_extractor is None:
            dspy_extractor = NERExtractor(use_cot=use_cot)
        self._extractor = dspy_extractor

    def extract(self, text: str, doc_id:str, chunk_id:str) -> List[MentionEntity]:
        entities = self._extractor(sentence=text, doc_id=doc_id, chunk_id=chunk_id)
        return entities


if __name__ == "__main__":
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    ner = DSPyNERExtractor()
    entities = ner.extract(text, "doc1","chunk1")
    print(entities)

