import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import re
import json
import dspy
import hashlib
import pandas as pd
import spacy
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter
from config.settings import settings
from shared.data_classes import Entity, CorefChain, Mention, Relation, EntityLabels, RelationLabels, DisambiguationStatus
from shared.utils import extract_exact_sentence, _locate_mention, NON_LINKABLE_TYPES
from ingestion.models.base import IExtractor


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


# ==================Name Entity Recognition=========================
_LABEL_LIST = ", ".join(EntityLabels.__members__.keys())
_LABEL_DEFS = "\n".join(
    f"      {name} = {name.lower().replace('_', ' ')}"  # or pull from a descriptions dict
    for name in EntityLabels.__members__.keys()
)

class NameEntityRecognition(dspy.Signature):
    __doc__ = f"""You are an expert Named Entity Recognition system.

    Read the provided tokenized text and extract ALL named entities.
    For each entity, provide:
    - text: the exact entity text as it appears in the input
    - label: one of [{_LABEL_LIST}]
    - confidence: your confidence from 0.0 to 1.0

    Entity label definitions:
    {_LABEL_DEFS}

    IMPORTANT: Return ONLY a valid JSON object. Do not write explanations,
    do not show your work, and do not wrap the output in markdown code blocks.
    """

    tokens: str = dspy.InputField(desc="text of strings")
    entities: List[Entity] = dspy.OutputField(
        desc="List of extracted entities with text, label, and confidence."
    )


class NERExtractor(dspy.Module):
    def __init__(self, use_cot: bool = False, nlp: Optional[spacy.Language] = None):
        super().__init__()
        if nlp is None:
            nlp = spacy.load(settings.SPACY_MODEL_PATH)
        self.nlp = nlp
        self.extractor = dspy.ChainOfThought(NameEntityRecognition) if use_cot else dspy.Predict(NameEntityRecognition)

    def forward(self, sentence: str, doc_id:str, chunk_id:str) -> List[Entity]:
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
            start, end = _locate_mention(sentence, ent.text, used_spans)

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
                Entity(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=ent.text,
                    canonical_name=ent.text,
                    status=DisambiguationStatus.AMBIGUOUS,
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
            nlp = spacy.load("en_core_web_sm")
        self.nlp = nlp
        self.extractor = NERExtractor(use_cot=False, nlp=self.nlp)
        self.n_passes = n_passes
        self.agreement_threshold = agreement_threshold

    def _entity_key(self, ent: Entity) -> str:
        # NOTE: ent is a pydantic Entity, not a dict — must use attribute
        # access. Item access (ent['text']) raises TypeError.
        return f"{ent.text.lower().strip()}::{ent.label}"

    def forward(self, sentence: str, doc_id:str, chunk_id:str):
        all_extractions: List[Entity] = []

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

        final_entities: List[Entity] = []
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


# ==================Coreference Extractor============================
class CoreferenceResolution(dspy.Signature):
    """
    You are a coreference resolution expert.

    Given a document and a list of named entities, resolve each PRONOUN
    to the correct antecedent entity. Consider:
    - Grammatical gender/number agreement
    - Syntactic subject preference
    - Semantic compatibility (e.g., "it" refers to organizations/products, not people)
    - Recency and salience (first-mention preference)
    - Entities inside quotation marks are usually titles/names and less likely to be the antecedent for a pronoun appearing outside those quotes

    For each pronoun, output the exact text of the antecedent entity it refers to.
    If uncertain, omit the entry.
    """

    document: str = dspy.InputField(desc="Full document text")
    entities: List[Dict] = dspy.InputField(desc="List of entities with text, label, start, end")
    pronouns: List[Dict] = dspy.InputField(desc="List of pronouns with text, start, end")
    resolutions: List[Dict] = dspy.OutputField(
        desc=(
            "JSON list of objects. Each object MUST have: "
            "pronoun_text (str), pronoun_start (int), antecedent_text (str), confidence (float 0-1). "
            "Example: [{\"pronoun_text\": \"He\", \"pronoun_start\": 45, \"antecedent_text\": \"John Smith\", \"confidence\": 0.95}]"
        )
    )


class CorefExtractor(dspy.Module):
    """DSPy module for LLM-based coreference extraction."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.extractor = dspy.ChainOfThought(CoreferenceResolution) if use_cot else dspy.Predict(CoreferenceResolution)

    def forward(self, document: str, entities: List[Dict], pronouns: List[Dict]) -> List[Dict]:
        if not pronouns or not entities:
            return []

        try:
            prediction = self.extractor(
                document=document,
                entities=entities,
                pronouns=pronouns
            )
            return prediction.resolutions if prediction.resolutions else []
        except Exception as e:
            print(f"[CorefExtractor LLM Error] {e}")
            return []

# ==================Relation Extraction==============================
# One LLM call per sentence (not per entity pair): the model sees every
# resolved entity in the sentence at once and returns all relations it can
# support from that context. This is both cheaper (O(sentences) calls
# instead of O(entities^2)) and gives the model more context per relation
# judgment than an isolated pairwise prompt would.

_ENTITY_DOCS = "\n".join(
    f"    {name}: {name.lower().replace('_', ' ')}"
    for name in EntityLabels.__members__
)

_RELATION_DOCS = "\n".join(
    f"    {e.value}"
    for e in RelationLabels
    if e not in {RelationLabels.NO_RELATION, RelationLabels.OTHER}
)


class RelationExtraction(dspy.Signature):
    __doc__ = f"""You are an expert Relation Extraction system.

Given a text and a list of resolved entities present in that text,
identify factual relations between pairs of entities.

Allowed Entity Labels:
{_ENTITY_DOCS}

Allowed Relation Predicates (use EXACTLY these snake_case values):
{_RELATION_DOCS}

Rules:
1. Use ONLY the provided entities as subjects and objects.
2. Use the `canonical_name` of the entity for the subject and object fields.
3. Do not invent new entities that are not in the provided list.
4. Extract clear, factual relations explicitly supported by the text.
5. Use ONLY the allowed predicates listed above. If uncertain, use "related_to".
6. If no relation exists between the entities in the text, return an empty list.

IMPORTANT: Return ONLY a valid JSON list. Do not write explanations,
do not show your work, and do not wrap the output in markdown code blocks.
"""

    text: str = dspy.InputField(desc="The sentence or text chunk containing the entities.")
    entities: List[Dict] = dspy.InputField(
        desc="List of entities with keys: canonical_name, entity_label, original_text."
    )
    relations: List[Dict] = dspy.OutputField(
        desc=(
            "JSON list of relation objects. Each object MUST have: "
            "subject (str: canonical_name), predicate (str: snake_case), "
            "object (str: canonical_name), confidence (float 0-1). "
            'Example: [{"subject": "Benyamin Bahadori", "predicate": "born_in", '
            '"object": "Tehran", "confidence": 0.95}]'
        )
    )


class RelationExtractor(dspy.Module):
    """DSPy module for LLM-based relation extraction. Outputs List[Relation]."""

    def __init__(self, use_cot: bool = False, min_confidence: float = 0.6):
        super().__init__()
        self.extractor = (
            dspy.ChainOfThought(RelationExtraction)
            if use_cot
            else dspy.Predict(RelationExtraction)
        )
        self.min_confidence = min_confidence

    @staticmethod
    def _get_canonical_name(e: Union[Entity, Dict]) -> str:
        if isinstance(e, Entity):
            return e.canonical_name
        if isinstance(e, Entity):
            return getattr(e, "canonical_name", e.text)
        return e.get("canonical_name", e.get("text", ""))

    @staticmethod
    def _get_label(e: Union[Entity, Dict]) -> str:
        if isinstance(e, Entity):
            return e.label
        if isinstance(e, Entity):
            return e.label
        return e.get("entity_label", e.get("label", "UNKNOWN"))

    @staticmethod
    def _get_original_text(e: Union[Entity, Dict]) -> str:
        if isinstance(e, Entity):
            return e.text
        if isinstance(e, Entity):
            return e.text
        return e.get("original_text", e.get("text", ""))

    def forward(self, text: str, entities: List[Union[Entity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
        if not entities or len(entities) < 2:
            return []

        # ── Build lookup maps + prompt payload ──
        entity_map: Dict[str, Dict[str, str]] = {}
        prompt_entities: List[Dict[str, str]] = []

        for e in entities:
            canon = self._get_canonical_name(e)
            label = self._get_label(e)
            orig = self._get_original_text(e)

            if canon:
                entity_map[canon.lower()] = {"canonical": canon, "label": label}
            if orig and orig.lower() != canon.lower():
                entity_map[orig.lower()] = {"canonical": canon, "label": label}

            prompt_entities.append(
                {
                    "canonical_name": canon,
                    "entity_label": label,
                    "original_text": orig,
                }
            )

        # ── LLM call ──
        try:
            prediction = self.extractor(text=text, entities=prompt_entities)
            raw_relations = prediction.relations if prediction.relations else []
        except Exception as e:
            print(f"[RelationExtractor LLM Error] {e}")
            return []

        # DSPy sometimes returns a JSON string instead of a parsed list
        if isinstance(raw_relations, str):
            try:
                raw_relations = json.loads(raw_relations)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", raw_relations, re.DOTALL)
                if match:
                    try:
                        raw_relations = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        raw_relations = []
                else:
                    raw_relations = []

        # ── Validate & build Relation objects ──
        validated: List[Relation] = []

        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue

            subj_raw = str(rel.get("subject", "")).strip()
            obj_raw = str(rel.get("object", "")).strip()
            pred_raw = str(rel.get("predicate", "")).strip()

            try:
                conf = float(rel.get("confidence", 0.5))
            except (ValueError, TypeError):
                conf = 0.5

            # Resolve to canonical names
            subj_canon = entity_map.get(subj_raw.lower(), {}).get("canonical", subj_raw)
            obj_canon = entity_map.get(obj_raw.lower(), {}).get("canonical", obj_raw)

            # Normalize predicate to snake_case
            pred_norm = re.sub(
                r"[^a-z0-9_]", "", pred_raw.lower().replace(" ", "_").replace("-", "_")
            )

            # ── Validation gates ──
            if not subj_canon or not obj_canon or not pred_norm:
                continue
            if subj_canon == obj_canon:
                continue
            if conf < self.min_confidence:
                continue
            if not RelationLabels.is_valid(pred_norm):
                continue  # strict: skip hallucinated predicates

            # ── Symmetric normalization (sync with BERT) ──
            if RelationLabels.is_symmetric(pred_norm) and obj_canon < subj_canon:
                subj_canon, obj_canon = obj_canon, subj_canon

            subj_label = entity_map.get(subj_canon.lower(), {}).get("label", "UNKNOWN")
            obj_label = entity_map.get(obj_canon.lower(), {}).get("label", "UNKNOWN")

            rel_id = hashlib.sha1(
                f"{subj_canon}|{pred_norm}|{obj_canon}".encode("utf-8")
            ).hexdigest()

            validated.append(
                Relation(
                    doc_id=doc_id,
                    subject=subj_canon,
                    subject_label=subj_label,
                    predicate=pred_norm,
                    object=obj_canon,
                    object_label=obj_label,
                    mention_sentence=text,
                    confidence=round(conf, 3),
                    needs_review=False,
                    evidence=[text],
                    relation_id=rel_id,
                    chunk_id=chunk_id,
                    provisional=False,
                )
            )

        return validated


class RelationResolver:
    """
    Pipeline-level orchestrator: groups resolved entities by (doc_id,
    chunk_id, sentence), calls RelationExtractor once per group, dedups
    symmetric predicates, and returns a Relation DataFrame.

    Mirrors CorefResolver's role relative to CorefExtractor: the *Extractor
    classes do one bounded unit of LLM work, the *Resolver classes handle
    batching, grouping, and DataFrame-level bookkeeping over a whole document.
    """

    def __init__(self,
                 dspy_extractor: Optional[RelationExtractor] = None,
                 use_cot: bool = True,
                 include_literals: bool = True):
        self.extractor = dspy_extractor or RelationExtractor(use_cot=use_cot)
        self.include_literals = include_literals
        self.literal_labels = {"DATE", "TIME", "MONEY", "PERCENT", "NUM", "CARDINAL", "ORDINAL", "QUANTITY"}
        self.symmetric_preds = {"spouse", "married_to", "sibling", "collaborates_with", "co_founder", "partner"}

    def _make_relation_id(self, doc_id: str, subject: str, predicate: str, obj: str) -> str:
        key = f"{doc_id}|{subject}|{predicate}|{obj}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def extract(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entrypoint. Takes the RESOLVED-only entity DataFrame produced by
        UnifiedEntityResolver.process_document() and returns a relations
        DataFrame built from the shared Relation dataclass.
        """
        empty_df = pd.DataFrame(columns=[f.name for f in Relation.__dataclass_fields__.values()])

        if clean_df is None or clean_df.empty:
            return empty_df

        df = clean_df.copy()

        required_cols = {"doc_id", "canonical_name", "entity_label", "mention_sentence"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"clean_df must contain columns: {required_cols}")

        if "chunk_id" not in df.columns:
            df["chunk_id"] = None
        if "original_text" not in df.columns:
            df["original_text"] = df["canonical_name"]

        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower().str.contains("resolved", na=False)]

        df = df.dropna(subset=["canonical_name", "mention_sentence"])

        relations: List[Relation] = []
        grouped = df.groupby(["doc_id", "chunk_id", "mention_sentence"], dropna=False)

        for (doc_id, chunk_id, mention_sentence), group in grouped:
            entities_payload = []
            seen_canons = set()

            for _, row in group.iterrows():
                canon = row["canonical_name"]
                label = str(row.get("entity_label", "UNKNOWN")).upper()

                if not self.include_literals and label in self.literal_labels:
                    continue
                if canon in seen_canons:
                    continue
                seen_canons.add(canon)

                entities_payload.append({
                    "canonical_name": canon,
                    "entity_label": label,
                    "original_text": row.get("original_text", canon)
                })

            if len(entities_payload) < 2:
                continue

            extracted = self.extractor(text=str(mention_sentence), entities=entities_payload)
            label_map = {e["canonical_name"]: e["entity_label"] for e in entities_payload}

            for rel in extracted:
                subj, obj = rel["subject"], rel["object"]
                pred = rel["predicate"]

                # Normalize symmetric relations so (A, spouse, B) and (B, spouse, A) match
                if pred in self.symmetric_preds and obj < subj:
                    subj, obj = obj, subj

                relations.append(Relation(
                    doc_id=doc_id,
                    subject=subj,
                    subject_label=label_map.get(subj, "UNKNOWN"),
                    predicate=pred,
                    object=obj,
                    object_label=label_map.get(obj, "UNKNOWN"),
                    mention_sentence=mention_sentence,
                    confidence=rel["confidence"],
                    source="dspy_llm",
                    evidence=[mention_sentence],
                    relation_id=self._make_relation_id(str(doc_id), subj, pred, obj),
                    chunk_id=chunk_id,
                ))

        if not relations:
            return empty_df

        return pd.DataFrame([r.to_dict() for r in relations])


# ==================Adapter Wrappers==============================

class DSPyNERExtractor(IExtractor):
    """Adapter: your existing NERExtractor becomes INERExtractor-compliant."""
    def __init__(self, use_cot:bool=False, dspy_extractor=None):
        if dspy_extractor is None:
            dspy_extractor = NERExtractor(use_cot=use_cot)
        self._extractor = dspy_extractor

    def extract(self, text: str, doc_id:str, chunk_id:str) -> List[Entity]:
        entities = self._extractor(sentence=text, doc_id=doc_id, chunk_id=chunk_id)
        return entities


class DSPyCorefResolver(IExtractor):
    """Hybrid coreference resolver combining fast NLP heuristics with LLM reasoning."""

    PRONOUN_MAP = {
        "he":       {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "him":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "his":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "she":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "her":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "hers":     {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},

        "they":     {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "them":     {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "their":    {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "theirs":   {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},

        "it":       {"types": ["ORG", "LOC", "MISC", "PRODUCT"], "category": "ambiguous", "confidence_boost": 0.0},
        "its":      {"types": ["ORG", "LOC", "MISC", "PRODUCT"], "category": "ambiguous", "confidence_boost": 0.0},

        "we":       {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "us":       {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "our":      {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "ours":     {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
    }

    DESCRIPTION_PATTERNS = [
        (r"\bthe\s+company\b",      ["ORG"]),
        (r"\bthe\s+firm\b",          ["ORG"]),
        (r"\bthe\s+organization\b", ["ORG"]),
        (r"\bthe\s+airline\b",      ["ORG"]),
        (r"\bthe\s+bank\b",         ["ORG"]),
        (r"\bthe\s+CEO\b",          ["PER", "PERSON"]),
        (r"\bthe\s+president\b",    ["PER", "PERSON"]),
        (r"\bthe\s+executive\b",    ["PER", "PERSON"]),
        (r"\bthe\s+writer\b",       ["PER", "PERSON"]),
        (r"\bthe\s+author\b",       ["PER", "PERSON"]),
        (r"\bthe\s+firm['']?s\b",   ["ORG"]),
        (r"\bthe\s+book\b",         ["MISC", "PRODUCT"]),
        (r"\bthe\s+novel\b",        ["MISC", "PRODUCT"]),
        (r"\bthe\s+movie\b",        ["MISC", "PRODUCT"]),
        (r"\bthe\s+album\b",        ["MISC", "PRODUCT"]),
    ]

    def __init__(self, nlp=None, llm_module: Optional[dspy.Module] = None,
                 mode: str = "hybrid", threshold: float = 0.92):
        self.nlp = nlp
        self.llm_module = llm_module
        self.mode = mode
        self.threshold = threshold

        if mode in ("llm", "hybrid") and llm_module is None:
            self.llm_resolver = CorefExtractor(use_cot=True)
        else:
            self.llm_resolver = llm_module

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def extract(self, text: str, entities: List[Entity]) -> List[Entity]:
        """
        Resolve pronouns and descriptions in text against the provided entities.
        Accepts List[Entity]; returns original entities + resolved coreferences.
        """
        if not entities:
            return []

        # ── 1. Base NER entities as Entity ──
        named_ents = self._get_named_entities(entities)

        # ── 2. Pronoun resolution ──
        doc = self.nlp(text) if self.nlp and hasattr(self.nlp, "__call__") else None
        pronouns = self._extract_pronouns(doc, text)

        if pronouns and named_ents:
            if self.mode == "nlp":
                chains = self._resolve_nlp(pronouns, named_ents, text)
            elif self.mode == "llm":
                chains = self._resolve_llm(pronouns, named_ents, text)
            else:
                chains = self._resolve_hybrid(pronouns, named_ents, text)

            for chain in chains:
                entities.append(self._chain_to_resolved(chain, text))

        # ── 3. Definite descriptions ──
        # resolved.extend(self._resolve_descriptions(text, named_ents))

        return entities

    # -----------------------------------------------------------------------
    # INTERNAL CONVERTERS
    # -----------------------------------------------------------------------

    def _get_named_entities(self, entities: List[Entity]) -> List[Entity]:
        """Return only base named entities (not previous coreference resolutions)."""
        return [r for r in entities if r.coref_to is None]

    def _chain_to_resolved(self, chain: CorefChain, text: str) -> Entity:
        mention_sent = extract_exact_sentence(
            doc_text=text,
            start_char=chain.pronoun.start,
            end_char=chain.pronoun.end,
        )
        return Entity(
            doc_id=chain.antecedent_doc_id,
            chunk_id=chain.antecedent_chunk_id,
            text=chain.pronoun.text,
            start=chain.antecedent_start,
            end=chain.antecedent_end,
            canonical_name=chain.antecedent_text,
            label=self._map_pronoun_to_label(chain.pronoun.text),
            mention_sentence=mention_sent,
            status=DisambiguationStatus.RESOLVED,
            confidence=round(min(chain.confidence, 1.0), 3),
            coref_to=chain.antecedent_text,
        )

    # -----------------------------------------------------------------------
    # NLP RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_nlp(self, pronouns: List[Mention],
                     named_ents: List[Entity],
                     text: str) -> List[CorefChain]:
        resolutions = []

        for pro in pronouns:
            cfg = self.PRONOUN_MAP.get(pro.text.lower(), {})
            compatible_types = cfg.get("types", ["MISC"])

            # Candidates must appear before the pronoun
            candidates = [e for e in named_ents if getattr(e, "end", 0) <= pro.start]
            if not candidates:
                continue

            # Label compatibility
            candidates = [e for e in candidates if e.label in compatible_types]
            if not candidates:
                continue

            # Score and pick best
            scored = [(e, self._score_antecedent(e, pro, text)) for e in candidates]
            best, best_score = max(scored, key=lambda x: x[1])

            if best_score > 0.3:
                resolutions.append(CorefChain(
                    antecedent_doc_id=best.doc_id,
                    antecedent_chunk_id=best.chunk_id,
                    pronoun=pro,
                    antecedent_text=best.text,
                    antecedent_start=getattr(best, "start", 0),
                    antecedent_end=getattr(best, "end", 0),
                    method="nlp",
                    confidence=min(best_score + cfg.get("confidence_boost", 0), 1.0)
                ))

        return resolutions

    def _score_antecedent(self, antecedent: Entity,
                          pronoun: Mention,
                          text: str) -> float:
        score = 0.0
        ant_start = getattr(antecedent, "start", 0)
        ant_end   = getattr(antecedent, "end", 0)
        ant_text  = antecedent.text

        distance = pronoun.start - ant_end
        score += 0.4 * (1.0 / (1.0 + 0.01 * distance))

        score += 0.2 * (1.0 / (1.0 + 0.001 * ant_start))

        sent_start = text.rfind(".", 0, ant_start) + 1
        if sent_start < 0:
            sent_start = 0
        dist_from_sent_start = ant_start - sent_start
        if dist_from_sent_start < 20:
            score += 0.2

        score += 0.1 * min(len(ant_text) / 20.0, 1.0)

        pro_lower = pronoun.text.lower()
        if pro_lower in ("he", "him", "his"):
            if any(n in ant_text.lower() for n in ["john", "michael", "david", "james", "robert"]):
                score += 0.1
        elif pro_lower in ("she", "her", "hers"):
            if any(n in ant_text.lower() for n in ["mary", "jennifer", "linda", "patricia"]):
                score += 0.1

        quote_before = text.rfind('"', 0, ant_start)
        quote_after = text.find('"', ant_end, pronoun.start)
        if quote_before != -1 and quote_after != -1:
            score -= 0.25

        return score

    # -----------------------------------------------------------------------
    # LLM RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_llm(self, pronouns: List[Mention],
                     named_ents: List[Entity],
                     text: str) -> List[CorefChain]:
        if not self.llm_resolver or not pronouns:
            return []

        entity_list = [
            {
                "text": e.text,
                "label": e.label,
                "start": getattr(e, "start", 0),
                "end": getattr(e, "end", 0),
            }
            for e in named_ents
        ]
        pronoun_list = [{"text": p.text, "start": p.start, "end": p.end} for p in pronouns]

        try:
            resolutions_raw = self.llm_resolver(
                document=text,
                entities=entity_list,
                pronouns=pronoun_list
            )
        except Exception as e:
            print(f"[LLM Coref Error] {e}")
            return []

        resolutions = []
        for res in resolutions_raw:
            if not res or not res.get("antecedent_text"):
                continue

            pro_start = res.get("pronoun_start")
            pro_text = res.get("pronoun_text", "")

            pro_match = None
            if pro_start is not None:
                pro_match = next((p for p in pronouns if p.start == pro_start), None)
            if not pro_match:
                pro_match = next((p for p in pronouns if p.text.lower() == pro_text.lower()), None)
            if not pro_match:
                continue

            ant_text = res["antecedent_text"]
            ant_candidates = [e for e in named_ents if e.text == ant_text]
            if not ant_candidates:
                ant_candidates = [e for e in named_ents if e.text.lower() == ant_text.lower()]

            if not ant_candidates:
                continue

            # Antecedent must appear before pronoun
            ant_candidates = [
                e for e in ant_candidates
                if getattr(e, "end", 0) <= pro_match.start
            ]
            if not ant_candidates:
                continue

            # Pick closest (minimum distance)
            ant = min(ant_candidates, key=lambda e: pro_match.start - getattr(e, "end", 0))

            resolutions.append(CorefChain(
                antecedent_doc_id=ant.doc_id,
                antecedent_chunk_id=ant.chunk_id,
                pronoun=pro_match,
                antecedent_text=ant.text,
                antecedent_start=getattr(ant, "start", 0),
                antecedent_end=getattr(ant, "end", 0),
                method="llm",
                confidence=min(res.get("confidence", 0.8), 1.0)
            ))

        return resolutions

    # -----------------------------------------------------------------------
    # HYBRID RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_hybrid(self, pronouns: List[Mention],
                        named_ents: List[Entity],
                        text: str) -> List[CorefChain]:
        person_pronouns = [
            p for p in pronouns
            if self.PRONOUN_MAP.get(p.text.lower(), {}).get("category") == "person"
        ]
        ambiguous_pronouns = [
            p for p in pronouns
            if self.PRONOUN_MAP.get(p.text.lower(), {}).get("category") in ("ambiguous", "org")
        ]

        resolutions: List[CorefChain] = []
        resolved_starts: set[int] = set()

        if person_pronouns:
            nlp_res = self._resolve_nlp(person_pronouns, named_ents, text)
            for r in nlp_res:
                resolutions.append(r)
                resolved_starts.add(r.pronoun.start)

            low_conf = [r for r in nlp_res if r.confidence < .6]
            if low_conf:
                llm_res = self._resolve_llm([r.pronoun for r in low_conf], named_ents, text)
                for lr in llm_res:
                    old = next((r for r in resolutions if r.pronoun.start == lr.pronoun.start), None)
                    if old and lr.confidence > old.confidence:
                        resolutions.remove(old)
                        resolutions.append(lr)

        if ambiguous_pronouns and self.llm_resolver:
            unresolved_ambiguous = [p for p in ambiguous_pronouns if p.start not in resolved_starts]
            if unresolved_ambiguous:
                llm_res = self._resolve_llm(unresolved_ambiguous, named_ents, text)
                resolutions.extend(llm_res)

        return resolutions

    # -----------------------------------------------------------------------
    # DEFINITE DESCRIPTIONS
    # -----------------------------------------------------------------------

    def _resolve_descriptions(self, text: str,
                              named_ents: List[Entity]) -> List[Entity]:
        results: List[Entity] = []

        for pattern, types in self.DESCRIPTION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                antecedent = self._find_nearest_compatible(m.start(), m.end(), named_ents, types)
                if antecedent is None:
                    continue

                mention_sent = extract_exact_sentence(
                    doc_text=text,
                    start_char=m.start(),
                    end_char=m.end())
                
                results.append(Entity(
                    text=m.group(),
                    canonical_name=antecedent.text,
                    entity_label=types[0],
                    mention_sentence=mention_sent,
                    status=DisambiguationStatus.AMBIGUOUS,
                    confidence=0.7,
                    coref_to=antecedent.text,
                ))

        return results

    def _find_nearest_compatible(self, start: int, end: int,
                                 named_ents: List[Entity],
                                 compatible_types: List[str]) -> Optional[Entity]:
        candidates = [
            e for e in named_ents
            if getattr(e, "end", 0) <= start and e.label in compatible_types
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: start - getattr(e, "end", 0))

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _extract_pronouns(self, doc, text: str) -> List[Mention]:
        pronouns = []
        sent_id = 0
        last_end = 0

        if doc and hasattr(doc, "sents"):
            sentences = list(doc.sents)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            if hasattr(sent, "text"):
                sent_text = sent.text
                sent_start = sent.start_char if hasattr(sent, "start_char") else text.find(sent_text, last_end)
            else:
                sent_text = sent
                sent_start = text.find(sent_text, last_end)

            if sent_start == -1:
                sent_start = last_end

            pattern = r'\b(' + '|'.join(re.escape(k) for k in self.PRONOUN_MAP.keys()) + r')\b'
            for match in re.finditer(pattern, sent_text, re.IGNORECASE):
                pronouns.append(Mention(
                    text=match.group(),
                    start=sent_start + match.start(),
                    end=sent_start + match.end(),
                    sent_id=sent_id,
                    is_pronoun=True,
                    pronoun_type=self.PRONOUN_MAP.get(match.group().lower(), {}).get("category")
                ))

            sent_id += 1
            last_end = sent_start + len(sent_text)

        return pronouns

    def _map_pronoun_to_label(self, pronoun: str) -> str:
        p = pronoun.lower()
        if p in ("he", "him", "his", "she", "her", "hers"):
            return "PER"
        elif p in ("it", "its"):
            return "MISC"
        elif p in ("we", "us", "our", "ours"):
            return "ORG"
        else:
            return "MISC"


class DSPyRelationExtractor(IExtractor):
    """Adapter: your existing dspy.Module becomes IRelationExtractor-compliant."""
    def __init__(self, dspy_extractor=None, use_cot:bool=False, min_confidence: float = 0.6):
        # Lazy import to avoid hard dependency
        if dspy_extractor is None:
            dspy_extractor = RelationExtractor(use_cot=use_cot, min_confidence=min_confidence)
        self._extractor = dspy_extractor

    def extract(self, text: str, entities: List[Union[Entity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
        if len(entities) < 2:
            return []
        result = self._extractor(text=text, entities=entities, doc_id=doc_id, chunk_id=chunk_id)
        return result

    
if __name__ == "__main__":
    ner = NERExtractor()
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    entities = ner(text)
    print(entities)