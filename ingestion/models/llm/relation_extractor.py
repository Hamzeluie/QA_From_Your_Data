import re
import json
import dspy
import hashlib
import pandas as pd
from typing import List, Dict, Optional, Union
from config.settings import settings
from domain import CanonicalEntity, Relation, EntityLabels, RelationLabels
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
    def _get_canonical_name(e: Union[CanonicalEntity, Dict]) -> str:
        if isinstance(e, CanonicalEntity):
            return getattr(e, "name", e.name)
        return e.get("canonical_name", e.get("text", ""))

    @staticmethod
    def _get_label(e: Union[CanonicalEntity, Dict]) -> str:
        if isinstance(e, CanonicalEntity):
            return e.label
        if isinstance(e, CanonicalEntity):
            return e.label
        return e.get("entity_label", e.get("label", "UNKNOWN"))
    
    def forward(self, text: str, entities: List[Union[CanonicalEntity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
        if len(entities) < 2:
            return []

        entity_map: Dict[str, Dict[str, str]] = {}
        prompt_entities: List[Dict[str, str]] = []
        seen_canonical_ids: set[str] = set()

        for entity in entities:
            canonical_id = entity.canonical_id

            if canonical_id in seen_canonical_ids:
                continue

            seen_canonical_ids.add(canonical_id)

            entity_map[entity.name.lower()] = {
                "canonical": entity.name,
                "label": entity.label,
            }

            prompt_entities.append({
                "name": entity.name,
                "label": entity.label,
            })

        if len(prompt_entities) < 2:
            return []
        
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


class DSPyRelationExtractor(IExtractor):
    """Adapter: your existing dspy.Module becomes IRelationExtractor-compliant."""
    def __init__(self, dspy_extractor=None, use_cot:bool=False, min_confidence: float = 0.6):
        # Lazy import to avoid hard dependency
        if dspy_extractor is None:
            dspy_extractor = RelationExtractor(use_cot=use_cot, min_confidence=min_confidence)
        self._extractor = dspy_extractor

    def extract(self, text: str, entities: List[Union[CanonicalEntity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
        if len(entities) < 2:
            return []
        result = self._extractor(text=text, entities=entities, doc_id=doc_id, chunk_id=chunk_id)
        return result
    

if __name__ == "__main__":
    from domain import DisambiguationStatus
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    resolved_entities = [
            CanonicalEntity(canonical_id="c_0",name='Apple Inc.', label=EntityLabels.ORG),
            CanonicalEntity(canonical_id="c_1",name='Steve Jobs', label=EntityLabels.PER),
            CanonicalEntity(canonical_id="c_3",name='Cupertino', label=EntityLabels.LOC ),
            CanonicalEntity(canonical_id="c_1",name='Steve Jobs', label=EntityLabels.PER),
            CanonicalEntity(canonical_id="c_0",name='Apple Inc.', label=EntityLabels.ORG),
            CanonicalEntity(canonical_id="c_0",name='Apple Inc.', label=EntityLabels.ORG)
            ]
    relation_extractor = DSPyRelationExtractor()
    relations = relation_extractor(text, resolved_entities, "user", "1")
    print(relations)


