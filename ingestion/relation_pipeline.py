import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import hashlib
import logging
from typing import List, Optional
from dataclasses import fields

import pandas as pd

from storage.factory import StorageFactory
from shared.data_classes import Relation

logger = logging.getLogger(__name__)


class RelationPipeline:
    """
    Production relation extraction.
    Groups resolved entities by sentence, calls LLM extractor,
    dedups symmetric predicates, and persists to Neo4j + ClickHouse.
    """

    def __init__(
        self,
        extractor,  # Callable: extractor(text=str, entities=List[Dict]) -> List[Dict]
        factory: Optional[StorageFactory] = None,
        include_literals: bool = True,
    ):
        self.extractor = extractor
        self.factory = factory or StorageFactory.from_env()
        self.include_literals = include_literals
        self.literal_labels = {
            "DATE", "TIME", "MONEY", "PERCENT", "NUM",
            "CARDINAL", "ORDINAL", "QUANTITY",
        }
        self.symmetric_preds = {
            "spouse", "married_to", "sibling",
            "collaborates_with", "co_founder", "partner",
        }

    def extract_and_store(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entrypoint. Takes the RESOLVED-only DataFrame from
        UnifiedEntityResolver.process_document().
        """
        empty_df = pd.DataFrame(columns=[f.name for f in fields(Relation)])

        if clean_df is None or clean_df.empty:
            return empty_df

        df = clean_df.copy()
        required = {"doc_id", "canonical_name", "entity_label", "mention_sentence"}
        if not required.issubset(df.columns):
            raise ValueError(f"clean_df must contain columns: {required}")

        if "chunk_id" not in df.columns:
            df["chunk_id"] = None
        if "original_text" not in df.columns:
            df["original_text"] = df["canonical_name"]

        # Only fully resolved entities
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower().str.contains("resolved", na=False)]

        df = df.dropna(subset=["canonical_name", "mention_sentence"])

        relations: List[Relation] = []
        grouped = df.groupby(["doc_id", "chunk_id", "mention_sentence"], dropna=False)

        for (doc_id, chunk_id, sentence), group in grouped:
            payload = []
            seen_canons = set()

            for _, row in group.iterrows():
                canon = row["canonical_name"]
                label = str(row.get("entity_label", "UNKNOWN")).upper()

                if not self.include_literals and label in self.literal_labels:
                    continue
                if canon in seen_canons:
                    continue
                seen_canons.add(canon)

                payload.append({
                    "canonical_name": canon,
                    "entity_label": label,
                    "original_text": row.get("original_text", canon),
                })

            if len(payload) < 2:
                continue

            try:
                extracted = self.extractor(text=str(sentence), entities=payload)
            except Exception as exc:
                logger.error(f"Relation extraction failed doc={doc_id}: {exc}")
                continue

            label_map = {e["canonical_name"]: e["entity_label"] for e in payload}

            for rel in extracted:
                subj, obj = rel["subject"], rel["object"]
                pred = rel["predicate"]

                # Normalize symmetric so (A,spouse,B) == (B,spouse,A)
                if pred in self.symmetric_preds and obj < subj:
                    subj, obj = obj, subj

                relation_id = self._make_relation_id(str(doc_id), subj, pred, obj)

                relations.append(Relation(
                    doc_id=str(doc_id),
                    subject=subj,
                    subject_label=label_map.get(subj, "UNKNOWN"),
                    predicate=pred,
                    object=obj,
                    object_label=label_map.get(obj, "UNKNOWN"),
                    mention_sentence=sentence,
                    confidence=float(rel.get("confidence", 0.0)),
                    source="dspy_llm",
                    needs_review=False,
                    evidence=[sentence],
                    relation_id=relation_id,
                    chunk_id=chunk_id,
                ))

        if not relations:
            return empty_df

        self._persist(relations)
        return pd.DataFrame([r.to_dict() for r in relations])

    # ── Persistence ─────────────────────────────────────────────────────────

    def _persist(self, relations: List[Relation]) -> None:
        # Neo4j graph edges (one by one; could batch with UNWIND in future)
        for rel in relations:
            try:
                self.factory.neo4j.create_relation(
                    subject=rel.subject,
                    predicate=rel.predicate,
                    obj=rel.object,
                    doc_id=rel.doc_id,
                    confidence=rel.confidence,
                    provisional=False,
                    relation_id=rel.relation_id,
                    evidence=rel.evidence,
                    chunk_id=rel.chunk_id,
                )
            except Exception as exc:
                logger.warning(f"Neo4j relation write failed {rel.relation_id}: {exc}")

        # ClickHouse analytics batch
        try:
            self.factory.clickhouse.insert_relations(relations)
        except Exception as exc:
            logger.error(f"ClickHouse relation batch failed: {exc}")

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_relation_id(doc_id: str, subject: str, predicate: str, obj: str) -> str:
        key = f"{doc_id}|{subject}|{predicate}|{obj}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()