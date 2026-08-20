"""
pipeline.py
============

EntityRelationPipeline wires together:
    EntityResolver      (NER -> coref -> NED/ER)   [entity_resolver.py]
    RelationResolver     (RE)                        [llm_extractors.py]
    PersistenceCoordinator (Neo4j / ClickHouse / ES / Qdrant fan-out)

Usage:
    pipeline = EntityRelationPipeline(
        resolver=resolver,                 # EntityResolver instance
        relation_extractor=relation_resolver,  # RelationResolver instance
        persistence=persistence,           # PersistenceCoordinator (optional)
    )
    clean_df, review_df, relations_df = pipeline.run(
        document=text,
        doc_id="doc_001",
        owner_id="user_123",
    )
    report = pipeline.last_report   # PersistenceReport, or None if persist=False

KNOWN UPSTREAM GAP (flagging rather than silently working around):
    EntityResolver.process_document(document, kg_threshold) does not accept
    doc_id/owner_id, and _name_entity_recognition / resolve_from_sentences
    hardcode doc_id="0" internally. This pipeline overwrites the doc_id
    column on the returned frames as a stopgap so downstream persistence is
    correctly attributed, but the real fix is threading doc_id/owner_id
    through EntityResolver.process_document -> _extract_entities_with_coref
    -> _name_entity_recognition, since right now every document you process
    collides on doc_id="0" inside the resolver's own bookkeeping (catalog
    mention counts, cross-doc resolution) even though the pipeline corrects
    the column on the *output* frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from typing import Optional, List, Tuple
import pandas as pd

from shared.data_classes import Chunk, Relation, ResolvedEntity, DisambiguationStatus
from ingestion.storage.persistence import PersistenceCoordinator, PersistenceReport
from ingestion.storage.db_writers import SearchIndexWriter, VectorStoreWriter, GraphWriter, OlapWriter
from ingestion.entity_resolver import EntityResolver
from ingestion.llm.llm_extractors import RelationResolver
from config.settings import settings
logger = logging.getLogger(__name__)


class EntityRelationPipeline:
    def __init__(self,
                 resolver:EntityResolver,
                 relation_extractor:RelationResolver,
                 persistence: Optional[PersistenceCoordinator] = None):
        self.resolver = resolver
        self.relation_extractor = relation_extractor
        self.persistence = persistence
        self.last_report: Optional[PersistenceReport] = None

    def run(self,
            document: str,
            doc_id: str,
            owner_id: str = "system",
            persist: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # ---- 1. NER -> coref -> NED/ER ---------------------------------
        # chunk_info, clean_df, review_df = self.resolver.process_document(document)

        
        chunk_info = [(1, "Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran po...s Cumanda ( 1879 ) .Nevertheless"), (2, ", thanks to the arguments of the well - known and respected Ecuadorian writer Alejand...rst novel .Riofrio died in exile in Peru .")]
        clean_df = resolver.get_all_unresolved()
        review_df = resolver.get_all_resolved()
        for df in (clean_df, review_df):
            if not df.empty:
                df["doc_id"] = doc_id  # stopgap, see module docstring

        # ---- 2. RE ------------------------------------------------------
        if clean_df.empty:
            relations_df = pd.DataFrame(columns=[f.name for f in Relation.__dataclass_fields__.values()])
        else:
            relations_df = self.relation_extractor.extract(clean_df)
            if not relations_df.empty:
                relations_df["doc_id"] = doc_id

        # ---- 3. persist ---------------------------------------------------
        self.last_report = None
        if persist and self.persistence is not None:
            chunks = self._build_chunks(document, doc_id, owner_id, clean_df)
            entities = self._entities_from_df(clean_df)
            relations = self._relations_from_df(relations_df)
            self.last_report = self.persistence.persist_document(doc_id, chunks, entities, relations)
            if not self.last_report.all_succeeded:
                logger.warning(self.last_report.summary())

        return clean_df, review_df, relations_df

    # ---- conversions: DataFrame rows -> dataclasses --------------------

    def _build_chunks(self, document: str, doc_id: str, owner_id: str, clean_df: pd.DataFrame) -> List[Chunk]:
        """One Chunk per distinct sentence/chunk_id that produced at least
        one resolved entity. If clean_df has no chunk_id column, falls back
        to a single chunk for the whole document."""
        if clean_df.empty:
            return [Chunk(doc_id=doc_id, chunk_id="0", owner_id=owner_id, sentence=document, date_time="")]

        if "chunk_id" not in clean_df.columns:
            unique_sentences = clean_df["mention_sentence"].dropna().unique()
            return [
                Chunk(doc_id=doc_id, chunk_id=str(i), owner_id=owner_id, sentence=s, date_time="")
                for i, s in enumerate(unique_sentences)
            ]

        chunks = []
        seen = set()
        for _, row in clean_df.dropna(subset=["mention_sentence"]).iterrows():
            cid = str(row.get("chunk_id", "0"))
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, owner_id=owner_id,
                                 sentence=row["mention_sentence"], date_time=""))
        return chunks

    def _entities_from_df(self, clean_df: pd.DataFrame) -> List[ResolvedEntity]:
        if clean_df.empty:
            return []
        entities = []
        for _, row in clean_df.iterrows():
            status_raw = row.get("status", DisambiguationStatus.RESOLVED.value)
            try:
                status = DisambiguationStatus(status_raw)
            except ValueError:
                status = DisambiguationStatus.RESOLVED

            entities.append(ResolvedEntity(
                original_text=row.get("original_text", row.get("text", "")),
                canonical_name=row["canonical_name"],
                entity_label=row.get("entity_label", "UNKNOWN"),
                mention_sentence=row.get("sentence", ""),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                status=status,
                source=row.get("source", "unknown"),
                kg_candidates=row.get("kg_candidates") or [],
                summary=row.get("summary"),
                context_clues=row.get("context_clues") or [],
                needs_review=bool(row.get("needs_review", False)),
                is_nil=bool(row.get("is_nil", False)),
                coref_to=row.get("coref_to"),
            ))
        return entities

    def _relations_from_df(self, relations_df: pd.DataFrame) -> List[Relation]:
        if relations_df.empty:
            return []
        relations = []
        for _, row in relations_df.iterrows():
            relations.append(Relation(
                doc_id=row["doc_id"],
                subject=row["subject"],
                subject_label=row.get("subject_label", "UNKNOWN"),
                predicate=row["predicate"],
                object=row["object"],
                object_label=row.get("object_label", "UNKNOWN"),
                sentence=row.get("sentence", ""),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                source=row.get("source", "dspy_llm"),
                needs_review=bool(row.get("needs_review", False)),
                evidence=row.get("evidence") or [],
                relation_id=row.get("relation_id"),
                chunk_id=row.get("chunk_id"),
            ))
        return relations


# --------------------------------------------------------------------------
# example wiring
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    PERSISTENCE_AVAILABLE = False
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)

    resolver = EntityResolver()
    resolver.load_full_state("/home/mehdi/Documents/projects/knowledge_graph_examples/full_state")
    relation_resolver = RelationResolver(use_cot=True, include_literals=True)
    if PERSISTENCE_AVAILABLE:
        persistence = PersistenceCoordinator(
            search_writer=SearchIndexWriter(hosts=["http://localhost:9200"]),
            vector_writer=VectorStoreWriter(url="http://localhost:6333", embedder=embedder),
            graph_writer=GraphWriter(uri="bolt://localhost:7687", user="neo4j", password="password"),
            olap_writer=OlapWriter(host="localhost", port=8123),
            outbox_path="kg_outbox.sqlite3",
        )

    pipeline = EntityRelationPipeline(
        resolver=resolver,
        relation_extractor=relation_resolver,
        persistence=persistence if PERSISTENCE_AVAILABLE else None,
    )

    text = "Benyamin Bahadori is a retired Iranian pop singer from Tehran. He released his first album '85' in 2006."
    clean_df, review_df, relations_df = pipeline.run(document=text, doc_id="doc_001", owner_id="user_123")

    print(pipeline.last_report.summary() if pipeline.last_report else "not persisted")
    persistence.close()
