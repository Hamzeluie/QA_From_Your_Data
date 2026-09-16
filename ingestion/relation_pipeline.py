import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import hashlib
import logging
from typing import List, Optional
from dataclasses import fields

from ingestion.models.base import IExtractor
from storage.factory import StorageFactory
from storage.data_classes import Relation, RelationLabels, EntityLabels, MentionEntity, Chunk

logger = logging.getLogger(__name__)


class RelationPipeline:
    """
    Production relation extraction.
    Groups resolved entities by sentence, calls LLM extractor,
    dedups symmetric predicates, and persists to Neo4j + postgres.
    """

    def __init__(
        self,
        extractor:IExtractor,
        factory: Optional[StorageFactory] = None,
        include_literals: bool = True,
    ):
        self.extractor = extractor
        self.factory = factory or StorageFactory.from_env()
        self.include_literals = include_literals

    def extract_and_store(self, entities: List[MentionEntity], chunks:List[Chunk]) -> List[Relation]:
        """
        Main entrypoint. Takes the RESOLVED-only DataFrame from
        UnifiedEntityResolver.process_document().
        """
        if not entities:
            return []

        relations: List[Relation] = []
        for chunk in chunks:
            chunk_entities = [ent for ent in entities if ent.chunk_id==chunk.chunk_id and ent.doc_id==chunk.doc_id]
            relations.extend(self.extractor(chunk.sentence, chunk_entities, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id))
        
        self._persist(relations)
        return relations

    # ── Persistence ─────────────────────────────────────────────────────────

    def _persist(self, relations: List[Relation]) -> None:
        # Neo4j graph edges (one by one; could batch with UNWIND in future)
        for rel in relations:
            try:
                self.factory.neo4j.create_relation(relation=rel)
            except Exception as exc:
                logger.warning(f"Neo4j relation write failed {rel.relation_id}: {exc}")
        try:
            self.factory.postgres.insert_relations(relations)
        except Exception as exc:
            logger.error(f"ClickHouse relation batch failed: {exc}")

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_relation_id(doc_id: str, subject: str, predicate: str, obj: str) -> str:
        key = f"{doc_id}|{subject}|{predicate}|{obj}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()
    
    
    