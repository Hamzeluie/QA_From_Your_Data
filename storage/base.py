import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Import domain objects from shared (cross-cutting)
from shared.data_classes import CandidateResult


class AbstractEntityStore(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def upsert_entity(
        self,
        canonical: str,
        label: str,
        aliases: List[str],
        summary: Optional[str] = None,
        source: str = "unknown",
        related_to: Optional[List[str]] = None,
        context_indicators: Optional[List[str]] = None,
        embedding_id: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def find_by_alias(self, normalized_alias: str) -> Optional[Dict]: ...

    @abstractmethod
    def find_by_canonical(self, canonical: str) -> Optional[Dict]: ...

    @abstractmethod
    def find_candidates_cypher(
        self, entity_text: str, entity_label: Optional[str] = None
    ) -> List[CandidateResult]: ...

    @abstractmethod
    def get_related(self, canonical: str) -> List[str]: ...

    @abstractmethod
    def merge_entities(self, absorb: str, keep: str) -> None: ...

    @abstractmethod
    def create_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        doc_id: str,
        confidence: float,
        provisional: bool,
        relation_id: str,
        evidence: List[str],
        chunk_id: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def create_outbox_event(self, event: Any) -> None: ...  # Any = OutboxEvent

    @abstractmethod
    def get_pending_outbox(self, limit: int = 100) -> List[Any]: ...

    @abstractmethod
    def mark_outbox_processed(self, event_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class AbstractVectorStore(ABC):
    """Qdrant abstraction for semantic search."""

    @abstractmethod
    def init_collections(self) -> None:
        """Create collections if missing."""

    @abstractmethod
    def upsert_entity_summary(
        self, canonical: str, vector: List[float], payload: Dict[str, Any]
    ) -> None:
        """Store entity summary embedding."""

    @abstractmethod
    def search_similar_entities(
        self, vector: List[float], top_k: int = 10, filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Return list of {canonical, score, payload}.
        Used by CandidateFinder (neural re-rank) and CrossDocResolver.
        """

    @abstractmethod
    def fetch_all(self, collection_name: str) -> List[Dict]:
        """Return all points with vectors. Used by EntityMerger."""

    @abstractmethod
    def delete_entity(self, canonical: str) -> None:
        """Remove from vector index (e.g., after merge)."""


class AbstractTextSearch(ABC):
    """Elasticsearch abstraction for fuzzy alias/summary search."""

    @abstractmethod
    def init_index(self) -> None:
        """Create index with mappings if missing."""

    @abstractmethod
    def index_entity(
        self,
        canonical: str,
        aliases: List[str],
        summary: str,
        context_indicators: List[str],
        related_to: List[str],
        label: str,
    ) -> None:
        """Upsert document to ES."""

    @abstractmethod
    def search_aliases(self, query: str, top_k: int = 10) -> List[Dict]:
        """Fuzzy multi-match over aliases, canonical, summary."""

    @abstractmethod
    def remove_entity(self, canonical: str) -> None:
        """Delete from ES (e.g., after merge)."""


class AbstractStateStore(ABC):
    """ClickHouse abstraction for pipeline state, queue, audit, relations."""

    @abstractmethod
    def init_tables(self) -> None:
        """CREATE TABLE IF NOT EXISTS all tables."""

    @abstractmethod
    def create_document(self, doc_id: str, owner_id: str) -> None:
        """Insert initial row with status='uploaded'."""

    @abstractmethod
    def transition_state(
        self, doc_id: str, expected: Optional[str], to_state: str, error_message: Optional[str] = None
    ) -> bool:
        """
        Optimistic state transition using version counter.
        Returns True if successful.
        """

    @abstractmethod
    def enqueue_unresolved(
        self,
        resolution_id: str,
        doc_id: str,
        entity_row: Any,  # ResolvedEntity or dict
        candidates_json: str,
    ) -> None:
        """Add to review queue."""

    @abstractmethod
    def resolve_unresolved(
        self, resolution_id: str, canonical: str, user_id: str
    ) -> None:
        """Mark queue item resolved and set canonical."""

    @abstractmethod
    def log_mention(
        self,
        doc_id: str,
        chunk_id: Optional[str],
        canonical_name: str,
        original_text: str,
        entity_label: str,
        mention_sentence: str,
        start: int,
        end: int,
        confidence: float,
        source: str,
    ) -> None:
        """Audit trail of every mention."""

    @abstractmethod
    def insert_relations(self, relations: List[Any]) -> None:
        """Batch insert to ClickHouse relations table."""

    @abstractmethod
    def get_documents_by_state(self, status: str, limit: int = 100) -> List[Dict]:
        """For backfill / reprocessing jobs."""


class AbstractCache(ABC):
    """Redis abstraction: alias cache, candidate cache, distributed locks, pub/sub."""

    @abstractmethod
    def get_alias(self, normalized_alias: str) -> Optional[str]:
        """Return canonical name or None."""

    @abstractmethod
    def set_alias(self, normalized_alias: str, canonical: str, ttl: int = 300) -> None:
        pass

    @abstractmethod
    def invalidate_alias(self, normalized_alias: str) -> None:
        pass

    @abstractmethod
    def invalidate_candidates(self, pattern: str = "candidates:*") -> None:
        pass

    @abstractmethod
    def get_entity(self, canonical: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set_entity(self, canonical: str, data: Dict, ttl: int = 300) -> None:
        pass

    @abstractmethod
    def acquire_lock(self, name: str, timeout: int = 60, blocking: bool = False) -> bool:
        pass

    @abstractmethod
    def release_lock(self, name: str) -> None:
        pass

    @abstractmethod
    def set_job_status(self, doc_id: str, status: str, meta: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def get_job_status(self, doc_id: str) -> Optional[Dict]:
        pass
    
    
    