import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from typing import List, Tuple, Optional
import pandas as pd

from storage.factory import StorageFactory
from ingestion.unified_resolver import UnifiedEntityResolver
from ingestion.relation_pipeline import RelationPipeline
from shared.data_classes import Chunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    High-level orchestrator for the full ingestion flow:
    NER → Entity Resolution → Relation Extraction → Chunk Indexing.
    Stateless; all side effects go through StorageFactory.
    """

    def __init__(
        self,
        relation_extractor,
        factory: Optional[StorageFactory] = None,
        embedder=None,
    ):
        self.factory = factory or StorageFactory.from_env()
        self.embedder = embedder or self._load_embedder()

        # Ensure backends are ready
        self.factory.init_all()

        self.resolver = UnifiedEntityResolver(
            factory=self.factory,
            embedder=self.embedder,
        )
        self.relation_pipeline = RelationPipeline(
            extractor=relation_extractor,
            factory=self.factory,
        )

    def run(
        self,
        doc_id: str,
        raw_text: str,
        owner_id: str,
    ) -> dict:
        """
        Synchronous pipeline run. Returns result manifest.
        For async production use, call this from a Celery task.
        """
        # 1. Document state: uploaded
        existing = self.factory.clickhouse.get_document_state(doc_id)
        if existing and existing.get("status") == "indexed":
            logger.info(f"Document {doc_id} already indexed; skipping.")
            return {"status": "skipped", "reason": "already_indexed", "doc_id": doc_id}

        if not existing:
            self.factory.clickhouse.create_document(doc_id, owner_id)

        # 2. Entity Resolution (NER + Coref + ER + Cross-doc + Merge)
        chunk_info, clean_df, review_df = self.resolver.process_document(
            doc_id, raw_text, owner_id
        )

        # 3. Relation Extraction (only on resolved)
        rel_df = pd.DataFrame()
        if not clean_df.empty:
            rel_df = self.relation_pipeline.extract_and_store(clean_df)

        # 4. Index chunks for hybrid search (ES + Qdrant)
        self._index_chunks(chunk_info)

        # 5. Final state
        self.factory.clickhouse.transition_state(doc_id, "re_done", "indexed")

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_indexed": len(chunk_info),
            "resolved_count": len(clean_df),
            "review_count": len(review_df),
            "relations_count": len(rel_df),
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        from config.settings import settings
        model_path = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_path)

    def _index_chunks(self, chunks: List[Chunk]) -> None:
        """Dense (Qdrant) + sparse (ES) indexing of text chunks."""
        if not chunks:
            return

        texts = [c.sentence for c in chunks]
        try:
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.error(f"Chunk embedding failed: {exc}")
            return

        for chunk, vec in zip(chunks, embeddings):
            # Qdrant dense vector
            try:
                self.factory.qdrant.upsert_chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=vec.tolist(),
                    text=chunk.sentence,
                    owner_id=chunk.owner_id,
                )
            except Exception as exc:
                logger.warning(f"Qdrant chunk index failed {chunk.chunk_id}: {exc}")

            # Elasticsearch sparse index
            try:
                self.factory.es.index_chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.sentence,
                    owner_id=chunk.owner_id,
                )
            except Exception as exc:
                logger.warning(f"ES chunk index failed {chunk.chunk_id}: {exc}")