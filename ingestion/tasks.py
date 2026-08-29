import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from celery import shared_task
from celery.exceptions import MaxRetriesExceededError

from storage.factory import StorageFactory
from storage.outbox import OutboxPoller
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def _get_relation_extractor():
    """
    Adjust this import to your actual RelationExtractor location.
    Example: from ingestion.llm.relation_extractor import RelationExtractor
    """
    try:
        from ingestion.llm.llm_extractors import RelationExtractor
        return RelationExtractor(use_cot=True)
    except ImportError:
        # Fallback stub so Celery doesn't crash on import if you haven't created it yet
        class _StubExtractor:
            def __call__(self, text, entities):
                return []
        logger.warning("RelationExtractor not found; using stub. Adjust _get_relation_extractor().")
        return _StubExtractor()


@shared_task(bind=True, max_retries=3)
def ingest_document_task(self, doc_id: str, raw_text: str, owner_id: str):
    """
    Main ingestion pipeline task.
    Idempotent: safe to retry or replay.
    """
    pipeline = IngestionPipeline(relation_extractor=_get_relation_extractor())

    try:
        result = pipeline.run(doc_id=doc_id, raw_text=raw_text, owner_id=owner_id)
        return result

    except Exception as exc:
        # Mark failed for observability
        try:
            factory = StorageFactory.from_env()
            factory.postgres.transition_state(
                doc_id, None, "failed", error_message=str(exc)
            )
        except Exception as inner:
            logger.error(f"Failed to mark failure state for {doc_id}: {inner}")

        # Exponential backoff: 60s, 120s, 180s
        countdown = 60 * (self.request.retries + 1)
        try:
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            logger.critical(f"Document {doc_id} ingestion failed permanently.")
            raise


@shared_task
def poll_outbox_task(limit: int = 100):
    """
    Celery beat task (schedule every 30s).
    Fans out Neo4j outbox events to ES/Qdrant/Redis.
    """
    poller = OutboxPoller()
    count = poller.process_batch(limit=limit)
    if count:
        logger.info(f"Outbox poll processed {count} events")
    return {"processed": count}


@shared_task(bind=True, max_retries=2)
def backfill_document_task(self, doc_id: str, raw_text: str):
    """
    Re-run full pipeline on a document after operator resolves entities.
    Call this when an unresolved entity gets a canonical mapping.
    """
    pipeline = IngestionPipeline(relation_extractor=_get_relation_extractor())
    try:
        # Force reset state to re-done so pipeline can re-index
        factory = StorageFactory.from_env()
        factory.postgres.transition_state(doc_id, None, "er_done")
        result = pipeline.run(doc_id=doc_id, raw_text=raw_text, owner_id="backfill")
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)


@shared_task
def merge_entities_task():
    """
    Periodic task (e.g., nightly or every 5 min) to deduplicate catalog.
    Uses Redis distributed lock so only one worker runs at a time.
    """
    from ingestion.unified_resolver import UnifiedEntityResolver
    resolver = UnifiedEntityResolver()
    resolver._merge_catalog()
    return {"status": "merge_attempted"}