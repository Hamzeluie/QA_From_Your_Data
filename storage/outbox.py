import json
import logging
from typing import Optional
from celery import shared_task
from storage.factory import StorageFactory

logger = logging.getLogger(__name__)


class OutboxPoller:
    """
    Side-effect broker. Reads pending OutboxEvent nodes from Neo4j
    and reliably projects them to ES, Qdrant, and Redis.
    """

    def __init__(self, factory: Optional[StorageFactory] = None):
        self.factory = factory or StorageFactory.from_env()

    def process_batch(self, limit: int = 100) -> int:
        events = self.factory.neo4j.get_pending_outbox(limit=limit)
        processed = 0

        for evt in events:
            try:
                self._handle_event(evt)
                self.factory.neo4j.mark_outbox_processed(evt.event_id)
                processed += 1
            except Exception as exc:
                logger.error(f"Outbox failed for {evt.event_id}: {exc}")
                self.factory.neo4j.increment_outbox_attempt(evt.event_id, str(exc))
                # Dead-letter after 5 attempts
                if evt.attempts >= 4:
                    logger.critical(f"Outbox dead-letter: {evt.event_id}")
        return processed

    def _handle_event(self, evt):
        payload = json.loads(evt.payload_json)

        if evt.event_type == "entity_upserted":
            canonical = payload["canonical"]
            label = payload["label"]
            aliases = payload["aliases"]
            summary = payload.get("summary", "")
            context_indicators = payload.get("context_indicators", [])

            # 1. Elasticsearch
            if "es" in evt.target_stores:
                self.factory.es.index_entity(
                    canonical=canonical,
                    aliases=aliases,
                    summary=summary,
                    context_indicators=context_indicators,
                    related_to=payload.get("related_to", []),
                    label=label,
                )

            # 2. Qdrant (embedding)
            if "qdrant" in evt.target_stores and self.factory.embedder:
                emb = self.factory.embedder.encode([summary])[0].tolist()
                self.factory.qdrant.upsert_entity_summary(
                    canonical, emb,
                    {"label": label, "aliases": aliases, "source": payload.get("source")}
                )

            # 3. Redis invalidation
            if "redis" in evt.target_stores:
                for alias in aliases:
                    self.factory.redis.invalidate_alias(alias.lower())
                self.factory.redis.invalidate_candidates()

        elif evt.event_type == "entity_merged":
            absorb = payload["absorb"]
            keep = payload["keep"]
            if "es" in evt.target_stores:
                self.factory.es.remove_entity(absorb)
            if "qdrant" in evt.target_stores:
                self.factory.qdrant.delete_entity(absorb)
            if "redis" in evt.target_stores:
                self.factory.redis.invalidate_alias(absorb.lower())


# Celery beat task (runs every 30 seconds)
@shared_task(bind=True, max_retries=3)
def run_outbox_poll(self, limit: int = 100):
    poller = OutboxPoller()
    count = poller.process_batch(limit=limit)
    if count:
        logger.info(f"Outbox processed {count} events")
    return count