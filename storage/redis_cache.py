import json
import logging
from typing import Dict, Optional
import redis
from redis.lock import Lock

from storage.base import AbstractCache

logger = logging.getLogger(__name__)


class RedisCache(AbstractCache):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, decode_responses: bool = True):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=decode_responses)

    # ── Alias Cache ───────────────────────────────────────────────────────────

    def get_alias(self, normalized_alias: str) -> Optional[str]:
        return self.client.get(f"alias:{normalized_alias}")

    def set_alias(self, normalized_alias: str, canonical: str, ttl: int = 300) -> None:
        self.client.set(f"alias:{normalized_alias}", canonical, ex=ttl)


    def invalidate_alias(self, normalized_alias: str) -> None:
        self.client.delete(f"alias:{normalized_alias}")

    def invalidate_candidates(self, pattern: str = "candidates:*") -> None:
        """Scan-delete all candidate caches. Use carefully."""
        for key in self.client.scan_iter(match=pattern):
            self.client.delete(key)

    # ── Entity Cache ─────────────────────────────────────────────────────────

    def get_entity(self, canonical: str) -> Optional[Dict]:
        raw = self.client.get(f"entity:{canonical.lower()}")
        return json.loads(raw) if raw else None

    def set_entity(self, canonical: str, data: Dict, ttl: int = 300) -> None:
        self.client.set(f"entity:{canonical.lower()}", json.dumps(data), ex=ttl)


    # ── Distributed Locks ─────────────────────────────────────────────────────

    def acquire_lock(self, name: str, timeout: int = 60, blocking: bool = False) -> bool:
        lock = self.client.lock(f"lock:{name}", timeout=timeout, thread_local=False)
        acquired = lock.acquire(blocking=blocking)
        if acquired:
            self.client.set(f"lock_meta:{name}", "1", ex=timeout + 10)
        return acquired

    def release_lock(self, name: str) -> None:
        # Best-effort release using redis-py lock reconstitution
        lock = Lock(self.client, f"lock:{name}", thread_local=False)
        try:
            lock.release()
        except Exception:
            pass
        self.client.delete(f"lock_meta:{name}")

    # ── Job Status ────────────────────────────────────────────────────────────

    def set_job_status(self, doc_id: str, status: str, meta: Optional[Dict] = None) -> None:
        payload = {"status": status, "meta": meta or {}}
        self.client.set(f"job:{doc_id}", json.dumps(payload), ex=86400)
    
    def get_job_status(self, doc_id: str) -> Optional[Dict]:
        raw = self.client.get(f"job:{doc_id}")
        return json.loads(raw) if raw else None
    
    
    