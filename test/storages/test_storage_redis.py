import pytest

pytestmark = [pytest.mark.integration, pytest.mark.redis]


class TestRedisCache:
    def test_alias_roundtrip(self, redis_cache, tid):
        redis_cache.set_alias(f"alias_{tid}", f"Canon_{tid}", ttl=300)
        assert redis_cache.get_alias(f"alias_{tid}") == f"Canon_{tid}"

    def test_invalidate_alias(self, redis_cache, tid):
        redis_cache.set_alias(f"inv_{tid}", f"C_{tid}")
        redis_cache.invalidate_alias(f"inv_{tid}")
        assert redis_cache.get_alias(f"inv_{tid}") is None

    def test_entity_roundtrip(self, redis_cache, tid):
        redis_cache.set_entity(f"E_{tid}", {"label": "PER"}, ttl=300)
        assert redis_cache.get_entity(f"E_{tid}")["label"] == "PER"

    def test_lock_acquire_release(self, redis_cache, tid):
        acquired = redis_cache.acquire_lock(f"lock_{tid}", timeout=10, blocking=False)
        assert acquired is True
        redis_cache.release_lock(f"lock_{tid}")

    def test_job_status(self, redis_cache, tid):
        redis_cache.set_job_status(tid, "running", {"worker": "test"})
        status = redis_cache.get_job_status(tid)
        assert status["status"] == "running"