"""
Shared fixtures for ingestion pipeline tests.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import sys
import uuid
import socket
import logging
from typing import List, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Service health checks ──────────────────────────────────────────────────
def _is_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

NEO4J_UP = _is_reachable("localhost", 7687)
POSTGRES_UP = _is_reachable("localhost", 5432)
QDRANT_UP = _is_reachable("localhost", 6333)
ES_UP = _is_reachable("localhost", 9200)
REDIS_UP = _is_reachable("localhost", 6379)

# ── Auto-skip integration tests when services are down ──────────────────────
def pytest_collection_modifyitems(config, items):
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "neo4j" in markers and not NEO4J_UP:
            item.add_marker(pytest.mark.skip(reason="Neo4j not reachable on localhost:7687"))
        if "postgres" in markers and not POSTGRES_UP:
            item.add_marker(pytest.mark.skip(reason="Postgres not reachable on localhost:5432"))
        if "qdrant" in markers and not QDRANT_UP:
            item.add_marker(pytest.mark.skip(reason="Qdrant not reachable on localhost:6333"))
        if "elasticsearch" in markers and not ES_UP:
            item.add_marker(pytest.mark.skip(reason="Elasticsearch not reachable on localhost:9200"))
        if "redis" in markers and not REDIS_UP:
            item.add_marker(pytest.mark.skip(reason="Redis not reachable on localhost:6379"))
        if "integration" in markers:
            # If any integration marker is present but no specific service marker,
            # still require at least one service to be up to avoid false passes
            if not any([NEO4J_UP, POSTGRES_UP, QDRANT_UP, ES_UP, REDIS_UP]):
                item.add_marker(pytest.mark.skip(reason="No Docker services reachable"))

# ── Unique ID per test ─────────────────────────────────────────────────────
@pytest.fixture
def tid() -> str:
    return f"t_{uuid.uuid4().hex[:8]}"

# ── Mock Embedder (fast, deterministic) ────────────────────────────────────
class MockEmbedder:
    vector_size: int = 384

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        vecs = []
        for s in sentences:
            seed = hash(s) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.vector_size).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        arr = np.stack(vecs)
        return arr if convert_to_numpy else arr.tolist()

@pytest.fixture
def mock_embedder():
    return MockEmbedder()

# ── Real Embedder (slow, optional) ─────────────────────────────────────────
@pytest.fixture
def real_embedder():
    from sentence_transformers import SentenceTransformer
    path = settings.EMBEDDING_MODEL_PATH
    if not os.path.isdir(path):
        pytest.skip(f"Embedder not found at {path}")
    return SentenceTransformer(path)

# ── Storage Fixtures ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def neo4j_store():
    if not NEO4J_UP:
        pytest.skip("Neo4j unavailable")
    from storage.neo4j_store import Neo4jEntityStore
    store = Neo4jEntityStore(
        uri=settings.GRAPH_DB_URL,
        user=settings.GRAPH_DB_USER,
        password=settings.GRAPH_DB_PASSWORD,
        database="neo4j",
    )
    store.init_schema()
    yield store
    with store._driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
    store.close()


@pytest.fixture(scope="session")
def postgres_store():
    if not POSTGRES_UP:
        pytest.skip("Postgres unavailable")
    from storage.postgres_store import PostgresStateStore
    store = PostgresStateStore(
        host=settings.POSTGRES_HOST,
        port=int(settings.POSTGRES_PORT),
        username=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB,
    )
    store.init_tables()
    yield store
    # Clean-up
    conn = store._get_conn()
    with conn.cursor() as cur:
        for tbl in ["document_state", "unresolved_queue", "mention_log", "relations"]:
            cur.execute(f"TRUNCATE TABLE {tbl} RESTART IDENTITY CASCADE")
    conn.commit()
    store.close()

@pytest.fixture(scope="session")
def qdrant_store():
    if not QDRANT_UP:
        pytest.skip("Qdrant unavailable")
    from storage.qdrant_store import QdrantVectorStore
    store = QdrantVectorStore(host="localhost", port=6333, vector_size=MockEmbedder.vector_size)
    store.init_collections()
    yield store
    for col in ["entity_summaries", "chunk_embeddings"]:
        if store.client.collection_exists(col):
            store.client.delete_collection(col)
    store.init_collections()

@pytest.fixture(scope="session")
def es_store():
    if not ES_UP:
        pytest.skip("Elasticsearch unavailable")
    from storage.elasticsearch_store import ElasticsearchEntitySearch
    store = ElasticsearchEntitySearch(hosts=[settings.ELASTIC_SEARCH_HOST])
    store.init_index()
    store.init_chunks_index()
    yield store
    if store.client.indices.exists(index=store.INDEX_NAME):
        store.client.delete_by_query(index=store.INDEX_NAME, body={"query": {"match_all": {}}}, refresh=True)
    if store.client.indices.exists(index=store.CHUNKS_INDEX):
        store.client.delete_by_query(index=store.CHUNKS_INDEX, body={"query": {"match_all": {}}}, refresh=True)

@pytest.fixture(scope="session")
def redis_cache():
    if not REDIS_UP:
        pytest.skip("Redis unavailable")
    from storage.redis_cache import RedisCache
    cache = RedisCache(host="localhost", port=6379, db=15)
    yield cache
    for key in cache.client.scan_iter(match="*"):
        cache.client.delete(key)

@pytest.fixture
def factory(mock_embedder, neo4j_store, postgres_store, qdrant_store, es_store, redis_cache):
    from storage.factory import StorageFactory
    f = MagicMock(spec=StorageFactory)
    f.neo4j = neo4j_store
    f.postgres = postgres_store
    f.qdrant = qdrant_store
    f.es = es_store
    f.redis = redis_cache
    f.embedder = mock_embedder
    return f