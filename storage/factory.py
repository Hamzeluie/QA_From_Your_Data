import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
from typing import Optional
from storage.neo4j_store import Neo4jEntityStore
from storage.postgres_store import PostgresStateStore
from storage.qdrant_store import QdrantVectorStore
from storage.elasticsearch_store import ElasticsearchEntitySearch
from storage.redis_cache import RedisCache


class StorageFactory:
    """
    Centralized wiring of all storage backends.
    Reads from environment variables with sensible defaults.
    """

    @staticmethod
    def from_env() -> "StorageFactory":
        return StorageFactory(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_user=os.getenv("POSTGRES_USER", "kg_user"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "kg_pass"),
            postgres_database=os.getenv("POSTGRES_DB", "knowledge_graph"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            es_hosts=os.getenv("ES_HOSTS", "http://localhost:9200").split(","),
            es_user=os.getenv("ES_USER"),
            es_password=os.getenv("ES_PASSWORD"),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            vector_size=int(os.getenv("VECTOR_SIZE", "384")),
        )

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str,
        postgres_host: str,
        postgres_port: int,
        postgres_user: str,
        postgres_password: str,
        postgres_database: str,
        qdrant_host: str,
        qdrant_port: int,
        es_hosts: list,
        es_user: Optional[str],
        es_password: Optional[str],
        redis_host: str,
        redis_port: int,
        vector_size: int = 384,
        embedder=None
    ):
        self._neo4j = None
        self._postgres = None
        self._qdrant = None
        self._es = None
        self._redis = None
        self.embedder = embedder


        self._neo4j_cfg = {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_password,
            "database": neo4j_database,
        }
        self._pg_cfg = {
            "host": postgres_host,
            "port": postgres_port,
            "username": postgres_user,
            "password": postgres_password,
            "database": postgres_database,
        }
        self._qdrant_cfg = {
            "host": qdrant_host,
            "port": qdrant_port,
            "vector_size": vector_size,
        }
        self._es_cfg = {
            "hosts": es_hosts,
            "username": es_user,
            "password": es_password,
        }
        self._redis_cfg = {
            "host": redis_host,
            "port": redis_port,
        }

    # ── Lazy singletons ───────────────────────────────────────────────────────

    @property
    def neo4j(self) -> Neo4jEntityStore:
        if self._neo4j is None:
            self._neo4j = Neo4jEntityStore(**self._neo4j_cfg)
        return self._neo4j

    @property
    def postgres(self) -> PostgresStateStore:
        if self._postgres is None:
            self._postgres = PostgresStateStore(**self._pg_cfg)
        return self._postgres
   
    @property
    def qdrant(self) -> QdrantVectorStore:
        if self._qdrant is None:
            self._qdrant = QdrantVectorStore(**self._qdrant_cfg)
        return self._qdrant

    @property
    def es(self) -> ElasticsearchEntitySearch:
        if self._es is None:
            self._es = ElasticsearchEntitySearch(
                self._es_cfg["hosts"],
                self._es_cfg.get("username"),
                self._es_cfg.get("password"),
            )
        return self._es

    @property
    def redis(self) -> RedisCache:
        if self._redis is None:
            self._redis = RedisCache(**self._redis_cfg)
        return self._redis

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def init_all(self) -> None:
        """Idempotent initialization of schema/indexes/collections."""
        self.neo4j.init_schema()
        self.postgres.init_tables()
        self.qdrant.init_collections()
        self.es.init_index()
        self.es.init_chunks_index()
        # Redis needs no schema init

    def close_all(self) -> None:
        if self._neo4j:
            self._neo4j.close()