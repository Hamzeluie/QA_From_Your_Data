from .base import *
from .neo4j_store import *
from .postgres_store import *
from .qdrant_store import *
from .elasticsearch_store import *
from .redis_cache import *
from .outbox import *
from .factory import *

def __getattr__(name):
    if name == "AbstractEntityStore":
        from storage.base import AbstractEntityStore
        return AbstractEntityStore
    if name == "AbstractVectorStore":
        from storage.base import AbstractVectorStore
        return AbstractVectorStore
    if name == "AbstractTextSearch":
        from storage.base import AbstractTextSearch
        return AbstractTextSearch
    if name == "AbstractStateStore":
        from storage.base import AbstractStateStore
        return AbstractStateStore
    if name == "AbstractCache":
        from storage.base import AbstractCache
        return AbstractCache
    if name == "Neo4jEntityStore":
        from storage.neo4j_store import Neo4jEntityStore
        return Neo4jEntityStore
    if name == "PostgresStateStore":
        from storage.postgres_store import PostgresStateStore
        return PostgresStateStore
    if name == "QdrantVectorStore":
        from storage.qdrant_store import QdrantVectorStore
        return QdrantVectorStore
    if name == "ElasticsearchEntitySearch":
        from storage.elasticsearch_store import ElasticsearchEntitySearch
        return ElasticsearchEntitySearch
    if name == "RedisCache":
        from storage.redis_cache import RedisCache
        return RedisCache
    if name == "OutboxPoller":
        from storage.outbox import OutboxPoller
        return OutboxPoller
    if name == "StorageFactory":
        from storage.factory import StorageFactory
        return StorageFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")