
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
    if name == "ClickHouseStateStore":
        from storage.clickhouse_store import ClickHouseStateStore
        return ClickHouseStateStore
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