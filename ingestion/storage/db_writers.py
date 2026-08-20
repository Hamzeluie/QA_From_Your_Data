"""
db_writers.py
=============

One writer class per storage engine. Each writer:
  - owns exactly one DB client/connection
  - exposes upsert_* methods that are IDEMPOTENT (safe to call twice with the
    same data — this is what lets PersistenceCoordinator retry on failure
    without worrying about duplicates)
  - never talks to the other writers; PersistenceCoordinator is the only
    thing that knows all four exist

Design note on "upsert":
  - Neo4j:        MERGE keyed on canonical_name (entities) / relation_id (edges)
  - Qdrant:       point id = deterministic UUID5 of the natural key; upsert
                   overwrites the point with that id
  - Elasticsearch: doc id = deterministic id of the natural key; index with
                   an explicit _id is an upsert (bulk `index` op_type)
  - ClickHouse:   has no native UPDATE/upsert. We use a ReplacingMergeTree
                   (versioned by an ingested_at column) and always INSERT a
                   new version. Reads should use FINAL or argMax(...) to get
                   the latest row. This is the standard ClickHouse pattern
                   for "upsert-like" semantics.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import asdict
from typing import List, Dict, Optional, Any

from shared.data_classes import Chunk, Relation, ResolvedEntity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def stable_uuid(*parts: str) -> str:
    """Deterministic UUID5 from arbitrary string parts — used as point IDs
    in Qdrant and doc IDs in Elasticsearch so re-ingesting the same entity
    or chunk overwrites rather than duplicates."""
    key = "::".join(str(p) for p in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

 
def sanitize_neo4j_type(predicate: str) -> str:
    """Neo4j relationship types can't be parameterized in Cypher — they have
    to be interpolated into the query string, so we whitelist characters
    defensively even though RelationExtractor already normalizes predicates
    to snake_case."""
    cleaned = "".join(c for c in predicate.upper() if c.isalnum() or c == "_")
    return cleaned or "RELATED_TO"


def sanitize_neo4j_label(label: str) -> str:
    cleaned = "".join(c for c in label.upper() if c.isalnum() or c == "_")
    return cleaned or "ENTITY"


# --------------------------------------------------------------------------
# Elasticsearch — lexical half of hybrid search
# --------------------------------------------------------------------------

class SearchIndexWriter:
    """Elasticsearch. Owns two indices: chunks (full text of sentences /
    chunks, for lexical retrieval) and entities (canonical name + aliases +
    summary, for entity lookup / autocomplete)."""

    def __init__(self,
                 hosts: List[str],
                 index_prefix: str = "kg",
                 api_key: Optional[str] = None,
                 basic_auth: Optional[tuple] = None,
                 request_timeout: int = 30):
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch.helpers import bulk
        except ImportError as e:
            raise ImportError(
                "pip install elasticsearch"
            ) from e

        self._bulk = bulk
        self.client = Elasticsearch(
            hosts,
            api_key=api_key,
            basic_auth=basic_auth,
            request_timeout=request_timeout,
        )
        self.chunks_index = f"{index_prefix}_chunks"
        self.entities_index = f"{index_prefix}_entities"
        self._ensure_indices()

    def _ensure_indices(self) -> None:
        chunk_mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "owner_id": {"type": "keyword"},
                    "sentence": {"type": "text"},
                    "date_time": {"type": "date", "ignore_malformed": True},
                }
            }
        }
        entity_mapping = {
            "mappings": {
                "properties": {
                    "canonical_name": {"type": "keyword"},
                    "entity_label": {"type": "keyword"},
                    "original_text": {"type": "text"},
                    "summary": {"type": "text"},
                    "doc_id": {"type": "keyword"},
                    "confidence": {"type": "float"},
                }
            }
        }
        for name, body in [(self.chunks_index, chunk_mapping), (self.entities_index, entity_mapping)]:
            if not self.client.indices.exists(index=name):
                self.client.indices.create(index=name, body=body)

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        actions = [
            {
                "_op_type": "index",  # index with explicit _id == upsert
                "_index": self.chunks_index,
                "_id": stable_uuid(c.doc_id, c.chunk_id),
                "_source": asdict(c),
            }
            for c in chunks
        ]
        success, errors = self._bulk(self.client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(f"Elasticsearch chunk upsert had {len(errors)} failures: {errors[:3]}")

    def upsert_entities(self, entities: List[ResolvedEntity], doc_id: str) -> None:
        if not entities:
            return
        actions = []
        for e in entities:
            source = e.to_dict()
            source["doc_id"] = doc_id
            actions.append({
                "_op_type": "index",
                "_index": self.entities_index,
                "_id": stable_uuid(e.canonical_name),
                "_source": source,
            })
        success, errors = self._bulk(self.client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(f"Elasticsearch entity upsert had {len(errors)} failures: {errors[:3]}")

    def close(self) -> None:
        self.client.close()


# --------------------------------------------------------------------------
# Qdrant — semantic half of hybrid search
# --------------------------------------------------------------------------

class VectorStoreWriter:
    """Qdrant. Embeds chunk text (and optionally entity summaries) and
    upserts them as points keyed by a deterministic UUID, so re-processing
    a document overwrites rather than duplicates."""

    def __init__(self,
                 url: str,
                 embedder,
                 collection: str = "kg_chunks",
                 entity_collection: str = "kg_entities",
                 vector_size: Optional[int] = None,
                 api_key: Optional[str] = None,
                 distance: str = "Cosine"):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as e:
            raise ImportError("pip install qdrant-client") from e

        self._PointStruct = PointStruct
        self.client = QdrantClient(url=url, api_key=api_key)
        self.embedder = embedder
        self.collection = collection
        self.entity_collection = entity_collection

        vector_size = vector_size or self.embedder.get_sentence_embedding_dimension()
        dist = getattr(Distance, distance.upper(), Distance.COSINE)
        for coll in (self.collection, self.entity_collection):
            if not self.client.collection_exists(coll):
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=vector_size, distance=dist),
                )

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.encode([c.sentence for c in chunks], convert_to_numpy=True)
        points = [
            self._PointStruct(
                id=stable_uuid(c.doc_id, c.chunk_id),
                vector=vec.tolist(),
                payload=asdict(c),
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def upsert_entities(self, entities: List[ResolvedEntity]) -> None:
        entities = [e for e in entities if e.summary]
        if not entities:
            return
        vectors = self.embedder.encode([e.summary for e in entities], convert_to_numpy=True)
        points = [
            self._PointStruct(
                id=stable_uuid(e.canonical_name),
                vector=vec.tolist(),
                payload=e.to_dict(),
            )
            for e, vec in zip(entities, vectors)
        ]
        self.client.upsert(collection_name=self.entity_collection, points=points)

    def close(self) -> None:
        self.client.close()


# --------------------------------------------------------------------------
# Neo4j — graph traversal
# --------------------------------------------------------------------------

class GraphWriter:
    """Neo4j. Entities become `(:Entity:<LABEL> {canonical_name, ...})`
    nodes; relations become typed edges. Everything is MERGE-based, so
    calling upsert_entities/upsert_relations twice with the same data is a
    no-op the second time."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError("pip install neo4j") from e

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT entity_canonical_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE"
            )

    def upsert_entities(self, entities: List[ResolvedEntity], doc_id: str) -> None:
        if not entities:
            return
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            label = sanitize_neo4j_label(e.entity_label)
            by_label.setdefault(label, []).append({
                "canonical_name": e.canonical_name,
                "summary": e.summary,
                "confidence": e.confidence,
                "source": e.source,
                "doc_id": doc_id,
            })

        with self.driver.session(database=self.database) as session:
            for label, rows in by_label.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (e:Entity:{label} {{canonical_name: row.canonical_name}})
                SET e.summary = coalesce(row.summary, e.summary),
                    e.confidence = row.confidence,
                    e.source = row.source,
                    e.last_seen_doc = row.doc_id,
                    e.updated_at = timestamp()
                """
                session.execute_write(lambda tx, q=query, r=rows: tx.run(q, rows=r).consume())

    def upsert_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in relations:
            rtype = sanitize_neo4j_type(r.predicate)
            by_type.setdefault(rtype, []).append({
                "subject": r.subject,
                "object": r.object,
                "relation_id": r.relation_id or stable_uuid(r.doc_id, r.subject, r.predicate, r.object),
                "confidence": r.confidence,
                "doc_id": r.doc_id,
                "sentence": r.sentence,
                "source": r.source,
            })

        with self.driver.session(database=self.database) as session:
            for rtype, rows in by_type.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (s:Entity {{canonical_name: row.subject}})
                MERGE (o:Entity {{canonical_name: row.object}})
                MERGE (s)-[rel:{rtype} {{relation_id: row.relation_id}}]->(o)
                SET rel.confidence = row.confidence,
                    rel.doc_id = row.doc_id,
                    rel.sentence = row.sentence,
                    rel.source = row.source,
                    rel.updated_at = timestamp()
                """
                session.execute_write(lambda tx, q=query, r=rows: tx.run(q, rows=r).consume())

    def close(self) -> None:
        self.driver.close()


# --------------------------------------------------------------------------
# ClickHouse — OLAP / analytics over entities & relations
# --------------------------------------------------------------------------

class OlapWriter:
    """ClickHouse. Tables use ReplacingMergeTree(ingested_at) keyed on the
    natural id, so repeated inserts of the same row are collapsed on
    background merge (or immediately with FINAL / argMax at query time).
    ClickHouse has no real UPDATE, so "upsert" here just means "insert a
    new version and let ClickHouse deduplicate later"."""

    ENTITIES_DDL = """
    CREATE TABLE IF NOT EXISTS entities (
        canonical_name String,
        entity_label   String,
        original_text  String,
        confidence     Float32,
        source         String,
        summary        Nullable(String),
        doc_id         String,
        ingested_at    DateTime64(3) DEFAULT now64(3)
    ) ENGINE = ReplacingMergeTree(ingested_at)
    ORDER BY (canonical_name)
    """

    RELATIONS_DDL = """
    CREATE TABLE IF NOT EXISTS relations (
        relation_id    String,
        doc_id         String,
        subject        String,
        subject_label  String,
        predicate      String,
        object         String,
        object_label   String,
        confidence     Float32,
        source         String,
        sentence       String,
        ingested_at    DateTime64(3) DEFAULT now64(3)
    ) ENGINE = ReplacingMergeTree(ingested_at)
    ORDER BY (relation_id)
    """

    def __init__(self, host: str, port: int = 8123, username: str = "default",
                 password: str = "", database: str = "kg"):
        try:
            import clickhouse_connect
        except ImportError as e:
            raise ImportError("pip install clickhouse-connect") from e

        # database must exist before we can point a client at it
        bootstrap = clickhouse_connect.get_client(host=host, port=port, username=username, password=password)
        bootstrap.command(f"CREATE DATABASE IF NOT EXISTS {database}")

        self.client = clickhouse_connect.get_client(
            host=host, port=port, username=username, password=password, database=database
        )
        self.client.command(self.ENTITIES_DDL)
        self.client.command(self.RELATIONS_DDL)

    def upsert_entities(self, entities: List[ResolvedEntity], doc_id: str) -> None:
        if not entities:
            return
        cols = ["canonical_name", "entity_label", "original_text", "confidence", "source", "summary", "doc_id"]
        rows = [[e.canonical_name, e.entity_label, e.original_text, float(e.confidence),
                  e.source, e.summary, doc_id] for e in entities]
        self.client.insert("entities", rows, column_names=cols)

    def upsert_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return
        cols = ["relation_id", "doc_id", "subject", "subject_label", "predicate",
                "object", "object_label", "confidence", "source", "sentence"]
        rows = [[
            r.relation_id or stable_uuid(r.doc_id, r.subject, r.predicate, r.object),
            r.doc_id, r.subject, r.subject_label, r.predicate, r.object, r.object_label,
            float(r.confidence), r.source, r.sentence,
        ] for r in relations]
        self.client.insert("relations", rows, column_names=cols)

    def close(self) -> None:
        self.client.close()
