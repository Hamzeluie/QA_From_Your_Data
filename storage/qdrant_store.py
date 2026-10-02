import logging
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timezone
import uuid
import json
import numpy as np
from domain import CandidateSource, CandidateResult, CandidateMatchMethod, CanonicalEntity
from storage.base import AbstractVectorStore
from qdrant_client import QdrantClient, models


logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Vector store, mirroring the other stores' conventions:
      - entities: one point per canonical entity. Vector is the summary
        embedding when a summary exists, else the name embedding;
        has_summary marks which (so resolvers can prefer the better one).
      - chunks:   one point per chunk. payload.canonical_ids lists the
        entities mentioned in the chunk (evidence lookups).

    Point IDs are deterministic: uuid5(canonical_id) / uuid5(chunk_id).
    Re-upserting the same key overwrites in place — same idempotency
    model as the Postgres and Neo4j stores.
    """

    ENTITIES = "entities"
    CHUNKS = "chunks"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        vector_size: int = 384,
        distance: models.Distance = models.Distance.COSINE,
    ):
        self.client = QdrantClient(
            host=host, port=port, grpc_port=grpc_port, prefer_grpc=prefer_grpc
        )
        self.vector_size = vector_size
        self.distance = distance
        # distinct namespaces so a chunk_id and canonical_id can never
        # collide onto the same UUID
        self._ns_entities = uuid.uuid5(uuid.NAMESPACE_DNS, "kg-entities")
        self._ns_chunks = uuid.uuid5(uuid.NAMESPACE_DNS, "kg-chunks")

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _point_id(self, namespace: uuid.UUID, key: str) -> str:
        return str(uuid.uuid5(namespace, key))

    _PRIMITIVE = (str, int, float, bool)

    @classmethod
    def _sanitize_props(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep payloads flat + filterable: drop None, JSON-encode nesting."""
        if not metadata:
            return {}
        out: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, dict):
                out[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (list, tuple)):
                out[key] = (
                    list(value)
                    if all(isinstance(x, cls._PRIMITIVE) for x in value)
                    else json.dumps(value, ensure_ascii=False)
                )
            elif isinstance(value, cls._PRIMITIVE):
                out[key] = value
            else:
                out[key] = str(value)
        return out

    def _ensure_collection(self, name: str) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=self.distance
                ),
            )
            logger.info("Created Qdrant collection: %s", name)

    def _search(self, collection: str, vector: List[float],
                flt: Optional[models.Filter], limit: int):
        """query_points (client >=1.10) with fallback to legacy search()."""
        if hasattr(self.client, "query_points"):
            return self.client.query_points(
                collection_name=collection, query=vector,
                query_filter=flt, limit=limit, with_payload=True,
            ).points
        return self.client.search(
            collection_name=collection, query_vector=vector,
            query_filter=flt, limit=limit, with_payload=True,
        )

    def _scroll(self, collection: str,
                flt: Optional[models.Filter] = None,
                limit: int = 1000, with_vectors: bool = False):
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                collection_name=collection, scroll_filter=flt,
                limit=limit, offset=offset,
                with_payload=True, with_vectors=with_vectors,
            )
            points.extend(batch)
            if offset is None:
                return points

    # ── schema ────────────────────────────────────────────────────────────

    def init_collections(self) -> None:
        self._ensure_collection(self.ENTITIES)
        self._ensure_collection(self.CHUNKS)

        indexes: Dict[str, List[Tuple[str, models.PayloadSchemaType]]] = {
            self.ENTITIES: [
                ("canonical_id", models.PayloadSchemaType.KEYWORD),
                ("label",        models.PayloadSchemaType.KEYWORD),
                ("has_summary",  models.PayloadSchemaType.BOOL),
            ],
            self.CHUNKS: [
                ("chunk_id",      models.PayloadSchemaType.KEYWORD),
                ("doc_id",        models.PayloadSchemaType.KEYWORD),
                ("canonical_ids", models.PayloadSchemaType.KEYWORD),
                ("owner_id",      models.PayloadSchemaType.KEYWORD),
            ],
        }
        for collection, fields in indexes.items():
            for field, schema in fields:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )

    # ═══════════════════════════ ENTITIES ═════════════════════════════════

    def upsert_entity(
        self,
        canonical_id: str,
        vector: List[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        has_summary: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create/update an entity point. Vector required — for payload-only
        changes use update_entity()."""
        existing = self.get_entity(canonical_id)
        old = existing["payload"] if existing else {}
        
        payload = {
            "canonical_id": canonical_id,
            "name": name if name is not None else old.get("name"),
            "label": label if label is not None else old.get("label"),
            "has_summary": has_summary if has_summary is not None
                           else bool(old.get("has_summary")),
            "updated_at": self._now_iso(),
        }
        payload.update(self._sanitize_props(metadata))
        payload = {k: v for k, v in payload.items() if v is not None}

        point = models.PointStruct(
            id=self._point_id(self._ns_entities, canonical_id),
            vector=vector,
            payload=payload,
        )
        self.client.upsert(collection_name=self.ENTITIES, points=[point])
        return canonical_id

    def get_entity(self, canonical_id: str, with_vector: bool = False) -> Optional[Dict]:
        records = self.client.retrieve(
            collection_name=self.ENTITIES,
            ids=[self._point_id(self._ns_entities, canonical_id)],
            with_payload=True,
            with_vectors=with_vector,
        )
        if not records:
            return None
        rec = records[0]
        return {
            "canonical_id": canonical_id,
            "payload": dict(rec.payload or {}),
            "vector": list(rec.vector) if with_vector and rec.vector else None,
        }

    def get_entities(self, canonical_ids: List[str]) -> Dict[str, Dict]:
        """Batch existence/payload lookup by canonical_id. Missing ids absent from result."""
        if not canonical_ids:
            return {}
        pid_to_cid = {self._point_id(self._ns_entities, c): c for c in canonical_ids}
        records = self.client.retrieve(
            collection_name=self.ENTITIES,
            ids=list(pid_to_cid),
            with_payload=True,
            with_vectors=False,
        )
        return {pid_to_cid[str(r.id)]: dict(r.payload or {})
                for r in records if str(r.id) in pid_to_cid}
        
    def update_entity(
        self,
        canonical_id: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        has_summary: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Payload-only update; the stored vector is untouched."""
        existing = self.get_entity(canonical_id)
        if not existing:
            return False
        old = existing["payload"]

        payload: Dict[str, Any] = {
            k: v for k, v in {
                "name": name, "label": label, "has_summary": has_summary,
            }.items() if v is not None
        }
        payload.update(self._sanitize_props(metadata))
        if not payload:
            return True

        self.client.set_payload(
            collection_name=self.ENTITIES,
            payload=payload,
            points=[self._point_id(self._ns_entities, canonical_id)],
        )
        return True

    def delete_entity(self, canonical_id: str) -> bool:
        if not self.get_entity(canonical_id):
            return False
        self.client.delete(
            collection_name=self.ENTITIES,
            points_selector=models.PointIdsList(
                points=[self._point_id(self._ns_entities, canonical_id)]
            ),
        )
        return True
    
    def search_entities(
        self,
        vector: List[float],
        top_k: int = 10,
        label: Optional[str] = None,
        has_summary: Optional[bool] = None,
    ) -> List[CandidateResult]:
        """
        Semantic candidate generation. Typical resolver flow: query with
        has_summary=True first (summaries embed better); if too few hits,
        retry without the filter.
        """
        must = []
        if label is not None:
            must.append(models.FieldCondition(
                key="label", match=models.MatchValue(value=label)))
        if has_summary is not None:
            must.append(models.FieldCondition(
                key="has_summary", match=models.MatchValue(value=has_summary)))
        flt = models.Filter(must=must) if must else None

        results = self._search(self.ENTITIES, vector, flt, top_k)
        return [
            CandidateResult(
                canonical_id=r.payload.get("canonical_id") or str(r.id),
                canonical_name=r.payload.get("name")
                               or r.payload.get("canonical_id") or str(r.id),
                label=r.payload.get("label") or "",
                summary=None,
                source=CandidateSource.QDRANT,
                match_method=CandidateMatchMethod.SEMANTIC,
                match_score=round(float(r.score), 3),
                label_match=bool(label and r.payload.get("label") == label),
            )
            for r in results
        ]

    # ═══════════════════════════ CHUNKS ═══════════════════════════════════

    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        vector: List[float],
        canonical_ids: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create/update a chunk point. chunk_id must be globally unique
        (your Ids.chunk(doc_id, index) guarantees this).
        """
        payload = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "canonical_ids": sorted(set(canonical_ids or [])),
            "owner_id": owner_id,
            "updated_at": self._now_iso(),
        }
        payload.update(self._sanitize_props(metadata))
        payload = {k: v for k, v in payload.items() if v is not None}

        point = models.PointStruct(
            id=self._point_id(self._ns_chunks, chunk_id),
            vector=vector,
            payload=payload,
        )
        self.client.upsert(collection_name=self.CHUNKS, points=[point])
        return chunk_id

    def get_chunk(self, chunk_id: str, with_vector: bool = False) -> Optional[Dict]:
        records = self.client.retrieve(
            collection_name=self.CHUNKS,
            ids=[self._point_id(self._ns_chunks, chunk_id)],
            with_payload=True,
            with_vectors=with_vector,
        )
        if not records:
            return None
        rec = records[0]
        return {
            "chunk_id": chunk_id,
            "payload": dict(rec.payload or {}),
            "vector": list(rec.vector) if with_vector and rec.vector else None,
        }

    def update_chunk(
        self,
        chunk_id: str,
        canonical_ids: Optional[List[str]] = None,
        replace_canonical_ids: bool = False,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Payload-only update. By default canonical_ids are MERGED with the
        existing list (a later resolution stage adds newly-linked entities);
        pass replace_canonical_ids=True to overwrite.
        """
        existing = self.get_chunk(chunk_id)
        if not existing:
            return False

        payload: Dict[str, Any] = {}
        if canonical_ids is not None:
            if replace_canonical_ids:
                payload["canonical_ids"] = sorted(set(canonical_ids))
            else:
                merged = set(existing["payload"].get("canonical_ids") or [])
                merged.update(canonical_ids)
                payload["canonical_ids"] = sorted(merged)
        if owner_id is not None:
            payload["owner_id"] = owner_id
        payload.update(self._sanitize_props(metadata))
        if not payload:
            return True

        payload["updated_at"] = self._now_iso()
        self.client.set_payload(
            collection_name=self.CHUNKS,
            payload=payload,
            points=[self._point_id(self._ns_chunks, chunk_id)],
        )
        return True

    def delete_chunk(self, chunk_id: str) -> bool:
        if not self.get_chunk(chunk_id):
            return False
        self.client.delete(
            collection_name=self.CHUNKS,
            points_selector=models.PointIdsList(
                points=[self._point_id(self._ns_chunks, chunk_id)]
            ),
        )
        return True

    def delete_chunks_for_doc(self, doc_id: str) -> int:
        """Delete every chunk of a document in one server-side call."""
        flt = models.Filter(must=[models.FieldCondition(
            key="doc_id", match=models.MatchValue(value=doc_id))])
        count = self.client.count(
            collection_name=self.CHUNKS, count_filter=flt, exact=True
        ).count
        if count:
            self.client.delete(
                collection_name=self.CHUNKS,
                points_selector=models.FilterSelector(filter=flt),
            )
        return count

    def search_similar_chunks(
        self,
        vector: List[float],
        top_k: int = 10,
        doc_id: Optional[str] = None,
        canonical_id: Optional[str] = None,   # restrict to chunks mentioning this entity
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        must = []
        if doc_id is not None:
            must.append(models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=doc_id)))
        if canonical_id is not None:
            must.append(models.FieldCondition(
                key="canonical_ids", match=models.MatchValue(value=canonical_id)))
        if owner_id is not None:
            must.append(models.FieldCondition(
                key="owner_id", match=models.MatchValue(value=owner_id)))
        flt = models.Filter(must=must) if must else None

        results = self._search(self.CHUNKS, vector, flt, top_k)
        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "doc_id": r.payload.get("doc_id"),
                "canonical_ids": r.payload.get("canonical_ids") or [],
                "score": round(float(r.score), 3),
                "payload": dict(r.payload or {}),
            }
            for r in results
        ]

    def get_chunks_for_canonical(self, canonical_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        All chunks mentioning an entity — pure payload filter via scroll
        (replaces the old dummy-vector trick; no vector needed, no score).
        """
        points = self._scroll(
            self.CHUNKS,
            flt=models.Filter(must=[models.FieldCondition(
                key="canonical_ids", match=models.MatchValue(value=canonical_id))]),
            limit=limit,
        )
        return [
            {
                "chunk_id": p.payload.get("chunk_id"),
                "doc_id": p.payload.get("doc_id"),
                "canonical_ids": p.payload.get("canonical_ids") or [],
                "payload": dict(p.payload or {}),
            }
            for p in points
        ]


    # ── misc ──────────────────────────────────────────────────────────────

    def fetch_all(self, collection_name: str, with_vectors: bool = False) -> List[Dict[str, Any]]:
        """Full scroll — merger / re-embedding jobs."""
        return [
            {
                "id": str(p.id),
                "payload": dict(p.payload or {}),
                "vector": list(p.vector) if with_vectors and p.vector else None,
            }
            for p in self._scroll(collection_name, with_vectors=with_vectors)
        ]

    def get_stats(self) -> Dict[str, int]:
        return {
            "entities": self.client.count(collection_name=self.ENTITIES, exact=True).count,
            "chunks": self.client.count(collection_name=self.CHUNKS, exact=True).count,
        }

    def close(self) -> None:
        self.client.close()
        

class QdrantIndexer:
    """
    Embeds + pushes into QdrantVectorStore:
      - canonical text -> entities collection
      - chunk text     -> chunks collection

    `embed` must be a BATCHED callable: List[str] -> List[List[float]].
    """

    def __init__(
        self,
        store: QdrantVectorStore,
        embed: Callable[[List[str]], List[List[float]]],
        batch_size: int = 64,
        max_summary_chars: int = 1000,
    ):
        self.store = store
        self.embed = embed
        self.batch_size = batch_size
        self.max_summary_chars = max_summary_chars

    # ── text builders ─────────────────────────────────────────────────────
    # NOTE: your format, with the "lable" typo fixed. Whatever template you
    # pick, NEVER change it later without re-embedding every entity —
    # embeddings are only comparable if the template is stable.

    def build_canonical_text(self, canonical: CanonicalEntity) -> str:
        parts = [f"name:{canonical.name}", f"label:{canonical.label}"]
        if canonical.summary:
            summary = canonical.summary.strip()
            if len(summary) > self.max_summary_chars:
                summary = summary[: self.max_summary_chars]
            parts.append(f"summary:{summary}")
        return ",".join(parts)   # "name:OpenAI,label:ORGANIZATION,summary:AI company"

    @staticmethod
    def build_chunk_text(text: str) -> str:
        return (text or "").strip()

    # ── embedding ─────────────────────────────────────────────────────────

    def _embed_all(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self.embed(texts[i : i + self.batch_size]))
        for v in vectors:
            if len(v) != self.store.vector_size:
                raise ValueError(
                    f"Embedding dim {len(v)} != collection dim "
                    f"{self.store.vector_size} — wrong model or vector_size config"
                )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Search-side: embed a query with the same model/batching."""
        return self._embed_all([text])[0]

    # ═══════════════════════════ CANONICALS ═══════════════════════════════

    def index_canonical(
        self,
        canonical: CanonicalEntity,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vector = self._embed_all([self.build_canonical_text(canonical)])[0]
        return self.store.upsert_entity(
            canonical_id=canonical.canonical_id,
            vector=vector,
            name=canonical.name,
            label=canonical.label,
            has_summary=bool(canonical.summary),
            metadata=metadata,
        )

    def index_canonicals(
        self,
        canonicals: List[CanonicalEntity],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,   # cid -> metadata
    ) -> int:
        """
        Embeds ONLY canonicals not yet stored. One batched retrieve decides;
        idempotent across retries, re-runs, and repeated human-resolve rounds.
        Re-embeds once if a summary was added since (has_summary False -> True),
        because summary vectors are the ones resolvers prefer.
        """
        if not canonicals:
            return 0
        unique = {c.canonical_id: c for c in canonicals if c.canonical_id}
        if not unique:
            return 0

        stored = self.store.get_entities(list(unique))
        todo = [
            c for cid, c in unique.items()
            if cid not in stored
            or (c.summary and not stored[cid].get("has_summary"))
        ]
        if not todo:
            return 0

        texts = [self.build_canonical_text(c) for c in todo]
        for c, vec in zip(todo, self._embed_all(texts)):
            self.store.upsert_entity(
                canonical_id=c.canonical_id,
                vector=vec,
                name=c.name,
                label=c.label,
                has_summary=bool(c.summary),
                metadata=(metadata or {}).get(c.canonical_id),
            )
        return len(todo)

    def index_canonical_from_postgres(
        self,
        pg,                       # PostgresStateStore
        canonical_id: str,
    ) -> bool:
        """Fetch canonical from Postgres, embed, mirror into Qdrant."""
        entity = pg.get_canonical(canonical_id)
        if not entity:
            return False
        self.index_canonical(entity)
        return True

    # ═══════════════════════════ CHUNKS ═══════════════════════════════════

    def index_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        canonical_ids: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vector = self._embed_all([self.build_chunk_text(text)])[0]
        return self.store.upsert_chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            vector=vector,
            canonical_ids=canonical_ids,
            owner_id=owner_id,
            metadata=metadata,
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Batched. Each dict: {chunk_id, doc_id, text,
                             canonical_ids?, owner_id?, metadata?}
        """
        if not chunks:
            return 0
        texts = [self.build_chunk_text(c["text"]) for c in chunks]
        vectors = self._embed_all(texts)
        for c, vec in zip(chunks, vectors):
            self.store.upsert_chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                vector=vec,
                canonical_ids=c.get("canonical_ids"),
                owner_id=c.get("owner_id"),
                metadata=c.get("metadata"),
            )
        return len(chunks)

    def update_chunk_canonicals(self, chunk_id: str, canonical_ids: List[str]) -> bool:
        """Merge canonical_ids into an existing chunk point (payload-only)."""
        return self.store.update_chunk(chunk_id=chunk_id, canonical_ids=canonical_ids)