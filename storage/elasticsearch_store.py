import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, NotFoundError


logger = logging.getLogger(__name__)


class ElasticsearchTextSearch:
    """
    BM25 full-text search over chunks, mirroring the other stores:
      - ES doc _id == chunk_id (deterministic, globally unique via Ids.chunk)
      - same upsert / partial-update / delete-by-doc API as QdrantVectorStore
    """

    INDEX = "chunks"

    def __init__(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        request_timeout: int = 15,
    ):
        kwargs: Dict[str, Any] = {"request_timeout": request_timeout}
        if username and password:
            kwargs["basic_auth"] = (username, password)
        self.client = Elasticsearch(hosts, **kwargs)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _refresh(self, refresh: bool) -> Optional[str]:
        """Pass refresh=True in tests to make writes immediately visible."""
        return "wait_for" if refresh else None

    # ── schema ────────────────────────────────────────────────────────────

    def init_index(self) -> None:
        if self.client.indices.exists(index=self.INDEX):
            return
        self.client.indices.create(
            index=self.INDEX,
            settings={"number_of_shards": 1, "number_of_replicas": 0},
            mappings={
                "properties": {
                    "chunk_id":      {"type": "keyword"},
                    "doc_id":        {"type": "keyword"},
                    "text":          {"type": "text", "analyzer": "english"},
                    "canonical_ids": {"type": "keyword"},
                    "owner_id":      {"type": "keyword"},
                    "created_at":    {"type": "date"},
                    "updated_at":    {"type": "date"},
                }
            },
        )
        logger.info("Created ES index: %s", self.INDEX)

    # ═══════════════════════════ CREATE ═══════════════════════════════════

    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        canonical_ids: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        refresh: bool = False,
    ) -> bool:
        """
        Create or overwrite a chunk. On overwrite:
          - original created_at is preserved (re-ingestion must not move it)
          - canonical_ids are MERGED unless replace_canonical_ids=True
        """
        return self._write_chunk(
            chunk_id, doc_id, text, canonical_ids, owner_id,
            created_at, updated_at, replace_canonical_ids=False,
            refresh=refresh,
        )

    def replace_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        canonical_ids: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        refresh: bool = False,
    ) -> bool:
        """Same as upsert_chunk but canonical_ids are REPLACED, not merged."""
        return self._write_chunk(
            chunk_id, doc_id, text, canonical_ids, owner_id,
            created_at, updated_at, replace_canonical_ids=True,
            refresh=refresh,
        )

    def _write_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        canonical_ids: Optional[List[str]],
        owner_id: Optional[str],
        created_at: Optional[str],
        updated_at: Optional[str],
        replace_canonical_ids: bool,
        refresh: bool,
    ) -> bool:
        existing = self.get_chunk(chunk_id)

        if existing and not replace_canonical_ids:
            merged = set(existing.get("canonical_ids") or [])
            merged.update(canonical_ids or [])
            canonical_ids = sorted(merged)

        document = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "canonical_ids": sorted(set(canonical_ids or [])),
            "owner_id": owner_id,
            "created_at": (existing or {}).get("created_at") or created_at or self._now_iso(),
            "updated_at": updated_at or self._now_iso(),
        }
        self.client.index(
            index=self.INDEX, id=chunk_id, document=document,
            refresh=self._refresh(refresh),
        )
        return True

    def index_chunks(self, chunks: List[Dict[str, Any]], refresh: bool = False) -> int:
        """
        Bulk ingestion — FULL overwrite per doc (no canonical_ids merge, no
        get round-trip). Use for the initial chunking stage where each dict
        is {chunk_id, doc_id, text, canonical_ids?, owner_id?, created_at?}.
        """
        if not chunks:
            return 0
        now = self._now_iso()
        actions = [
            {
                "_index": self.INDEX,
                "_id": c["chunk_id"],
                "_source": {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "text": c["text"],
                    "canonical_ids": sorted(set(c.get("canonical_ids") or [])),
                    "owner_id": c.get("owner_id"),
                    "created_at": c.get("created_at") or now,
                    "updated_at": now,
                },
            }
            for c in chunks
        ]
        from elasticsearch.helpers import bulk
        success, _ = bulk(self.client, actions, refresh=refresh, raise_on_error=True)
        return success

    # ── READ ──────────────────────────────────────────────────────────────

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = self.client.get(index=self.INDEX, id=chunk_id)
        except NotFoundError:
            return None
        return {"_score": None, **(doc.get("_source") or {})}

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        doc_id: Optional[str] = None,
        canonical_ids: Optional[List[str]] = None,   # ANY of these (terms)
        owner_id: Optional[str] = None,
        fuzziness: str = "AUTO",
        min_score: Optional[float] = None,
        highlight: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        BM25 search over chunk text. Filters are exact (keyword) and combine
        with AND; canonical_ids is any-of.

        `offset` paging caps at 10,000 (ES limit) — for deep pagination use
        search_after; for this workload offset is fine.
        """
        filter_conditions: List[Dict[str, Any]] = []
        if doc_id is not None:
            filter_conditions.append({"term": {"doc_id": doc_id}})
        if owner_id is not None:
            filter_conditions.append({"term": {"owner_id": owner_id}})
        if canonical_ids:
            filter_conditions.append({"terms": {"canonical_ids": canonical_ids}})

        es_query: Dict[str, Any] = {
            "bool": {
                "must": [{
                    "match": {
                        "text": {
                            "query": query,
                            "fuzziness": fuzziness,
                            "prefix_length": 1,
                        }
                    }
                }],
                "filter": filter_conditions,
            }
        }

        kwargs: Dict[str, Any] = {}
        if min_score is not None:
            kwargs["min_score"] = min_score
        if highlight:
            kwargs["highlight"] = {
                "fields": {"text": {"fragment_size": 180, "number_of_fragments": 2}},
                "boundary_scanner": "sentence",
            }

        resp = self.client.search(
            index=self.INDEX,
            query=es_query,
            size=top_k,
            from_=offset,
            **kwargs,
        )
        return [self._hit_to_dict(h) for h in resp["hits"]["hits"]]

    def get_chunks_for_canonical(
        self, canonical_id: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """All chunks mentioning an entity — filter-only, unscored."""
        resp = self.client.search(
            index=self.INDEX,
            query={
                "bool": {"filter": [
                    {"term": {"canonical_ids": canonical_id}}
                ]}
            },
            size=limit,
        )
        return [self._hit_to_dict(h) for h in resp["hits"]["hits"]]

    def count_chunks_for_doc(self, doc_id: str) -> int:
        return self.client.count(
            index=self.INDEX,
            query={"term": {"doc_id": doc_id}},
        ).count

    # ═══════════════════════════ UPDATE ═══════════════════════════════════

    def update_chunk(
        self,
        chunk_id: str,
        text: Optional[str] = None,
        canonical_ids: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
        merge_canonical_ids: bool = True,
        refresh: bool = False,
    ) -> bool:
        """
        Partial update — text stays untouched unless given. The document must
        already exist (False if not). For text changes prefer replace_chunk /
        upsert_chunk so the full doc stays consistent.
        """
        existing = self.get_chunk(chunk_id)
        if not existing:
            return False

        doc: Dict[str, Any] = {"updated_at": self._now_iso()}
        if text is not None:
            doc["text"] = text
        if owner_id is not None:
            doc["owner_id"] = owner_id
        if canonical_ids is not None:
            if merge_canonical_ids:
                merged = set(existing.get("canonical_ids") or [])
                merged.update(canonical_ids)
                doc["canonical_ids"] = sorted(merged)
            else:
                doc["canonical_ids"] = sorted(set(canonical_ids))

        self.client.update(
            index=self.INDEX, id=chunk_id, doc=doc,
            retry_on_conflict=3,
            refresh=self._refresh(refresh),
        )
        return True

    # ═══════════════════════════ DELETE ═══════════════════════════════════

    def delete_chunk(self, chunk_id: str, refresh: bool = False) -> bool:
        try:
            self.client.delete(
                index=self.INDEX, id=chunk_id,
                refresh=self._refresh(refresh),
            )
            return True
        except NotFoundError:
            return False

    def delete_chunks_for_doc(self, doc_id: str, refresh: bool = False) -> int:
        """Server-side delete-by-query — one call for the whole document."""
        resp = self.client.delete_by_query(
            index=self.INDEX,
            query={"term": {"doc_id": doc_id}},
            refresh=self._refresh(refresh),
        )
        return resp.get("deleted", 0)

    # ── response assembly ─────────────────────────────────────────────────

    @staticmethod
    def _hit_to_dict(hit: Dict[str, Any]) -> Dict[str, Any]:
        src = hit.get("_source", {})
        highlights = hit.get("highlight", {}).get("text") or []
        return {
            "chunk_id": src.get("chunk_id") or hit.get("_id"),
            "doc_id": src.get("doc_id"),
            "text": src.get("text"),
            "canonical_ids": src.get("canonical_ids") or [],
            "owner_id": src.get("owner_id"),
            "created_at": src.get("created_at"),
            "updated_at": src.get("updated_at"),
            "score": round(float(hit["_score"]), 3) if hit.get("_score") else None,
            "highlights": highlights,
        }