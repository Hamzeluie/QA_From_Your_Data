import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from typing import List, Dict, Optional, Any
import uuid
import numpy as np
from storage.base import AbstractVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(AbstractVectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        vector_size: int = 384,
    ):
        self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=prefer_grpc)
        self.vector_size = vector_size
        self._base_url = f"http://{host}:{port}" 

    @staticmethod
    def _to_uuid(key: str) -> str:
        """Deterministic UUID from any string key."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def init_collections(self) -> None:
        collections = {
            "entity_summaries": {
                "vectors": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
            "chunk_embeddings": {
                "vectors": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
        }
        for name, config in collections.items():
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=config["vectors"],
                )
                logger.info(f"Created Qdrant collection: {name}")

    def upsert_entity_summary(
        self, canonical: str, vector: List[float], payload: Dict[str, Any]
    ) -> None:
        payload = {**payload, "canonical": canonical}
        point = PointStruct(
            id=self._to_uuid(canonical),
            vector=vector,
            payload=payload,
        )
        self.client.upsert(collection_name="entity_summaries", points=[point])

    def upsert_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        vector: List[float],
        text: str,
        owner_id: str,
    ) -> None:
        chunk_key = f"{doc_id}_{chunk_id}"
        point = PointStruct(
            id=self._to_uuid(chunk_key),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "owner_id": owner_id,
            }
        )
        self.client.upsert(collection_name="chunk_embeddings", points=[point])

    def search_similar_entities(
        self, vector: List[float], top_k: int = 10, filter: Optional[Dict] = None
    ) -> List[Dict]:
        import httpx

        body = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
        }
        if filter:
            body["filter"] = {
                "must": [{"key": k, "match": {"value": v}} for k, v in filter.items()]
            }

        resp = httpx.post(
            f"{self._base_url}/collections/entity_summaries/points/search",
            json=body,
        )
        resp.raise_for_status()
        results = resp.json()["result"]

        return [
            {
                "canonical": r.get("payload", {}).get("canonical", r.get("id")),
                "score": r.get("score"),
                "payload": r.get("payload", {}),
            }
            for r in results
        ]
            
    def fetch_all(self, collection_name: str) -> List[Dict]:
        """Scroll entire collection. Used by EntityMerger."""
        all_points = []
        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=next_offset,
                with_vectors=True,
                with_payload=True,
            )
            for p in batch:
                all_points.append({
                    "canonical": p.payload.get("canonical", p.id),
                    "vector": p.vector,
                    "payload": p.payload,
                })
            if next_offset is None:
                break
        return all_points

    def delete_entity(self, canonical: str) -> None:
        self.client.delete(
            collection_name="entity_summaries",
            points_selector=[self._to_uuid(canonical)],
        )
    