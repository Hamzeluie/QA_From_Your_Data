import pytest
import numpy as np

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]


class TestQdrantVectorStore:
    def test_upsert_and_search_entity(self, qdrant_store, tid, mock_embedder):
        vec = mock_embedder.encode(["test entity"])[0].tolist()
        qdrant_store.upsert_entity_summary(
            canonical=f"Ent_{tid}",
            vector=vec,
            payload={"label": "PER", "aliases": [f"Ent_{tid}"]},
        )
        hits = qdrant_store.search_similar_entities(vec, top_k=1)
        assert len(hits) == 1
        assert hits[0]["canonical"] == f"Ent_{tid}"
   
    def test_delete_entity(self, qdrant_store, tid, mock_embedder):
        vec = mock_embedder.encode(["delete me"])[0].tolist()
        qdrant_store.upsert_entity_summary(f"Del_{tid}", vec, {})
        qdrant_store.delete_entity(f"Del_{tid}")
        hits = qdrant_store.search_similar_entities(vec, top_k=10)
        assert not any(h["canonical"] == f"Del_{tid}" for h in hits)

    def test_fetch_all(self, qdrant_store, tid, mock_embedder):
        vec = mock_embedder.encode(["fetch all"])[0].tolist()
        qdrant_store.upsert_entity_summary(f"FA_{tid}", vec, {})
        all_pts = qdrant_store.fetch_all("entity_summaries")
        assert any(p["canonical"] == f"FA_{tid}" for p in all_pts)

    def test_upsert_chunk(self, qdrant_store, tid, mock_embedder):
        vec = mock_embedder.encode(["chunk text"])[0].tolist()
        qdrant_store.upsert_chunk(
            doc_id=tid, chunk_id="c1", vector=vec,
            text="chunk text", owner_id="user",
        )