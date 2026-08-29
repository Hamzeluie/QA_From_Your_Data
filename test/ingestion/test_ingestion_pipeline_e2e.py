import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from ingestion.pipeline import IngestionPipeline
from shared.data_classes import Chunk

pytestmark = pytest.mark.integration


@pytest.fixture
def stub_extractor():
    def _extract(text, entities):
        if len(entities) >= 2:
            return [{
                "subject": entities[0]["canonical_name"],
                "predicate": "author_of",
                "object": entities[1]["canonical_name"],
                "confidence": 0.92,
            }]
        return []
    return _extract


@pytest.fixture
def pipeline(factory, stub_extractor, mock_embedder):
    p = IngestionPipeline(
        relation_extractor=stub_extractor,
        factory=factory,
        embedder=mock_embedder,
    )
    return p


class TestIngestionPipelineE2E:
    def test_full_pipeline_happy_path(self, pipeline, tid):
        doc_id = f"e2e_doc_{tid}"
        text = "Miguel Riofrio wrote La Emancipada. He was born in Loja."

        # Mock NER to avoid LLM/spacy heavy lifting in E2E
        fake_df = pd.DataFrame([
            {"doc_id": doc_id, "chunk_id": "c1", "text": "Miguel Riofrio",
             "label": "PER", "start": 0, "end": 14,
             "mention_sentence": "Miguel Riofrio wrote La Emancipada.", "confidence": 1.0},
            {"doc_id": doc_id, "chunk_id": "c1", "text": "La Emancipada",
             "label": "MISC", "start": 20, "end": 33,
             "mention_sentence": "Miguel Riofrio wrote La Emancipada.", "confidence": 1.0},
            {"doc_id": doc_id, "chunk_id": "c1", "text": "Loja",
             "label": "LOC", "start": 52, "end": 56,
             "mention_sentence": "He was born in Loja.", "confidence": 1.0},
        ])
        fake_chunks = [
            Chunk(doc_id=doc_id, chunk_id="c1", owner_id="u1",
                  sentence="Miguel Riofrio wrote La Emancipada.", date_time=""),
            Chunk(doc_id=doc_id, chunk_id="c2", owner_id="u1",
                  sentence="He was born in Loja.", date_time=""),
        ]

        with patch.object(pipeline.resolver, '_extract_entities_with_coref', return_value=(fake_df, fake_chunks)):
            result = pipeline.run(doc_id=doc_id, raw_text=text, owner_id="u1")

        assert result["status"] == "success"
        assert result["doc_id"] == doc_id
        assert result["resolved_count"] > 0
        # Verify Postgres state
        state = pipeline.factory.postgres.get_document_state(doc_id)
        assert state["status"] == "indexed"

    def test_idempotent_skip_already_indexed(self, pipeline, tid):
        doc_id = f"e2e_idem_{tid}"
        pipeline.factory.postgres.create_document(doc_id, "u1")
        pipeline.factory.postgres.transition_state(doc_id, "uploaded", "indexed")

        result = pipeline.run(doc_id=doc_id, raw_text="any", owner_id="u1")
        assert result["status"] == "skipped"

    def test_unresolved_entity_goes_to_review(self, pipeline, tid):
        doc_id = f"e2e_rev_{tid}"
        text = "Some unknown person named Xyzabc did something."

        fake_df = pd.DataFrame([
            {"doc_id": doc_id, "chunk_id": "c1", "text": "Xyzabc",
             "label": "PER", "start": 0, "end": 6,
             "mention_sentence": text, "confidence": 1.0},
        ])
        fake_chunks = [Chunk(doc_id=doc_id, chunk_id="c1", owner_id="u1",
                             sentence=text, date_time="")]

        with patch.object(pipeline.resolver, '_extract_entities_with_coref', return_value=(fake_df, fake_chunks)):
            result = pipeline.run(doc_id=doc_id, raw_text=text, owner_id="u1")

        assert result["review_count"] > 0
        state = pipeline.factory.postgres.get_document_state(doc_id)
        assert state["status"] == "review_pending"

    def test_chunk_indexing_creates_vectors_and_es_docs(self, pipeline, tid, mock_embedder):
        doc_id = f"e2e_chunk_{tid}"
        text = "Chunk one. Chunk two."

        fake_df = pd.DataFrame()  # no entities
        fake_chunks = [
            Chunk(doc_id=doc_id, chunk_id="c1", owner_id="u1", sentence="Chunk one.", date_time=""),
            Chunk(doc_id=doc_id, chunk_id="c2", owner_id="u1", sentence="Chunk two.", date_time=""),
        ]

        with patch.object(pipeline.resolver, '_extract_entities_with_coref', return_value=(fake_df, fake_chunks)):
            pipeline.run(doc_id=doc_id, raw_text=text, owner_id="u1")

        # Qdrant check
        pts = pipeline.factory.qdrant.fetch_all("chunk_embeddings")
        assert any(p["payload"]["doc_id"] == doc_id for p in pts)

        # ES check
        import time
        time.sleep(1.5)
        es_hits = pipeline.factory.es.client.search(
            index=pipeline.factory.es.CHUNKS_INDEX,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
        assert es_hits["hits"]["total"]["value"] == 2