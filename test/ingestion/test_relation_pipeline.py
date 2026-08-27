import pytest
import pandas as pd
from unittest.mock import MagicMock

from ingestion.relation_pipeline import RelationPipeline
from shared.data_classes import Relation

pytestmark = pytest.mark.integration


class TestRelationPipeline:
    @pytest.fixture
    def stub_extractor(self):
        def _extract(text, entities):
            # Deterministic stub
            if len(entities) >= 2:
                return [
                    {
                        "subject": entities[0]["canonical_name"],
                        "predicate": "knows",
                        "object": entities[1]["canonical_name"],
                        "confidence": 0.95,
                    }
                ]
            return []
        return _extract

    def test_extract_and_store(self, factory, stub_extractor, tid):
        pipeline = RelationPipeline(extractor=stub_extractor, factory=factory)
        clean_df = pd.DataFrame([
            {
                "doc_id": tid, "chunk_id": "c1",
                "canonical_name": "Alice", "entity_label": "PER",
                "mention_sentence": "Alice knows Bob.", "status": "RESOLVED",
                "original_text": "Alice",
            },
            {
                "doc_id": tid, "chunk_id": "c1",
                "canonical_name": "Bob", "entity_label": "PER",
                "mention_sentence": "Alice knows Bob.", "status": "RESOLVED",
                "original_text": "Bob",
            },
        ])
        rel_df = pipeline.extract_and_store(clean_df)
        assert not rel_df.empty
        assert rel_df.iloc[0]["predicate"] == "knows"

    def test_symmetric_dedup(self, factory, stub_extractor, tid):
        # Stub that returns symmetric pair
        def sym_extractor(text, entities):
            return [
                {"subject": "Bob", "predicate": "spouse", "object": "Alice", "confidence": 0.9},
            ]
        pipeline = RelationPipeline(extractor=sym_extractor, factory=factory)
        clean_df = pd.DataFrame([
            {"doc_id": tid, "chunk_id": "c1", "canonical_name": "Alice", "entity_label": "PER",
             "mention_sentence": "x", "status": "RESOLVED", "original_text": "Alice"},
            {"doc_id": tid, "chunk_id": "c1", "canonical_name": "Bob", "entity_label": "PER",
             "mention_sentence": "x", "status": "RESOLVED", "original_text": "Bob"},
        ])
        rel_df = pipeline.extract_and_store(clean_df)
        # Should normalize to Alice < Bob
        row = rel_df.iloc[0]
        assert row["subject"] == "Alice"
        assert row["object"] == "Bob"

    def test_empty_clean_df(self, factory, stub_extractor):
        pipeline = RelationPipeline(extractor=stub_extractor, factory=factory)
        df = pipeline.extract_and_store(pd.DataFrame())
        assert df.empty

    def test_literal_exclusion(self, factory, tid):
        def lit_extractor(text, entities):
            return [{"subject": "E1", "predicate": "on", "object": "E2", "confidence": 0.5}]
        pipeline = RelationPipeline(
            extractor=lit_extractor, factory=factory, include_literals=False
        )
        clean_df = pd.DataFrame([
            {"doc_id": tid, "chunk_id": "c1", "canonical_name": "2024", "entity_label": "DATE",
             "mention_sentence": "x", "status": "RESOLVED", "original_text": "2024"},
            {"doc_id": tid, "chunk_id": "c1", "canonical_name": "E2", "entity_label": "PER",
             "mention_sentence": "x", "status": "RESOLVED", "original_text": "E2"},
        ])
        rel_df = pipeline.extract_and_store(clean_df)
        # DATE should be excluded, so < 2 entities remain
        assert rel_df.empty