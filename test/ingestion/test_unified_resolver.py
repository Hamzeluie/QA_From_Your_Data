import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from ingestion.unified_resolver import UnifiedEntityResolver
from shared.data_classes import DisambiguationStatus, Entity, Chunk


@pytest.fixture
def resolver(factory, mock_embedder):
    r = UnifiedEntityResolver(
        factory=factory,
        embedder=mock_embedder,
        merge_threshold=0.99,  # high threshold so merges don't happen accidentally
    )
    return r


class TestUnifiedResolverUnit:
    """Fast unit tests with mocked NER."""

    def test_normalize_literal_money(self, resolver, tid):
        row = pd.Series({
            "text": "$100", "label": "MONEY", "start": 0, "end": 4,
            "mention_sentence": "It costs $100.", "confidence": 1.0,
        })
        ent = resolver._resolve_entity(row)
        assert ent.status == DisambiguationStatus.RESOLVED
        assert ent.source == "normalized"

    def test_normalize_literal_date(self, resolver, tid):
        row = pd.Series({
            "text": "2024-01-01", "label": "DATE", "start": 0, "end": 10,
            "mention_sentence": "On 2024-01-01.", "confidence": 1.0,
        })
        ent = resolver._resolve_entity(row)
        assert ent.status == DisambiguationStatus.RESOLVED

    def test_disambiguate_single_candidate(self, resolver, tid):
        from shared.data_classes import CandidateResult
        cands = [
            CandidateResult(
                canonical=f"C_{tid}", label="PER", aliases=[f"A_{tid}"],
                summary="s", context_indicators=[], related_to=[],
                match_score=0.95, match_method="exact",
            )
        ]
        result = resolver._disambiguate(f"A_{tid}", "PER", "ctx", cands)
        assert result.status == DisambiguationStatus.RESOLVED
        assert result.canonical_name == f"C_{tid}"

    def test_disambiguate_ambiguous_low_score(self, resolver, tid):
        from shared.data_classes import CandidateResult
        cands = [
            CandidateResult(
                canonical=f"C_{tid}", label="PER", aliases=[f"A_{tid}"],
                summary="s", context_indicators=[], related_to=[],
                match_score=0.40, match_method="partial",
            )
        ]
        result = resolver._disambiguate(f"A_{tid}", "PER", "ctx", cands)
        assert result.status == DisambiguationStatus.AMBIGUOUS


class TestUnifiedResolverIntegration:
    """Tests against real stores (NER mocked to avoid LLM calls)."""

    def test_process_document_resolves_known_entity(self, resolver, tid, factory):
        # Pre-seed the catalog
        factory.neo4j.upsert_entity(
            canonical="Albert Einstein",
            label="PER",
            aliases=["Einstein", "Albert"],
            summary="Physicist",
            source="test",
        )
        # Mock NER to return Einstein
        fake_df = pd.DataFrame([
            {
                "doc_id": tid, "chunk_id": "c1", "text": "Einstein",
                "label": "PER", "start": 0, "end": 8,
                "mention_sentence": "Einstein discovered relativity.",
                "confidence": 1.0,
            }
        ])
        fake_chunks = [Chunk(doc_id=tid, chunk_id="c1", owner_id="u1",
                            sentence="Einstein discovered relativity.", date_time="")]

        with patch.object(resolver, '_extract_entities_with_coref', return_value=(fake_df, fake_chunks)):
            chunks, clean_df, review_df = resolver.process_document(tid, "dummy text", "u1")

        assert not clean_df.empty
        assert clean_df.iloc[0]["canonical_name"] == "Albert Einstein"

    def test_add_user_resolution_visible_to_next_document(self, resolver, tid, factory):
        # Operator resolves a new entity
        resolver.add_user_resolution(
            canonical_name="TestCorp",
            entity_label="ORG",
            summary="A test corporation",
            aliases=["TC", "Test Corp"],
        )
        # Poll outbox so ES/Qdrant catch up
        from storage.outbox import OutboxPoller
        poller = OutboxPoller(factory)
        poller.process_batch(limit=100)

        # Next document mentions alias
        fake_df = pd.DataFrame([
            {
                "doc_id": f"{tid}_2", "chunk_id": "c1", "text": "TC",
                "label": "ORG", "start": 0, "end": 2,
                "mention_sentence": "TC is growing.", "confidence": 1.0,
            }
        ])
        fake_chunks = [Chunk(doc_id=f"{tid}_2", chunk_id="c1", owner_id="u1",
                            sentence="TC is growing.", date_time="")]

        with patch.object(resolver, '_extract_entities_with_coref', return_value=(fake_df, fake_chunks)):
            _, clean_df2, review_df2 = resolver.process_document(f"{tid}_2", "dummy", "u1")

        assert not clean_df2.empty
        assert clean_df2.iloc[0]["canonical_name"] == "TestCorp"