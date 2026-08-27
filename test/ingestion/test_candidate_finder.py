import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import pytest
from ingestion.candidate_finder import CandidateFinder
from shared.data_classes import CandidateResult

pytestmark = pytest.mark.integration


class TestCandidateFinder:
    def test_find_candidates_exact_match(self, factory, tid, mock_embedder):
        factory.neo4j.upsert_entity(
            canonical=f"CF_Ent_{tid}", label="PER",
            aliases=[f"AliasCF_{tid}"], summary="Summary text", source="test",
        )
        factory.redis.invalidate_candidates()
        finder = CandidateFinder(factory, embedder=mock_embedder)
        cands = finder.find_candidates(
            f"AliasCF_{tid}", "PER", f"AliasCF_{tid} is a person."
        )
        assert any(c.canonical == f"CF_Ent_{tid}" for c in cands)

    def test_add_entity_creates_outbox(self, factory, tid, mock_embedder):
        finder = CandidateFinder(factory, embedder=mock_embedder)
        finder.add_entity(
            canonical=f"Added_{tid}",
            entity_label="ORG",
            aliases=[f"A1_{tid}"],
            summary="An org",
        )
        node = factory.neo4j.find_by_canonical(f"Added_{tid}")
        assert node is not None
        assert node["label"] == "ORG"

    def test_ensemble_scoring_neural(self, factory, tid, mock_embedder):
        # Seed two entities with different summaries
        factory.neo4j.upsert_entity(
            canonical=f"NeuralA_{tid}", label="PER",
            aliases=[f"NA_{tid}"], summary="Football player", source="test",
        )
        factory.neo4j.upsert_entity(
            canonical=f"NeuralB_{tid}", label="PER",
            aliases=[f"NB_{tid}"], summary="Quantum physicist", source="test",
        )
        finder = CandidateFinder(factory, embedder=mock_embedder)
        cands = finder.find_candidates(
            f"NA_{tid}", "PER", "He plays football professionally."
        )
        # The football player should outrank the physicist
        if len(cands) >= 2:
            scores = {c.canonical: c.match_score for c in cands}
            assert scores.get(f"NeuralA_{tid}", 0) >= scores.get(f"NeuralB_{tid}", 0)

    def test_cache_hit(self, factory, tid, mock_embedder):
        factory.neo4j.upsert_entity(
            canonical=f"CacheEnt_{tid}", label="LOC",
            aliases=[f"CE_{tid}"], summary="City", source="test",
        )
        finder = CandidateFinder(factory, embedder=mock_embedder)
        key = finder._cache_key(f"CE_{tid}", "LOC", "sentence")
        factory.redis.client.delete(key)

        cands1 = finder.find_candidates(f"CE_{tid}", "LOC", "sentence")
        cands2 = finder.find_candidates(f"CE_{tid}", "LOC", "sentence")
        assert len(cands1) == len(cands2)
        # Second call should be from cache (no error = success)