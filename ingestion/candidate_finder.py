import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import hashlib
import logging
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from storage.factory import StorageFactory
from shared.data_classes import CandidateResult
from shared.utils import jaccard_similarity, levenshtein_ratio

logger = logging.getLogger(__name__)


class CandidateFinder:
    """
    Stateless coordinator replacing NEDEngine.
    Queries Redis → Elasticsearch → Neo4j → Qdrant in sequence,
    then applies the original ensemble scoring weights.
    """

    def __init__(self, factory: StorageFactory, embedder=None, threshold: float = 0.5):
        self.factory = factory
        self.embedder = embedder
        self.threshold = threshold

    def find_candidates(
        self,
        entity_text: str,
        entity_label: str,
        entity_sentence: str,
        threshold: Optional[float] = None,
    ) -> List[CandidateResult]:
        """
        Four-phase retrieval with neural re-rank.
        Returns sorted CandidateResult list.
        """
        thresh = threshold if threshold is not None else self.threshold
        cache_key = self._cache_key(entity_text, entity_label, entity_sentence)

        # ── Phase 0: Redis cache ──────────────────────────────────────────────
        cached = self.factory.redis.client.get(cache_key)
        if cached:
            return [CandidateResult(**c) for c in json.loads(cached)]

        # ── Phase 1: Neo4j graph candidates (exact + partial + word overlap) ─
        candidates = self.factory.neo4j.find_candidates_cypher(entity_text, entity_label)
        seen = {c.canonical for c in candidates}

        # ── Phase 2: Elasticsearch fuzzy recall boost ─────────────────────────
        es_hits = self.factory.es.search_aliases(entity_text, top_k=10)
        for hit in es_hits:
            if hit["canonical"] not in seen:
                seen.add(hit["canonical"])
                candidates.append(CandidateResult(
                    canonical=hit["canonical"],
                    label=hit.get("label", "UNKNOWN"),
                    aliases=hit.get("aliases", []),
                    summary=hit.get("summary", "")[:120] + "...",
                    context_indicators=hit.get("context_indicators", []),
                    related_to=hit.get("related_to", []),
                    match_score=0.3,  # base ES score, will be re-ranked
                    match_method="es_fuzzy",
                    source="elasticsearch",
                ))

        if not candidates:
            return []

        # ── Phase 3: Neural + context ensemble scoring ───────────────────────
        candidates = self._ensemble_score(candidates, entity_text, entity_label, entity_sentence)

        # ── Phase 4: Threshold & sort ────────────────────────────────────────
        candidates = [c for c in candidates if c.match_score >= thresh]
        candidates.sort(key=lambda x: x.match_score, reverse=True)

        # Cache for 60 seconds
        self.factory.redis.client.set(cache_key, json.dumps([c.to_dict() for c in candidates]), ex=60)
        return candidates

    def add_entity(
        self,
        canonical: str,
        entity_label: str,
        aliases: List[str],
        summary: str = "",
        context_indicators: List[str] = None,
        related_to: List[str] = None,
    ) -> None:
        """
        Write-through to Neo4j (which auto-creates outbox event).
        The outbox poller will sync to ES/Qdrant/Redis asynchronously.
        """
        all_aliases = list(set([canonical] + [a.strip() for a in aliases if a.strip()]))
        self.factory.neo4j.upsert_entity(
            canonical=canonical.strip(),
            label=entity_label,
            aliases=all_aliases,
            summary=summary,
            source="catalog",
            related_to=related_to,
            context_indicators=context_indicators,
        )
        # Redis immediate invalidation (best effort)
        for alias in all_aliases:
            self.factory.redis.invalidate_alias(alias.lower())

    # ── Internal ─────────────────────────────────────────────────────────────

    def _cache_key(self, entity_text: str, entity_label: str, sentence: str) -> str:
        h = hashlib.sha1(f"{entity_text.lower()}|{entity_label}|{sentence}".encode()).hexdigest()
        return f"candidates:{h}"

    def _ensemble_score(
        self,
        candidates: List[CandidateResult],
        entity_text: str,
        entity_label: str,
        entity_sentence: str,
    ) -> List[CandidateResult]:
        if not self.embedder:
            return candidates

        # Batch encode all candidate summaries + the query sentence
        summaries = [c.summary for c in candidates if c.summary]
        if not summaries:
            return candidates

        try:
            sent_vec = self.embedder.encode([entity_sentence], convert_to_numpy=True)
            sum_vecs = self.embedder.encode(summaries, convert_to_numpy=True)
            sims = sk_cosine_similarity(sent_vec, sum_vecs)[0]
        except Exception:
            sims = [0.0] * len(candidates)

        sim_idx = 0
        for cand in candidates:
            neural_sim = 0.0
            if cand.summary and cand.summary != "...":
                neural_sim = float(sims[sim_idx])
                sim_idx += 1

            jaccard_ctx = jaccard_similarity(entity_sentence, cand.summary) if cand.summary else 0.0
            lev_name = levenshtein_ratio(entity_text.lower(), cand.summary.lower())

            label_bonus = 0.1 if (entity_label and cand.label.upper() == entity_label.upper()) else 0.0

            # Original weights from your NEDEngine
            final_score = (cand.match_score * 0.1) + \
                          (neural_sim * 0.70) + \
                          (jaccard_ctx * 0.05) + \
                          (lev_name * 0.05) + \
                          label_bonus

            if cand.match_method == 'exact_match':
                final_score = max(final_score, 1.0)
                
            cand.neural_sim = round(neural_sim, 3)
            cand.jaccard_ctx = round(jaccard_ctx, 3)
            cand.levenshtein_name = round(lev_name, 3)
            cand.label_match = label_bonus > 0
            cand.match_score = round(min(final_score, 1.0), 3)
            cand.match_method = f"{cand.match_method}+ensemble"

        return candidates