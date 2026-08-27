# ingestion/unified_resolver.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from ingestion.llm.llm_extractors import NERExtractor, NERWithConfidence, CorefResolver
from storage.factory import StorageFactory
from storage.outbox import OutboxPoller
from ingestion.candidate_finder import CandidateFinder
from shared.data_classes import (
    Entity, EntityLabels, DisambiguationStatus, ResolvedEntity, Relation, Chunk, CandidateResult
)
from shared.utils import (
    WikipediaEntitySummarizer, ValueNormalizer, semantic_sentence_chunk, NON_LINKABLE_TYPES
)
from config.settings import settings

logger = logging.getLogger(__name__)


class UnifiedEntityResolver:
    """
    Production-grade entity resolver.
    - Stateless: all persistence goes through StorageFactory.
    - ACID outbox: user resolutions are journaled in Neo4j and fanned out async.
    """

    def __init__(
        self,
        factory: Optional[StorageFactory] = None,
        embedder=None,
        merge_threshold: float = 0.88,
        cross_doc_threshold: float = 0.82,
        use_ner_with_confidence: bool = False,
        use_cot: bool = True,
        run_demo: bool = False,
    ):
        self.factory = factory or StorageFactory.from_env()
        self.embedder = embedder or self._load_embedder()

        # Ensure storage backends are initialized
        self.factory.init_all()

        # Candidate finder replaces NEDEngine
        self.finder = CandidateFinder(self.factory, embedder=self.embedder)

        # Wiki summarizer (kept from your original code)
        self.wiki = WikipediaEntitySummarizer(self.embedder)

        self.nlp = spacy.load(settings.SPACY_MODEL_PATH)
        if run_demo == False:
            self.ner_llm = NERWithConfidence() if use_ner_with_confidence else NERExtractor(use_cot=use_cot)
            self.coref = CorefResolver(self.nlp, mode="nlp")
        else:
            self.ner_llm = None
            self.coref = None

        self.merge_threshold = merge_threshold
        self.cross_doc_threshold = cross_doc_threshold
        self.run_demo = run_demo

    def _load_embedder(self):
        model_path = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_path)

    # ── Public API ───────────────────────────────────────────────────────────

    def process_document(self, doc_id: str, document: str, owner_id: str) -> Tuple[List[Chunk], pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: NER → Coref → Resolve → Cross-doc → Merge → Queue.
        Returns: (chunk_info, clean_df, review_df)
        """
        # 1. State: document uploaded
        self.factory.clickhouse.create_document(doc_id, owner_id)

        # 2. NER + Coref (your existing logic)
        if self.run_demo:
            chunk_info, entities_df = self._fake_ner_for_structure(doc_id, document)
        else:
            entities_df, chunk_info = self._extract_entities_with_coref(document)
            entities_df = self._remove_subsumed_entities(entities_df)
        # ---

        self.factory.clickhouse.transition_state(doc_id, "uploaded", "ner_done")

        # 3. Resolve each entity
        resolved_pairs = []
        for _, row in entities_df.iterrows():
            resolved = self._resolve_entity(row)
            resolved_pairs.append((resolved, row))

            # Audit log
            self.factory.clickhouse.log_mention(
                doc_id=doc_id,
                chunk_id=row.get("chunk_id"),
                canonical_name=resolved.canonical_name,
                original_text=row["text"],
                entity_label=resolved.entity_label,
                mention_sentence=row["mention_sentence"],
                start=row["start"],
                end=row["end"],
                confidence=resolved.confidence,
                source=resolved.source,
            )

        self.factory.clickhouse.transition_state(doc_id, "ner_done", "er_done")

        # 4. Split clean vs review
        clean_pairs = [p for p in resolved_pairs if p[0].status == DisambiguationStatus.RESOLVED]
        review_pairs = [p for p in resolved_pairs if p[0].status != DisambiguationStatus.RESOLVED]

        # 5. Cross-document resolution on unresolved
        if review_pairs:
            review_entities = [p[0] for p in review_pairs]
            review_entities = self._cross_doc_resolve(review_entities)
            newly_resolved = [(e, row) for e, (_, row) in zip(review_entities, review_pairs) if e.status == DisambiguationStatus.RESOLVED]
            still_review = [(e, row) for e, (_, row) in zip(review_entities, review_pairs) if e.status != DisambiguationStatus.RESOLVED]
            clean_pairs.extend(newly_resolved)
            review_pairs = still_review

        # 6. Merge similar nodes (Redis distributed lock)
        self._merge_catalog()

        # 7. Persist unresolved queue
        for r, row in review_pairs:
            self.factory.clickhouse.enqueue_unresolved(
                resolution_id=f"{doc_id}_{r.original_text}_{row.get('start', 0)}",
                doc_id=doc_id,
                entity_row=r,
                candidates_json=json.dumps([c.to_dict() for c in r.kg_candidates]),
            )
        # 8. State transition
        if review_pairs:
            self.factory.clickhouse.transition_state(doc_id, "er_done", "review_pending")
        else:
            self.factory.clickhouse.transition_state(doc_id, "er_done", "re_done")

        clean_df = pd.DataFrame([
            {**r.to_dict(), "doc_id": row.get("doc_id"), "mention_sentence": row.get("mention_sentence")}
            for r, row in clean_pairs
        ]) if clean_pairs else pd.DataFrame()

        review_df = pd.DataFrame([
            {**r.to_dict(), "doc_id": row.get("doc_id"), "mention_sentence": row.get("mention_sentence")}
            for r, row in review_pairs
        ]) if review_pairs else pd.DataFrame()

        return chunk_info, clean_df, review_df

    def add_user_resolution(
        self,
        canonical_name: str,
        entity_label: EntityLabels,
        summary: str,
        aliases: List[str] = None,
        context_indicators: Optional[List[str]] = None,
        related_to: Optional[List[str]] = None,
    ) -> None:
        """
        Operator resolves an entity. SINGLE write path for user-confirmed entities.
        Neo4j upsert is atomic and creates an OutboxEvent in the same tx.
        """
        canonical = canonical_name.strip()
        alias_list = list(set([canonical] + [a.strip() for a in (aliases or []) if a.strip()]))

        # 1. Graph source of truth + outbox journal (ACID)
        self.factory.neo4j.upsert_entity(
            canonical=canonical,
            label=entity_label.value if hasattr(entity_label, "value") else entity_label,
            aliases=alias_list,
            summary=summary,
            source="user",
            related_to=related_to or [],
            context_indicators=context_indicators or [],
        )

        # 2. Immediate Redis invalidation (best effort; outbox poller will redo)
        for alias in alias_list:
            self.factory.redis.invalidate_alias(alias.lower())
        self.factory.redis.invalidate_candidates()

        # 3. ClickHouse: mark any queue items for this alias as resolved
        # (Implementation depends on your UI matching logic)

        logger.info(f"[User Feedback] '{canonical}' ({entity_label.value}) registered. Outbox will sync to ES/Qdrant.")

    # ── NED + Coreference Resolution ─────────────────────────────────────────────────────
     
    def _name_entity_recognition(self, doc:str)-> List[Dict]:
        all_preds = []
        char_offset = 0
        
        for idx, chunk in enumerate(semantic_sentence_chunk(doc)):
            result = self.ner_llm(chunk)
            chunk = result['sentence']
            
            chunk_pred = {}
            chunk_ents = []           
            chunk_pred["chunk_text"] = chunk
            chunk_pred["chunk_id"] = idx + 1
            
            # CRITICAL: Adjust offsets to document-level
            for ent in result.get("entities", []):
                ent.start += char_offset
                ent.end += char_offset
                chunk_ents.append(ent)                

            char_offset += len(chunk) + 1
            chunk_pred["entities"] = chunk_ents
            all_preds.append(chunk_pred)
        return all_preds
    
    def _chunk_entity_splitter(self, ner_result:list[Dict]):
        all_entities = []
        chunk_info = []
        for chunk in ner_result:
            chunk_info.append((chunk["chunk_id"], chunk["chunk_text"]))
            all_entities.extend(chunk["entities"])

        return all_entities, chunk_info
    
    def _extract_entities_with_coref(self, document:str):
        entity_extracted = self._name_entity_recognition(document)
        all_entities, chunk_info = self._chunk_entity_splitter(entity_extracted)
        return self.coref.resolve_from_sentences(document, all_entities), chunk_info
    
    def _remove_subsumed_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop entities whose character span is fully contained inside a larger
        entity span in the same document. Prefer longer spans.
        """
        if df.empty:
            return df

        rows = df.to_dict("records")

        # Sort by span length descending, then by start position
        rows_sorted = sorted(
            rows,
            key=lambda r: (r["end"] - r["start"], r["start"]),
            reverse=True,
        )

        kept = []
        kept_spans = []  # (doc_id, start, end)

        for row in rows_sorted:
            doc_id, s, e = row["doc_id"], row["start"], row["end"]

            # Is this row fully contained inside an already-kept span?
            is_subsumed = any(
                doc_id == kd and s >= ks and e <= ke and (s != ks or e != ke)
                for kd, ks, ke in kept_spans
            )

            if not is_subsumed:
                kept.append(row)
                kept_spans.append((doc_id, s, e))

        # Restore original document order
        kept_sorted = sorted(kept, key=lambda r: (r["doc_id"], r["start"]))
        return pd.DataFrame(kept_sorted)
    
    # ── Resolution Logic ─────────────────────────────────────────────────────

    def _resolve_entity(self, row: pd.Series) -> ResolvedEntity:
        entity_text = row["text"]
        entity_label = row["label"]
        sentence = row["mention_sentence"]
        confidence = row.get("confidence", 1.0)

        # Non-linkable types (DATE, MONEY, etc.) — keep your existing logic
        if entity_label in NON_LINKABLE_TYPES:
            return self._normalize_literal(entity_text, entity_label, sentence, confidence)

        # 1. Catalog lookup (Redis hot path)
        cached_canon = self.factory.redis.get_alias(entity_text.lower().strip())
        if cached_canon:
            node = self.factory.neo4j.find_by_canonical(cached_canon)
            if node:
                return ResolvedEntity(
                    original_text=entity_text,
                    canonical_name=node["canonical"],
                    entity_label=node["label"],
                    mention_sentence=sentence,
                    confidence=confidence,
                    status=DisambiguationStatus.RESOLVED,
                    source="catalog",
                    summary=node.get("summary"),
                    context_clues=["Redis alias cache hit"],
                    needs_review=False,
                )

        # 2. CandidateFinder (Neo4j + ES + Qdrant ensemble)
        candidates = self.finder.find_candidates(entity_text, entity_label, sentence)

        if not candidates:
            # 3. Wikipedia fallback
            if self.wiki:
                wiki = self.wiki.summarize(entity=entity_text, context=sentence, ner_label=entity_label)
                if wiki:
                    # Auto-register to catalog via outbox
                    self.add_user_resolution(
                        canonical_name=wiki.canonical_name,
                        entity_label=EntityLabels(wiki.entity_label),
                        summary=wiki.summary or "",
                        aliases=[wiki.original_text, wiki.canonical_name],
                    )
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=wiki.canonical_name,
                        entity_label=wiki.entity_label,
                        mention_sentence=sentence,
                        confidence=wiki.confidence,
                        status=DisambiguationStatus.RESOLVED,
                        source="wiki",
                        summary=wiki.summary,
                        needs_review=False,
                    )

            return ResolvedEntity(
                original_text=entity_text,
                canonical_name=entity_text,
                entity_label=entity_label,
                mention_sentence=sentence,
                confidence=confidence,
                status=DisambiguationStatus.UNKNOWN,
                source="unresolved",
                kg_candidates=[],
                context_clues=["No match in KG, Wikipedia, or Catalog"],
                needs_review=True,
                is_nil=True,
            )

        # 4. Disambiguate from candidates
        return self._disambiguate(entity_text, entity_label, sentence, candidates)

    def _disambiguate(
        self,
        entity_text: str,
        entity_label_hint: Optional[str],
        context: str,
        candidates: List[CandidateResult],
    ) -> ResolvedEntity:
        # Single candidate fast path
        if len(candidates) == 1:
            cand = candidates[0]
            label_match = self._check_label_match(entity_label_hint, cand.label)
            if cand.match_score >= 0.88:
                status = DisambiguationStatus.RESOLVED
                conf = cand.match_score * (0.9 if label_match else 0.85)
                needs_review = False
            else:
                status = DisambiguationStatus.AMBIGUOUS
                conf = cand.match_score
                needs_review = True

            return ResolvedEntity(
                original_text=entity_text,
                canonical_name=cand.canonical,
                entity_label=cand.label,
                mention_sentence=context,
                confidence=conf,
                status=status,
                source="kg",
                kg_candidates=[c.to_dict() for c in candidates],
                context_clues=[f"Single candidate: {cand.match_method}"],
                needs_review=needs_review,
            )

        # Multi-candidate: pick best
        best = candidates[0]
        label_match = self._check_label_match(entity_label_hint, best.label)
        if best.match_score > 0.88 and label_match:
            status = DisambiguationStatus.RESOLVED
        else:
            status = DisambiguationStatus.AMBIGUOUS

        return ResolvedEntity(
            original_text=entity_text,
            canonical_name=best.canonical,
            entity_label=best.label,
            mention_sentence=context,
            confidence=best.match_score,
            status=status,
            source="kg",
            kg_candidates=[c.to_dict() for c in candidates],
            context_clues=[f"Best candidate score: {best.match_score}"],
            needs_review=not (label_match and best.match_score > 0.88),
        )

    def _cross_doc_resolve(self, unresolved: List[ResolvedEntity]) -> List[ResolvedEntity]:
        """Qdrant semantic search for unresolved mentions."""
        if not self.embedder:
            return unresolved

        for ent in unresolved:
            if ent.status != DisambiguationStatus.UNKNOWN and not ent.is_nil:
                continue
            vec = self.embedder.encode([ent.mention_sentence], convert_to_numpy=True)[0].tolist()
            hits = self.factory.qdrant.search_similar_entities(vec, top_k=1)
            if hits and hits[0]["score"] >= self.cross_doc_threshold:
                canonical = hits[0]["canonical"]
                node = self.factory.neo4j.find_by_canonical(canonical)
                if node:
                    ent.canonical_name = canonical
                    ent.confidence = hits[0]["score"]
                    ent.status = DisambiguationStatus.RESOLVED
                    ent.source = "cross_doc"
                    ent.summary = node.get("summary")
                    ent.is_nil = False
                    ent.context_clues.append(f"Cross-doc match to {canonical}")
        return unresolved

    def _merge_catalog(self) -> None:
        """Entity merge with distributed lock."""
        if not self.factory.redis.acquire_lock("entity_merge", timeout=60, blocking=False):
            logger.info("Merge lock held by another worker; skipping.")
            return

        try:
            # Fetch all vectors from Qdrant
            points = self.factory.qdrant.fetch_all("entity_summaries")
            if len(points) < 2:
                return

            vectors = np.array([p["vector"] for p in points])
            sim_matrix = sk_cosine_similarity(vectors)
            merged = set()

            for i in range(len(points)):
                if points[i]["canonical"] in merged:
                    continue
                for j in range(i + 1, len(points)):
                    if points[j]["canonical"] in merged:
                        continue
                    sim = sim_matrix[i][j]
                    if sim >= self.merge_threshold:
                        absorb = points[j]["canonical"]
                        keep = points[i]["canonical"]
                        # Prefer higher mention count (from ClickHouse or Neo4j)
                        self.factory.neo4j.merge_entities(absorb, keep)
                        merged.add(absorb)
                        logger.info(f"Merged '{absorb}' -> '{keep}' (sim={sim:.3f})")
        finally:
            self.factory.redis.release_lock("entity_merge")

    def _normalize_literal(self, text, label, sentence, confidence):
        # ... (keep your existing MONEY/DATE/TIME normalization logic) ...
        if label == "MONEY":
            normalized = ValueNormalizer.normalize_money(text)
            return ResolvedEntity(
                original_text=text, canonical_name=normalized["canonical"],
                entity_label=label, mention_sentence=sentence,
                confidence=confidence, status=DisambiguationStatus.RESOLVED,
                source="normalized", needs_review=False,
            )
        elif label == "DATE":
            normalized = ValueNormalizer.normalize_date(text)
            return ResolvedEntity(
                original_text=text, canonical_name=normalized["canonical"],
                entity_label=label, mention_sentence=sentence,
                confidence=confidence, status=DisambiguationStatus.RESOLVED,
                source="normalized", needs_review=False,
            )
            
        # elif label == "TIME":
        #     normalized = ValueNormalizer.normalize_time(text)
        #     return ResolvedEntity(
        #         original_text=text, canonical_name=normalized["canonical"],
        #         entity_label=label, mention_sentence=sentence,
        #         confidence=confidence, status=DisambiguationStatus.RESOLVED,
        #         source="normalized", needs_review=False,
        #     )
        # ... etc ...
        return ResolvedEntity(
            original_text=text, canonical_name=text, entity_label=label,
            mention_sentence=sentence, confidence=confidence,
            status=DisambiguationStatus.RESOLVED, source="normalized", needs_review=False,
        )

    def _check_label_match(self, hint, candidate_label):
        if not hint or hint == "UNKNOWN":
            return True
        return hint.upper() == candidate_label.upper()

    def _fake_ner_for_structure(self, doc_id, document):
        """Replace this with your real _extract_entities_with_coref."""
        # Returns (chunk_info, entities_df)
        from shared.data_classes import Chunk
        chunks = [Chunk(doc_id=doc_id, chunk_id="c1", owner_id="user", sentence=document, date_time="")]
        df = pd.DataFrame([{
            "doc_id": doc_id, "chunk_id": "c1", "text": "Miguel Riofrio",
            "label": "PER", "start": 0, "end": 14,
            "mention_sentence": document, "confidence": 1.0
        }])
        return chunks, df