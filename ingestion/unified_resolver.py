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
from ingestion.models.base import IExtractor
from storage.factory import StorageFactory
from ingestion.candidate_finder import CandidateFinder
from shared.data_classes import (
    EntityLabels, DisambiguationStatus, Chunk, CandidateResult, Entity
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
        ner_model:IExtractor,
        coref_model:IExtractor,
        factory: Optional[StorageFactory] = None,
        embedder=None,
        merge_threshold: float = 0.88,
        cross_doc_threshold: float = 0.82,
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
            self.ner = ner_model
            self.coref = coref_model
        else:
            self.ner = None
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
        self.factory.postgres.create_document(doc_id, owner_id)

        # 2. NER + Coref (your existing logic)
        if self.run_demo:
            chunk_info, entities = self._fake_ner_for_structure(doc_id, document)
        else:
            entities, chunk_info = self._extract_entities_with_coref(document)
            entities = self._remove_subsumed_entities(entities)
        # ---

        self.factory.postgres.transition_state(doc_id, "uploaded", "ner_done")

        # 3. Resolve each entity
        resolved_pairs = []
        for _, row in entities_df.iterrows():
            resolved = self._resolve_entity(row)
            resolved_pairs.append((resolved, row))

            # Audit log
            self.factory.postgres.log_mention(
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

        self.factory.postgres.transition_state(doc_id, "ner_done", "er_done")

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
            self.factory.postgres.enqueue_unresolved(
                resolution_id=f"{doc_id}_{r.original_text}_{row.get('start', 0)}",
                doc_id=doc_id,
                entity_row=r,
                candidates_json=json.dumps([c.to_dict() for c in r.kg_candidates]),
            )
        # 8. State transition
        if review_pairs:
            self.factory.postgres.transition_state(doc_id, "er_done", "review_pending")
        else:
            self.factory.postgres.transition_state(doc_id, "er_done", "re_done")

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
    def _name_entity_recognition(self, doc: str) -> List[Entity]:
        all_preds = []

        for idx, chunk_info in enumerate(semantic_sentence_chunk(doc, self.embedder)):
            chunk = chunk_info["text"]
            offset = chunk_info["start"]  # exact, no drift

            result = self.ner(chunk)
            chunk_pred = {
                "chunk_text": chunk,
                "chunk_id": idx + 1,
                "entities": [],
            }

            for ent in result:
                ent.start += offset
                ent.end += offset
                chunk_pred["entities"].append(ent)

            all_preds.append(chunk_pred)

        return all_preds
    
    def _chunk_entity_splitter(self, ner_result:list[Dict]):
        all_entities = []
        chunk_info = []
        for chunk in ner_result:
            chunk_info.append((chunk["chunk_id"], chunk["chunk_text"]))
            all_entities.extend(chunk["entities"])

        return all_entities, chunk_info
    
    def _extract_entities_with_coref(self, doc_id: str, document: str, owner_id: str):
        entity_extracted = self._name_entity_recognition(document)
        all_entities, chunk_info = self._chunk_entity_splitter(entity_extracted)
        return self.coref.extract(document, all_entities), chunk_info
    
    def _remove_subsumed_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Drop entities whose character span is fully contained inside a larger
        entity span in the same document. Prefer longer spans.
        """
        if not entities:
            return entities

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

    def _resolve_entity(self, row: pd.Series) -> Entity:
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
                return Entity(
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
                    return Entity(
                        original_text=entity_text,
                        canonical_name=wiki.canonical_name,
                        entity_label=EntityLabels(wiki.entity_label),
                        mention_sentence=sentence,
                        confidence=wiki.confidence,
                        status=DisambiguationStatus.RESOLVED,
                        source="wiki",
                        summary=wiki.summary,
                        needs_review=False,
                    )

            return Entity(
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
    ) -> Entity:
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

            return Entity(
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

        return Entity(
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

    def _cross_doc_resolve(self, unresolved: List[Entity]) -> List[Entity]:
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
            return Entity(
                original_text=text, canonical_name=normalized["canonical"],
                entity_label=label, mention_sentence=sentence,
                confidence=confidence, status=DisambiguationStatus.RESOLVED,
                source="normalized", needs_review=False,
            )
        elif label == "DATE":
            normalized = ValueNormalizer.normalize_date(text)
            return Entity(
                original_text=text, canonical_name=normalized["canonical"],
                entity_label=label, mention_sentence=sentence,
                confidence=confidence, status=DisambiguationStatus.RESOLVED,
                source="normalized", needs_review=False,
            )
            
        # elif label == "TIME":
        #     normalized = ValueNormalizer.normalize_time(text)
        #     return Entity(
        #         original_text=text, canonical_name=normalized["canonical"],
        #         entity_label=label, mention_sentence=sentence,
        #         confidence=confidence, status=DisambiguationStatus.RESOLVED,
        #         source="normalized", needs_review=False,
        #     )
        # ... etc ...
        return Entity(
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
    
    
if __name__ == "__main__":
    from ingestion.models.factory import get_ner_extractor, get_coref_resolver
    
    ner = get_ner_extractor(model_name=settings.BERT_NER_MODEL_NAME, local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"), use_onnx=True, onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner"))
    coref = get_coref_resolver()

    resolver = UnifiedEntityResolver(ner_model=ner, coref_model=coref)  # Assuming UnifiedResolver is the class name
    doc_id = "doc1"
    record = {"doc_id": 6, "document": "Ezri Dax ( ) is a fictional character who appears in the of the American science fiction TV series .Portrayed by Nicole de Boer , she is a counselor aboard the Bajoran space station Deep Space Nine .The character is a member of the Trill species , and is formed of both a host and a symbiont – referred to as Dax .Ezri was introduced to the series following the death of the previous Dax host , Jadzia ( Terry Farrell ) at the end of .It had been the producers ' intention to introduce a new female character bearing the symbiont in order to ensure that Nana Visitor as Kira Nerys was not the only female member of the main cast .There were difficulties in casting initially , and the character changed from one who was intended to be \" spooky \" to one that was struggling to deal with all her previous personalities as a result of unexpectedly taking on the Dax symbiont .De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role .The character made her first appearance in the first episode of the seventh season , \" Image in the Sand \" .The character continued to appear throughout the final season of the series , with her final appearance in the series finale \" What You Leave Behind \" .Her character stepped into the void left by Jadzia amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with Jadzia 's widower , Worf ( Michael Dorn ) .During the course of the season , Ezri becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with Dr. Julian Bashir ( Alexander Siddig ) .The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for \" \" – an episode described as \" just a mess \" by writer Ronald D. Moore .The relationship between Ezri and both Worf and Bashir was described as one of five \" great geek TV love triangles \" .The inclusion of the character was criticised on the internet , with Ezri being referred to as both an \" ill - conceived idea \" and a \" replacement Dax \" .", "sentence": [["Ezri", "Dax", "(", ")", "is", "a", "fictional", "character", "who", "appears", "in", "the", "of", "the", "American", "science", "fiction", "TV", "series", "."], ["Portrayed", "by", "Nicole", "de", "Boer", ",", "she", "is", "a", "counselor", "aboard", "the", "Bajoran", "space", "station", "Deep", "Space", "Nine", "."], ["The", "character", "is", "a", "member", "of", "the", "Trill", "species", ",", "and", "is", "formed", "of", "both", "a", "host", "and", "a", "symbiont", "–", "referred", "to", "as", "Dax", "."], ["Ezri", "was", "introduced", "to", "the", "series", "following", "the", "death", "of", "the", "previous", "Dax", "host", ",", "Jadzia", "(", "Terry", "Farrell", ")", "at", "the", "end", "of", "."], ["It", "had", "been", "the", "producers", "'", "intention", "to", "introduce", "a", "new", "female", "character", "bearing", "the", "symbiont", "in", "order", "to", "ensure", "that", "Nana", "Visitor", "as", "Kira", "Nerys", "was", "not", "the", "only", "female", "member", "of", "the", "main", "cast", "."], ["There", "were", "difficulties", "in", "casting", "initially", ",", "and", "the", "character", "changed", "from", "one", "who", "was", "intended", "to", "be", "\"", "spooky", "\"", "to", "one", "that", "was", "struggling", "to", "deal", "with", "all", "her", "previous", "personalities", "as", "a", "result", "of", "unexpectedly", "taking", "on", "the", "Dax", "symbiont", "."], ["De", "Boer", "was", "not", "considered", "for", "the", "part", "until", "co", "-", "producer", "Hans", "Beimler", "suggested", "that", "she", "should", "submit", "an", "audition", "tape", ",", "which", "resulted", "in", "her", "invitation", "to", "meet", "with", "the", "producers", "in", "Los", "Angeles", "and", "in", "her", "gaining", "the", "role", "."], ["The", "character", "made", "her", "first", "appearance", "in", "the", "first", "episode", "of", "the", "seventh", "season", ",", "\"", "Image", "in", "the", "Sand", "\"", "."], ["The", "character", "continued", "to", "appear", "throughout", "the", "final", "season", "of", "the", "series", ",", "with", "her", "final", "appearance", "in", "the", "series", "finale", "\"", "What", "You", "Leave", "Behind", "\"", "."], ["Her", "character", "stepped", "into", "the", "void", "left", "by", "Jadzia", "amongst", "the", "crew", ",", "but", "found", "that", "she", "had", "to", "redevelop", "those", "previous", "relationships", "and", "learn", "to", "get", "along", "with", "Jadzia", "'s", "widower", ",", "Worf", "(", "Michael", "Dorn", ")", "."], ["During", "the", "course", "of", "the", "season", ",", "Ezri", "becomes", "less", "nervous", "of", "her", "role", "over", "time", "and", "learns", "from", "the", "Dax", "symbiont", "and", "becomes", "involved", "romantically", "with", "Dr.", "Julian", "Bashir", "(", "Alexander", "Siddig", ")", "."], ["The", "fan", "reaction", "to", "the", "character", "was", "reported", "as", "positive", ",", "but", "several", "of", "the", "Ezri", "-", "centric", "episodes", "came", "in", "for", "criticism", ",", "with", "producer", "Ira", "Steven", "Behr", "apologising", "to", "de", "Boer", "for", "\"", "\"", "–", "an", "episode", "described", "as", "\"", "just", "a", "mess", "\"", "by", "writer", "Ronald", "D.", "Moore", "."], ["The", "relationship", "between", "Ezri", "and", "both", "Worf", "and", "Bashir", "was", "described", "as", "one", "of", "five", "\"", "great", "geek", "TV", "love", "triangles", "\"", "."], ["The", "inclusion", "of", "the", "character", "was", "criticised", "on", "the", "internet", ",", "with", "Ezri", "being", "referred", "to", "as", "both", "an", "\"", "ill", "-", "conceived", "idea", "\"", "and", "a", "\"", "replacement", "Dax", "\"", "."]], "label_sents": [[{"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}, {"name": "Ezri", "sent_id": 3, "pos": [0, 1], "type": "PER"}, {"name": "Ezri", "sent_id": 13, "pos": [12, 13], "type": "PER"}, {"name": "Ezri", "sent_id": 12, "pos": [3, 4], "type": "PER"}, {"name": "Ezri", "sent_id": 11, "pos": [15, 16], "type": "PER"}, {"name": "Ezri", "sent_id": 10, "pos": [7, 8], "type": "PER"}, {"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}], [{"name": "American", "sent_id": 0, "pos": [14, 15], "type": "LOC"}], [{"name": "Nicole de Boer", "sent_id": 1, "pos": [2, 5], "type": "PER"}], [{"name": "Bajoran", "sent_id": 1, "pos": [12, 13], "type": "LOC"}], [{"name": "Deep Space Nine", "sent_id": 1, "pos": [15, 18], "type": "MISC"}], [{"name": "Trill", "sent_id": 2, "pos": [7, 8], "type": "MISC"}], [{"name": "Dax", "sent_id": 10, "pos": [20, 21], "type": "MISC"}, {"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "MISC"}, {"name": "Dax", "sent_id": 5, "pos": [41, 42], "type": "MISC"}], [{"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "PER"}, {"name": "Dax", "sent_id": 3, "pos": [12, 13], "type": "PER"}, {"name": "Dax", "sent_id": 13, "pos": [29, 30], "type": "PER"}], [{"name": "Terry Farrell", "sent_id": 3, "pos": [17, 19], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [29, 30], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [8, 9], "type": "PER"}, {"name": "Jadzia", "sent_id": 3, "pos": [15, 16], "type": "PER"}], [{"name": "Nana Visitor", "sent_id": 4, "pos": [21, 23], "type": "PER"}], [{"name": "Kira Nerys", "sent_id": 4, "pos": [24, 26], "type": "PER"}], [{"name": "De Boer", "sent_id": 6, "pos": [0, 2], "type": "PER"}, {"name": "de Boer", "sent_id": 11, "pos": [31, 33], "type": "PER"}], [{"name": "Hans Beimler", "sent_id": 6, "pos": [12, 14], "type": "PER"}], [{"name": "Los Angeles", "sent_id": 6, "pos": [34, 36], "type": "LOC"}], [{"name": "Image in the Sand", "sent_id": 7, "pos": [16, 20], "type": "MISC"}], [{"name": "What You Leave Behind", "sent_id": 8, "pos": [22, 26], "type": "MISC"}], [{"name": "Worf", "sent_id": 12, "pos": [6, 7], "type": "PER"}, {"name": "Worf", "sent_id": 9, "pos": [33, 34], "type": "PER"}], [{"name": "Michael Dorn", "sent_id": 9, "pos": [35, 37], "type": "PER"}], [{"name": "Julian Bashir", "sent_id": 10, "pos": [28, 30], "type": "PER"}], [{"name": "Alexander Siddig", "sent_id": 10, "pos": [31, 33], "type": "PER"}], [{"name": "Ira Steven Behr", "sent_id": 11, "pos": [26, 29], "type": "PER"}], [{"name": "Ronald D. Moore", "sent_id": 11, "pos": [48, 51], "type": "PER"}], [{"name": "Bashir", "sent_id": 12, "pos": [8, 9], "type": "PER"}], [{"name": "five", "sent_id": 12, "pos": [14, 15], "type": "NUM"}]], "label_doc": [{"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "Ezri", "label": "PER", "start": 314, "end": 318, "sents_id": 3}, {"text": "Ezri", "label": "PER", "start": 2207, "end": 2211, "sents_id": 13}, {"text": "Ezri", "label": "PER", "start": 2045, "end": 2049, "sents_id": 12}, {"text": "Ezri", "label": "PER", "start": 1842, "end": 1846, "sents_id": 11}, {"text": "Ezri", "label": "PER", "start": 1602, "end": 1606, "sents_id": 10}, {"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "American", "label": "LOC", "start": 64, "end": 72, "sents_id": 0}, {"text": "Nicole de Boer", "label": "PER", "start": 113, "end": 127, "sents_id": 1}, {"text": "Bajoran", "label": "LOC", "start": 160, "end": 167, "sents_id": 1}, {"text": "Deep Space Nine", "label": "MISC", "start": 182, "end": 197, "sents_id": 1}, {"text": "Trill", "label": "MISC", "start": 232, "end": 237, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 1670, "end": 1673, "sents_id": 10}, {"text": "Dax", "label": "MISC", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 859, "end": 862, "sents_id": 5}, {"text": "Dax", "label": "PER", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "PER", "start": 384, "end": 387, "sents_id": 3}, {"text": "Dax", "label": "PER", "start": 2286, "end": 2289, "sents_id": 13}, {"text": "Terry Farrell", "label": "PER", "start": 404, "end": 417, "sents_id": 3}, {"text": "Jadzia", "label": "PER", "start": 1525, "end": 1531, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 1406, "end": 1412, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 395, "end": 401, "sents_id": 3}, {"text": "Nana Visitor", "label": "PER", "start": 554, "end": 566, "sents_id": 4}, {"text": "Kira Nerys", "label": "PER", "start": 570, "end": 580, "sents_id": 4}, {"text": "De Boer", "label": "PER", "start": 873, "end": 880, "sents_id": 6}, {"text": "de Boer", "label": "PER", "start": 1935, "end": 1942, "sents_id": 11}, {"text": "Hans Beimler", "label": "PER", "start": 933, "end": 945, "sents_id": 6}, {"text": "Los Angeles", "label": "LOC", "start": 1061, "end": 1072, "sents_id": 6}, {"text": "Image in the Sand", "label": "MISC", "start": 1189, "end": 1206, "sents_id": 7}, {"text": "What You Leave Behind", "label": "MISC", "start": 1337, "end": 1358, "sents_id": 8}, {"text": "Worf", "label": "PER", "start": 2059, "end": 2063, "sents_id": 12}, {"text": "Worf", "label": "PER", "start": 1545, "end": 1549, "sents_id": 9}, {"text": "Michael Dorn", "label": "PER", "start": 1552, "end": 1564, "sents_id": 9}, {"text": "Julian Bashir", "label": "PER", "start": 1726, "end": 1739, "sents_id": 10}, {"text": "Alexander Siddig", "label": "PER", "start": 1742, "end": 1758, "sents_id": 10}, {"text": "Ira Steven Behr", "label": "PER", "start": 1904, "end": 1919, "sents_id": 11}, {"text": "Ronald D. Moore", "label": "PER", "start": 2003, "end": 2018, "sents_id": 11}, {"text": "Bashir", "label": "PER", "start": 2068, "end": 2074, "sents_id": 12}, {"text": "five", "label": "NUM", "start": 2099, "end": 2103, "sents_id": 12}]}

    document = record["document"]
    chunks, entities_df = resolver.process_document(doc_id=doc_id, document=document, owner_id="system")
    print(chunks)
    print(entities_df)