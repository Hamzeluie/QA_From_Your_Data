# ingestion/unified_resolver.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from ingestion.models.base import IExtractor
from storage.factory import StorageFactory
from ingestion.candidate_finder import CandidateFinder
from shared.data_classes import (
    EntityLabels, DisambiguationStatus, Chunk, CandidateResult, Entity, NEResult
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
        model_path = getattr(settings, "EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")
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
            ner_results = self._fake_ner_for_structure(doc_id, document)
        else:
            ner_results = self._extract_entities_with_coref(document=document, doc_id=doc_id, relative_loc=False)
        
        self.factory.postgres.transition_state(doc_id, "uploaded", "ner_done")

        # 3. Resolve each entity
        clean_entities = []
        review_entities = []
        for ent in ner_results.entities:
            resolved_entity = self._resolve_entity(ent)
            if resolved_entity.status == DisambiguationStatus.RESOLVED:
                clean_entities.append(resolved_entity)
            else:
                review_entities.append(resolved_entity)

            # Audit log
            self.factory.postgres.log_mention(resolved_entity)

        self.factory.postgres.transition_state(doc_id, "ner_done", "er_done")

        # 5. Cross-document resolution on unresolved
        if review_entities:
            review_entities = self._cross_doc_resolve(review_entities)
            still_review = []
            for ent in review_entities:
                if ent.status == DisambiguationStatus.RESOLVED:
                    clean_entities.append(ent)
                else:
                    still_review.append(ent)
            review_entities = still_review

        # 6. Merge similar nodes (Redis distributed lock)
        self._merge_catalog()

        # 7. Persist unresolved queue
        for ent in review_entities:
            self.factory.postgres.enqueue_unresolved(
                resolution_id=f"{doc_id}_{ent.text}_{ent.start}",
                doc_id=doc_id,
                entity_row=ent,
                candidates_json=json.dumps([c if isinstance(c, Dict) else c.to_dict() for c in ent.kg_candidates]),
            )
        # 8. State transition
        if review_entities:
            self.factory.postgres.transition_state(doc_id, "er_done", "review_pending")
        else:
            self.factory.postgres.transition_state(doc_id, "er_done", "re_done")
        
        return ner_results.chunks, clean_entities, review_entities

    def add_user_resolution(self, entity: Entity, aliases: List[str] = None, context_indicators: Optional[List[str]] = None, related_to: Optional[List[str]] = None) -> None:
        """
        Operator resolves an entity. SINGLE write path for user-confirmed entities.
        Neo4j upsert is atomic and creates an OutboxEvent in the same tx.
        """
        canonical = entity.canonical_name.strip()
        alias_list = list(set([canonical] + [a.strip() for a in (aliases or []) if a.strip()]))

        # 1. Graph source of truth + outbox journal (ACID)
        self.factory.neo4j.upsert_entity(
            canonical=canonical,
            label=entity.label,
            aliases=alias_list,
            summary=entity.summary,
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

        logger.info(f"[User Feedback] '{canonical}' ({entity.label}) registered. Outbox will sync to ES/Qdrant.")

    # ── NED + Coreference Resolution ─────────────────────────────────────────────────────    
    def _name_entity_recognition(self, doc: str, doc_id:str, owner_id:str="system", relative_loc:bool=False) -> NEResult:
        """name entity recognition

        Args:
            doc (str): input text document
            doc_id (str): document if
            owner_id (str, optional): how imported the document. Defaults to "system".
            relative_loc (bool, optional): address entity start and end base on chunk(relative_loc == True) or document(relative_loc == False). Defaults to False.

        Returns:
            NEResult
        """
        ner_result = NEResult()
        for idx, chunk_info in enumerate(semantic_sentence_chunk(doc, self.embedder)):
            chunk_id = idx + 1
            chunk = chunk_info["text"]
            offset =  0 if relative_loc else chunk_info["start"] # exact, no drift
            ner_result.chunks.append(Chunk(doc_id=doc_id, 
                                             chunk_id=chunk_id,
                                             owner_id=owner_id, 
                                             sentence=chunk, 
                                             date_time=datetime.now().strftime("%y/%m/%d/ %H:%M:%S"),
                                             start_offset=chunk_info["start"],
                                             end_offset=chunk_info["end"],
                                             )
                                       )
            result = self.ner(chunk,doc_id=doc_id, chunk_id=chunk_id)
            for ent in result:
                ent.start += offset
                ent.end += offset
                ner_result.entities.append(ent)

        return ner_result
    
    
    def _extract_entities_with_coref(self, doc_id: str, document: str, relative_loc:bool=False)->NEResult:
        entity_extracted = self._name_entity_recognition(document, doc_id=doc_id, relative_loc=relative_loc)
        
        if relative_loc:
        
            group_entities = lambda entities: [
                [e for e in entities if e.doc_id == doc_id and e.chunk_id == chunk_id]
                for doc_id, chunk_id in sorted(list(set((e.doc_id, e.chunk_id) for e in entities)))
                ]
        
            entities = []
            for idx, ents in enumerate(group_entities(entity_extracted.entities)):
                coref_entities = self.coref.extract(entity_extracted.chunks[idx].sentence, ents)
                entities.extend(self._remove_subsumed_entities(coref_entities))
            entity_extracted = NEResult(chunks=entity_extracted.chunks, entities=entities)
        
        else:
        
            coref_entities = self.coref.extract(document, entity_extracted.entities)
            entity_extracted.entities = self._remove_subsumed_entities(coref_entities)
        
        return entity_extracted
    
    def _remove_subsumed_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Drop entities whose character span is fully contained inside a larger
        entity span in the same document. Prefer longer spans.
        """
        if not entities:
            return entities

        # Sort by span length descending, then by start position
        entitiy_sorted = sorted(
            entities,
            key=lambda r: (r.end - r.start, r.start),
            reverse=True,
        )

        kept = []
        kept_spans = []  # (doc_id, start, end)

        for ent in entitiy_sorted:
            doc_id, s, e = ent.doc_id, ent.start, ent.end

            # Is this row fully contained inside an already-kept span?
            is_subsumed = any(
                doc_id == kd and s >= ks and e <= ke and (s != ks or e != ke)
                for kd, ks, ke in kept_spans
            )

            if not is_subsumed:
                kept.append(ent)
                kept_spans.append((doc_id, s, e))

        return sorted(kept, key=lambda r: (r.doc_id, r.start))
    
    # ── Resolution Logic ─────────────────────────────────────────────────────

    def _resolve_entity(self, entity: Entity) -> Entity:
        # Non-linkable types (DATE, MONEY, etc.) — keep your existing logic
        if entity.label in NON_LINKABLE_TYPES:
            return self._normalize_literal(entity)

        # 1. Catalog lookup (Redis hot path)
        cached_canon = self.factory.redis.get_alias(entity.text.lower().strip())
        if cached_canon:
            node = self.factory.neo4j.find_by_canonical(cached_canon)
            if node:
                entity.status = DisambiguationStatus.RESOLVED
                entity.context_clues = ["Redis alias cache hit"]
                entity.needs_review=False
                return entity

        # 2. CandidateFinder (Neo4j + ES + Qdrant ensemble)
        candidates = self.finder.find_candidates(entity)

        if not candidates:
            # 3. Wikipedia fallback
            if self.wiki:
                wiki_ent = self.wiki.summarize(entity=entity)
                if wiki_ent is not None:
                    # Auto-register to catalog via outbox
                    self.add_user_resolution(wiki_ent, aliases=[wiki_ent.text, wiki_ent.canonical_name])
                    return wiki_ent
            
            entity.status = DisambiguationStatus.UNRESOLVED
            entity.context_clues = ["No match in KG, Wikipedia, or Catalog"]
            entity.needs_review = True
            entity.is_nil = True
            return entity

        # 4. Disambiguate from candidates
        return self._disambiguate(entity, candidates)

    def _disambiguate(
        self,
        entity:Entity,
        candidates: List[CandidateResult],
    ) -> Entity:
        # Single candidate fast path
        if len(candidates) == 1:
            cand = candidates[0]
            label_match = self._check_label_match(entity.label, cand.label)
            if cand.match_score >= 0.88:
                entity.status = DisambiguationStatus.RESOLVED
                entity.confidence = cand.match_score * (0.9 if label_match else 0.85)
                entity.needs_review = False
            else:
                entity.status = DisambiguationStatus.UNRESOLVED
                entity.confidence = cand.match_score
                entity.needs_review = True

            entity.kg_candidates = [c.to_dict() for c in candidates]
            entity.context_clues = [f"Single candidate: {cand.match_method}"]
            return entity

        # Multi-candidate: pick best
        best = candidates[0]
        label_match = self._check_label_match(entity.label, best.label)
        if best.match_score > 0.88 and label_match:
            entity.status = DisambiguationStatus.RESOLVED
        else:
            entity.status = DisambiguationStatus.UNRESOLVED

        entity.label = best.label
        entity.canonical_name = best.canonical
        entity.confidence = best.match_score
        entity.kg_candidates=[c.to_dict() for c in candidates],
        entity.context_clues=[f"Best candidate score: {best.match_score}"],
        entity.needs_review=not (label_match and best.match_score > 0.88),
        return entity

    def _cross_doc_resolve(self, unresolved: List[Entity]) -> List[Entity]:
        """Qdrant semantic search for unresolved mentions."""
        if not self.embedder:
            return unresolved

        for ent in unresolved:
            if ent.status == DisambiguationStatus.RESOLVED and not ent.is_nil:
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

    def _normalize_literal(self, entity:Entity):
        if entity.label == "MONEY":
            normalized = ValueNormalizer.normalize_money(entity.text)
            entity.canonical_name=normalized["canonical"]
            entity.status=DisambiguationStatus.RESOLVED
            return entity
        
        elif entity.label == "DATE":
            normalized = ValueNormalizer.normalize_date(entity.text)
            entity.canonical_name=normalized["canonical"]
            entity.status=DisambiguationStatus.RESOLVED
            return entity
            
        # elif entity.label == "TIME":
        #     normalized = ValueNormalizer.normalize_time(entity.text)
        #     entity.canonical_name=normalized["canonical"]
        #     entity.status=DisambiguationStatus.RESOLVED
        #     return entity
        entity.status=DisambiguationStatus.RESOLVED
        return entity

    def _check_label_match(self, hint, candidate_label):
        if not hint or hint == "UNRESOLVED":
            return True
        return hint.upper() == candidate_label.upper()

    def _fake_ner_for_structure(self, doc_id, document):
        """Replace this with your real _extract_entities_with_coref."""
        # Returns (chunk_info, entities_df)
        
        ber_result = NEResult(
                chunks=[
                    Chunk(doc_id='doc1', chunk_id=1, owner_id='system', sentence='Ezri Dax ( ) is a fictional character who appears in the of the American science fiction TV series .Portrayed by Nicole de Boer , she is a counselor aboard the Bajoran space station Deep Space Nine ', date_time='26/09/05/ 17:06:21', start_offset=0, end_offset=198), 
                    Chunk(doc_id='doc1', chunk_id=2, owner_id='system', sentence='The character is a member of the Trill species , and is formed of both a host and a symbiont – referred to as Dax .Ezri was introduced to the series following the death of the previous Dax host , Jadzia ( Terry Farrell ) at the end of .It had been the producers \' intention to introduce a new female character bearing the symbiont in order to ensure that Nana Visitor as Kira Nerys was not the only female member of the main cast .There were difficulties in casting initially , and the character changed from one who was intended to be " spooky " to one that was struggling to deal with all her previous personalities as a result of unexpectedly taking on the Dax symbiont .De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role .The character made her first appearance in the first episode of the seventh season , " Image in the Sand " .The character continued to appear throughout the final season of the series , with her final appearance in the series finale " What You Leave Behind " .Her character stepped into the void left by Jadzia amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with Jadzia \'s widower , Worf ( Michael Dorn ) .During the course of the season , Ezri becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with Dr. Julian Bashir ( Alexander Siddig ) .The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for " " – an episode described as " just a mess " by writer Ronald D. Moore ', date_time='26/09/05/ 17:06:21', start_offset=199, end_offset=2019), 
                    Chunk(doc_id='doc1', chunk_id=3, owner_id='system', sentence='The relationship between Ezri and both Worf and Bashir was described as one of five " great geek TV love triangles " .The inclusion of the character was criticised on the internet , with Ezri being referred to as both an " ill - conceived idea " and a " replacement Dax " .', date_time='26/09/05/ 17:06:21', start_offset=2020, end_offset=2293)
                    ],
                entities= [Entity(text='Ezri Dax', label=EntityLabels.PER, start=0, end=8, mention_sentence='<Ezri Dax> ( ) is a fictional character who appears in the of the American science fiction TV series', confidence=1.0, canonical_name='Ezri Dax', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Nicole de Boer', label=EntityLabels.PER, start=113, end=127, mention_sentence='Portrayed by <Nicole de Boer> , she is a counselor aboard the Bajoran space station Deep Space Nine', confidence=1.0, canonical_name='Nicole de Boer', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='she', label=EntityLabels.PER, start=130, end=133, mention_sentence='Portrayed by Nicole de Boer , <she> is a counselor aboard the Bajoran space station Deep Space Nine', confidence=0.75, canonical_name='Ezri Dax', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Ezri Dax'),
                    Entity(text='Deep Space Nine', label=EntityLabels.ORG, start=182, end=197, mention_sentence='Portrayed by Nicole de Boer , she is a counselor aboard the Bajoran space station <Deep Space Nine>', confidence=0.64, canonical_name='Deep Space Nine', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='.The character', label=EntityLabels.PER, start=198, end=212, mention_sentence='<The character> is a member of the Trill species , and is formed of both a host and a symbiont – referred to as Dax', confidence=0.75, canonical_name='Ezri Dax', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Ezri Dax'),
                    Entity(text='Ezri', label=EntityLabels.PER, start=314, end=318, mention_sentence='<Ezri> was introduced to the series following the death of the previous Dax host , Jadzia ( Terry Farrell ) at the end of', confidence=0.85, canonical_name='Ezri', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='the previous Dax host , Jadzia ( Terry Farrell )', label=EntityLabels.PER, start=371, end=419, mention_sentence='Ezri was introduced to the series following the death of <the previous Dax host , Jadzia ( Terry Farrell )> at the end of', confidence=0.72, canonical_name='Jadzia', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Jadzia'),
                    Entity(text='Nana Visitor', label=EntityLabels.PER, start=554, end=566, mention_sentence="It had been the producers ' intention to introduce a new female character bearing the symbiont in order to ensure that <Nana Visitor> as Kira Nerys was not the only female member of the main cast", confidence=1.0, canonical_name='Nana Visitor', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Kira Nerys', label=EntityLabels.PER, start=570, end=580, mention_sentence="It had been the producers ' intention to introduce a new female character bearing the symbiont in order to ensure that Nana Visitor as <Kira Nerys> was not the only female member of the main cast", confidence=1.0, canonical_name='Kira Nerys', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='the character', label=EntityLabels.PER, start=681, end=694, mention_sentence='There were difficulties in casting initially , and <the character> changed from one who was intended to be " spooky " to one that was struggling to deal with all her previous personalities as a result of unexpectedly taking on the Dax symbiont', confidence=0.75, canonical_name='Ezri Dax', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Ezri Dax'),
                    Entity(text='her', label=EntityLabels.PER, start=790, end=793, mention_sentence='There were difficulties in casting initially , and the character changed from one who was intended to be " spooky " to one that was struggling to deal with all <her> previous personalities as a result of unexpectedly taking on the Dax symbiont', confidence=0.75, canonical_name='Ezri Dax', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Ezri Dax'),
                    Entity(text='De Boer', label=EntityLabels.PER, start=873, end=880, mention_sentence='<De Boer> was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role', confidence=1.0, canonical_name='De Boer', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Hans Beimler', label=EntityLabels.PER, start=933, end=945, mention_sentence='De Boer was not considered for the part until co - producer <Hans Beimler> suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role', confidence=1.0, canonical_name='Hans Beimler', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='she', label=EntityLabels.PER, start=961, end=964, mention_sentence='De Boer was not considered for the part until co - producer Hans Beimler suggested that <she> should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='her', label=EntityLabels.PER, start=1016, end=1019, mention_sentence='De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in <her> invitation to meet with the producers in Los Angeles and in her gaining the role', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='Los Angeles', label=EntityLabels.LOC, start=1061, end=1072, mention_sentence='De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in <Los Angeles> and in her gaining the role', confidence=1.0, canonical_name='Los Angeles', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='her', label=EntityLabels.PER, start=1080, end=1083, mention_sentence='De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in <her> gaining the role', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='her', label=EntityLabels.PER, start=1121, end=1124, mention_sentence='The character made <her> first appearance in the first episode of the seventh season , " Image in the Sand "', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='her', label=EntityLabels.PER, start=1293, end=1296, mention_sentence='The character continued to appear throughout the final season of the series , with <her> final appearance in the series finale " What You Leave Behind "', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='Jadzia', label=EntityLabels.PER, start=1406, end=1412, mention_sentence="Her character stepped into the void left by <Jadzia> amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with Jadzia 's widower , Worf ( Michael Dorn )", confidence=1.0, canonical_name='Jadzia', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='she', label=EntityLabels.PER, start=1447, end=1450, mention_sentence="Her character stepped into the void left by Jadzia amongst the crew , but found that <she> had to redevelop those previous relationships and learn to get along with Jadzia 's widower , Worf ( Michael Dorn )", confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text="Jadzia 's widower , Worf ( Michael Dorn )", label=EntityLabels.PER, start=1525, end=1566, mention_sentence="Her character stepped into the void left by Jadzia amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with <Jadzia 's widower , Worf ( Michael Dorn )>", confidence=0.68, canonical_name='Worf', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Worf'),
                    Entity(text='Ezri', label=EntityLabels.PER, start=1602, end=1606, mention_sentence='During the course of the season , <Ezri> becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with Dr.', confidence=1.0, canonical_name='Ezri', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='her', label=EntityLabels.PER, start=1631, end=1634, mention_sentence='During the course of the season , Ezri becomes less nervous of <her> role over time and learns from the Dax symbiont and becomes involved romantically with Dr.', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='Dr. Julian Bashir ( Alexander Siddig )', label=EntityLabels.PER, start=1722, end=1760, mention_sentence='During the course of the season , Ezri becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with <Dr.>', confidence=0.68, canonical_name='Bashir', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Bashir'),
                    Entity(text='the character', label=EntityLabels.PER, start=1782, end=1795, mention_sentence='The fan reaction to <the character> was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for " " – an episode described as " just a mess " by writer Ronald D.', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='Ira Steven Behr', label=EntityLabels.PER, start=1904, end=1919, mention_sentence='The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer <Ira Steven Behr> apologising to de Boer for " " – an episode described as " just a mess " by writer Ronald D.', confidence=0.99, canonical_name='Ira Steven Behr', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='de Boer', label=EntityLabels.PER, start=1935, end=1942, mention_sentence='The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to <de Boer> for " " – an episode described as " just a mess " by writer Ronald D.', confidence=1.0, canonical_name='de Boer', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Ronald D. Moore', label=EntityLabels.PER, start=2003, end=2018, mention_sentence='The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for " " – an episode described as " just a mess " by writer <Ronald D.>', confidence=1.0, canonical_name='Ronald D. Moore', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=2, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Ezri', label=EntityLabels.PER, start=2045, end=2049, mention_sentence='The relationship between <Ezri> and both Worf and Bashir was described as one of five " great geek TV love triangles "', confidence=0.99, canonical_name='Ezri', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Worf', label=EntityLabels.PER, start=2059, end=2063, mention_sentence='The relationship between Ezri and both <Worf> and Bashir was described as one of five " great geek TV love triangles "', confidence=1.0, canonical_name='Worf', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Bashir', label=EntityLabels.PER, start=2068, end=2074, mention_sentence='The relationship between Ezri and both Worf and <Bashir> was described as one of five " great geek TV love triangles "', confidence=0.99, canonical_name='Bashir', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='the character', label=EntityLabels.PER, start=2155, end=2168, mention_sentence='The inclusion of <the character> was criticised on the internet , with Ezri being referred to as both an " ill - conceived idea " and a " replacement Dax " .', confidence=0.85, canonical_name='Nicole de Boer', status=DisambiguationStatus.RESOLVED, doc_id='doc1', chunk_id=1, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Nicole de Boer'),
                    Entity(text='Ezri', label=EntityLabels.PER, start=2207, end=2211, mention_sentence='The inclusion of the character was criticised on the internet , with <Ezri> being referred to as both an " ill - conceived idea " and a " replacement Dax " .', confidence=1.0, canonical_name='Ezri', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                    Entity(text='Dax', label=EntityLabels.PER, start=2286, end=2289, mention_sentence='The inclusion of the character was criticised on the internet , with Ezri being referred to as both an " ill - conceived idea " and a " replacement <Dax> " .', confidence=0.99, canonical_name='Dax', status=DisambiguationStatus.UNRESOLVED, doc_id='doc1', chunk_id=3, kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None)]
        )
        return ber_result
    
    
if __name__ == "__main__":
    from ingestion.models.factory import get_ner_extractor, get_coref_resolver
    is_demo = True
    if is_demo:
        ner = None
        coref = None
    else:
        ner = get_ner_extractor(backend="bert", model_name=settings.BERT_NER_MODEL_NAME, local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"), use_onnx=True, onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner"))
        coref = get_coref_resolver(backend="bert", nlp=settings.SPACY_MODEL_PATH)
        
    resolver = UnifiedEntityResolver(ner_model=ner, coref_model=coref, run_demo=is_demo)  # Assuming UnifiedResolver is the class name
    # resolver = UnifiedEntityResolver(ner_model=None, coref_model=None, run_demo=True)  # Assuming UnifiedResolver is the class name
    doc_id = "doc1"
    record = {"doc_id": 6, "document": "Ezri Dax ( ) is a fictional character who appears in the of the American science fiction TV series .Portrayed by Nicole de Boer , she is a counselor aboard the Bajoran space station Deep Space Nine .The character is a member of the Trill species , and is formed of both a host and a symbiont – referred to as Dax .Ezri was introduced to the series following the death of the previous Dax host , Jadzia ( Terry Farrell ) at the end of .It had been the producers ' intention to introduce a new female character bearing the symbiont in order to ensure that Nana Visitor as Kira Nerys was not the only female member of the main cast .There were difficulties in casting initially , and the character changed from one who was intended to be \" spooky \" to one that was struggling to deal with all her previous personalities as a result of unexpectedly taking on the Dax symbiont .De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role .The character made her first appearance in the first episode of the seventh season , \" Image in the Sand \" .The character continued to appear throughout the final season of the series , with her final appearance in the series finale \" What You Leave Behind \" .Her character stepped into the void left by Jadzia amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with Jadzia 's widower , Worf ( Michael Dorn ) .During the course of the season , Ezri becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with Dr. Julian Bashir ( Alexander Siddig ) .The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for \" \" – an episode described as \" just a mess \" by writer Ronald D. Moore .The relationship between Ezri and both Worf and Bashir was described as one of five \" great geek TV love triangles \" .The inclusion of the character was criticised on the internet , with Ezri being referred to as both an \" ill - conceived idea \" and a \" replacement Dax \" .", "sentence": [["Ezri", "Dax", "(", ")", "is", "a", "fictional", "character", "who", "appears", "in", "the", "of", "the", "American", "science", "fiction", "TV", "series", "."], ["Portrayed", "by", "Nicole", "de", "Boer", ",", "she", "is", "a", "counselor", "aboard", "the", "Bajoran", "space", "station", "Deep", "Space", "Nine", "."], ["The", "character", "is", "a", "member", "of", "the", "Trill", "species", ",", "and", "is", "formed", "of", "both", "a", "host", "and", "a", "symbiont", "–", "referred", "to", "as", "Dax", "."], ["Ezri", "was", "introduced", "to", "the", "series", "following", "the", "death", "of", "the", "previous", "Dax", "host", ",", "Jadzia", "(", "Terry", "Farrell", ")", "at", "the", "end", "of", "."], ["It", "had", "been", "the", "producers", "'", "intention", "to", "introduce", "a", "new", "female", "character", "bearing", "the", "symbiont", "in", "order", "to", "ensure", "that", "Nana", "Visitor", "as", "Kira", "Nerys", "was", "not", "the", "only", "female", "member", "of", "the", "main", "cast", "."], ["There", "were", "difficulties", "in", "casting", "initially", ",", "and", "the", "character", "changed", "from", "one", "who", "was", "intended", "to", "be", "\"", "spooky", "\"", "to", "one", "that", "was", "struggling", "to", "deal", "with", "all", "her", "previous", "personalities", "as", "a", "result", "of", "unexpectedly", "taking", "on", "the", "Dax", "symbiont", "."], ["De", "Boer", "was", "not", "considered", "for", "the", "part", "until", "co", "-", "producer", "Hans", "Beimler", "suggested", "that", "she", "should", "submit", "an", "audition", "tape", ",", "which", "resulted", "in", "her", "invitation", "to", "meet", "with", "the", "producers", "in", "Los", "Angeles", "and", "in", "her", "gaining", "the", "role", "."], ["The", "character", "made", "her", "first", "appearance", "in", "the", "first", "episode", "of", "the", "seventh", "season", ",", "\"", "Image", "in", "the", "Sand", "\"", "."], ["The", "character", "continued", "to", "appear", "throughout", "the", "final", "season", "of", "the", "series", ",", "with", "her", "final", "appearance", "in", "the", "series", "finale", "\"", "What", "You", "Leave", "Behind", "\"", "."], ["Her", "character", "stepped", "into", "the", "void", "left", "by", "Jadzia", "amongst", "the", "crew", ",", "but", "found", "that", "she", "had", "to", "redevelop", "those", "previous", "relationships", "and", "learn", "to", "get", "along", "with", "Jadzia", "'s", "widower", ",", "Worf", "(", "Michael", "Dorn", ")", "."], ["During", "the", "course", "of", "the", "season", ",", "Ezri", "becomes", "less", "nervous", "of", "her", "role", "over", "time", "and", "learns", "from", "the", "Dax", "symbiont", "and", "becomes", "involved", "romantically", "with", "Dr.", "Julian", "Bashir", "(", "Alexander", "Siddig", ")", "."], ["The", "fan", "reaction", "to", "the", "character", "was", "reported", "as", "positive", ",", "but", "several", "of", "the", "Ezri", "-", "centric", "episodes", "came", "in", "for", "criticism", ",", "with", "producer", "Ira", "Steven", "Behr", "apologising", "to", "de", "Boer", "for", "\"", "\"", "–", "an", "episode", "described", "as", "\"", "just", "a", "mess", "\"", "by", "writer", "Ronald", "D.", "Moore", "."], ["The", "relationship", "between", "Ezri", "and", "both", "Worf", "and", "Bashir", "was", "described", "as", "one", "of", "five", "\"", "great", "geek", "TV", "love", "triangles", "\"", "."], ["The", "inclusion", "of", "the", "character", "was", "criticised", "on", "the", "internet", ",", "with", "Ezri", "being", "referred", "to", "as", "both", "an", "\"", "ill", "-", "conceived", "idea", "\"", "and", "a", "\"", "replacement", "Dax", "\"", "."]], "label_sents": [[{"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}, {"name": "Ezri", "sent_id": 3, "pos": [0, 1], "type": "PER"}, {"name": "Ezri", "sent_id": 13, "pos": [12, 13], "type": "PER"}, {"name": "Ezri", "sent_id": 12, "pos": [3, 4], "type": "PER"}, {"name": "Ezri", "sent_id": 11, "pos": [15, 16], "type": "PER"}, {"name": "Ezri", "sent_id": 10, "pos": [7, 8], "type": "PER"}, {"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}], [{"name": "American", "sent_id": 0, "pos": [14, 15], "type": "LOC"}], [{"name": "Nicole de Boer", "sent_id": 1, "pos": [2, 5], "type": "PER"}], [{"name": "Bajoran", "sent_id": 1, "pos": [12, 13], "type": "LOC"}], [{"name": "Deep Space Nine", "sent_id": 1, "pos": [15, 18], "type": "MISC"}], [{"name": "Trill", "sent_id": 2, "pos": [7, 8], "type": "MISC"}], [{"name": "Dax", "sent_id": 10, "pos": [20, 21], "type": "MISC"}, {"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "MISC"}, {"name": "Dax", "sent_id": 5, "pos": [41, 42], "type": "MISC"}], [{"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "PER"}, {"name": "Dax", "sent_id": 3, "pos": [12, 13], "type": "PER"}, {"name": "Dax", "sent_id": 13, "pos": [29, 30], "type": "PER"}], [{"name": "Terry Farrell", "sent_id": 3, "pos": [17, 19], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [29, 30], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [8, 9], "type": "PER"}, {"name": "Jadzia", "sent_id": 3, "pos": [15, 16], "type": "PER"}], [{"name": "Nana Visitor", "sent_id": 4, "pos": [21, 23], "type": "PER"}], [{"name": "Kira Nerys", "sent_id": 4, "pos": [24, 26], "type": "PER"}], [{"name": "De Boer", "sent_id": 6, "pos": [0, 2], "type": "PER"}, {"name": "de Boer", "sent_id": 11, "pos": [31, 33], "type": "PER"}], [{"name": "Hans Beimler", "sent_id": 6, "pos": [12, 14], "type": "PER"}], [{"name": "Los Angeles", "sent_id": 6, "pos": [34, 36], "type": "LOC"}], [{"name": "Image in the Sand", "sent_id": 7, "pos": [16, 20], "type": "MISC"}], [{"name": "What You Leave Behind", "sent_id": 8, "pos": [22, 26], "type": "MISC"}], [{"name": "Worf", "sent_id": 12, "pos": [6, 7], "type": "PER"}, {"name": "Worf", "sent_id": 9, "pos": [33, 34], "type": "PER"}], [{"name": "Michael Dorn", "sent_id": 9, "pos": [35, 37], "type": "PER"}], [{"name": "Julian Bashir", "sent_id": 10, "pos": [28, 30], "type": "PER"}], [{"name": "Alexander Siddig", "sent_id": 10, "pos": [31, 33], "type": "PER"}], [{"name": "Ira Steven Behr", "sent_id": 11, "pos": [26, 29], "type": "PER"}], [{"name": "Ronald D. Moore", "sent_id": 11, "pos": [48, 51], "type": "PER"}], [{"name": "Bashir", "sent_id": 12, "pos": [8, 9], "type": "PER"}], [{"name": "five", "sent_id": 12, "pos": [14, 15], "type": "NUM"}]], "label_doc": [{"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "Ezri", "label": "PER", "start": 314, "end": 318, "sents_id": 3}, {"text": "Ezri", "label": "PER", "start": 2207, "end": 2211, "sents_id": 13}, {"text": "Ezri", "label": "PER", "start": 2045, "end": 2049, "sents_id": 12}, {"text": "Ezri", "label": "PER", "start": 1842, "end": 1846, "sents_id": 11}, {"text": "Ezri", "label": "PER", "start": 1602, "end": 1606, "sents_id": 10}, {"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "American", "label": "LOC", "start": 64, "end": 72, "sents_id": 0}, {"text": "Nicole de Boer", "label": "PER", "start": 113, "end": 127, "sents_id": 1}, {"text": "Bajoran", "label": "LOC", "start": 160, "end": 167, "sents_id": 1}, {"text": "Deep Space Nine", "label": "MISC", "start": 182, "end": 197, "sents_id": 1}, {"text": "Trill", "label": "MISC", "start": 232, "end": 237, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 1670, "end": 1673, "sents_id": 10}, {"text": "Dax", "label": "MISC", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 859, "end": 862, "sents_id": 5}, {"text": "Dax", "label": "PER", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "PER", "start": 384, "end": 387, "sents_id": 3}, {"text": "Dax", "label": "PER", "start": 2286, "end": 2289, "sents_id": 13}, {"text": "Terry Farrell", "label": "PER", "start": 404, "end": 417, "sents_id": 3}, {"text": "Jadzia", "label": "PER", "start": 1525, "end": 1531, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 1406, "end": 1412, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 395, "end": 401, "sents_id": 3}, {"text": "Nana Visitor", "label": "PER", "start": 554, "end": 566, "sents_id": 4}, {"text": "Kira Nerys", "label": "PER", "start": 570, "end": 580, "sents_id": 4}, {"text": "De Boer", "label": "PER", "start": 873, "end": 880, "sents_id": 6}, {"text": "de Boer", "label": "PER", "start": 1935, "end": 1942, "sents_id": 11}, {"text": "Hans Beimler", "label": "PER", "start": 933, "end": 945, "sents_id": 6}, {"text": "Los Angeles", "label": "LOC", "start": 1061, "end": 1072, "sents_id": 6}, {"text": "Image in the Sand", "label": "MISC", "start": 1189, "end": 1206, "sents_id": 7}, {"text": "What You Leave Behind", "label": "MISC", "start": 1337, "end": 1358, "sents_id": 8}, {"text": "Worf", "label": "PER", "start": 2059, "end": 2063, "sents_id": 12}, {"text": "Worf", "label": "PER", "start": 1545, "end": 1549, "sents_id": 9}, {"text": "Michael Dorn", "label": "PER", "start": 1552, "end": 1564, "sents_id": 9}, {"text": "Julian Bashir", "label": "PER", "start": 1726, "end": 1739, "sents_id": 10}, {"text": "Alexander Siddig", "label": "PER", "start": 1742, "end": 1758, "sents_id": 10}, {"text": "Ira Steven Behr", "label": "PER", "start": 1904, "end": 1919, "sents_id": 11}, {"text": "Ronald D. Moore", "label": "PER", "start": 2003, "end": 2018, "sents_id": 11}, {"text": "Bashir", "label": "PER", "start": 2068, "end": 2074, "sents_id": 12}, {"text": "five", "label": "NUM", "start": 2099, "end": 2103, "sents_id": 12}]}

    document = record["document"]
    chunks, entities_df = resolver.process_document(doc_id=doc_id, document=document, owner_id="system")
    print(chunks)
    print(entities_df)
    
    
    


