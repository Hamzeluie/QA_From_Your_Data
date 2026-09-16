import sys
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import requests
import json
import hashlib
import logging
from typing import List, Optional, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from sentence_transformers import SentenceTransformer
from config.settings import settings
from storage.factory import StorageFactory
from storage.data_classes import CandidateResult, MentionEntity, EntityLabels, CandidateMatchMethod, DisambiguationStatus
from shared.utils import (char_ngram_jaccard,
                          levenshtein_ratio,
                          jaro_winkler,
                          NON_LINKABLE_TYPES, 
                          ValueNormalizer,)


logger = logging.getLogger(__name__)

class WikipediaEntitySummarizer:
    LABEL_SIGNATURES = {
        EntityLabels.PER.value: [
            "born", "politician", "businessman", "businesswoman", "actor", "actress",
            "author", "scientist", "footballer", "musician", "singer", "CEO",
            "entrepreneur", "engineer", "inventor", "philanthropist", "artist",
            "is a ", "was a ", "is an ", "was an "
        ],
        EntityLabels.ORG.value: [
            "company", "corporation", "inc.", "ltd", "organization", "firm",
            "multinational", "headquartered in", "founded in", "subsidiary of",
            "publicly traded", "listed on", "stock exchange", "enterprise"
        ],
        EntityLabels.LOC.value: [
            "country", "city", "state", "capital", "republic", "kingdom",
            "province", "county", "municipality", "located in", "population of",
            "island", "continent", "territory"
        ],
        EntityLabels.FAC.value: [
            "airport", "bridge", "highway", "building", "station", "hospital",
            "university", "museum", "stadium", "located in", "built in"
        ],
        EntityLabels.PRODUCT.value: [
            "software", "device", "car", "phone", "game", "console", "product",
            "launched in", "released by", "developed by", "chatbot", "model",
            "series", "platform", "application", "app"
        ],
        EntityLabels.TECHNOLOGY.value: [
            "technology", "artificial intelligence", "machine learning",
            "deep learning", "neural network", "algorithm", "computational",
            "software framework", "model", "system", "platform", "architecture",
            "is a field of", "is a branch of", "is a type of"
        ],
        EntityLabels.EVENT.value: [
            "war", "battle", "conference", "festival", "olympics", "tournament",
            "held in", "took place", "anniversary", "celebration"
        ],
        EntityLabels.WORK_OF_ART.value: [
            "novel", "book", "film", "movie", "song", "album", "painting",
            "written by", "directed by", "composed by", "published in"
        ],
        EntityLabels.LAW.value: [
            "act", "treaty", "constitution", "amendment", "law", "bill",
            "signed into law", "ratified", "legal"
        ],
        EntityLabels.LANGUAGE.value: [
            "language", "dialect", "spoken in", "official language", "lingua franca"
        ],
        EntityLabels.DATE.value: [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ],
        EntityLabels.TIME.value: [
            "morning", "afternoon", "evening", "night", "midnight", "noon",
            "a.m.", "p.m.", "o'clock", "hour", "minute", "second"
        ],
        EntityLabels.MONEY.value: [
            "dollar", "euro", "pound", "yen", "usd", "eur", "gbp",
            "million", "billion", "trillion", "budget", "revenue", "cost"
        ],
        EntityLabels.PERCENT.value: [
            "percent", "percentage", "%", "proportion", "rate", "share"
        ],
        EntityLabels.QUANTITY.value: [
            "meter", "kilometer", "mile", "kilogram", "ton", "liter",
            "degree", "celsius", "fahrenheit", "inch", "foot", "pound"
        ],
        EntityLabels.CARDINAL.value: [
            "one", "two", "three", "hundred", "thousand", "million"
        ],
        EntityLabels.ORDINAL.value: [
            "first", "second", "third", "fourth", "fifth", "last"
        ],
        EntityLabels.NORP.value: [
            "american", "european", "asian", "african", "christian", "muslim",
            "jewish", "buddhist", "hindu", "democrat", "republican", "conservative",
            "liberal", "socialist", "nationality", "ethnic"
        ],
        EntityLabels.MISC.value: [
            "award", "honor", "title", "degree", "religion", "ideology",
            "culture", "tradition", "custom", "mythology", "legend"
        ],
        EntityLabels.NUM.value: [
            "number", "amount", "total", "sum", "count", "quantity"
        ],
    }

    def __init__(self, embedding:SentenceTransformer=None):
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "EntityLinkerBot/1.0"}
        if embedding:
            self.embedder = embedding
        else:
            if os.path.isdir(settings.EMBEDDING_MODEL_PATH):
                self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
            else:
                self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                self.embedder.save(settings.EMBEDDING_MODEL_PATH)
        
    def search_wikipedia(self, entity: str, limit: int = 5) -> List[Dict]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": entity,
            "srlimit": limit,
            "format": "json"
        }
        try:
            response = requests.get(self.wiki_api, params=params,
                                   headers=self.headers, timeout=10)
            data = response.json()
        except Exception:
            return []

        candidates = []
        for result in data.get("query", {}).get("search", []):
            title = result["title"]
            snippet = self._clean_html(result.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            candidates.append({
                "title": title,
                "pageid": result["pageid"],
                "snippet": snippet,
                "url": url
            })
        return candidates

    def _clean_html(self, text: str) -> str:
        text = re.sub(r'<span class="searchmatch">(.*?)</span>', r'\1', text)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def _rank_candidates(self,
                         entity: str,
                         context: str,
                         candidates: List[Dict],
                         ner_label: Optional[str] = None) -> Optional[Dict]:
        if not candidates:
            return None

        label_hints = {
            "ORG": ["Inc.", "Company", "Corporation", "Ltd", "Group", "Airlines", "Bank"],
            "PERSON": ["born", "politician", "actor", "author", "scientist", "footballer"],
            "GPE": ["country", "city", "state", "capital", "republic", "kingdom"],
            "PRODUCT": ["software", "device", "car", "phone", "game", "console"],
            "TECHNOLOGY": ["software", "algorithm", "intelligence", "learning", "network", "model"],
        }

        context_vec = None
        if self.embedder and context:
            context_vec = self.embedder.encode([context], convert_to_numpy=True)

        scored = []
        for cand in candidates:
            score = 0.0
            snippet = cand["snippet"]
            title = cand["title"]

            if context_vec is not None and snippet:
                snippet_vec = self.embedder.encode([snippet], convert_to_numpy=True)
                sim = float(sk_cosine_similarity(context_vec, snippet_vec)[0][0])
                score += sim * 0.6

            context_words = set(context.lower().split())
            candidate_text = (title + " " + snippet).lower()
            overlap = len(context_words & set(candidate_text.split()))
            score += (overlap / max(len(context_words), 1)) * 0.3

            if ner_label and ner_label in label_hints:
                if any(h.lower() in candidate_text for h in label_hints[ner_label]):
                    score += 0.1

            if "disambiguation" in title.lower():
                score -= 0.5

            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def extract_summary(self, title: str, max_sentences: int = 2) -> str:
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "exintro": True,
            "exsentences": max_sentences,
            "explaintext": True,
            "format": "json"
        }
        try:
            response = requests.get(self.wiki_api, params=params,
                                   headers=self.headers, timeout=10)
            data = response.json()
        except Exception:
            return ""

        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            extract = page_data.get("extract", "")
            if extract:
                return self._normalize_summary(extract)
        return ""

    def _normalize_summary(self, text: str) -> str:
        text = re.sub(r'\s*\([^)]*(disambiguation|company|fruit|disambiguation page)[^)]*\)', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'For (other uses|the company|the fruit)[,.].*?(?=\n|$)', '', text)
        text = " ".join(text.split())
        return text.strip()

    def classify_from_summary(self, summary: str, fallback_label: Optional[str] = None) -> str:
        """
        Scan the Wikipedia summary for label-indicative keywords.
        Returns the best-matching label or the fallback.
        """
        if not summary:
            return fallback_label or "UNKNOWN"

        summary_lower = summary.lower()
        scores = {}

        for ent_label, phrases in self.LABEL_SIGNATURES.items():
            score = sum(1 for phrase in phrases if phrase.lower() in summary_lower)
            if score:
                scores[ent_label] = score

        if scores:
            return max(scores, key=scores.get)

        # Heuristic: if the summary mentions years of birth/death, it's likely a person
        if re.search(r'\b\d{4}\s*–\s*\d{4}\b', summary) or "born" in summary_lower:
            return "PERSON"

        return fallback_label or "UNKNOWN"

    def summarize(self,entity: MentionEntity) -> Optional[MentionEntity]:
        candidates = self.search_wikipedia(entity.text)
        if not candidates:
            return None

        best = self._rank_candidates(entity.text, entity.mention_sentence, candidates, entity.label)
        if not best:
            return None

        entity.summary = self.extract_summary(best["title"])

        # ── INFER LABEL FROM SUMMARY (override spaCy guess if confident) ──
        wiki_inferred_label = self.classify_from_summary(entity.summary, fallback_label=entity.label)

        # confidence = 0.
        if self.embedder and entity.summary:
            ctx_vec = self.embedder.encode([entity.mention_sentence], convert_to_numpy=True)
            sum_vec = self.embedder.encode([entity.summary], convert_to_numpy=True)
            # confidence = float(sk_cosine_similarity(ctx_vec, sum_vec)[0][0])

        entity.status = DisambiguationStatus.UNRESOLVED
        entity.context_clues = [f"Wikipedia match: {best['title']}", f"Label inferred from summary: {wiki_inferred_label}"]
        return entity
    

class CandidateFinder:
    """
    Stateless coordinator replacing NEDEngine.
    Queries Redis → Elasticsearch → Neo4j → Qdrant in sequence,
    then applies the original ensemble scoring weights.
    """

    def __init__(self, factory: StorageFactory, embedder=None, threshold: float = 0.92):
        self.factory = factory
        self.embedder = embedder
        self.threshold = min(max(threshold, 0.92), 1.0)
    
    def find_candidates(self, entity:MentionEntity, cache_entities:List[MentionEntity])->CandidateResult:
        candidates = []
        # 1.Lookup in-doc cache entities
        if cache_entities:
            candidates.extend(self._lookup_in_doc(entity, cache_entities))
            if len(candidates) == 1:
                return candidates 
            
        # 2.Lookup in Redis
        candidates.extend(self._lookup_in_redis(entity=entity))
        if len(candidates) == 1:
            return candidates
        elif len(candidates) > 1:
            candidates = self._ensemble_score(candidates, entity)
            if len(candidates):
                return candidates
            
        
        # 3.Lookup in Postgres
        candidates.extend(self._lookup_in_postgres(entity=entity))
        if len(candidates) == 1:
            return candidates
        elif len(candidates) > 1:
            candidates = self._ensemble_score(candidates, entity)
            if len(candidates):
                return candidates
        
        # 4.Lookup in ES
        candidates.extend(self._lookup_in_es(entity=entity))
        if len(candidates) == 1:
            return candidates
        elif len(candidates) > 1:
            candidates = self._ensemble_score(candidates, entity)
            if len(candidates):
                return candidates
        
        # 5.Lookup in Graph
        candidates.extend(self._lookup_in_graph(entity=entity))
        if len(candidates) == 1:
            return candidates
        elif len(candidates) > 1:
            candidates = self._ensemble_score(candidates, entity)
            if len(candidates):
                return candidates

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

    # ── Internal functions ─────────────────────────────────────────────────────────────
    
    def _find_candidates_in_retrieval(self, entity:MentionEntity, threshold: Optional[float] = None) -> List[CandidateResult]:
        """
        Four-phase retrieval with neural re-rank.
        Returns sorted CandidateResult list.
        """
        thresh = threshold if threshold is not None else self.threshold
        cache_key = self._cache_key(entity.text, entity.label, entity.mention_sentence)

        # ── Phase 0: Redis cache ──────────────────────────────────────────────
        cached = self.factory.redis.client.get(cache_key)
        if cached:
            return [CandidateResult(**c) for c in json.loads(cached)]

        # ── Phase 1: postgress (exact + partial + word overlap) ─
        candidates = self._lookup_in_postgres(entity=entity)
        # ── Phase 2: Neo4j graph candidates (exact + partial + word overlap) ─
        candidates = self.factory.neo4j.find_candidates_cypher(entity.text)
        seen = {c.canonical for c in candidates}

        # ── Phase 2: Elasticsearch fuzzy recall boost ─────────────────────────
        es_hits = self.factory.es.search_aliases(entity.text, top_k=10)
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
                ))

        if not candidates:
            return []

        # ── Phase 3: Neural + context ensemble scoring ───────────────────────
        candidates = self._ensemble_score(candidates, entity)

        # ── Phase 4: Threshold & sort ────────────────────────────────────────
        candidates = [c for c in candidates if c.match_score >= thresh]
        candidates.sort(key=lambda x: x.match_score, reverse=True)

        # Cache for 60 seconds
        self.factory.redis.client.set(cache_key, json.dumps([c.to_dict() for c in candidates]), ex=60)
        return candidates

    def _cache_key(self, entity_text: str, entity_label: str, sentence: str) -> str:
        h = hashlib.sha1(f"{entity_text.lower()}|{entity_label}|{sentence}".encode()).hexdigest()
        return f"candidates:{h}"

    def _ensemble_score(
        self,
        candidates: List[CandidateResult],
        entity:MentionEntity
    ) -> List[CandidateResult]:
        if not self.embedder:
            return candidates

        # Batch encode all candidate summaries + the query sentence
        summaries = [c.summary for c in candidates if c.summary]
        if not summaries:
            return candidates

        try:
            sent_vec = self.embedder.encode([entity.mention_sentence], convert_to_numpy=True)
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

            jaccard_ctx = jaro_winkler(entity.mention_sentence, cand.summary) if cand.summary else 0.0
            lev_name = levenshtein_ratio(entity.text.lower(), cand.summary.lower())

            label_bonus = 0.1 if (entity.label and cand.label.upper() == entity.label.upper()) else 0.0

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

        # ── Phase 4: Threshold & sort ────────────────────────────────────────
        candidates = [c for c in candidates if c.match_score >= self.threshold]
        candidates.sort(key=lambda x: x.match_score, reverse=True)
        return candidates
    
    # ── Lookup functions ─────────────────────────────────────────────────────────────
    def _lookup_in_doc(self, entity:MentionEntity, entities:List[MentionEntity]=None) -> List[CandidateResult]:
        if entities is None:
            return []
        
        result = []
        for erc in entities:
            if entity.text == erc.text and entity.label == erc.label:
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=1., 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FULL_MATCH)
                    )
                continue
            
            if entity.text == erc.canonical_name and entity.label == erc.label:
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=1., 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FULL_MATCH)
                    )
                continue

            if entity.canonical_name == erc.canonical_name and entity.label == erc.label:
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=1., 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FULL_MATCH)
                    )
                continue
            
            jaccard_score = char_ngram_jaccard(entity.text, erc.text) * .25
            jaro_winkler_score = jaro_winkler(entity.text, erc.text) * .25
            label_score = 0.5 if entity.label == erc.label else 0
            
            if jaccard_score + jaro_winkler_score + label_score >= self.threshold:
                score = max(jaccard_score + jaro_winkler_score + label_score, 1)
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=score, 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FUZZY_MATCH)
                              )
                continue
            
            leveshtine_score = levenshtein_ratio(entity.text, erc.text) * .25
            if leveshtine_score + jaro_winkler_score + label_score >= self.threshold:
                score = max(leveshtine_score + jaro_winkler_score + label_score, 1)
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=score, 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FUZZY_MATCH)
                    )
                continue
            
            if leveshtine_score + jaccard_score + label_score >= self.threshold:
                score = max(leveshtine_score + jaccard_score + label_score, 1)
                result.append(
                    CandidateResult(canonical=erc.canonical_name, 
                                    aliases=list(dict.fromkeys([erc.text, entity.text])), 
                                    summary=erc.summary, 
                                    match_score=score, 
                                    label_match=entity.label == erc.label,
                                    match_method=CandidateMatchMethod.FUZZY_MATCH)
                )
                continue
            
        return []
       
    def _lookup_in_redis(self, entity:MentionEntity)->List[CandidateResult]:
            pass
        
    def _lookup_in_postgres(self, entity:MentionEntity):
        candidates = []
        candidates.extend(self.factory.postgres.get_resolved_entities_by_canonical(entity.canonical_name))
        candidates.extend(self.factory.postgres.get_resolved_entities_by_text(entity.text))
        return [
            CandidateResult(canonical=cnt.canonical_name, 
                            aliases=list(dict.fromkeys([cnt.text, entity.text])), 
                            summary=cnt.summary, 
                            match_score=1., 
                            label_match=entity.label == cnt.label)
            for cnt in candidates
        ]
    
    def _lookup_in_es(self, entity:MentionEntity)->List[CandidateResult]:
        pass
    
    def _lookup_in_graph(self, entity:MentionEntity)->List[CandidateResult]:
        pass
    