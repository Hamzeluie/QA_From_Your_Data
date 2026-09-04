import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from QA_From_Your_Data.ingestion.models.llm.llm_extractors import NERExtractor, NERWithConfidence, CorefResolver
import spacy
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from config.settings import settings
from sentence_transformers import SentenceTransformer
from shared.data_classes import (Entity,
                                 EntityLabels, 
                                 DisambiguationStatus, 
                                 ResolvedEntity)

from shared.utils import (WikipediaEntitySummarizer, 
                          ValueNormalizer,
                          jaccard_similarity,
                          levenshtein_ratio,
                          NON_LINKABLE_TYPES)


class NEDEngine:
    def __init__(self, kg_entities: Optional[Dict] = None, embedder=None):
        self.kg = kg_entities or {}
        self.embedder = embedder
        self._build_index()

    def _build_index(self):
        self.alias_index: Dict[str, str] = {}
        self.name_index: Dict[str, str] = {}  # lowercase canonical name -> canonical

        for canonical, data in self.kg.items():
            self.alias_index[canonical.lower()] = canonical
            self.name_index[canonical.lower()] = canonical
            for alias in data.get("aliases", []):
                self.alias_index[alias.lower()] = canonical

    def add_entity(self,
                   canonical: str,
                   entity_label: str,
                   aliases: List[str],
                   summary: str = "",
                   context_indicators: List[str] = None,
                   related_to: List[str] = None) -> None:
        """
        Dynamically inject a user-resolved (or Wikipedia-discovered) entity
        into the live KG so find_candidates() can match it in future docs.
        """
        canonical = canonical.strip()
        all_aliases = list(set(
            [canonical] + [a.strip() for a in aliases if a.strip()]
        ))

        self.kg[canonical] = {
            "label": entity_label,
            "aliases": all_aliases,
            "summary": summary,
            "context_indicators": list(context_indicators or []),
            "related_to": list(related_to or []),
        }
        # Rebuild indices so the new entity is immediately searchable
        self._build_index()
        
    def find_candidates(self, entity_text: str, entity_label: str, entity_sentence: str, threshold:float=.5) -> List[Dict]:
        entity_lower = entity_text.lower().strip()
        candidates = []
        seen = set()

        # ── 1. Exact alias match ─────────────────────────────────────────────
        if entity_lower in self.alias_index:
            canonical = self.alias_index[entity_lower]
            seen.add(canonical)
            candidates.append(self._make_candidate(canonical, 1.0, "exact_match"))

        # ── 2. Partial / substring match ───────────────────────────────────────
        for alias, canonical in self.alias_index.items():
            if canonical in seen:
                continue
            if entity_lower in alias or alias in entity_lower:
                score = len(entity_lower) / len(alias) if len(alias) > 0 else 0
                seen.add(canonical)
                candidates.append(self._make_candidate(canonical, score, "partial_match"))

        # ── 3. asymmetric word overlap ───
        words = re.findall(r'[\u0600-\u06FF\w]+', entity_lower)
        word_matches: Dict[str, int] = {}
        for word in words:
            if word in self.alias_index.keys():
                canonical = self.alias_index[word]
                if canonical not in seen:
                    word_matches[canonical] = word_matches.get(canonical, 0) + 1

        for canonical, match_count in word_matches.items():
            total_words = len(re.findall(r'[\u0600-\u06FF\w]+', canonical))
            score = match_count / max(total_words, 1)
            if score >= 0.5:
                candidates.append(self._make_candidate(canonical, score, "word_overlap"))

        # ── 4. Ensemble SCORING: Neural + Jaccard(sentence,summary) + Levenshtein ──
        for cand in candidates:
            summary = self.kg[cand["canonical"]].get("summary", "")

            # NEURAL: sentence vs KG summary
            neural_sim = 0.0
            if self.embedder and summary:
                try:
                    sent_vec = self.embedder.encode([entity_sentence], convert_to_numpy=True)
                    sum_vec = self.embedder.encode([summary], convert_to_numpy=True)
                    neural_sim = float(sk_cosine_similarity(sent_vec, sum_vec)[0][0])
                except Exception:
                    neural_sim = 0.0

            # JACCARD (context level): sentence vs summary
            jaccard_ctx = jaccard_similarity(entity_sentence, summary) if summary else 0.0

            # LEVENSHTEIN (mention level): already computed above, reuse
            lev_name = levenshtein_ratio(entity_sentence.lower(), summary.lower())

            # LABEL MATCH
            label_bonus = 0.1 if (entity_label and cand["label"].upper() == entity_label.upper()) else 0.0

            # ENSEMBLE weights
            
            final_score = (cand["match_score"] * 0.1) + \
                          (neural_sim * 0.70) + \
                          (jaccard_ctx * 0.05) + \
                          (lev_name * 0.05) + \
                          label_bonus
            cand["neural_sim"] = round(neural_sim, 3)
            cand["jaccard_ctx"] = round(jaccard_ctx, 3)
            cand["levenshtein_name"] = round(lev_name, 3)
            cand["label_match"] = label_bonus > 0
            cand["match_score"] = round(min(final_score, 1.0), 3)
            cand["match_method"] = f"{cand['match_method']}+ensemble"

        # ── 5. THRESHOLD FILTER ──────────────────────────────────────────────
        candidates = [c for c in candidates if c["match_score"] >= threshold]
        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        return candidates
    
    def _make_candidate(self, canonical: str, score: float, method: str) -> Dict:
        data = self.kg[canonical]
        return {
            "canonical": canonical,
            "label": data["label"],
            "aliases": data.get("aliases", []),
            "summary": data.get("summary", "")[:120] + "...",
            "context_indicators": data.get("context_indicators", []),
            "related_to": data.get("related_to", []),
            "match_score": round(score, 3),
            "match_method": method,
        }

    def disambiguate(self,
                     entity_text: str,
                     confidence_score:float,
                     entity_label_hint: Optional[str] = None,
                     context: str = "",
                     kg_threshold:float=0.88,
                     extracted_entities: Optional[List[Dict]] = None) -> ResolvedEntity:
        candidates = self.find_candidates(entity_text, entity_label_hint or "UNKNOWN", context, threshold=kg_threshold)

        if not candidates:
            return ResolvedEntity(
                original_text=entity_text,
                canonical_name=entity_text,
                entity_label=entity_label_hint or "UNKNOWN",
                mention_sentence=context,
                confidence=confidence_score,
                status=DisambiguationStatus.NEW_ENTITY,
                source="kg",
                context_clues=["No match in Knowledge Graph"],
                needs_review=True,
                is_nil=True,
            )

        if len(candidates) == 1:
            candidate = candidates[0]
            label_match = self._check_label_match(entity_label_hint, candidate["label"])
            if label_match or candidate["match_score"] >= kg_threshold:
                status = DisambiguationStatus.RESOLVED
                confidence = candidate["match_score"] * (0.9 if label_match else 0.85)
            else:
                status = DisambiguationStatus.AMBIGUOUS
                
            return ResolvedEntity(
                original_text=entity_text,
                canonical_name=candidate["canonical"],
                entity_label=candidate["label"],                
                mention_sentence=context,
                confidence=confidence,
                status=status,
                source="kg",
                kg_candidates=candidates,
                context_clues=[
                    f"Single candidate: {candidate['match_method']}",
                    f"Embedding sim: {candidate.get('embedding_sim', 'N/A')}"
                ],
                needs_review=True if not label_match or candidate["match_score"] < kg_threshold else False,
            )

        best_candidate, clues = self._select_by_context(
            candidates, entity_label_hint, context, extracted_entities or []
        )

        confidence = best_candidate["match_score"]
        label_match = self._check_label_match(entity_label_hint, best_candidate["label"])
        if confidence > kg_threshold and label_match:
            status = DisambiguationStatus.RESOLVED 
        else:
            status = DisambiguationStatus.AMBIGUOUS

        return ResolvedEntity(
            original_text=entity_text,
            canonical_name=best_candidate["canonical"],
            entity_label=best_candidate["label"],
            mention_sentence=context,
            confidence=confidence,
            status=status,
            source="kg",
            kg_candidates=candidates,
            context_clues=clues,
            needs_review=True if not label_match or confidence < kg_threshold else False,
        )

    def _check_label_match(self, hint: Optional[str], candidate_label: str) -> bool:
        if not hint or hint == "UNKNOWN":
            return True
        return hint.upper() == candidate_label.upper()

    def _select_by_context(self,
                          candidates: List[Dict],
                          label_hint: Optional[str],
                          context: str,
                          other_entities: List[Dict]) -> Tuple[Dict, List[str]]:
        context_lower = context.lower()
        scored = []

        for cand in candidates:
            score = cand["match_score"]
            clues = []

            if label_hint and cand["label"].upper() == label_hint.upper():
                score += 0.3
                clues.append(f"Label match: {label_hint}")

            for indicator in cand.get("context_indicators", []):
                if indicator.lower() in context_lower:
                    score += 0.25
                    clues.append(f"Context indicator: '{indicator}'")
                    break

            if other_entities:
                other_canonicals = {e.get("canonical_name", e.get("text", "")).lower()
                                 for e in other_entities}
                for related in cand.get("related_to", []):
                    if related.lower() in other_canonicals:
                        score += 0.2
                        clues.append(f"Related entity present: {related}")
                        break

            cand_words = set(re.findall(r'[\u0600-\u06FF\w]+', cand["canonical"].lower()))
            context_words = set(re.findall(r'[\u0600-\u06FF\w]+', context_lower))
            overlap = len(cand_words & context_words) / max(len(cand_words), 1)
            if overlap > 0.5:
                score += 0.15
                clues.append("High word overlap with context")

            scored.append((cand, score, clues))

        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score, best_clues = scored[0]
        best["match_score"] = min(best_score, 1.0)
        return best, best_clues


class EntityCatalog:
    def __init__(self, embedder=None):
        self.nodes: Dict[str, Dict] = {}
        self.alias_map: Dict[str, str] = {}
        self.embedder = embedder

    def register(self,
                 canonical: str,
                 entity_label: str,
                 alias: str,
                 summary: Optional[str] = None):
        canonical = canonical.strip()
        alias_clean = alias.strip().lower()

        embedding = None
        if self.embedder and summary:
            embedding = self.embedder.encode([summary], convert_to_numpy=True)[0]

        if canonical not in self.nodes:
            self.nodes[canonical] = {
                "canonical": canonical,
                "label": entity_label,
                "aliases": set(),
                "summary": summary,
                "mentions": 0,
                "embedding": embedding,
            }
        self.nodes[canonical]["aliases"].add(alias_clean)
        self.nodes[canonical]["mentions"] += 1
        self.alias_map[alias_clean] = canonical
        self.alias_map[canonical.lower()] = canonical

        if embedding is not None and self.nodes[canonical]["embedding"] is None:
            self.nodes[canonical]["embedding"] = embedding

    def lookup(self, text: str) -> Optional[Dict]:
        return self.nodes.get(self.alias_map.get(text.lower().strip()))

    def get_summary(self, canonical: str) -> Optional[str]:
        node = self.nodes.get(canonical)
        return node["summary"] if node else None

    def merge_similar_nodes(self, threshold: float = 0.88) -> List[Tuple[str, str, float]]:
        if not self.embedder or len(self.nodes) < 2:
            return []

        merges = []
        node_list = list(self.nodes.items())
        embeddings = []
        valid_keys = []

        for key, node in node_list:
            if node.get("embedding") is not None:
                embeddings.append(node["embedding"])
                valid_keys.append(key)

        if len(valid_keys) < 2:
            return []

        embeddings = np.array(embeddings)
        sim_matrix = sk_cosine_similarity(embeddings)
        merged = set()

        for i in range(len(valid_keys)):
            if valid_keys[i] in merged:
                continue
            for j in range(i + 1, len(valid_keys)):
                if valid_keys[j] in merged:
                    continue
                sim = sim_matrix[i][j]
                if sim >= threshold:
                    key_i, key_j = valid_keys[i], valid_keys[j]
                    node_i, node_j = self.nodes[key_i], self.nodes[key_j]

                    if node_j["mentions"] > node_i["mentions"]:
                        key_i, key_j = key_j, key_i
                        node_i, node_j = node_j, node_i

                    for alias in node_j["aliases"]:
                        node_i["aliases"].add(alias)
                        self.alias_map[alias] = key_i

                    node_i["mentions"] += node_j["mentions"]

                    if not node_i["summary"] and node_j["summary"]:
                        node_i["summary"] = node_j["summary"]
                        node_i["embedding"] = node_j["embedding"]

                    del self.nodes[key_j]
                    merged.add(key_j)
                    merges.append((key_j, key_i, float(sim)))

        return merges

    def stats(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "canonical": c,
                "label": d["label"],
                "aliases": ", ".join(sorted(d["aliases"])),
                "mentions": d["mentions"],
                "has_summary": d["summary"] is not None,
                "has_embedding": d.get("embedding") is not None,
            }
            for c, d in self.nodes.items()
        ])


class CrossDocumentResolver:
    def __init__(self, catalog: EntityCatalog, embedder, threshold: float = 0.82):
        self.catalog = catalog
        self.embedder = embedder
        self.threshold = threshold

    def resolve_cross_document(self, all_results: List[ResolvedEntity]) -> List[ResolvedEntity]:
        if not self.embedder:
            return all_results

        catalog_items = list(self.catalog.nodes.items())
        catalog_embeddings = []
        catalog_keys = []

        for key, node in catalog_items:
            if node.get("embedding") is not None:
                catalog_embeddings.append(node["embedding"])
                catalog_keys.append(key)

        if not catalog_keys:
            return all_results

        catalog_embeddings = np.array(catalog_embeddings)
        updated = []

        for ent in all_results:
            if ent.status != DisambiguationStatus.UNKNOWN and not ent.is_nil:
                updated.append(ent)
                continue

            sent_vec = self.embedder.encode([ent.sentence], convert_to_numpy=True)
            sims = sk_cosine_similarity(sent_vec, catalog_embeddings)[0]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.threshold:
                matched_key = catalog_keys[best_idx]
                matched_node = self.catalog.nodes[matched_key]
                ent.canonical_name = matched_key
                ent.confidence = best_sim
                ent.status = DisambiguationStatus.RESOLVED
                ent.source = "cross_doc"
                ent.summary = matched_node.get("summary")
                ent.is_nil = False
                ent.context_clues.append(f"Cross-doc match to {matched_key} (sim={best_sim:.3f})")
                self.catalog.register(
                    canonical=matched_key,
                    entity_label=matched_node["label"],
                    alias=ent.original_text,
                )
            else:
                ent.context_clues.append(f"No cross-doc match (best sim={best_sim:.3f})")

            updated.append(ent)

        return updated


class EntityResolver:
    def __init__(self,
                 kg_entities: Optional[Dict] = None,
                 embedding: SentenceTransformer=None,
                 merge_threshold: float = 0.88,
                 cross_doc_threshold: float = 0.82,
                 use_ner_with_confidence:bool=False,
                 use_cot:bool=True):
        self.ner_llm = NERWithConfidence() if use_ner_with_confidence else NERExtractor(use_cot=use_cot)
        self.wiki = WikipediaEntitySummarizer(embedding)
        embedder = self.wiki.embedder if self.wiki else None
        
        self.ned = NEDEngine(kg_entities, embedder=embedder)
        self.catalog = EntityCatalog(embedder=embedder)
        
        self.nlp = spacy.load(settings.SPACY_MODEL_PATH)
        self.coref = CorefResolver(self.nlp, mode="nlp")
        
        self.merge_threshold = merge_threshold
        self.cross_doc_threshold = cross_doc_threshold
        
    def _fake_llm_response(self, doc:str):
        return [{'sentence': "Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator .He was born in the city of Loja .He is best known today as the author of Ecuador 's first novel La Emancipada ( 1863 ) .Owing to the book 's length , usually less than 100 pages long , many experts have argued that it is really a novella rather than a full novel , and that Ecuador 's first novel is Juan León Mera 's Cumanda ( 1879 ) .Nevertheless",
                 'entities': [Entity(text='Miguel Riofrio Sánchez', label='PER', start=0, end=22, mention_sentence="<Miguel Riofrio Sánchez> ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator .He was born in the city of Loja .He is best known today as the author of Ecuador 's first novel La Emancipada ( 1863 )", confidence=1.0),
                              Entity(text='September 7', label='TIME', start=25, end=36, mention_sentence='Miguel Riofrio Sánchez ( <September 7> , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator', confidence=1.0),
                              Entity(text='1822', label='TIME', start=39, end=43, mention_sentence='Miguel Riofrio Sánchez ( September 7 , <1822> – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator', confidence=1.0),
                              Entity(text='October 11', label='TIME', start=46, end=56, mention_sentence='Miguel Riofrio Sánchez ( September 7 , 1822 – <October 11> , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator', confidence=1.0),
                              Entity(text='1879', label='TIME', start=59, end=63, mention_sentence='Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , <1879> ) was an Ecuadoran poet , novelist , journalist , orator , and educator', confidence=1.0),
                              Entity(text='Ecuador', label='LOC', start=73, end=80, mention_sentence="Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an <Ecuador>an poet , novelist , journalist , orator , and educator .He was born in the city of Loja .He is best known today as the author of Ecuador 's first novel La Emancipada ( 1863 )", confidence=1.0),
                              Entity(text='Loja', label='LOC', start=164, end=168, mention_sentence="Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator .He was born in the city of <Loja> .He is best known today as the author of Ecuador 's first novel La Emancipada ( 1863 )", confidence=1.0),
                              Entity(text='La Emancipada', label='MISC', start=233, end=246, mention_sentence="Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator .He was born in the city of Loja .He is best known today as the author of Ecuador 's first novel <La Emancipada> ( 1863 )", confidence=1.0),
                              Entity(text='1863', label='TIME', start=249, end=253, mention_sentence="He is best known today as the author of Ecuador 's first novel La Emancipada ( <1863> )", confidence=1.0),
                              Entity(text='Juan León Mera', label='PER', start=437, end=451, mention_sentence=".Owing to the book 's length , usually less than 100 pages long , many experts have argued that it is really a novella rather than a full novel , and that Ecuador 's first novel is <Juan León Mera> 's Cumanda ( 1879 ) .Nevertheless", confidence=1.0),
                              Entity(text='Cumanda', label='MISC', start=455, end=462, mention_sentence=".Owing to the book 's length , usually less than 100 pages long , many experts have argued that it is really a novella rather than a full novel , and that Ecuador 's first novel is Juan León Mera 's <Cumanda> ( 1879 ) .Nevertheless", confidence=1.0),
                              Entity(text='1879', label='TIME', start=465, end=469, mention_sentence="Owing to the book 's length , usually less than 100 pages long , many experts have argued that it is really a novella rather than a full novel , and that Ecuador 's first novel is Juan León Mera 's Cumanda ( <1879> )", confidence=1.0)
                              ]} ,    
            {'sentence': ", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru .", 
             'entities': [Entity(text='Alejandro Carrión', label='PER', start=78, end=95, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer <Alejandro Carrión> ( 1915 – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru .", confidence=1.0),
                          Entity(text='Miguel Riofrío', label='PER', start=114, end=128, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , <Miguel Riofrío> 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru .", confidence=1.0),
                          Entity(text='Ecuador', label='LOC', start=60, end=67, mention_sentence=", thanks to the arguments of the well - known and respected <Ecuador>ian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru .", confidence=1.0),
                          Entity(text='Peru', label='LOC', start=216, end=220, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in <Peru> .", confidence=1.0),
                          Entity(text='1915', label='TIME', start=98, end=102, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( <1915> – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel", confidence=1.0),
                          Entity(text='1992', label='TIME', start=105, end=109, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – <1992> ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel", confidence=1.0),
                          Entity(text='La Emancipada', label='MISC', start=132, end=145, mention_sentence=", thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's <La Emancipada> has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru .", confidence=1.0)
                          ]}]
    
    def _name_entity_recognition(self, doc:str)-> List[Dict]:
        all_preds = []
        char_offset = 0
        # for idx, chunk in enumerate(semantic_sentence_chunk(doc)):
            # result = self.ner_llm(chunk)
        for idx, result in enumerate(self._fake_llm_response(doc)):
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
         
    def _resolve_entity(self,
                       entity_text: str,
                       entity_label: str,
                       sentence: str,
                       confidence_score:float,
                       other_entities: Optional[List[Dict]] = None) -> ResolvedEntity:
            if entity_label in NON_LINKABLE_TYPES:
                if entity_label == "MONEY":
                    normalized = ValueNormalizer.normalize_money(entity_text)
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=normalized["canonical"],
                        entity_label=entity_label,
                        mention_sentence=sentence,
                        confidence=confidence_score,
                        status=DisambiguationStatus.RESOLVED,
                        source="normalized",
                        context_clues=[f"Normalized money: {normalized['currency']} {normalized['value']}"],
                        needs_review=False,
                    )
    
                elif entity_label == "DATE":
                    normalized = ValueNormalizer.normalize_date(entity_text)
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=normalized["canonical"],
                        entity_label=entity_label,
                        mention_sentence=sentence,
                        confidence=confidence_score,
                        status=DisambiguationStatus.RESOLVED,
                        source="normalized",
                        context_clues=[f"Normalized DATE: {normalized['canonical']} ({normalized['granularity']})"],
                        needs_review=False,
                    )
    
                elif entity_label == "TIME":
                    # You can add a time normalizer later; for now keep literal
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=entity_text,
                        entity_label=entity_label,
                        mention_sentence=sentence,
                        confidence=confidence_score,
                        status=DisambiguationStatus.RESOLVED,
                        source="normalized",
                        needs_review=False,
                    )
    
                else:
                    # QUANTITY, CARDINAL, ORDINAL, PERCENT — do NOT parse as date
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=entity_text,
                        entity_label=entity_label,
                        mention_sentence=sentence,
                        confidence=confidence_score,
                        status=DisambiguationStatus.RESOLVED,
                        source="normalized",
                        needs_review=False,
                    )
    
            catalog_hit = self.catalog.lookup(entity_text)
            if catalog_hit:
                return ResolvedEntity(
                    original_text=entity_text,
                    canonical_name=catalog_hit["canonical"],
                    entity_label=catalog_hit["label"],
                    mention_sentence=sentence,
                    confidence=confidence_score,
                    status=DisambiguationStatus.RESOLVED,
                    source="catalog",
                    summary=catalog_hit.get("summary"),
                    context_clues=["Retrieved from EntityCatalog"],
                    needs_review=False,
                )
    
            kg_candidates: List[Dict] = []
            if self.ned:
                kg_result = self.ned.disambiguate(
                    entity_text=entity_text,
                    entity_label_hint=entity_label,
                    context=sentence,
                    confidence_score=confidence_score,
                    extracted_entities=other_entities or []
                )
                kg_candidates = kg_result.kg_candidates
                if kg_result.status == DisambiguationStatus.RESOLVED:
                    self.catalog.register(
                        canonical=kg_result.canonical_name,
                        entity_label=kg_result.entity_label,
                        alias=entity_text,
                    )
                    return kg_result
    
            if self.wiki:
                wiki_result = self.wiki.summarize(entity=entity_text, context=sentence, ner_label=entity_label)
                if wiki_result:
                    kg_result = self.ned.disambiguate(
                        entity_text=wiki_result.original_text,
                        confidence_score=wiki_result.confidence,
                        entity_label_hint=wiki_result.entity_label,
                        context=wiki_result.mention_sentence,
                        extracted_entities=other_entities or []
                        )
                    # FIX: Only register to catalog if actually resolved
                    resolved_obj = ResolvedEntity(original_text=wiki_result.original_text,
                                                  canonical_name=wiki_result.canonical_name,
                                                  entity_label=wiki_result.entity_label,
                                                  mention_sentence=kg_result.mention_sentence,
                                                  confidence=wiki_result.confidence,
                                                  status=DisambiguationStatus.RESOLVED if kg_result.status == DisambiguationStatus.RESOLVED else wiki_result.status,
                                                  source=wiki_result.source,
                                                  kg_candidates=kg_result.kg_candidates,
                                                  summary=wiki_result.summary,
                                                  context_clues=wiki_result.context_clues,
                                                  needs_review=True,
                                                  is_nil=wiki_result.is_nil,
                                                  coref_to=kg_result.coref_to
                                                  )
                    if resolved_obj.status == DisambiguationStatus.RESOLVED:
                        self.catalog.register(
                            canonical=resolved_obj.canonical_name,
                            entity_label=resolved_obj.entity_label,
                            alias=entity_text,
                            summary=resolved_obj.summary,
                        )
                    return resolved_obj
    
            
            if kg_candidates and len(kg_candidates) == 1:
                cand = kg_candidates[0]
                if cand["match_method"].startswith("exact_match"):
                    # Force-resolve via catalog so future docs hit the cache
                    self.catalog.register(
                        canonical=cand["canonical"],
                        entity_label=cand["type"],  # use KG type, not NER type
                        alias=entity_text,
                    )
                    return ResolvedEntity(
                        original_text=entity_text,
                        canonical_name=cand["canonical"],
                        entity_label=cand["type"],
                        mention_sentence=sentence,
                        confidence=max(cand["match_score"], 0.90),
                        status=DisambiguationStatus.RESOLVED,
                        source="kg",
                        kg_candidates=kg_candidates,
                        context_clues=[f"Exact alias match (type override: {entity_label}→{cand['type']})"],
                        needs_review=False,
                    )
                
            nil_entity = ResolvedEntity(
                original_text=entity_text,
                canonical_name=entity_text,
                entity_label=entity_label,
                mention_sentence=sentence,
                confidence=confidence_score,
                status=DisambiguationStatus.UNKNOWN,
                source="unresolved",
                kg_candidates=kg_candidates,
                context_clues=["No match in KG, Wikipedia, or Catalog"],
                needs_review=True,
                is_nil=True,
            )
            self.catalog.register(
                canonical=entity_text,
                entity_label=entity_label,
                alias=entity_text,
            )
            return nil_entity
    
    def _resolve_dataframe(self, entity_list: List[Dict]) -> Tuple[pd.DataFrame, List[ResolvedEntity]]:
        results: List[ResolvedEntity] = []

        for idx, row in entity_list.iterrows():
            if "coref_to" in row and pd.notna(row["coref_to"]):
                antecedent_text = row["coref_to"]
                catalog_hit = self.catalog.lookup(antecedent_text)
                antecedent_canonical = catalog_hit["canonical"] if catalog_hit else antecedent_text

                resolved = ResolvedEntity(
                    original_text=row["text"],
                    canonical_name=antecedent_canonical,
                    entity_label=row["label"],
                    mention_sentence=row["mention_sentence"],
                    confidence=row["confidence"],
                    status=DisambiguationStatus.RESOLVED,
                    source="coref",
                    context_clues=[f"Coreference to '{antecedent_text}' -> {antecedent_canonical}"],
                )
                resolved.coref_to = antecedent_canonical
                results.append(resolved)
                continue

            same_doc = entity_list[entity_list["doc_id"] == row["doc_id"]]
            other_entities = [
                {"text": r["text"], "type": r["label"], "canonical_name": r["text"]}
                for _, r in same_doc.iterrows() if r.name != idx
            ]

            resolved = self._resolve_entity(
                entity_text=row["text"],
                entity_label=row["label"],
                sentence=row["mention_sentence"],
                confidence_score=row["confidence"],
                other_entities=other_entities)
            results.append(resolved)

        result_df = pd.DataFrame([r.to_dict() for r in results])
        # Drop columns already present in df to avoid duplicates
        result_df = result_df.drop(columns=["sentence", "coref_to"], errors="ignore")

        combined = pd.concat([entity_list.reset_index(drop=True), result_df], axis=1)
        return combined, results
    
    def _chunk_entity_splitter(self, ner_result:list[Dict]):
        all_entities = []
        chunk_info = []
        for chunk in ner_result:
            chunk_info.append((chunk["chunk_id"], chunk["chunk_text"]))
            all_entities.extend(chunk["entities"])

        entences(document, all_entities), chunk_info
        
    def process_document(self, document: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: extract → resolve → cross-doc → merge → return.

        Returns:
            clean_df: Resolved / new entities with context (input for Relation Extraction)
            review_df: Ambiguous / unknown entities with candidates (for human review)
        """
        # ── Step 1: Extract (NER , Coref) ─────────────────────────────────────────────
        entities_df, chunk_info = self._extract_entities_with_coref(document)
        
        # ── Step 2: Resolve (per-entity: KG → Wiki → Catalog) (NED, ER) ───────────────────
        _, all_results = self._resolve_dataframe(entities_df)

        # ── Step 3: Split into clean (RE input) vs review (human check) ────────
        resolved_new_indices: List[int] = []
        review_indices: List[int] = []
        resolved_new_entities: List[ResolvedEntity] = []
        review_entities: List[ResolvedEntity] = []

        for i, entity in enumerate(all_results):
            is_clean = entity.status == DisambiguationStatus.RESOLVED
            if is_clean:
                resolved_new_indices.append(i)
                resolved_new_entities.append(entity)
            else:
                review_indices.append(i)
                review_entities.append(entity)

        # ── Step 4: Catalog merge (only on resolved/new nodes) ──────────────────
        merges = self.catalog.merge_similar_nodes(threshold=self.merge_threshold)
        if merges:
            print(f"\n[Catalog Merge] Merged {len(merges)} duplicate node pairs:")
            for src, dst, sim in merges:
                print(f"    '{src}' -> '{dst}' (sim={sim:.3f})")

        # ── Step 5: Cross-document resolution — ONLY on resolved/new entities ───
        cross_resolver = CrossDocumentResolver(
            self.catalog,
            self.wiki.embedder if self.wiki else None,
            self.cross_doc_threshold
        )
        updated_resolved = cross_resolver.resolve_cross_document(resolved_new_entities)

        # ── Step 6: Build CLEAN DataFrame for Relation Extraction ───────────────
        clean_records = []
        for idx, entity in zip(resolved_new_indices, updated_resolved):
            orig = entities_df.iloc[idx]
            clean_records.append({
                "doc_id": orig["doc_id"],
                "original_text": entity.original_text,
                "canonical_name": entity.canonical_name,
                "entity_label": entity.entity_label,
                "start": orig["start"],
                "end": orig["end"],
                "mention_sentence": orig["mention_sentence"],
                "confidence": entity.confidence,
                "status": entity.status.value,
                "source": entity.source,
                "summary": entity.summary,
                "needs_review": entity.needs_review,
                "is_nil": entity.is_nil,
                "coref_to": entity.coref_to,
                "context_clues": entity.context_clues,
                "kg_candidates": entity.kg_candidates,
            })
        clean_df = pd.DataFrame(clean_records)

        # ── Step 7: Build REVIEW DataFrame for human correction ─────────────────
        review_records = []
        for idx, entity in zip(review_indices, review_entities):
            orig = entities_df.iloc[idx]
            review_records.append({
                "resolution_id": f"review_{len(review_records)}",
                "doc_id": orig["doc_id"],
                "original_text": entity.original_text,
                "canonical_name": entity.canonical_name,
                "entity_label": entity.entity_label,
                "start": orig["start"],
                "end": orig["end"],
                "mention_sentence": orig["mention_sentence"],
                "confidence": entity.confidence,
                "status": entity.status.value,
                "source": entity.source,
                "summary": entity.summary,
                "needs_review": entity.needs_review,
                "is_nil": entity.is_nil,
                "coref_to": entity.coref_to,
                "context_clues": entity.context_clues,
                "kg_candidates": entity.kg_candidates,
                "review_reason": (
                    "Ambiguous: multiple candidates or low confidence"
                    if entity.status == DisambiguationStatus.AMBIGUOUS
                    else "Unknown: no match in KG, Wikipedia, or Catalog"
                ),
            })
        review_df = pd.DataFrame(review_records)
        return chunk_info, clean_df, review_df

    def get_entity_clusters(self) -> pd.DataFrame:
        return self.catalog.stats()

    def add_user_resolution(self,
                        canonical_name: str,
                        entity_label: EntityLabels,
                        summary: str,
                        aliases: List[str] = None,
                        context_indicators: Optional[List[str]] = None,
                        related_to: Optional[List[str]] = None) -> None:
        """
        Persist a USER-CONFIRMED resolution into Catalog + KG.
        This is the ONLY way non-KG entities enter persistent storage.
        """
        canonical = canonical_name.strip()
        alias_list = list(set([
            canonical_name.strip(),
            canonical
        ] + [a.strip() for a in (aliases or []) if a.strip()]))

        # ── 1. Exact-match cache ───────────────────────────────────────────
        for alias in alias_list:
            self.catalog.register(
                canonical=canonical,
                entity_label=entity_label,
                alias=alias,
                summary=summary,
            )

        # ── 2. Fuzzy-match index (NEDEngine) ──────────────────────────────
        if self.ned is not None:
            self.ned.add_entity(
                canonical=canonical,
                entity_label=entity_label,
                aliases=alias_list,
                summary=summary or "",
                context_indicators=list(context_indicators or []),
                related_to=list(related_to or []),
            )

        print(f"[User Feedback] -> '{canonical}' "
              f"({entity_label}) registered with {len(alias_list)} alias(es).")
    
    def save_full_state(self,
                        path: str,
                        clean_df: pd.DataFrame = None,
                        review_df: pd.DataFrame = None) -> None:
        """
        Serialize catalog, KG, resolved entities (clean_df), and unresolved
        review items (review_df) into separate JSON files within a directory.
        """
        # Ensure the target directory exists
        os.makedirs(path, exist_ok=True)

        # ── 1. Prepare Catalog Data ──
        catalog_data = {
            k: {
                "label": v["label"],
                "aliases": list(v["aliases"]),
                "summary": v.get("summary"),
                "url": v.get("url"),
                "mentions": v["mentions"],
            }
            for k, v in self.catalog.nodes.items()
        }

        # ── 2. Prepare KG Data ──
        kg_data = self.ned.kg if self.ned else {}

        # ── 3. Helper: convert DataFrame → clean JSON-serializable records ──
        def _df_to_records(df: pd.DataFrame) -> List[Dict]:
            if df is None or df.empty:
                return []
            # Replace NaN/NaT with None so json.dump doesn't choke
            clean = df.copy()
            clean = clean.astype(object).where(pd.notnull(clean), None)
            return clean.to_dict("records")

        resolved_records = _df_to_records(clean_df)
        unresolved_records = _df_to_records(review_df)

        # ── 4. Define File Paths ──
        catalog_path = os.path.join(path, "catalog.json")
        kg_path = os.path.join(path, "kg.json")
        resolved_path = os.path.join(path, "resolved.json")
        unresolved_path = os.path.join(path, "unresolved.json")

        # ── 5. Write to Individual Files ──
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False, default=str)

        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False, default=str)

        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(resolved_records, f, indent=2, ensure_ascii=False, default=str)

        with open(unresolved_path, "w", encoding="utf-8") as f:
            json.dump(unresolved_records, f, indent=2, ensure_ascii=False, default=str)

        # ── 6. Summary Logging ──
        print(f"[Save] {len(catalog_data)} catalog | "
              f"{len(kg_data)} KG | "
              f"{len(resolved_records)} resolved | "
              f"{len(unresolved_records)} unresolved → {path}/")

    def load_full_state(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Hydrate catalog + KG from individual JSON files.
        Returns (resolved_df, unresolved_df) and stores them as instance vars.
        """
        assert os.path.isdir(path), f"path must be a directory: {path}"

        catalog_path   = os.path.join(path, "catalog.json")
        kg_path        = os.path.join(path, "kg.json")
        resolved_path  = os.path.join(path, "resolved.json")
        unresolved_path= os.path.join(path, "unresolved.json")

        # ── 1. Catalog ──
        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
            for canonical, data in catalog_data.items():
                for alias in data.get("aliases", []):
                    self.catalog.register(
                        canonical=canonical,
                        entity_label=data.get("label", "UNKNOWN"),
                        alias=alias,
                        summary=data.get("summary"),
                    )

        # ── 2. KG ──
        if self.ned and os.path.exists(kg_path):
            with open(kg_path, "r", encoding="utf-8") as f:
                kg_data = json.load(f)
            for canonical, data in kg_data.items():
                self.ned.add_entity(
                    canonical=canonical,
                    entity_label=data.get("label", "UNKNOWN"),
                    aliases=data.get("aliases", []),
                    summary=data.get("summary", ""),
                    context_indicators=data.get("context_indicators", []),
                    related_to=data.get("related_to", []),
                )

        # ── 3. DataFrames ──
        resolved_df = pd.DataFrame()
        if os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                resolved_df = pd.DataFrame(json.load(f))

        unresolved_df = pd.DataFrame()
        if os.path.exists(unresolved_path):
            with open(unresolved_path, "r", encoding="utf-8") as f:
                unresolved_df = pd.DataFrame(json.load(f))

        # Cache on instance so getters work without reloading from disk
        self._resolved_df   = resolved_df
        self._unresolved_df = unresolved_df

        print(f"[Load] {path} — catalog:{len(self.catalog.nodes)}  "
            f"kg:{len(self.ned.kg) if self.ned else 0}  "
            f"resolved:{len(resolved_df)}  unresolved:{len(unresolved_df)}")
        return resolved_df, unresolved_df

    def get_all_resolved(self) -> pd.DataFrame:
        """
        Union of Catalog + KG entries as a single DataFrame.
        Catalog entries take precedence (they have mention counts & embeddings).
        """
        records = []
        seen = set()

        # 1. Catalog nodes (runtime cache — includes auto + user resolved)
        for canonical, data in self.catalog.nodes.items():
            seen.add(canonical.lower())
            records.append({
                "layer": "catalog",
                "canonical_name": canonical,
                "entity_type": data.get("type"),
                "aliases": sorted(data.get("aliases", set())),
                "mentions": data.get("mentions", 0),
                "summary": data.get("summary"),
                "url": data.get("url"),
            })

        # 2. KG-only entries (seed entities not yet seen in any document)
        if self.ned:
            for canonical, data in self.ned.kg.items():
                if canonical.lower() in seen:
                    continue
                records.append({
                    "layer": "kg_only",
                    "canonical_name": canonical,
                    "entity_type": data.get("type"),
                    "aliases": data.get("aliases", []),
                    "mentions": 0,
                    "summary": data.get("summary"),
                    "url": None,
                })

        return pd.DataFrame(records)

    def get_all_unresolved(self) -> pd.DataFrame:
        """Return the last loaded unresolved review queue."""
        return getattr(self, "_unresolved_df", pd.DataFrame())

    def get_all_kg(self) -> pd.DataFrame:
        """Return the curated Knowledge Graph as a DataFrame."""
        if not self.ned:
            return pd.DataFrame()
        records = []
        for canonical, data in self.ned.kg.items():
            records.append({
                "canonical_name": canonical,
                "entity_type": data.get("type"),
                "aliases": data.get("aliases", []),
                "summary": data.get("summary"),
                "context_indicators": data.get("context_indicators", []),
                "related_to": data.get("related_to", []),
            })
        return pd.DataFrame(records)

    def get_all_catalogs(self) -> pd.DataFrame:
        """Alias for get_entity_clusters()."""
        return self.catalog.stats()




















