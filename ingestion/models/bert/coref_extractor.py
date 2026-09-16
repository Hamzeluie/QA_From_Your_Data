import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import os
from typing import List, Dict, Tuple

from ingestion.models.base import IExtractor
from storage.data_classes import MentionEntity
from shared.utils import extract_exact_sentence
from config.settings import settings


class BertCorefResolver(IExtractor):
    """
    fastcoref-based neural coreference resolution.
    Input:  full document text + DataFrame or List[Entity] from NER.
    Output: List[ResolvedEntity] — original entities + resolved pronouns/descriptions.
    """

    def __init__(self, nlp: str = "", use_neural: bool = True, threshold:float=.92):
        self.use_neural = use_neural
        self.threshold = threshold
        self.neural_coref = None
        nlp = nlp if nlp else os.path.join(settings.SPACY_MODEL_PATH, settings.SPACY_MODEL_NAME)
        if use_neural:
            try:
                from fastcoref import FCoref
                self.neural_coref = FCoref(device="cpu", nlp=nlp)
            except ImportError:
                print("[warn] fastcoref not installed; install with: pip install fastcoref")
                self.use_neural = False

    def extract(
        self,
        text: str,
        entities: List[MentionEntity],
    ) -> List[MentionEntity]:
        """
        Extract coreference-resolved entities from the text.
        Accepts a List[Entity].
        """
        # ── 1. asserting entities and models ──
        if not entities:
            return entities
        
        if not self.use_neural or self.neural_coref is None:
            return entities
        
        # ── 2. Neural coref (pronouns + descriptions) ──
        preds = self.neural_coref.predict(texts=[text])[0]
        clusters = preds.get_clusters(as_strings=False)
        text_clusters = preds.get_clusters(as_strings=True)

        # Map mention span -> (canonical_name, label) from NER
        mention_to_canon: Dict[Tuple[int, int], Tuple[str, str]] = {}
        for e in entities:
            key = (e.start, e.end)
            mention_to_canon[key] = (
                getattr(e, "canonical_name", e.text),
                getattr(e, "label", e.label),
                getattr(e, "doc_id", e.doc_id),
                getattr(e, "chunk_id", e.chunk_id),
            )

        for cluster_strs, cluster_spans in zip(text_clusters, clusters):
            if not cluster_spans:
                continue

            # Find named entities in this cluster
            canon_ents = []
            for (start, end) in cluster_spans:
                canon = self._find_canon_for_span(start, end, mention_to_canon, text)
                if canon:
                    canon_ents.append(canon)

            # Only need 1 named entity to serve as antecedent
            if len(canon_ents) < 1:
                continue

            # Pick antecedent: longest canonical name, carry its label
            antecedent, antecedent_label, antecedent_doc_id, antecedent_chunk_id = max(canon_ents, key=lambda x: len(x[0]))
            coref_confidence = self._calculate_coref_confidence(cluster_spans, mention_to_canon)

            # Add all non-named-entity spans in this cluster as resolved mentions
            for (start, end) in cluster_spans:
                if self._find_canon_for_span(start, end, mention_to_canon, text):
                    continue    # skip named entities already added above

                mention_text = extract_exact_sentence(doc_text=text,
                                       start_char=start,
                                       end_char=end,
                                       use_nlp=False,
                                       nlp=None)
                entities.append(MentionEntity(
                    doc_id=antecedent_doc_id,
                    chunk_id=antecedent_chunk_id,
                    text=text[start:end],
                    label=antecedent_label,
                    mention_sentence=mention_text,
                    confidence=coref_confidence,
                    coref_to=antecedent,
                    start=start,
                    end=end,
                    
                ))

        return entities

    def _calculate_coref_confidence(
        self, 
        cluster_spans: List[Tuple[int, int]], 
        mention_to_canon: Dict[Tuple[int, int], Tuple[str, str]]
    ) -> float:
        """
        Calculate a proxy confidence score for a coreference cluster.
        Since fastcoref doesn't expose raw logits, we use cluster heuristics.
        """
        base_score = 0.75
        num_mentions = len(cluster_spans)
        
        # 1. Size bonus: More mentions in a cluster increase confidence
        # (e.g., "Apple", "the company", "it", "they" = very confident)
        size_bonus = min(num_mentions * 0.04, 0.15)
        
        # 2. Distance penalty: Mentions far apart are less likely to be coreferent
        distance_penalty = 0.0
        if num_mentions >= 2:
            # Calculate average token distance between consecutive mentions
            distances = [
                cluster_spans[i+1][0] - cluster_spans[i][1] 
                for i in range(num_mentions - 1)
            ]
            avg_distance = sum(distances) / len(distances)
            
            if avg_distance > 100:
                distance_penalty = 0.15
            elif avg_distance > 50:
                distance_penalty = 0.08
            elif avg_distance > 20:
                distance_penalty = 0.03
                
        # 3. Antecedent quality bonus: If the first mention is a strong Named Entity
        first_mention_key = (cluster_spans[0][0], cluster_spans[0][1])
        antecedent_bonus = 0.10 if first_mention_key in mention_to_canon else 0.0
        
        # Calculate final score and clamp between 0.0 and 1.0
        final_score = base_score + size_bonus - distance_penalty + antecedent_bonus
        return round(max(0.0, min(1.0, final_score)), 3)

    def _find_canon_for_span(self, start: int, end: int, mention_to_canon: dict, text: str):
        """Match a coref span to a canonical NER entity, tolerating punctuation drift."""
        # 1. exact match
        key = (start, end)
        if key in mention_to_canon:
            return mention_to_canon[key]

        # 2. normalized text match
        coref_text = text[start:end].strip(" .,()[]{}\"';:-").lower()
        if not coref_text:
            return None

        for (s, e), canon in mention_to_canon.items():
            ner_text = text[s:e].strip(" .,()[]{}\"';:-").lower()
            if coref_text == ner_text:
                return canon
        return None


# ── Main: end-to-end demo with coreference ─────────────────────────
if __name__ == "__main__":
    from config.settings import settings
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    coref = BertCorefResolver(nlp=os.path.join(settings.SPACY_MODEL_PATH, settings.SPACY_MODEL_NAME), use_neural=True)
    entities = [
        MentionEntity(doc_id="1", chunk_id="1" , text='Apple Inc.', label='ORGANIZATION', start=0, end=10, mention_sentence='Ap...dquartered in Cupertino.', confidence=1.0), 
        MentionEntity(doc_id="1", chunk_id="1",  text='Steve Jobs', label='PERSON', start=26, end=36, mention_sentence='Apple I...dquartered in Cupertino.', confidence=1.0), 
        MentionEntity(doc_id="1", chunk_id="1" , text='Cupertino', label='LOCATION', start=104, end=113, mention_sentence='Appl...dquartered in Cupertino.', confidence=1.0)]
    
    resolved_entities = coref(text, entities)
    print("\n--- Coref Output (List[ResolvedEntity]) ---")
    for r in resolved_entities:
        coref_info = f" -> coref_to={r.coref_to!r}" if r.coref_to else ""
        print(f"  {r.original_text!r:<20} | canonical={r.canonical_name!r:<20} | {r.entity_label:<6} | src={r.source}{coref_info}")

        