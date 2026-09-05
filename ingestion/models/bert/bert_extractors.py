# ingestion/models/bert/bert_extractors.py
import sys
import warnings
import hashlib
from pathlib import Path

from httpx import get

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from optimum.onnxruntime import (
    ORTModelForTokenClassification,
    ORTModelForSequenceClassification,
)

from ingestion.models.base import IExtractor
from shared.data_classes import (Entity, Relation, RelationLabels, EntityLabels, DisambiguationStatus)
from shared.utils import extract_exact_sentence, NON_LINKABLE_TYPES
from config.settings import settings
# ── Robust loading helpers ─────────────────────────────────────────

def _is_valid_local_model(path: str) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    return (p / "config.json").exists() or (p / "tokenizer_config.json").exists()


def _load_tokenizer_robust(load_path: str, model_name: str, local_dir: Optional[str]):
    kwargs = {"use_fast": True, "trust_remote_code": True}

    if _is_valid_local_model(load_path):
        try:
            return AutoTokenizer.from_pretrained(load_path, local_files_only=True, **kwargs)
        except Exception as e:
            print(f"[warn] Local tokenizer at {load_path} failed: {e}")

    print(f"[info] Downloading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)

    if local_dir:
        save_path = Path(local_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(save_path)
        print(f"[info] Tokenizer cached to {save_path}")

    return tok


def _load_model_robust(
    load_path: str,
    model_name: str,
    local_dir: Optional[str],
    onnx_path: Optional[Path],
    use_onnx: bool,
    model_cls,
    ort_cls,
    resize_tokens: int = 0,
):
    # 1. Existing ONNX
    if use_onnx and onnx_path and (onnx_path / "model.onnx").exists():
        if not (onnx_path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json missing from {onnx_path}. "
                f"Copy it from your trained checkpoint."
            )
        print(f"[info] Loading ONNX from {onnx_path}")
        return ort_cls.from_pretrained(str(onnx_path)), True

    # 2. Local PyTorch checkpoint
    if _is_valid_local_model(load_path):
        try:
            if use_onnx:
                print(f"[info] Exporting local model to ONNX: {load_path}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ort_cls.from_pretrained(load_path, export=True)
                if onnx_path:
                    onnx_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(onnx_path))
                return model, True
            else:
                model = model_cls.from_pretrained(load_path, local_files_only=True)
                model.eval()
                return model, True
        except Exception as e:
            print(f"[warn] Local model load failed: {e}")

    # 3. Download from Hub
    print(f"[info] Downloading model: {model_name}")
    model = model_cls.from_pretrained(model_name)
    model.eval()

    if resize_tokens > 0 and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(resize_tokens)

    if local_dir:
        save_path = Path(local_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"[info] Model cached to {save_path}")

    if use_onnx and onnx_path:
        onnx_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(onnx_path))
        print(f"[info] ONNX saved to {onnx_path}")
        return ort_cls.from_pretrained(str(onnx_path)), False

    return model, False



# ── NER ──────────────────────────────────────────────────────────────

class BertNERExtractor(IExtractor):
    """
    DistilBERT-based NER. Runs on CPU via ONNX.
    Expects pre-chunked text (chunking happens upstream in UnifiedEntityResolver).
    Returns List[Entity] (pydantic objects).
    """

    def __init__(
        self,
        model_name: str,
        local_dir: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        use_onnx: bool = True,
    ):
        self.use_onnx = use_onnx
        self.onnx_path = Path(onnx_dir) if onnx_dir else None
        load_path = str(Path(local_dir)) if local_dir else model_name

        self.tokenizer = _load_tokenizer_robust(load_path, model_name, local_dir)
        self.model, _ = _load_model_robust(
            load_path, model_name, local_dir, self.onnx_path, use_onnx,
            AutoModelForTokenClassification, ORTModelForTokenClassification,
            resize_tokens=0,
        )
        self.id2label = self.model.config.id2label

    def extract(self, text: str, doc_id:str, chunk_id:str) -> List[Entity]:
        if not text or not text.strip():
            return []

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].numpy()

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).numpy()

        preds = torch.argmax(logits, dim=-1).numpy()
        
        all_entities: List[Entity] = []
        current_ent: Optional[Entity] = None

        for idx, pred_id in enumerate(preds):
            if idx == 0 or idx == len(preds) - 1:
                continue

            label = self.id2label.get(int(pred_id), "O")

            if label.startswith("B-"):
                if current_ent:
                    all_entities.append(current_ent)
                
                ent_label = label.split("-")[1]
                ent_label = EntityLabels.find_best_match(ent_label)
                if ent_label is None:
                    ent_label = EntityLabels.MISC
                    
                start = int(offset_mapping[idx][0])
                end = int(offset_mapping[idx][1])
                conf = round(float(probs[idx][pred_id]), 2)
                
                current_ent = Entity(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text[start:end],
                    canonical_name=text[start:end],
                    status=DisambiguationStatus.UNRESOLVED,
                    label=ent_label,
                    start=start,
                    end=end,
                    confidence=conf,
                    mention_sentence="",
                )

            elif label.startswith("I-") and current_ent is not None:
                new_end = int(offset_mapping[idx][1])
                current_ent.text = text[current_ent.start:new_end]
                current_ent.end = new_end
                current_ent.confidence = round((current_ent.confidence + float(probs[idx][pred_id])) / 2, 2)

            else:
                if current_ent:
                    all_entities.append(current_ent)
                    current_ent = None

        if current_ent:
            all_entities.append(current_ent)

        for e in all_entities:
            e.canonical_name = e.text
            e.mention_sentence = extract_exact_sentence(doc_text=text,
                                                        start_char=e.start,
                                                        end_char=e.end,
                                                        use_nlp=e.label not in NON_LINKABLE_TYPES,
                                                        nlp=None)
        return all_entities


# ── Coref ────────────────────────────────────────────────────────────

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
        nlp = nlp if nlp else settings.SPACY_MODEL_PATH
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
        entities: List[Entity],
    ) -> List[Entity]:
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
                entities.append(Entity(
                    doc_id=antecedent_doc_id,
                    chunk_id=antecedent_chunk_id,
                    text=text[start:end],
                    canonical_name=antecedent,
                    label=antecedent_label,
                    mention_sentence=mention_text,
                    confidence=coref_confidence,
                    status=DisambiguationStatus.RESOLVED,
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
# ── RE ───────────────────────────────────────────────────────────────

class BertRelationExtractor(IExtractor):
    """
    BERT-based sentence-level RE with ONNX.
    Input:  sentence text + List[Entity] or List[ResolvedEntity] or List[Dict].
    Output: List[Relation] dataclass objects.
    """

    def __init__(
        self,
        model_name: str,
        local_dir: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        use_onnx: bool = True,
        max_length: int = 256,
        threshold: float = 0.5,
        label_list: Optional[List[str]] = None,
    ):
        self.max_length = max_length
        self.threshold = threshold
        self.onnx_path = Path(onnx_dir) if onnx_dir else None
        load_path = str(Path(local_dir)) if local_dir else model_name
        # 1. Load tokenizer
        self.tokenizer = _load_tokenizer_robust(load_path, model_name, local_dir)

        # 2. Add entity markers BEFORE any model loading
        special_tokens = {"additional_special_tokens": ["<s>", "</s>", "<o>", "</o>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        new_vocab_size = len(self.tokenizer)

        # 3. Load model with resized embeddings
        self.model, _ = _load_model_robust(
            load_path, model_name, local_dir, self.onnx_path, use_onnx,
            AutoModelForSequenceClassification, ORTModelForSequenceClassification,
            resize_tokens=new_vocab_size if num_added > 0 else 0,
        )

        # 4. Save updated tokenizer (with new tokens) alongside ONNX
        if use_onnx and self.onnx_path:
            self.onnx_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(self.onnx_path))

        # ── Labels ──
        if label_list is not None:
            self.label_list = label_list
        elif hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            self.label_list = [self.id2label[i] for i in range(len(self.id2label))]
        else:
            from shared.data_classes import RelationLabels
            self.label_list = ["no_relation"] + [e.value for e in RelationLabels]

        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self._no_relation_id = self.label2id.get("no_relation", 0)

    @staticmethod
    def _get_canonical_name(e: Union[Entity, Dict]) -> str:
        if isinstance(e, Entity):
            return e.canonical_name
        return e.get("canonical_name", e.get("text", ""))

    @staticmethod
    def _get_label(e: Union[Entity, Dict]) -> str:
        if isinstance(e, Entity):
            return e.label
        return e.get("label", "UNKNOWN")

    def _mark_entities(self, text: str, subj: str, obj: str) -> str:
        text = text.replace(subj, f"<s> {subj} </s>", 1)
        text = text.replace(obj, f"<o> {obj} </o>", 1)
        return text

    def extract(self, text: str, entities: List[Union[Entity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
        if len(entities) < 2:
            return []

        label_map: dict[str, str] = {}
        for e in entities:
            label_map[self._get_canonical_name(e)] = self._get_label(e)

        names = [n for n in label_map.keys() if n]
        relations: List[Relation] = []

        for i, subj_name in enumerate(names):
            for obj_name in names[i + 1:]:
                marked_text = self._mark_entities(text, subj_name, obj_name)

                inputs = self.tokenizer(
                    marked_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )

                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1).numpy()

                pred_id = int(np.argmax(probs))
                confidence = float(probs[pred_id])

                if pred_id == self._no_relation_id:
                    continue
                if confidence < self.threshold:
                    continue

                raw_label = self.id2label[pred_id]

                # ── 2. Fuzzy canonical match for everything else ──
              
                match = RelationLabels.find_best_match(raw_label, threshold=0.75)
                if match is None:
                    # Below threshold → treat as no_relation
                    continue
                pred_label = match.value

                # ── 3. Optional strict gate (comment out if you want to keep unknowns as RELATED_TO) ──
                # if not RelationLabels.is_valid(pred_label):
                #     pred_label = RelationLabels.RELATED_TO.value

                # ── 4. Symmetric normalization ──
                subj_out, obj_out = subj_name, obj_name
                if RelationLabels.is_symmetric(pred_label) and obj_out < subj_out:
                    subj_out, obj_out = obj_out, subj_out

                rel_id = hashlib.sha1(
                    f"{subj_out}|{pred_label}|{obj_out}".encode("utf-8")
                ).hexdigest()

                relations.append(Relation(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    subject=subj_out,
                    subject_label=label_map.get(subj_out, "UNKNOWN"),
                    predicate=pred_label,
                    object=obj_out,
                    object_label=label_map.get(obj_out, "UNKNOWN"),
                    mention_sentence=text,
                    confidence=round(confidence, 3),
                    needs_review=False,
                    evidence=[text],
                    relation_id=rel_id,
                    provisional=False,
                ))

        return relations
    

# ── Download utility ─────────────────────────────────────────────────

def download_and_save_models(
    bert_save_path: str,
    ner_model_name: str = "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    re_model_name: str = "distilbert-base-uncased",
):
    """
    Download models from Hub to local directories.
    NOTE: re_model_name must be a BERT/DistilBERT classification model.
          Do NOT use seq2seq models like Babelscape/rebel-large here.
    """
    save_root = Path(bert_save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    models = {
        "ner": (ner_model_name, AutoModelForTokenClassification),
        "re": (re_model_name, AutoModelForSequenceClassification),
    }

    for subdir, (hub_name, model_cls) in models.items():
        target = save_root / subdir
        if _is_valid_local_model(str(target)):
            print(f"[skip] {subdir} already exists at {target}")
            continue

        print(f"[download] {hub_name} -> {target}")
        tok = AutoTokenizer.from_pretrained(hub_name, use_fast=True, trust_remote_code=True)
        model = model_cls.from_pretrained(hub_name)

        target.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(target)
        model.save_pretrained(target)
        print(f"[saved] {subdir}")

    print("Done.")


# ── Main: end-to-end demo with coreference ─────────────────────────
if __name__ == "__main__":
    from config.settings import settings
    # -----------------------------------------------------------------
    # 1. Initialize extractors
    # -----------------------------------------------------------------
    ner = BertNERExtractor(
        model_name="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"),
        use_onnx=True,
        onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner"),
    )

    coref = BertCorefResolver(nlp=settings.SPACY_MODEL_PATH, use_neural=True)

    re_ext = BertRelationExtractor(
        model_name="distilbert-base-uncased",
        local_dir=str(Path(settings.BERT_MODEL_PATH) / "best_re"),
        use_onnx=False,
        onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "best_onnx_re"),
        threshold=0
    )

    # -----------------------------------------------------------------
    # 2. Text with coreference
    # -----------------------------------------------------------------
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    print("=" * 60)
    print("INPUT TEXT:")
    print(text)
    print("=" * 60)

    # -----------------------------------------------------------------
    # 3. NER
    # -----------------------------------------------------------------
    entities = ner(text)
    print("\n--- NER Output (List[Entity]) ---")
    for e in entities:
        print(f"  {e.text!r:<20} | {e.label:<6} | ({e.start}, {e.end})")

    # -----------------------------------------------------------------
    # 4. Coref
    # -----------------------------------------------------------------
    
    entities = [Entity(text='Apple Inc.', label='ORGANIZATION', start=0, end=10, mention_sentence='Ap...dquartered in Cupertino.', confidence=1.0), Entity(text='Steve Jobs', label='PERSON', start=26, end=36, mention_sentence='Apple I...dquartered in Cupertino.', confidence=1.0), Entity(text='Cupertino', label='LOCATION', start=104, end=113, mention_sentence='Appl...dquartered in Cupertino.', confidence=1.0)]
    resolved_entities = coref.resolve(text, entities)
    print("\n--- Coref Output (List[ResolvedEntity]) ---")
    for r in resolved_entities:
        coref_info = f" -> coref_to={r.coref_to!r}" if r.coref_to else ""
        print(f"  {r.original_text!r:<20} | canonical={r.canonical_name!r:<20} | {r.entity_label:<6} | src={r.source}{coref_info}")

    # -----------------------------------------------------------------
    # 5. RE on resolved entities
    # -----------------------------------------------------------------
    
    
    # resolved_entities = [
    #     ResolvedEntity(original_text='Apple Inc.', canonical_name='Apple Inc.', entity_label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to=None),
    #     ResolvedEntity(original_text='Steve Jobs', canonical_name='Steve Jobs', entity_label='PERSON', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to=None),
    #     ResolvedEntity(original_text='Cupertino', canonical_name='Cupertino', entity_label='LOCATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to=None),
    #     ResolvedEntity(original_text='He', canonical_name='Steve Jobs', entity_label='PERSON', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to='Steve Jobs'),
    #     ResolvedEntity(original_text='the company', canonical_name='Apple Inc.', entity_label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to='Apple Inc.'),
    #     ResolvedEntity(original_text='The firm', canonical_name='Apple Inc.', entity_label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', source='bert_ner', status=DisambiguationStatus.RESOLVED, confidence=1.0, needs_review=False, is_nil=False, coref_to='Apple Inc.'),
    #     ]

    relations = re_ext(text, resolved_entities)
    print("\n--- RE Output (List[Relation]) ---")
    for rel in relations:
        print(f"  ({rel.subject!r}, {rel.predicate!r}, {rel.object!r}) | conf={rel.confidence} | id={rel.relation_id[:8]}...")
        
        
        