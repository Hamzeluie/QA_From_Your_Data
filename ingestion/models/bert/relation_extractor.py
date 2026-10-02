
import hashlib
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from transformers import AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from ingestion.models.bert.utils import _load_tokenizer_robust, _load_model_robust
from ingestion.models.base import IExtractor
from domain import (MentionEntity, Relation, RelationLabels)


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
            from QA_From_Your_Data.storage.data_classes import RelationLabels
            self.label_list = ["no_relation"] + [e.value for e in RelationLabels]

        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self._no_relation_id = self.label2id.get("no_relation", 0)

    @staticmethod
    def _get_canonical_name(e: Union[MentionEntity, Dict]) -> str:
        if isinstance(e, MentionEntity):
            return e.canonical_name
        return e.get("canonical_name", e.get("text", ""))

    @staticmethod
    def _get_label(e: Union[MentionEntity, Dict]) -> str:
        if isinstance(e, MentionEntity):
            return e.label
        return e.get("label", "UNKNOWN")

    def _mark_entities(self, text: str, subj: str, obj: str) -> str:
        text = text.replace(subj, f"<s> {subj} </s>", 1)
        text = text.replace(obj, f"<o> {obj} </o>", 1)
        return text

    def extract(self, text: str, entities: List[Union[MentionEntity, Dict]], doc_id:str, chunk_id:str) -> List[Relation]:
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
                    
                    evidence=[text],
                    relation_id=rel_id,
                    provisional=False,
                ))

        return relations


# ── Main: end-to-end demo with coreference ─────────────────────────
if __name__ == "__main__":
    from config.settings import settings
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    re_ext = BertRelationExtractor(
    model_name="distilbert-base-uncased",
    local_dir=str(Path(settings.BERT_MODEL_PATH) / "best_re"),
    use_onnx=False,
    onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "best_onnx_re"),
    threshold=0)
    
    resolved_entities = [
        MentionEntity(text='Apple Inc.', canonical_name='Apple Inc.', label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.',  status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to=None),
        MentionEntity(text='Steve Jobs', canonical_name='Steve Jobs', label='PERSON', mention_sentence='Apple Inc. is a company headquartered in Cupertino.',  status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to=None),
        MentionEntity(text='Cupertino', canonical_name='Cupertino', label='LOCATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.',  status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to=None),
        MentionEntity(text='He', canonical_name='Steve Jobs', label='PERSON', mention_sentence='Apple Inc. is a company headquartered in Cupertino.',  status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to='Steve Jobs'),
        MentionEntity(text='the company', canonical_name='Apple Inc.', label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.',  status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to='Apple Inc.'),
        MentionEntity(text='The firm', canonical_name='Apple Inc.', label='ORGANIZATION', mention_sentence='Apple Inc. is a company headquartered in Cupertino.', status=DisambiguationStatus.RESOLVED, confidence=1.0, coref_to='Apple Inc.'),
        ]

    relations = re_ext(text, resolved_entities)
    print("\n--- RE Output (List[Relation]) ---")
    for rel in relations:
        print(f"  ({rel.subject!r}, {rel.predicate!r}, {rel.object!r}) | conf={rel.confidence} | id={rel.relation_id[:8]}...")
