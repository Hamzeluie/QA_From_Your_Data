from pathlib import Path
import torch
from typing import List, Optional
from transformers import AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from ingestion.models.base import IExtractor
from domain import (MentionEntity, EntityLabels)
from shared.utils import extract_exact_sentence, NON_LINKABLE_TYPES
from ingestion.models.bert.utils import _load_model_robust, _load_tokenizer_robust


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

    def extract(self, text: str, doc_id:str, chunk_id:str) -> List[MentionEntity]:
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
        
        all_entities: List[MentionEntity] = []
        current_ent: Optional[MentionEntity] = None

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
                
                current_ent = MentionEntity(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text[start:end],
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


# ── Main: end-to-end demo with coreference ─────────────────────────
if __name__ == "__main__":
    from config.settings import settings
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    ner = BertNERExtractor(
    model_name="Davlan/distilbert-base-multilingual-cased-ner-hrl",
    local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"),
    use_onnx=True,
    onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner"),)
    entities = ner(text)
    print("\n--- NER Output (List[Entity]) ---")
    for e in entities:
        print(f"  {e.text!r:<20} | {e.label:<6} | ({e.start}, {e.end})")
    
    