import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm

from train_docred import DocREDSentenceDataset  # reuse dataset class


class REDEvaluator:
    """
    Evaluates Relation Extraction with standard metrics:
    - Micro F1 (includes no_relation)
    - Ignored F1 (excludes no_relation)
    - Per-relation F1
    """

    def __init__(self, model_dir: str, batch_size: int = 32):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "label_map.json", "r") as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        self.label2id = maps["label2id"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def evaluate(self, test_path: str):
        dataset = DocREDSentenceDataset(test_path, self.tokenizer, self.label2id)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Micro F1 (all classes)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )

        # Ignored F1 (exclude no_relation)
        mask = all_labels != self.label2id["no_relation"]
        if mask.sum() > 0:
            p_ign, r_ign, f1_ign, _ = precision_recall_fscore_support(
                all_labels[mask], all_preds[mask], average="micro", zero_division=0
            )
        else:
            p_ign = r_ign = f1_ign = 0.0

        print("\n" + "="*50)
        print("DocRED Relation Extraction Results")
        print("="*50)
        print(f"Micro F1 (with NA):    {f1_micro:.4f}  (P={p_micro:.4f}, R={r_micro:.4f})")
        print(f"Ignored F1 (no NA):    {f1_ign:.4f}  (P={p_ign:.4f}, R={r_ign:.4f})")
        print("="*50)

        # Per-relation breakdown
        print("\nPer-relation metrics:")
        print(classification_report(
            all_labels, all_preds,
            target_names=[self.id2label[i] for i in range(len(self.id2label))],
            digits=3,
            zero_division=0
        ))

        return {
            "micro_f1": f1_micro,
            "micro_precision": p_micro,
            "micro_recall": r_micro,
            "ignored_f1": f1_ign,
        }


if __name__ == "__main__":
    import sys
    evaluator = REDEvaluator("./docred_re_model/best")
    evaluator.evaluate("data/docred/dev.jsonl")