#!/usr/bin/env python3
# train_docred.py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from pathlib import Path
from typing import Dict, List


# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"  # or "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
OUTPUT_DIR = "./docred_re_model"


# ── Dataset ─────────────────────────────────────────────────────────

class DocREDSentenceDataset(Dataset):
    def __init__(self, path: str, tokenizer, label2id: Dict[str, int], max_length: int = 256):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        text = ex["text"]
        subj = ex["subject"]
        obj = ex["object"]
        relation = ex["relation"]

        # Mark entities
        marked = self._mark_entities(text, subj, obj)

        encoding = self.tokenizer(
            marked,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_id = self.label2id.get(relation, self.label2id["no_relation"])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

    def _mark_entities(self, text: str, subj: str, obj: str) -> str:
        # Replace first occurrence only
        text = text.replace(subj, f"<s> {subj} </s>", 1)
        text = text.replace(obj, f"<o> {obj} </o>", 1)
        return text


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    # Build label list from data
    train_samples = []
    with open("data/docred/train.jsonl", "r") as f:
        for line in f:
            train_samples.append(json.loads(line))

    all_relations = sorted({ex["relation"] for ex in train_samples})
    if "no_relation" not in all_relations:
        all_relations = ["no_relation"] + all_relations

    label2id = {r: i for i, r in enumerate(all_relations)}
    id2label = {i: r for r, i in label2id.items()}

    print(f"Relations: {len(all_relations)}")
    print(f"Train samples: {len(train_samples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Add special tokens for entity marking
    tokenizer.add_special_tokens({"additional_special_tokens": ["<s>", "</s>", "<o>", "</o>"]})

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_relations),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = DocREDSentenceDataset("data/docred/train.jsonl", tokenizer, label2id, MAX_LENGTH)
    dev_dataset = DocREDSentenceDataset("data/docred/dev.jsonl", tokenizer, label2id, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_steps=int(0.1 * EPOCHS * (len(train_dataset) // BATCH_SIZE)),  # ← FIXED (was warmup_ratio=0.1)
        load_best_model_at_end=True,
        metric_for_best_model="f1_ignored",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")

    # Save label mapping
    import json
    with open(f"{OUTPUT_DIR}/best/label_map.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    print(f"Model saved to {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()