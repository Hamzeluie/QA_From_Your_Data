"""
NER Evaluation Metrics Module
==============================
Supports both Standard (CoNLL-style) and GEPA evaluation schemes.

GEPA = Global Entity-level Partial-match Aggregation
- Global:   Aggregated across all documents (micro) or per-type (macro)
- Entity:   Evaluates at entity span level (not token-level BIO)
- Partial:  Supports exact, partial, and overlap matching criteria
- Aggregation: Micro / Macro / Weighted averaging

Ground Truth Format (from your dataset):
    {"text": str, "label": str, "start": int, "end": int, "sents_id": int}

Prediction Format (from your DSPy NER):
    {"text": str, "type": str, "start_char": int, "end_char": int, "confidence": float}
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

from QA_From_Your_Data.utils.utils import Entity, MatchResult



# ============================================================================
# MATCHING ENGINE
# ============================================================================

def match_entities(
    preds: List[Entity],
    gts: List[Entity],
    criteria: str = "exact"
) -> Tuple[List[MatchResult], List[Entity], List[Entity]]:
    """
    Match predictions to ground truth based on criteria.

    Args:
        preds: Predicted entities
        gts: Ground truth entities
        criteria: "exact" | "partial" | "type_partial" | "boundary" | "overlap"

    Returns:
        (matched_pairs, unmatched_preds, unmatched_gts)
    """
    matched_preds = set()
    matched_gts = set()
    results = []

    # Sort by confidence descending (prioritize high-confidence predictions)
    sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1].confidence)

    for p_idx, pred in sorted_preds:
        if p_idx in matched_preds:
            continue

        best_match = None
        best_iou = 0.0
        best_g_idx = -1

        for g_idx, gt in enumerate(gts):
            if g_idx in matched_gts:
                continue

            iou = pred.span_overlap(gt)

            if criteria == "exact":
                valid = pred.exact_match(gt) and pred.label == gt.label
            elif criteria == "partial":
                valid = pred.partial_match(gt, 0.5) and pred.label == gt.label
            elif criteria == "type_partial":
                valid = pred.any_overlap(gt) and pred.label == gt.label
            elif criteria == "boundary":
                valid = pred.exact_match(gt)
            elif criteria == "overlap":
                valid = pred.any_overlap(gt)
            else:
                valid = False

            if valid and iou > best_iou:
                best_iou = iou
                best_match = gt
                best_g_idx = g_idx

        if best_match:
            matched_preds.add(p_idx)
            matched_gts.add(best_g_idx)
            results.append(MatchResult(pred, best_match, criteria, best_iou))
        else:
            results.append(MatchResult(pred, None, "none", 0.0))

    unmatched_preds = [preds[i] for i in range(len(preds)) if i not in matched_preds]
    unmatched_gts = [gts[i] for i in range(len(gts)) if i not in matched_gts]

    return results, unmatched_preds, unmatched_gts


def _compute_prf(tp: int, fp: int, fn: int) -> Dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ============================================================================
# STANDARD (NORMAL) EVALUATION — CoNLL-style Exact Match
# ============================================================================

class StandardNEREvaluator:
    """
    Standard NER evaluation: strict exact-match boundary + exact type.
    This is the CoNLL-2003 / SemEval standard.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def add_document(self, preds: List[Dict], gts: List[Dict]):
        pred_ents = [Entity.from_pred(p) for p in preds]
        gt_ents = [Entity.from_gt(g) for g in gts]

        matched, unmatched_preds, unmatched_gts = match_entities(
            pred_ents, gt_ents, criteria="exact"
        )

        for m in matched:
            if m.matched_gt:
                self.tp += 1
                self.per_type[m.matched_gt.label]["tp"] += 1

        self.fp += len(unmatched_preds)
        for p in unmatched_preds:
            self.per_type[p.label]["fp"] += 1

        self.fn += len(unmatched_gts)
        for g in unmatched_gts:
            self.per_type[g.label]["fn"] += 1

    def compute(self) -> Dict:
        micro = _compute_prf(self.tp, self.fp, self.fn)
        micro["tp"] = self.tp
        micro["fp"] = self.fp
        micro["fn"] = self.fn

        type_metrics = {}
        for label, counts in self.per_type.items():
            type_metrics[label] = _compute_prf(
                counts["tp"], counts["fp"], counts["fn"]
            )

        return {
            "scheme": "standard_exact",
            "micro": micro,
            "per_type": type_metrics
        }


# ============================================================================
# GEPA EVALUATION — Global Entity-level Partial-match Aggregation
# ============================================================================

class GEPANEREvaluator:
    """
    GEPA: Global Entity-level Partial-match Aggregation

    Evaluates NER across multiple matching criteria:
      - Exact:      exact boundary + exact type  (strict)
      - Partial:    >=50% IoU overlap + exact type
      - Type:       any overlap + exact type
      - Boundary:   exact boundary (type can differ)
      - Overlap:    any overlap (type can differ)

    Aggregation modes:
      - Micro:      aggregate counts globally then compute P/R/F1
      - Macro:      compute P/R/F1 per document then average
      - Weighted:   macro weighted by document entity count
    """

    CRITERIA = ["exact", "partial", "type_partial", "boundary", "overlap"]

    def __init__(self, aggregation: str = "micro"):
        self.aggregation = aggregation
        self.reset()

    def reset(self):
        self.doc_results = []
        self.global_counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in self.CRITERIA}
        self.per_type_counts = {c: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) 
                                for c in self.CRITERIA}

    def add_document(self, preds: List[Dict], gts: List[Dict]):
        pred_ents = [Entity.from_pred(p) for p in preds]
        gt_ents = [Entity.from_gt(g) for g in gts]

        doc_result = {"n_gts": len(gt_ents), "criteria": {}}

        for criteria in self.CRITERIA:
            matched, unmatched_preds, unmatched_gts = match_entities(
                pred_ents, gt_ents, criteria=criteria
            )

            tp = sum(1 for m in matched if m.matched_gt is not None)
            fp = len(unmatched_preds)
            fn = len(unmatched_gts)

            doc_result["criteria"][criteria] = {"tp": tp, "fp": fp, "fn": fn}

            self.global_counts[criteria]["tp"] += tp
            self.global_counts[criteria]["fp"] += fp
            self.global_counts[criteria]["fn"] += fn

            for m in matched:
                if m.matched_gt:
                    label = m.matched_gt.label
                    self.per_type_counts[criteria][label]["tp"] += 1
            for p in unmatched_preds:
                self.per_type_counts[criteria][p.label]["fp"] += 1
            for g in unmatched_gts:
                self.per_type_counts[criteria][g.label]["fn"] += 1

        self.doc_results.append(doc_result)

    def compute(self) -> Dict:
        results = {"scheme": "gepa", "aggregation": self.aggregation}

        # Micro aggregation
        results["micro"] = {}
        for criteria in self.CRITERIA:
            c = self.global_counts[criteria]
            results["micro"][criteria] = {
                **_compute_prf(c["tp"], c["fp"], c["fn"]),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]
            }

        # Macro / Weighted aggregation
        if self.doc_results:
            macro_results = {c: {"precision": [], "recall": [], "f1": []} for c in self.CRITERIA}

            for doc in self.doc_results:
                weight = doc["n_gts"] if self.aggregation == "weighted" else 1
                for criteria in self.CRITERIA:
                    c = doc["criteria"][criteria]
                    prf = _compute_prf(c["tp"], c["fp"], c["fn"])
                    for _ in range(max(weight, 1)):
                        macro_results[criteria]["precision"].append(prf["precision"])
                        macro_results[criteria]["recall"].append(prf["recall"])
                        macro_results[criteria]["f1"].append(prf["f1"])

            results[self.aggregation] = {}
            for criteria in self.CRITERIA:
                n = len(macro_results[criteria]["f1"])
                results[self.aggregation][criteria] = {
                    "precision": round(sum(macro_results[criteria]["precision"]) / n, 4),
                    "recall": round(sum(macro_results[criteria]["recall"]) / n, 4),
                    "f1": round(sum(macro_results[criteria]["f1"]) / n, 4)
                }

        # Per-type metrics (micro)
        results["per_type"] = {}
        for criteria in self.CRITERIA:
            results["per_type"][criteria] = {}
            for label, counts in self.per_type_counts[criteria].items():
                results["per_type"][criteria][label] = _compute_prf(
                    counts["tp"], counts["fp"], counts["fn"]
                )

        return results


# ============================================================================
# DSPY METRICS (for optimization)
# ============================================================================

def ner_f1_metric(example, pred, trace=None) -> float:
    """
    DSPy-compatible metric. Returns F1 score for a single example.
    Use with dspy.BootstrapFewShot or other teleprompters.

    Expects:
        example.entities = ground truth list of dicts
        pred.entities    = predicted list of dicts
    """
    gt_ents = [Entity.from_gt(e) for e in example.entities]
    pred_ents = [Entity.from_pred(e) for e in pred.entities]

    matched, unmatched_preds, unmatched_gts = match_entities(
        pred_ents, gt_ents, criteria="exact"
    )

    tp = sum(1 for m in matched if m.matched_gt is not None)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def ner_gepa_metric(example, pred, trace=None, criteria: str = "partial") -> float:
    """
    DSPy-compatible GEPA metric. Uses partial matching for more stable optimization.

    Args:
        criteria: "exact" | "partial" | "type_partial" — which GEPA criterion to use
    """
    gt_ents = [Entity.from_gt(e) for e in example.entities]
    pred_ents = [Entity.from_pred(e) for e in pred.entities]

    matched, unmatched_preds, unmatched_gts = match_entities(
        pred_ents, gt_ents, criteria=criteria
    )

    tp = sum(1 for m in matched if m.matched_gt is not None)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


# ============================================================================
# FULL PIPELINE
# ============================================================================

def evaluate_dataset(data: List[Dict], ner_module, chunker=None) -> Dict:
    """
    Run full evaluation on dataset.

    Args:
        data: List of documents with 'document', 'label_doc'
        ner_module: Your DSPy NER module (must have .forward() or be callable)
        chunker: Optional semantic chunking function

    Returns:
        Combined evaluation report
    """
    standard_eval = StandardNEREvaluator()
    gepa_eval = GEPANEREvaluator(aggregation="micro")

    for row in data:
        doc_text = row["document"]
        gt_entities = row["label_doc"]

        if chunker:
            all_preds = []
            for chunk in chunker(doc_text):
                result = ner_module(chunk)
                all_preds.extend(result.get("entities", []))
        else:
            result = ner_module(doc_text)
            all_preds = result.get("entities", [])

        standard_eval.add_document(all_preds, gt_entities)
        gepa_eval.add_document(all_preds, gt_entities)

    return {
        "standard": standard_eval.compute(),
        "gepa": gepa_eval.compute()
    }
    
