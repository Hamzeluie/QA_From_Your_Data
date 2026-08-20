"""
Integration Demo: NER Evaluation with your existing DSPy pipeline
=================================================================

This shows how to plug the evaluation metrics into your existing code.
"""

import json
from QA_From_Your_Data.ingestion.llm.llm_evaluation import (
    StandardNEREvaluator,
    GEPANEREvaluator,
    ner_f1_metric,
    ner_gepa_metric,
    evaluate_dataset
)


# ---------------------------------------------------------------------------
# OPTION 1: Manual evaluation (after running your NER)
# ---------------------------------------------------------------------------

def manual_evaluate(data, ner_module, chunker=None):
    """Run NER and evaluate manually."""
    # Run evaluation
    report = evaluate_dataset(data, ner_module, chunker=chunker)

    # Print results
    print("=" * 60)
    print("STANDARD EVALUATION (Exact Match)")
    print("=" * 60)
    std = report["standard"]["micro"]
    print(f"  Precision: {std['precision']}")
    print(f"  Recall:    {std['recall']}")
    print(f"  F1:        {std['f1']}")
    print(f"  TP/FP/FN:  {std['tp']}/{std['fp']}/{std['fn']}")
    print()
    print("Per-Type:")
    for label, metrics in report["standard"]["per_type"].items():
        print(f"  {label:6s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    print()
    print("=" * 60)
    print("GEPA EVALUATION (Multi-Criteria)")
    print("=" * 60)
    gepa = report["gepa"]["micro"]
    for criteria, metrics in gepa.items():
        print(f"  {criteria:15s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    return report


# ---------------------------------------------------------------------------
# OPTION 2: DSPy Optimization (using metrics as objective)
# ---------------------------------------------------------------------------

def optimize_with_dspy(train_data, val_data, ner_module_class):
    """
    Use evaluation metric to optimize your DSPy NER module.

    This requires wrapping your data as dspy.Example objects.
    """
    import dspy

    # Convert your data to dspy.Examples
    # Each example needs: tokens (input text) and entities (ground truth)
    train_examples = []
    for row in train_data:
        ex = dspy.Example(
            tokens=row["document"],
            entities=row["label_doc"]  # ground truth entity list
        ).with_inputs("tokens")
        train_examples.append(ex)

    # --- Strategy A: Strict Exact-Match F1 ---
    # Best for: Final reporting, benchmarking against CoNLL/SemEval papers
    # Risk: Very sparse signal for DSPy optimizer (hard to improve)
    optimizer_strict = dspy.BootstrapFewShot(
        metric=ner_f1_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8
    )

    # --- Strategy B: GEPA Partial-Match F1 ---
    # Best for: DSPy optimization (denser reward signal)
    # The optimizer gets partial credit for almost-correct predictions
    optimizer_partial = dspy.BootstrapFewShot(
        metric=lambda ex, pred, trace=None: ner_gepa_metric(ex, pred, trace, criteria="partial"),
        max_bootstrapped_demos=4,
        max_labeled_demos=8
    )

    # Compile (optimize) your module
    print("Optimizing with STRICT metric...")
    optimized_strict = optimizer_strict.compile(ner_module_class(), train=train_examples)

    print("Optimizing with GEPA-PARTIAL metric...")
    optimized_partial = optimizer_partial.compile(ner_module_class(), train=train_examples)

    # Evaluate both on validation set
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for name, model in [("Strict-Optimized", optimized_strict), 
                        ("GEPA-Optimized", optimized_partial)]:
        report = evaluate_dataset(val_data, model)
        f1 = report["standard"]["micro"]["f1"]
        print(f"  {name:20s}  Standard F1 = {f1:.4f}")

    return optimized_strict, optimized_partial


# ---------------------------------------------------------------------------
# OPTION 3: Per-chunk evaluation (if using semantic chunking)
# ---------------------------------------------------------------------------

def evaluate_per_chunk(data, ner_module, chunker):
    """
    Evaluate NER when you chunk documents semantically.
    Important: predictions must be mapped back to document-level offsets!
    """
    standard_eval = StandardNEREvaluator()
    gepa_eval = GEPANEREvaluator(aggregation="micro")

    for row in data:
        doc_text = row["document"]
        gt_entities = row["label_doc"]
        all_preds = []

        char_offset = 0
        for chunk in chunker(doc_text):
            result = ner_module(chunk)

            # CRITICAL: Adjust offsets to document-level
            for ent in result.get("entities", []):
                ent.start += char_offset
                ent.end += char_offset
                all_preds.append(ent)

            char_offset += len(chunk) + 1  # +1 for space separator

        standard_eval.add_document(all_preds, gt_entities)
        gepa_eval.add_document(all_preds, gt_entities)

    return {
        "standard": standard_eval.compute(),
        "gepa": gepa_eval.compute()
    }


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace with your actual module
    from QA_From_Your_Data.ingestion.llm.llm_extractors import NERExtraction, NERWithConfidence, semantic_sentence_chunk
    ner = NERExtraction(use_cot=True)

    data_path = "/home/mehdi/Documents/projects/knowledge_graph_examples/datasets/QA_your_data/ner_dataset.jsonl"
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                    
    train_data, val_data = data[:100], data[100:110]
    ner = NERExtraction(use_cot=True)
    # ner_voter = NERWithConfidence(n_passes=3)
    # manual_evaluate(val_data, ner_module=ner,chunker=semantic_sentence_chunk)
    # optimize_with_dspy(train_data, val_data, ner)
    evaluate_per_chunk(val_data, ner, semantic_sentence_chunk)
    
    