import dspy
from QA_From_Your_Data.ingestion.models.llm.llm_extractors import NERExtraction
from QA_From_Your_Data.ingestion.models.llm.llm_evaluation import ner_gepa_metric


def optimize_ner(train_data, save_dir:str="./optimized_ner.json"):
    # Wrap your data as dspy.Examples
    train_examples = []
    for row in train_data:
        ex = dspy.Example(
            tokens=row["document"],
            entities=row["label_doc"]
        ).with_inputs("tokens")
        train_examples.append(ex)

    # Optimize with GEPA partial (denser signal for DSPy)
    optimizer = dspy.BootstrapFewShot(
        metric=lambda ex, pred, trace=None: ner_gepa_metric(ex, pred, trace, criteria="partial"),
        max_bootstrapped_demos=4
    )
    optimized_ner = optimizer.compile(NERExtraction(), train=train_examples)
    optimized_ner.save(save_dir)
    print("Optimized NER model saved to:", save_dir)
    return optimized_ner
    