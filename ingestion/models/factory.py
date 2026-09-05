import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import inspect
from ingestion.models.base import IExtractor
from config.settings import settings


def get_relation_extractor(backend: str = "bert", **kwargs) -> IExtractor:
    if backend == "llm":
        from ingestion.models.llm.llm_extractors import DSPyRelationExtractor
        target_cls = DSPyRelationExtractor
    elif backend == "bert":
        from ingestion.models.bert.bert_extractors import BertRelationExtractor
        target_cls = BertRelationExtractor
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ── 1. Inspect the target class's __init__ signature ──
    sig = inspect.signature(target_cls.__init__)
    
    # Get all valid parameter names (excluding 'self')
    valid_params = {
        name for name, param in sig.parameters.items() 
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    
    # Check if the class natively accepts **kwargs
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )

    # ── 2. Filter kwargs dynamically ──
    if accepts_var_kwargs:
        # If the class has **kwargs, pass everything through
        filtered_kwargs = kwargs
    else:
        # Otherwise, only pass keys that match the class signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    # ── 3. Instantiate ──
    return target_cls(**filtered_kwargs)

def get_ner_extractor(backend: str = "bert", **kwargs) -> IExtractor:
    if backend == "llm":
        from ingestion.models.llm.llm_extractors import DSPyNERExtractor
        target_cls = DSPyNERExtractor
    elif backend == "bert":
        from ingestion.models.bert.bert_extractors import BertNERExtractor
        target_cls = BertNERExtractor
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ── 1. Inspect the target class's __init__ signature ──
    sig = inspect.signature(target_cls.__init__)
    
    # Get all valid parameter names (excluding 'self')
    valid_params = {
        name for name, param in sig.parameters.items() 
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    
    # Check if the class natively accepts **kwargs
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )

    # ── 2. Filter kwargs dynamically ──
    if accepts_var_kwargs:
        # If the class has **kwargs, pass everything through
        filtered_kwargs = kwargs
    else:
        # Otherwise, only pass keys that match the class signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    # ── 3. Instantiate ──
    return target_cls(**filtered_kwargs)

def get_coref_resolver(backend: str = "bert", **kwargs) -> IExtractor:
    if backend == "llm":
        from ingestion.models.llm.llm_extractors import DSPyCorefResolver
        target_cls = DSPyCorefResolver
    elif backend == "bert":
        from ingestion.models.bert.bert_extractors import BertCorefResolver
        target_cls = BertCorefResolver
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ── 1. Inspect the target class's __init__ signature ──
    sig = inspect.signature(target_cls.__init__)
    
    # Get all valid parameter names (excluding 'self')
    valid_params = {
        name for name, param in sig.parameters.items() 
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    
    # Check if the class natively accepts **kwargs
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )

    # ── 2. Filter kwargs dynamically ──
    if accepts_var_kwargs:
        # If the class has **kwargs, pass everything through
        filtered_kwargs = kwargs
    else:
        # Otherwise, only pass keys that match the class signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    # ── 3. Instantiate ──
    return target_cls(**filtered_kwargs)
    
    
if __name__ == "__main__":
    from shared.data_classes import (Entity, Relation, EntityLabels, DisambiguationStatus)
    # ── CONFIG: switch here ──
    BACKEND = "llm"  #  "bert" or "llm"
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    # ── Run NER ──    
    """
    ner = get_ner_extractor(BACKEND, 
                            model_name="Davlan/distilbert-base-multilingual-cased-ner-hrl",
                            local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"),
                            use_onnx=True,
                            onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner")
                            )
    entities = ner.extract(text, doc_id="user", chunk_id="1")
    print(entities)
    """
    # ── Run COREF ──
    """
    entities = [Entity(text='Apple Inc.', label=EntityLabels.ORG, start=0, end=10, mention_sentence='<Apple Inc.> was founded by Steve Jobs.', confidence=0.99, canonical_name='Apple Inc.', status=DisambiguationStatus.UNRESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                Entity(text='Steve Jobs', label=EntityLabels.PER, start=26, end=36, mention_sentence='Apple Inc. was founded by <Steve Jobs>.', confidence=1.0, canonical_name='Steve Jobs', status=DisambiguationStatus.UNRESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
                Entity(text='Cupertino', label=EntityLabels.LOC, start=104, end=113, mention_sentence='The firm is headquartered in <Cupertino>.', confidence=1.0, canonical_name='Cupertino', status=DisambiguationStatus.UNRESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None)
                ]
    
    coref = get_coref_resolver(BACKEND, nlp=settings.SPACY_MODEL_PATH)
    resolved = coref.extract(text, entities)
    print(resolved)
    """
    
    # ── Run RELATION EXTRACTION ──
    """
    resolved_entities = [
        Entity(text='Apple Inc.', label=EntityLabels.ORG, start=0, end=10, mention_sentence='<Apple Inc.> was founded by Steve Jobs.', confidence=0.99, canonical_name='Apple Inc.', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
        Entity(text='Steve Jobs', label=EntityLabels.PER, start=26, end=36, mention_sentence='Apple Inc. was founded by <Steve Jobs>.', confidence=1.0, canonical_name='Steve Jobs', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
        Entity(text='Cupertino', label=EntityLabels.LOC, start=104, end=113, mention_sentence='The firm is headquartered in <Cupertino>.', confidence=1.0, canonical_name='Cupertino', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to=None),
        Entity(text='He', label=EntityLabels.PER, start=38, end=40, mention_sentence='<He> served as the CEO of the company.', confidence=0.93, canonical_name='Steve Jobs', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Steve Jobs'),
        Entity(text='the company', label=EntityLabels.ORG, start=62, end=73, mention_sentence='He served as the CEO of <the company>.', confidence=0.94, canonical_name='Apple Inc.', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Apple Inc.'),
        Entity(text='The firm', label=EntityLabels.ORG, start=75, end=83, mention_sentence='<The firm> is headquartered in Cupertino.', confidence=0.94, canonical_name='Apple Inc.', status=DisambiguationStatus.RESOLVED, doc_id='user', chunk_id='1', kg_candidates=[], summary=None, context_clues=[], needs_review=False, is_nil=False, coref_to='Apple Inc.')
        ]
    relation_extractor = get_relation_extractor(BACKEND, 
                                                model_name="distilbert-base-uncased",
                                                local_dir=str(Path(settings.BERT_MODEL_PATH) / "best_re"),
                                                use_onnx=False,
                                                onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "best_onnx_re"),
                                                threshold=0)
    relations = relation_extractor(text, resolved_entities, "user", "1")
    print(relations)
    """
