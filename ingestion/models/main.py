import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from ingestion.models.factory import get_ner_extractor, get_relation_extractor, get_coref_resolver
from config.settings import settings

# ── CONFIG: switch here ──
BACKEND = "bert"  # or "llm"

# ── Initialize ──
ner = get_ner_extractor(BACKEND, 
                        model_name=settings.BERT_NER_MODEL_NAME, 
                        onnx_dir=settings.BERT_ONNX_MODEL_PATH)


# ── Run ──
text = "Apple Inc. was founded by Steve Jobs in Cupertino. he passed away in 2011."
entities = ner(text)
print("sdf")

[Entity(text='Apple Inc.', label='ORGANIZATION', start=0, end=10, mention_sentence='Ap... he passed away in 2011.', confidence=1.0), Entity(text='Steve Jobs', label='PERSON', start=26, end=36, mention_sentence='Apple I... he passed away in 2011.', confidence=1.0), Entity(text='Cupertino', label='LOCATION', start=40, end=49, mention_sentence='Apple ... he passed away in 2011.', confidence=1.0)]
exit()
# entities -> [{"text":"Apple Inc.","label":"ORG","start":0,"end":10,...}, ...]


re_ext = get_relation_extractor(BACKEND, 
                                model_name=settings.BERT_RE_MODEL_NAME, 
                                onnx_dir=settings.BERT_ONNX_MODEL_PATH, 
                                threshold=0.6)
relations = re_ext(text, entities)

coref = get_coref_resolver(BACKEND)
df = pd.DataFrame([{"text": "Apple Inc.", "start": 0, "end": 10, "canonical_name": "Apple Inc."},
                   {"text": "Steve Jobs", "start": 27, "end": 37, "canonical_name": "Steve Jobs"},
                   {"text": "Cupertino", "start": 41, "end": 50, "canonical_name": "Cupertino"},
                   {"text": "he", "start": 52, "end": 54, "canonical_name": "Steve Jobs"},
                   {"text": "2011", "start": 67, "end": 71, "canonical_name": "2011"}])  # convert to your resolver format
resolved = coref(text, df)

# relations -> [{"subject":"Apple Inc.","predicate":"founded","object":"Steve Jobs",...}]