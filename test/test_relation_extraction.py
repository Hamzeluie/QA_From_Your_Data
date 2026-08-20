import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.settings import settings
import spacy
import json
from sentence_transformers import SentenceTransformer
from shared.utils import semantic_sentence_chunk
from ingestion.entity_resolver import NERExtractor, CorefResolver
from ingestion.llm.llm_extractors import RelationResolver


# 1. Initialize models
print("Loading models...")
embedding = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)

try:
    nlp = spacy.load(settings.SPACY_MODEL_PATH)
except OSError:
    print("Warning: en_core_web_sm not found. CorefResolver will fallback to regex splitting.")
    nlp = None

ner = NERExtractor(use_cot=True, nlp=nlp)
coref_resolver = CorefResolver(nlp=nlp, mode="hybrid")
relation_resolver = RelationResolver(use_cot=True, include_literals=True)

# 2. Load data (falls back to a dummy document if the path doesn't exist)
dataset_path = settings.DATASET_PATH
data = []
try:
    with open(dataset_path, "r") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
except FileNotFoundError:
    print(f"Dataset not found at {dataset_path}. Using dummy data for demonstration.")
    data = [{"document": "Benyamin Bahadori is a retired Iranian pop singer from Tehran. He released his first album '85' in 2006. It was a massive success."}]

if not data:
    print("No data to process.")
    raise SystemExit(0)

row = data[0]
doc_text = row["document"]
doc_id = "doc_001"

print(f"\n{'=' * 20} Processing Document: {doc_id} {'=' * 20}")
print(f"Text snippet: {doc_text[:100]}...\n")

# 3. NER extraction per chunk
all_entities = []
chunks = list(semantic_sentence_chunk(doc_text, embedding, nlp, True, threshold=0.1))

print("=== Step 1: NER Extraction ===")
for chunk_idx, chunk_text in enumerate(chunks):
    result = ner(chunk_text)

    print(f"\n--- Chunk {chunk_idx} ---")
    for ent in result["entities"]:
        print(f"  [{ent.label}] '{ent.text}' (conf={ent.confidence}) @ [{ent.start}:{ent.end}]")

        all_entities.append({
            "doc_id": doc_id,
            "chunk_id": chunk_idx,
            "text": ent.text,
            "label": ent.label,
            "start": ent.start,
            "end": ent.end,
            "confidence": ent.confidence,
            "mention_sentence": chunk_text
        })

if not all_entities:
    print("No entities found. Exiting.")
    raise SystemExit(0)

entities_df = pd.DataFrame(all_entities)
print(f"\nTotal raw entities extracted: {len(entities_df)}")

# 4. Coreference resolution
print("\n=== Step 2: Coreference Resolution ===")
resolved_entities_df = coref_resolver.resolve_document(doc_text, entities_df)

coref_results = resolved_entities_df[resolved_entities_df["coref_to"].notna()]
print(f"Found {len(coref_results)} coreferences.")
for _, res in coref_results.head(5).iterrows():
    print(f"  Pronoun: '{res['text']}' -> Antecedent: '{res['coref_to']}' (Method: {res['coref_method']}, Conf: {res['coref_confidence']})")

# 5. Mock entity resolution (UnifiedEntityResolver.process_document() would
#    normally produce this clean_df — this stands in for that step so
#    the RE stage can be exercised standalone)
print("\n=== Step 3: Mocking Entity Resolution (Canonical Names) ===")
clean_df = resolved_entities_df.copy()

clean_df["canonical_name"] = clean_df.apply(
    lambda r: r["coref_to"] if pd.notna(r["coref_to"]) else r["text"], axis=1
)
clean_df["original_text"] = clean_df["text"]
clean_df["entity_label"] = clean_df["label"]
clean_df["status"] = "RESOLVED"

# Coreferenced pronoun rows are now merged into their antecedents
clean_df = clean_df[clean_df["coref_to"].isna()].copy()
print(f"Resolved entities ready for RE: {len(clean_df)}")

# 6. Relation extraction
print("\n=== Step 4: Relation Extraction ===")
relations_df = relation_resolver.extract(clean_df)

print(f"\n{'=' * 20} Extracted {len(relations_df)} Relations {'=' * 20}")
if not relations_df.empty:
    for _, rel in relations_df.iterrows():
        print(f"\n  Subject: {rel['subject']} ({rel['subject_label']})")
        print(f"  Predicate: {rel['predicate']}")
        print(f"  Object: {rel['object']} ({rel['object_label']})")
        print(f"  Confidence: {rel['confidence']}")
        print(f"  Evidence: \"{rel['sentence'][:80]}...\"")
else:
    print("No relations extracted.")