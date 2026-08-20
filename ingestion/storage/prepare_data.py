import os
from pathlib import Path
from datasets import load_dataset



output_dir = Path("/home/mehdi/Documents/projects/knowledge_graph_examples/datasets/docred")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading DocRED from auto-converted Parquet...")

# Load via the parquet revision (no trust_remote_code needed)
ds = load_dataset("thunlp/docred", revision="refs/convert/parquet")

import os
import json

def prepare_NER_dataset(ds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for idx, doc in enumerate(ds):
        # Reconstruct sentence texts and full document
        sent_texts = [" ".join(sent) for sent in doc["sents"]]
        document = "".join(sent_texts)  # sentences glued with no separator
        
        # Precompute character offset where each sentence starts
        sent_offsets = []
        off = 0
        for st in sent_texts:
            sent_offsets.append(off)
            off += len(st)
        
        label_doc = []
        for entity in doc["vertexSet"]:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]          # e.g. [3] or [0, 2]
                name = mention["name"]
                ent_type = mention["type"]
                
                tokens = doc["sents"][sent_id]
                
                # Build token char offsets inside this sentence
                tok_offsets = []
                char_ptr = 0
                for i, tok in enumerate(tokens):
                    tok_offsets.append(char_ptr)
                    char_ptr += len(tok) + 1  # +1 for the space between tokens
                
                start_tok = pos[0]
                start_in_sent = tok_offsets[start_tok]
                start = sent_offsets[sent_id] + start_in_sent
                
                # KEY FIX: end is determined by the exact name length,
                # not by token boundaries (which can overshoot)
                end = start + len(name)
                
                # Sanity check (optional)
                # assert document[start:end] == name, f"{document[start:end]!r} != {name!r}"
                
                label_doc.append({
                    "text": name,
                    "label": ent_type,
                    "start": start,
                    "end": end,
                    "sents_id": sent_id
                })
        
        result = {
            "doc_id": idx,
            "document": document,
            "sentence": doc["sents"],
            "label_sents": doc["vertexSet"],
            "label_doc": label_doc
        }
        results.append(result)
    
    out_path = os.path.join(output_dir, "ner_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    return results
    

prepare_NER_dataset(ds["test"], "/home/mehdi/Documents/projects/knowledge_graph_examples/datasets/QA_your_data")
