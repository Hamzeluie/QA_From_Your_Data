import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_docred(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sentencize_docred(doc: Dict) -> List[Dict]:
    """
    Convert one DocRED document into sentence-level training examples.
    Each example: (sentence_text, entities_in_sent, relations_in_sent)
    """
    sentences = doc["sents"]  # list of token lists
    title = doc.get("title", "")

    # Build sentence texts
    sent_texts = [" ".join(tokens) for tokens in sentences]

    # Map entity_id -> list of mentions
    vertex_set = doc.get("vertexSet", [])

    # Map (sent_idx, start_tok, end_tok) -> entity_id
    mention_to_ent = {}
    ent_id_to_name = {}
    for ent_id, mentions in enumerate(vertex_set):
        names = [m["name"] for m in mentions]
        ent_id_to_name[ent_id] = max(names, key=len)  # longest as canonical
        for m in mentions:
            key = (m["sent_id"], m["pos"][0], m["pos"][1])
            mention_to_ent[key] = ent_id

    # Group labels by evidence sentence
    sent_relations = defaultdict(list)
    for label in doc.get("labels", []):
        h, t, r = label["h"], label["t"], label["r"]
        for ev_sent in label.get("evidence", []):
            sent_relations[ev_sent].append({
                "h": h,
                "t": t,
                "relation": r,
            })

    examples = []
    for sent_idx, sent_text in enumerate(sent_texts):
        # Find entities mentioned in this sentence
        sent_entities = []
        for ent_id, mentions in enumerate(vertex_set):
            for m in mentions:
                if m["sent_id"] == sent_idx:
                    sent_entities.append({
                        "id": ent_id,
                        "name": m["name"],
                        "pos": m["pos"],
                        "type": m.get("type", "MISC"),
                    })
                    break  # one mention per entity per sentence is enough

        if len(sent_entities) < 2:
            continue

        # Build positive pairs
        pos_pairs = set()
        for rel in sent_relations.get(sent_idx, []):
            pos_pairs.add((rel["h"], rel["t"], rel["relation"]))

        # Build all pairs (positive + negative)
        sent_entity_ids = [e["id"] for e in sent_entities]
        for i, h in enumerate(sent_entity_ids):
            for t in sent_entity_ids[i+1:]:
                # Check if (h,t) or (t,h) has a relation
                rel_type = "no_relation"
                if (h, t, r) in [(a, b, c) for a, b, c in pos_pairs]:
                    # Need to match exact relation
                    for pr in sent_relations.get(sent_idx, []):
                        if pr["h"] == h and pr["t"] == t:
                            rel_type = pr["relation"]
                            break
                elif (t, h, r) in [(a, b, c) for a, b, c in pos_pairs]:
                    for pr in sent_relations.get(sent_idx, []):
                        if pr["h"] == t and pr["t"] == h:
                            rel_type = pr["relation"]
                            break

                examples.append({
                    "text": sent_text,
                    "subject": ent_id_to_name[h],
                    "object": ent_id_to_name[t],
                    "subject_type": next(e["type"] for e in sent_entities if e["id"] == h),
                    "object_type": next(e["type"] for e in sent_entities if e["id"] == t),
                    "relation": rel_type,
                    "doc_title": title,
                    "sent_idx": sent_idx,
                })

    return examples


def prepare_docred_dataset(docred_path: str, out_dir: str):
    """Convert DocRED JSON to sentence-level train/dev/test JSONL."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_docred(docred_path)
    all_examples = []
    for doc in data:
        all_examples.extend(sentencize_docred(doc))

    # Split 80/10/10 if no dev/test provided
    # (DocRED provides dev.json separately; use that for val)
    import random
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    train = all_examples[:int(0.8*n)]
    dev = all_examples[int(0.8*n):int(0.9*n)]
    test = all_examples[int(0.9*n):]

    for split, examples in [("train", train), ("dev", dev), ("test", test)]:
        with open(out_path / f"{split}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    return all_examples