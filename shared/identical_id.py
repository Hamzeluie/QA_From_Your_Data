import hashlib
import json
import unicodedata
from typing import Any
from uuid import NAMESPACE_DNS, uuid5
from domain import EntityLabels, RelationLabels


_SEP = "\x1f"  # ASCII unit separator: can't collide with real text


def _clean_text(value: Any) -> str:
    """Canonical text form so 'Open AI', 'openai ', 'OpenAI' all hash identically."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.casefold().split())


def make_id(*parts: Any, prefix: str, length: int = 32) -> str:
    """
    Deterministic ID: sha256 of the parts, truncated to `length` hex chars.

    - _SEP prevents (\"ab\",\"c\") vs (\"a\",\"bc\") collisions
    - 32 hex chars = 128 bits → collisions are effectively impossible
      (16 would already be fine up to ~100M entities)
    - prefix makes IDs self-describing in logs / Neo4j / ES
    """
    material = _SEP.join("" if p is None else str(p) for p in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:length]}"


def digest_any(payload: Any) -> str:
    """Stable hash of arbitrary JSON-able data (dict key order ignored)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    


def _norm_label(value: str, enum_cls) -> str:
    """'ORG' and 'ORGANIZATION' must map to the SAME id → normalize via your enum."""
    best = enum_cls.find_best_match(value) if value else None
    return best.value if best else _clean_text(value)


def _norm_label_predicate(predicate: str):
    return _norm_label(predicate, RelationLabels)


class Ids:
    @staticmethod
    def canonical(name: str, label: str, namespace: str = "") -> str:
        # name+label in the key → the same real-world string resolves to the
        # same id across ALL documents → automatic dedup on upsert.
        return make_id(
            namespace,
            _clean_text(name),
            _norm_label(label, EntityLabels),
            prefix="can",
        )

    @staticmethod
    def alias(canonical_id: str, name: str) -> str:
        return make_id(canonical_id, _clean_text(name), prefix="als")

    @staticmethod
    def document(owner_id: str, source_name: str) -> str:
        return make_id(owner_id, _clean_text(source_name), prefix="doc")

    @staticmethod
    def document_from_content(content: bytes, owner_id: str = "") -> str:
        # re-uploading the same file → same doc_id → whole pipeline idempotent
        return make_id(owner_id, hashlib.sha256(content).hexdigest(), prefix="doc")

    @staticmethod
    def chunk(doc_id: str, index: int) -> str:
        return make_id(doc_id, index, prefix="chk")

    @staticmethod
    def mention(doc_id: str, chunk_id: str, start: int, end: int) -> str:
        # offsets are already unique within a chunk; DON'T include the text,
        # otherwise a new NER model version changes ids and breaks idempotency.
        return make_id(doc_id, chunk_id, start, end, prefix="mnt")

    @staticmethod
    def relation(doc_id: str, chunk_id: str, subject: str, predicate: str,
                 obj: str, sentence: str = "") -> str:
        # sentence in the key: the same triple in two sentences = two
        # relations with different evidence, which you want to keep.
        return make_id(
            doc_id, chunk_id,
            _clean_text(subject),
            _norm_label_predicate(predicate),
            _clean_text(obj),
            _clean_text(sentence),
            prefix="rel",
        )

    @staticmethod
    def event(doc_id: str, event_type: str, stage_key: str = "",
              payload: dict | None = None) -> str:
        # stage_key e.g. "chunk_3": re-running a stage re-emits the SAME
        # event id → ON CONFLICT DO NOTHING swallows duplicates.
        return make_id(doc_id, event_type, stage_key,
                       digest_any(payload) if payload else "", prefix="evt")

    # optional: uuid5 variant if you need real UUID columns
    @staticmethod
    def uuid(*parts: Any) -> str:
        ns = uuid5(NAMESPACE_DNS, "knowledge-graph")
        return str(uuid5(ns, _SEP.join(map(str, parts))))