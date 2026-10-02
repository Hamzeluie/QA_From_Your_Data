from ..entities import CanonicalEntity, MentionEntity
from ..documents import Chunk
from ..relations import Relation
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class EventPayload:
    canonicals:            Optional[List[CanonicalEntity]] = None
    mentions:              Optional[List[MentionEntity]]   = None
    resolved_mentions:     Optional[List[CanonicalEntity]]   = None
    chunks:                Optional[List[Chunk]]           = None
    chunk_canonical_links: Optional[List[Dict[str, str]]]  = None
    relations:             Optional[List[Relation]]        = None
    raw_text:              Optional[str]                   = None

    def to_dict(self) -> Dict:
        return {
            "canonicals":            [c.to_dict() for c in (self.canonicals or [])],
            "mentions":              [m.to_dict() for m in (self.mentions or [])],
            "resolved_mentions":     [m.to_dict() for m in (self.resolved_mentions or [])],
            "chunks":                [ch.to_dict() for ch in (self.chunks or [])],
            "chunk_canonical_links": list(self.chunk_canonical_links or []),
            "relations":             [r.to_dict() for r in (self.relations or [])],
            "raw_text":              self.raw_text,
        }

    
    @classmethod
    def from_dict(cls, data: Dict) -> "EventPayload":
        return cls(
            canonicals=[CanonicalEntity.from_dict(c) for c in (data.get("canonicals") or [])],
            mentions=[MentionEntity.from_dict(m) for m in (data.get("mentions") or [])],
            resolved_mentions=[CanonicalEntity.from_dict(m) for m in (data.get("resolved_mentions") or [])],
            chunks=[Chunk.from_dict(ch) for ch in (data.get("chunks") or [])],
            chunk_canonical_links=data.get("chunk_canonical_links") or [],
            relations=[Relation.from_dict(r) for r in (data.get("relations") or [])],
            raw_text= data.get("raw_text") or None,
        )
