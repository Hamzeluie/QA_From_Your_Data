from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MentionEntity:
    # What NER actually detected
    name: str
    label: str

    # Character offsets in the source text
    start: int
    end: int

    # Local context used for resolution
    mention_sentence: str
    

    # NER confidence
    confidence: float

    # Source location
    doc_id: str
    chunk_id: str

    # Coreference information
    coref_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "label": self.label,
            "start":self.start,
            "end":self.end,
            "mention_sentence": self.mention_sentence,
            "confidence":self.confidence,
            "doc_id":self.doc_id,
            "chunk_id":self.chunk_id,
            "coref_to":self.coref_to,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MentionEntity":
        return cls(**data)
