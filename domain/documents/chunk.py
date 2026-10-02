from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    owner_id: str
    text: str
    start_offset:int
    end_offset:int
    updated_at: str
    metadata: Optional[dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "owner_id": self.owner_id,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": self.metadata,
            "updated_at": self.updated_at
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        return cls(**data)
