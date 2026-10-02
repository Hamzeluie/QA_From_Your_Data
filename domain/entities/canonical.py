from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class CanonicalEntity:
    canonical_id: str
    name: str
    label: str
    aliases: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "label": self.label,
            "aliases": self.aliases,
            "summary": self.summary,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CanonicalEntity":
        return cls(**data)
    