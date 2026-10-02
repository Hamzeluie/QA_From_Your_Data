from .enums import CandidateSource, CandidateMatchMethod
from .canonical import CanonicalEntity
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class CandidateResult:
    # Canonical identity
    canonical_id: str
    canonical_name: str

    # Canonical metadata
    label: str
    aliases: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    # How this candidate was found
    source: CandidateSource = CandidateSource.POSTGRES
    match_method: CandidateMatchMethod = CandidateMatchMethod.FUZZY

    # Candidate-specific scores
    match_score: float = 0.0
    label_match: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "label": self.label,
            "aliases": self.aliases,
            "summary": self.summary,
            "source": self.source,
            "match_method": self.match_method,
            "match_score": round(self.match_score, 3),
            "label_match": self.label_match,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> CanonicalEntity:
        return cls(**data)
