from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from pydantic import BaseModel, ConfigDict

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    owner_id: str
    sentence: str
    date_time: str


@dataclass
class Relation:
    doc_id: str
    subject: str
    subject_label: str
    predicate: str
    object: str
    object_label: str
    mention_sentence: str
    confidence: float = 0.0
    source: str = "dspy"
    needs_review: bool = False
    evidence: List[str] = field(default_factory=list)
    relation_id: Optional[str] = None
    chunk_id: Optional[str] = None
    provisional: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class EntityLabels(str, Enum):
    PER = "PERSON"
    ORG = "ORGANIZATION"
    LOC = "LOCATION"
    FAC = "FACILITY"          # Buildings, airports, highways
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATE = "DATE"
    TIME = "TIME"
    PERCENT = "PERCENT"
    MONEY = "MONEY"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    NORP = "NORP"             # Nationalities, religious/political groups
    MISC = "MISCELLANEOUS"
    TECHNOLOGY = "TECHNOLOGY"
    NUM = "NUMBER"

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None
        # Normalize common NER label variants (spaCy, etc.)
        mapping = {
            "PER": cls.PER,
            "PERSON": cls.PER,
            "ORG": cls.ORG,
            "ORGANIZATION": cls.ORG,
            "LOC": cls.LOC,
            "LOCATION": cls.LOC,
            "FAC": cls.FAC,
            "FACILITY": cls.FAC,
            "PRODUCT": cls.PRODUCT,
            "EVENT": cls.EVENT,
            "WORK_OF_ART": cls.WORK_OF_ART,
            "LAW": cls.LAW,
            "LANGUAGE": cls.LANGUAGE,
            "DATE": cls.DATE,
            "TIME": cls.TIME,
            "PERCENT": cls.PERCENT,
            "MONEY": cls.MONEY,
            "QUANTITY": cls.QUANTITY,
            "ORDINAL": cls.ORDINAL,
            "CARDINAL": cls.CARDINAL,
            "NORP": cls.NORP,
            "MISC": cls.MISC,
            "MISCELLANEOUS": cls.MISC,
            "TECHNOLOGY": cls.TECHNOLOGY,
            "NUM": cls.NUM,
            "NUMBER": cls.NUM,
        }
        return mapping.get(value.upper())



class Entity(BaseModel):
    """Unified entity representation."""
    text: str
    label: str
    start: int
    end: int
    mention_sentence: str = ""
    confidence: float = 1.0
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_gt(cls, gt: Dict) -> "Entity":
        if isinstance(gt, Entity):
            return gt
        return cls(
            text=gt["text"],
            label=gt["label"].upper().strip(),
            start=int(gt["start"]),
            end=int(gt["end"]),
            mention_sentence=gt.get("mention_sentence", ""),
            confidence=1.0,
        )

    @classmethod
    def from_pred(cls, pred: Dict) -> "Entity":
        if isinstance(pred, Entity):
            return pred
        return cls(
            text=pred["text"],
            label=pred["label"].upper().strip(),
            start=int(pred["start_char"]),
            end=int(pred["end_char"]),
            mention_sentence=pred.get("mention_sentence", ""),
            confidence=float(pred.get("confidence", 1.0)),
        )

    def span_overlap(self, other: "Entity") -> float:
        """Return IoU (Intersection over Union) of spans."""
        inter_start = max(self.start, other.start)
        inter_end = min(self.end, other.end)
        if inter_start >= inter_end:
            return 0.0
        intersection = inter_end - inter_start
        union = max(self.end, other.end) - min(self.start, other.start)
        return intersection / union if union > 0 else 0.0

    def exact_match(self, other: "Entity") -> bool:
        return self.start == other.start and self.end == other.end

    def partial_match(self, other: "Entity", min_overlap: float = 0.5) -> bool:
        return self.span_overlap(other) >= min_overlap

    def any_overlap(self, other: "Entity") -> bool:
        return not (self.end <= other.start or self.start >= other.end)

    def same_sentence(self, other: "Entity") -> bool:
        """Whether two entities were mentioned in the same sentence."""
        if not self.mention_sentence or not other.mention_sentence:
            return False
        return self.mention_sentence.strip() == other.mention_sentence.strip()


@dataclass
class Mention:
    text: str
    start: int
    end: int
    sent_id: int
    is_pronoun: bool = False
    pronoun_type: Optional[str] = None  # "person", "org", "ambiguous"
    resolved_to: Optional[str] = None
    resolved_entity_idx: Optional[int] = None


@dataclass
class CorefChain:
    """A coreference chain: pronoun -> antecedent entity."""
    pronoun: Mention
    antecedent_text: str
    antecedent_start: int
    antecedent_end: int
    method: str  # "nlp" | "llm" | "rule"
    confidence: float


@dataclass
class MatchResult:
    """Result of matching one prediction against ground truth."""
    pred: Entity
    matched_gt: Optional[Entity]
    match_label: str
    iou: float


class DisambiguationStatus(Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NEW_ENTITY = "new_entity"
    UNKNOWN = "unknown"


@dataclass
class ResolvedEntity:
    original_text: str
    canonical_name: str
    entity_label: str
    mention_sentence: str
    confidence: float
    status: DisambiguationStatus
    source: str
    kg_candidates: List[Dict] = field(default_factory=list)
    summary: Optional[str] = None
    context_clues: List[str] = field(default_factory=list)
    needs_review: bool = False
    is_nil: bool = False
    coref_to: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "original_text": self.original_text,
            "canonical_name": self.canonical_name,
            "entity_label": self.entity_label,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "source": self.source,
            "summary": self.summary,
            "needs_review": self.needs_review,
            "is_nil": self.is_nil,
            "coref_to": self.coref_to,
        }


@dataclass
class CandidateResult:
    """
    Structured output of CandidateFinder.find_candidates().
    Mirrors the Dict returned by legacy NEDEngine._make_candidate(),
    but is type-safe and cross-cutting.
    """
    canonical: str
    label: str
    aliases: List[str]
    summary: str
    context_indicators: List[str]
    related_to: List[str]
    match_score: float
    match_method: str
    neural_sim: Optional[float] = None
    jaccard_ctx: Optional[float] = None
    levenshtein_name: Optional[float] = None
    label_match: bool = False
    source: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "canonical": self.canonical,
            "label": self.label,
            "aliases": self.aliases,
            "summary": self.summary,
            "context_indicators": self.context_indicators,
            "related_to": self.related_to,
            "match_score": round(self.match_score, 3),
            "match_method": self.match_method,
            "neural_sim": self.neural_sim,
            "jaccard_ctx": self.jaccard_ctx,
            "levenshtein_name": self.levenshtein_name,
            "label_match": self.label_match,
            "source": self.source,
        }


@dataclass
class OutboxEvent:
    """
    ACID outbox entry. Written to Neo4j in the SAME TRANSACTION
    as the entity mutation, then asynchronously fanned out to
    ES, Qdrant, and Redis by a poller.
    """
    event_id: str
    event_type: str          # "entity_upserted", "entity_merged", "relation_created"
    canonical: str           # target entity
    payload_json: str        # serialized parameters
    target_stores: List[str] # ["es", "qdrant", "redis"]
    processed: bool = False
    attempts: int = 0
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    processed_at: Optional[str] = None

