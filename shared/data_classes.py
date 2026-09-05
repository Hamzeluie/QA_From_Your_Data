from typing import List, Dict, Optional, Tuple, ClassVar
import difflib
from dataclasses import dataclass, field, asdict
from enum import Enum


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    owner_id: str
    sentence: str
    date_time: str
    start_offset:int
    end_offset:int


class DisambiguationStatus(Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NEW_ENTITY = "new_entity"
    # AMBIGUOUS = "ambiguous"
    # UNKNOWN = "unknown"


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    mention_sentence: str
    confidence: float
    canonical_name: str
    status: DisambiguationStatus
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    kg_candidates: List[Dict] = field(default_factory=list)
    summary: Optional[str] = None
    context_clues: List[str] = field(default_factory=list)
    needs_review: bool = False
    is_nil: bool = False
    coref_to: Optional[str] = None
    
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
            canonical_name=gt.get("canonical_name", ""),
            status=DisambiguationStatus(gt.get("status", "unknown")),
            doc_id=gt.get("doc_id"),
            chunk_id=gt.get("chunk_id"),
            kg_candidates=gt.get("kg_candidates", []),
            summary=gt.get("summary"),
            context_clues=gt.get("context_clues", []),
            needs_review=gt.get("needs_review", False),
            is_nil=gt.get("is_nil", False),
            coref_to=gt.get("coref_to"),
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
            canonical_name=pred.get("canonical_name", ""),
            status=DisambiguationStatus(pred.get("status", "unknown")),
            doc_id=pred.get("doc_id"),
            chunk_id=pred.get("chunk_id"),
            kg_candidates=pred.get("kg_candidates", []),
            summary=pred.get("summary"),
            context_clues=pred.get("context_clues", []),
            needs_review=pred.get("needs_review", False),
            is_nil=pred.get("is_nil", False),
            coref_to=pred.get("coref_to"),
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

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "mention_sentence": self.mention_sentence,
            "confidence": round(self.confidence, 3),
            "canonical_name": self.canonical_name,
            "status": self.status.value,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "kg_candidates": self.kg_candidates,
            "summary": self.summary,
            "context_clues": self.context_clues,
            "needs_review": self.needs_review,
            "is_nil": self.is_nil,
            "coref_to": self.coref_to,
        }


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
    FAC = "FACILITY"
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
    NORP = "NORP"
    MISC = "MISCELLANEOUS"
    TECHNOLOGY = "TECHNOLOGY"
    NUM = "NUMBER"

    @classmethod
    def find_best_match(cls, value: str, threshold: float = 0.75) -> Optional["EntityLabels"]:
        if not isinstance(value, str) or not value.strip():
            return None

        cleaned = value.lower().strip().replace(" ", "_").replace("-", "_")

        # 1. Exact / alias match
        exact = cls._missing_(cleaned)
        if exact is not None:
            return exact

        # 2. Fuzzy match against canonical values + known aliases
        candidates = list({e.value for e in cls} | set(cls._alias_keys()))
        matches = difflib.get_close_matches(cleaned, candidates, n=1, cutoff=threshold)
        if matches:
            resolved = cls._missing_(matches[0])
            if resolved is not None:
                return resolved
            # Defensive: only construct if it's a valid canonical member
            if cls.is_valid(matches[0]):
                return cls(matches[0])
        return None

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None
        mapping = {
            "PER": cls.PER, "PERSON": cls.PER,
            "ORG": cls.ORG, "ORGANIZATION": cls.ORG,
            "LOC": cls.LOC, "LOCATION": cls.LOC,
            "FAC": cls.FAC, "FACILITY": cls.FAC,
            "PRODUCT": cls.PRODUCT, "EVENT": cls.EVENT,
            "WORK_OF_ART": cls.WORK_OF_ART, "LAW": cls.LAW,
            "LANGUAGE": cls.LANGUAGE, "DATE": cls.DATE,
            "TIME": cls.TIME, "PERCENT": cls.PERCENT,
            "MONEY": cls.MONEY, "QUANTITY": cls.QUANTITY,
            "ORDINAL": cls.ORDINAL, "CARDINAL": cls.CARDINAL,
            "NORP": cls.NORP, "MISC": cls.MISC,
            "MISCELLANEOUS": cls.MISC, "TECHNOLOGY": cls.TECHNOLOGY,
            "NUM": cls.NUM, "NUMBER": cls.NUM,
        }
        return mapping.get(value.upper())

    @classmethod
    def is_valid(cls, value: str) -> bool:
        try:
            cls(value)
            return True
        except ValueError:
            return False
        
        
class RelationLabels(str, Enum):
    # ── Symmetric Relations ──
    SPOUSE = "spouse"
    MARRIED_TO = "married_to"
    SIBLING = "sibling"
    COLLABORATES_WITH = "collaborates_with"
    CO_FOUNDER = "co_founder"
    PARTNER = "partner"
    COLLEAGUE = "colleague"
    FRIEND = "friend"
    INTERACTS_WITH = "interacts_with"

    # ── Asymmetric / Hierarchical Relations ──
    EMPLOYEE_OF = "employee_of"
    MEMBER_OF = "member_of"
    FOUNDED = "founded"
    HEADQUARTERS = "headquarters"
    LOCATED_IN = "located_in"
    PARENT_COMPANY = "parent_company"
    SUBSIDIARY = "subsidiary"
    ACQUIRED = "acquired"
    ACQUIRED_BY = "acquired_by"
    AUTHOR_OF = "author_of"
    DIRECTED_BY = "directed_by"
    BORN_IN = "born_in"
    DIED_IN = "died_in"
    CAPITAL_OF = "capital_of"
    PART_OF = "part_of"
    CONTAINS = "contains"
    OPERATES_IN = "operates_in"
    WORKS_FOR = "works_for"

    # ── Knowledge Graph / Wikidata Specific Relations ──
    NO_RELATION = "no_relation"
    APPLIES_TO_JURISDICTION = "applies_to_jurisdiction"
    AUTHOR = "author"
    AWARD_RECEIVED = "award_received"
    BASIN_COUNTRY = "basin_country"
    CAPITAL = "capital"
    CAST_MEMBER = "cast_member"
    CHAIRPERSON = "chairperson"
    CHARACTERS = "characters"

    # ── Fallbacks ──
    RELATED_TO = "related_to"
    OTHER = "other"

    _SYMMETRIC: ClassVar[frozenset[str]] = frozenset({
        "spouse", "married_to", "sibling", "collaborates_with",
        "co_founder", "partner", "colleague", "friend", "interacts_with",
    })

    @classmethod
    def is_symmetric(cls, value: str) -> bool:
        try:
            member = cls(value)
            return member.value in cls._SYMMETRIC
        except ValueError:
            return False

    @classmethod
    def find_best_match(cls, value: str, threshold: float = 0.75) -> Optional["RelationLabels"]:
        if not isinstance(value, str) or not value.strip():
            return None

        cleaned = value.lower().strip().replace(" ", "_").replace("-", "_")

        # 1. Exact / alias match
        exact = cls._missing_(cleaned)
        if exact is not None:
            return exact

        # 2. Fuzzy match against canonical values + known aliases
        candidates = list({e.value for e in cls} | set(cls._alias_keys()))
        matches = difflib.get_close_matches(cleaned, candidates, n=1, cutoff=threshold)
        if matches:
            resolved = cls._missing_(matches[0])
            if resolved is not None:
                return resolved
            # Defensive: only construct if it's a valid canonical member
            if cls.is_valid(matches[0]):
                return cls(matches[0])
        return None


    @classmethod
    def _alias_keys(cls) -> list[str]:
        return [
            "spouse", "husband", "wife", "married", "married_to",
            "sibling", "brother", "sister", "collaborates", "collaborates_with",
            "cofounder", "co_founder", "partner", "colleague", "friend",
            "interacts_with", "employee_of", "employee", "works_for", "works_at",
            "member_of", "member", "founded", "founder", "headquarters", "hq",
            "headquarters_location", "headquarters location",
            "located_in", "location", "resides_in", "located in",
            "parent_company", "parent",
            "subsidiary", "acquired", "acquired_by",
            "author_of", "directed_by", "director",
            "born_in", "birthplace", "place_of_birth", "place of birth",
            "died_in", "deathplace", "place_of_death", "place of death",
            "capital_of", "part_of", "part", "contains", "operates_in",
            "no_relation", "applies_to_jurisdiction", "jurisdiction",
            "author", "award_received", "award",
            "basin_country", "capital", "cast_member", "cast",
            "chairperson", "chair", "characters", "character",
            "related_to", "related", "other",
            # BERT variants that are direction-agnostic
            "country_of_citizenship", "country of citizenship",
        ]

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None

        cleaned = value.lower().strip().replace(" ", "_").replace("-", "_")

        # ── Direction-sensitive aliases (kept here for exact-match only) ──
        # These are also listed in _alias_keys so fuzzy matching can reach them,
        # but the EXTRACTOR is responsible for subject/object swaps.
        mapping = {
            # Symmetric
            "spouse": cls.SPOUSE, "husband": cls.SPOUSE, "wife": cls.SPOUSE,
            "married": cls.MARRIED_TO, "married_to": cls.MARRIED_TO,
            "sibling": cls.SIBLING, "brother": cls.SIBLING, "sister": cls.SIBLING,
            "collaborates": cls.COLLABORATES_WITH, "collaborates_with": cls.COLLABORATES_WITH,
            "cofounder": cls.CO_FOUNDER, "co_founder": cls.CO_FOUNDER,
            "partner": cls.PARTNER, "colleague": cls.COLLEAGUE,
            "friend": cls.FRIEND, "interacts_with": cls.INTERACTS_WITH,

            # Asymmetric
            "employee_of": cls.EMPLOYEE_OF, "employee": cls.EMPLOYEE_OF,
            "works_for": cls.WORKS_FOR, "works_at": cls.WORKS_FOR,
            "member_of": cls.MEMBER_OF, "member": cls.MEMBER_OF,
            "founded": cls.FOUNDED, "founder": cls.FOUNDED,
            "headquarters": cls.HEADQUARTERS, "hq": cls.HEADQUARTERS,
            "headquarters_location": cls.HEADQUARTERS,
            "headquarters location": cls.HEADQUARTERS,
            "located_in": cls.LOCATED_IN, "location": cls.LOCATED_IN,
            "resides_in": cls.LOCATED_IN, "located in": cls.LOCATED_IN,
            "parent_company": cls.PARENT_COMPANY, "parent": cls.PARENT_COMPANY,
            "subsidiary": cls.SUBSIDIARY,
            "acquired": cls.ACQUIRED, "acquired_by": cls.ACQUIRED_BY,
            "author_of": cls.AUTHOR_OF, "directed_by": cls.DIRECTED_BY,
            "director": cls.DIRECTED_BY,
            "born_in": cls.BORN_IN, "birthplace": cls.BORN_IN,
            "place_of_birth": cls.BORN_IN, "place of birth": cls.BORN_IN,
            "died_in": cls.DIED_IN, "deathplace": cls.DIED_IN,
            "place_of_death": cls.DIED_IN, "place of death": cls.DIED_IN,
            "capital_of": cls.CAPITAL_OF,
            "part_of": cls.PART_OF, "part": cls.PART_OF,
            "contains": cls.CONTAINS, "operates_in": cls.OPERATES_IN,

            # Knowledge Graph
            "no_relation": cls.NO_RELATION,
            "applies_to_jurisdiction": cls.APPLIES_TO_JURISDICTION,
            "jurisdiction": cls.APPLIES_TO_JURISDICTION,
            "author": cls.AUTHOR, "award_received": cls.AWARD_RECEIVED,
            "award": cls.AWARD_RECEIVED, "basin_country": cls.BASIN_COUNTRY,
            "capital": cls.CAPITAL, "cast_member": cls.CAST_MEMBER,
            "cast": cls.CAST_MEMBER, "chairperson": cls.CHAIRPERSON,
            "chair": cls.CHAIRPERSON, "characters": cls.CHARACTERS,
            "character": cls.CHARACTERS,

            # Fallbacks
            "related_to": cls.RELATED_TO, "related": cls.RELATED_TO,
            "other": cls.OTHER,
        }
        return mapping.get(cleaned)

    @classmethod
    def is_valid(cls, value: str) -> bool:
        try:
            cls(value)
            return True
        except ValueError:
            return False


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
    antecedent_doc_id: str
    antecedent_chunk_id:str
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


@dataclass
class NEResult:
    chunks: List['Chunk'] = field(default_factory=list)
    entities: List['Entity'] = field(default_factory=list)
