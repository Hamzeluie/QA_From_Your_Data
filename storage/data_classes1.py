from typing import List, Dict, Optional, Union, ClassVar, Any
from datetime import datetime
import difflib
from dataclasses import dataclass, field, asdict
from enum import Enum



# ========== Label DataClasses
class CandidateSource(str, Enum):
    REDIS = "redis"
    POSTGRES = "postgres"
    ELASTICSEARCH = "elasticsearch"
    QDRANT = "qdrant"
    NEO4J = "neo4j"
    
    
class CandidateMatchMethod(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    
    
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


class DisambiguationStatus(Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"

# ========== NER DataClasses
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
    def from_dict(cls, data: Dict) -> "CanonicalEntity":
        return cls(**data)

@dataclass
class NEResult:
    chunks: List[Chunk] = field(default_factory=list)
    entities: List[MentionEntity] = field(default_factory=list)

# ========== Coreference DataClasses
# @dataclass
# class Mention:
#     text: str
#     start: int
#     end: int
#     sent_id: int
#     is_pronoun: bool = False
#     pronoun_type: Optional[str] = None  # "person", "org", "ambiguous"
#     token: Optional[str] = None
#     # resolved_to: Optional[str] = None
#     # resolved_entity_idx: Optional[int] = None

@dataclass
class CorefMention:
    text: str
    start: int
    end: int
    head: str
    head_lemma: str
    pos: str
    label: EntityLabels | None
    sentence_id: int
    number: str
    gender: str | None
    is_pronoun: bool
    is_named_entity: bool

@dataclass
class CorefChain:
    """A coreference chain: pronoun -> antecedent entity."""
    pronoun: CorefMention
    antecedent_doc_id: str
    antecedent_chunk_id:str
    pronoun_token:str
    antecedent_text: str
    antecedent_start: int
    antecedent_end: int
    method: str  # "nlp" | "llm" | "rule"
    confidence: float

# ========== RelationExtraction DataClasses
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
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CanonicalEntity":
        return cls(**data)


# ========== OutBox

class ProcessSteps(str, Enum):
    SAVE_CANONICAL_POSTGRES = "save_canonical_posgres"
    SAVE_CANONICAL_NEO4J = "save_canonical_neo4j"
    SAVE_CANONICAL_QDRANT = "save_canonical_qdrant"
    SAVE_MENTION_POSTGRES = "save_mention_postgres"
    SAVE_CHUNKS_QDRANT = "save_chunks_gdrant"
    UPDATE_CHUNK_CANONICALS_QDRANT = "update_chunk_canonicals_qdrant"
    SAVE_CHUNKS_ELASTICSEARCH = "save_chunks_elasticsearch"   
    SAVE_RELATION_POSTGRES = "save_relation_postgres"
    SAVE_RELATION_NEO4J = "save_relation_neo4j"
    DELETE_RESOLVED_MENTION_POSTGRES = "delete_resolved_mention_postgres"
    DELETE_MENTION_PAYLOAD = "delete_mention_payload"

class EventType(str, Enum):
    UPLOAD_RAW_TEXT = "upload_raw_text"
    RESOLVED_ENTITY = "resolved_entity"
    HUMAN_RESOLVED_MENTION = "human_resolved_mention"
    RELATION_EXTRACTION = "realtion_extraction"
    DELETE_CANONICAL = "delete_canonical"
    DELETE_MENTION = "delete_mention"
    UPDATE_CANONICAL = "update_canonical"
    UPDATE_CHUNK_METADATA = "update_chunk_metadata"
    UPDATE_RELATION = "update_relation"
    

class DocumentStatus(str, Enum):
    FAILD = "faild"
    UPLOADED = "upload"
    ENTITIES_RESOLVING = "entities_resolving"
    ENTITIES_RESOLVED = "entities_resolved"
    NEEDS_REVIEW = "needs_review"
    EXTRACTING_RELATIONS = "extracting_relations"
    RELATION_EXTRACTED = "relation_extracted"
    

@dataclass
class OutboxDocument:
    doc_id: str
    owner_id: str
    status: DocumentStatus
    pending_unresolved_count:int = 0
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EventPayload:
    canonicals:            Optional[List[CanonicalEntity]] = None
    mentions:              Optional[List[MentionEntity]]   = None
    resolved_mentions:     Optional[List[MentionEntity]]   = None
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
            resolved_mentions=[MentionEntity.from_dict(m) for m in (data.get("resolved_mentions") or [])],
            chunks=[Chunk.from_dict(ch) for ch in (data.get("chunks") or [])],
            chunk_canonical_links=data.get("chunk_canonical_links") or [],
            relations=[Relation.from_dict(r) for r in (data.get("relations") or [])],
            raw_text= data.get("raw_text") or None,
        )

@dataclass
class OutboxEvent:
    doc_id: str
    event_id: str
    event_type: EventType
    
    # Must be a list! A chunk needs to go to BOTH Qdrant and ES.
    processed_steps: List[ProcessSteps] = field(default_factory=list)  # list of steps whcih is processed
    
    # Generic payload. The poller will parse this based on event_type.
    # e.g., {"canonicals": [...], "mentions": [...]}
    payload: Union[Dict[str, Any], EventPayload] = field(default_factory=dict) 
    
    # Poller tracking fields
    processed: bool = False
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    @property
    def typed_payload(self) -> EventPayload:
        if isinstance(self.payload, EventPayload):
            return self.payload
        return EventPayload.from_dict(self.payload or {})


