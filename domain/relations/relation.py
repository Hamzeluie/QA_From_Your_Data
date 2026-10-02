from enum import Enum
from typing import List, Dict, Optional, ClassVar
import difflib
from dataclasses import dataclass, field, asdict


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
    def from_dict(cls, data: Dict) -> "Relation":
        return cls(**data)
