from enum import Enum
from typing import List, Dict, Optional, Union, ClassVar, Any
import difflib

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

class DisambiguationStatus(Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


