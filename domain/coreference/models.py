from domain.entities import EntityLabels
from dataclasses import dataclass


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
