import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import re
import dspy
import spacy
from spacy.tokens import Doc, Token
from typing import List, Dict, Optional
from config.settings import settings
from storage.data_classes import MentionEntity, CorefChain, Mention, EntityLabels, DisambiguationStatus
from ingestion.models.base import IExtractor
from shared.utils import extract_exact_sentence


model_id = "openai/" + settings.LLM_MODEL_NAME

lm = dspy.LM(
    model=model_id,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    max_tokens=4096,
    temperature=0.0,
    top_p=1.0,
    n=1,
    stop=None
)
dspy.configure(lm=lm)


class CoreferenceResolution_v1(dspy.Signature):
    """
    You are a coreference resolution expert.

    Given a document and a list of named entities, resolve each PRONOUN
    to the correct antecedent entity. Consider:
    - Grammatical gender/number agreement
    - Syntactic subject preference
    - Semantic compatibility (e.g., "it" refers to organizations/products, not people)
    - Recency and salience (first-mention preference)
    - Entities inside quotation marks are usually titles/names and less likely to be the antecedent for a pronoun appearing outside those quotes

    For each pronoun, output the exact text of the antecedent entity it refers to.
    If uncertain, omit the entry.
    """

    document: str = dspy.InputField(desc="Full document text")
    entities: List[Dict] = dspy.InputField(desc="List of entities with text, label, start, end")
    pronouns: List[Dict] = dspy.InputField(desc="List of pronouns with text, start, end")
    resolutions: List[Dict] = dspy.OutputField(
        desc=(
            "JSON list of objects. Each object MUST have: "
            "pronoun_text (str), pronoun_start (int), antecedent_text (str), confidence (float 0-1). "
            "Example: [{\"pronoun_text\": \"He\", \"pronoun_start\": 45, \"antecedent_text\": \"John Smith\", \"confidence\": 0.95}]"
        )
    )

class CoreferenceResolution(dspy.Signature):
    """
    You are a coreference resolution expert.

    Task:
    1. Scan the 'document' and identify all personal pronouns (e.g., he, she, it, they, we, his, her, its, their, him, them).
    2. For each pronoun, resolve it to the correct antecedent from the provided 'entities' list.

    Rules:
    - Grammatical gender/number agreement is mandatory.
    - Semantic compatibility (e.g., "it" refers to organizations/products, not people).
    - Recency and salience (first-mention preference).
    - If a pronoun does not clearly refer to any provided entity, omit it.
    """
    document: str = dspy.InputField(desc="Full document text")
    entities: List[Dict] = dspy.InputField(desc="List of named entities with text, label, start, end")
    
    resolutions: List[Dict] = dspy.OutputField(
        desc=(
            "JSON list of resolved pronouns. Each object MUST have: "
            "'pronoun_text' (str), 'sentence_context' (str, the full sentence containing the pronoun), "
            "'antecedent_text' (str), 'confidence' (float 0-1). "
            "Example: [{'pronoun_text': 'He', 'sentence_context': 'He served as the CEO.', 'antecedent_text': 'Steve Jobs', 'confidence': 0.95}]"
        )
    )
    
    
class CorefExtractor(dspy.Module):
    """DSPy module for LLM-based coreference extraction."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.extractor = dspy.ChainOfThought(CoreferenceResolution) if use_cot else dspy.Predict(CoreferenceResolution)

    def forward(self, document: str, entities: List[Dict], pronouns: List[Dict]) -> List[Dict]:
        if not pronouns or not entities:
            return []

        try:
            prediction = self.extractor(
                document=document,
                entities=entities,
                pronouns=pronouns
            )
            return prediction.resolutions if prediction.resolutions else []
        except Exception as e:
            print(f"[CorefExtractor LLM Error] {e}")
            return []


class DSPyCorefResolver_v1(IExtractor):
    """Hybrid coreference resolver combining fast NLP heuristics with LLM reasoning."""

    PRONOUN_MAP = {
        "he":       {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "him":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "his":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "she":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "her":      {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},
        "hers":     {"types": ["PER", "PERSON"],     "category": "person",     "confidence_boost": 0.2},

        "they":     {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "them":     {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "their":    {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},
        "theirs":   {"types": ["PER", "ORG", "MISC"], "category": "ambiguous", "confidence_boost": 0.0},

        "it":       {"types": ["ORG", "LOC", "MISC", "PRODUCT"], "category": "ambiguous", "confidence_boost": 0.0},
        "its":      {"types": ["ORG", "LOC", "MISC", "PRODUCT"], "category": "ambiguous", "confidence_boost": 0.0},

        "we":       {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "us":       {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "our":      {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
        "ours":     {"types": ["ORG"],               "category": "org",        "confidence_boost": 0.15},
    }

    DESCRIPTION_PATTERNS = [
        (r"\bthe\s+company\b",      ["ORG"]),
        (r"\bthe\s+firm\b",          ["ORG"]),
        (r"\bthe\s+organization\b", ["ORG"]),
        (r"\bthe\s+airline\b",      ["ORG"]),
        (r"\bthe\s+bank\b",         ["ORG"]),
        (r"\bthe\s+CEO\b",          ["PER", "PERSON"]),
        (r"\bthe\s+president\b",    ["PER", "PERSON"]),
        (r"\bthe\s+executive\b",    ["PER", "PERSON"]),
        (r"\bthe\s+writer\b",       ["PER", "PERSON"]),
        (r"\bthe\s+author\b",       ["PER", "PERSON"]),
        (r"\bthe\s+firm['']?s\b",   ["ORG"]),
        (r"\bthe\s+book\b",         ["MISC", "PRODUCT"]),
        (r"\bthe\s+novel\b",        ["MISC", "PRODUCT"]),
        (r"\bthe\s+movie\b",        ["MISC", "PRODUCT"]),
        (r"\bthe\s+album\b",        ["MISC", "PRODUCT"]),
    ]

    def __init__(self, nlp=None, llm_module: Optional[dspy.Module] = None,
                 mode: str = "hybrid", threshold: float = 0.92):
        self.nlp = nlp
        self.llm_module = llm_module
        self.mode = mode
        self.threshold = threshold

        if mode in ("llm", "hybrid") and llm_module is None:
            self.llm_resolver = CorefExtractor(use_cot=True)
        else:
            self.llm_resolver = llm_module

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def extract(self, text: str, entities: List[MentionEntity]) -> List[MentionEntity]:
        """
        Resolve pronouns and descriptions in text against the provided entities.
        Accepts List[Entity]; returns original entities + resolved coreferences.
        """
        if not entities:
            return []

        # ── 1. Base NER entities as Entity ──
        named_ents = self._get_named_entities(entities)

        # ── 2. Pronoun resolution ──
        doc = self.nlp(text) if self.nlp and hasattr(self.nlp, "__call__") else None
        pronouns = self._extract_pronouns(doc, text)

        if pronouns and named_ents:
            if self.mode == "nlp":
                chains = self._resolve_nlp(pronouns, named_ents, text)
            elif self.mode == "llm":
                chains = self._resolve_llm(pronouns, named_ents, text)
            else:
                chains = self._resolve_hybrid(pronouns, named_ents, text)

            for chain in chains:
                entities.append(self._chain_to_resolved(chain, text))

        # ── 3. Definite descriptions ──
        # resolved.extend(self._resolve_descriptions(text, named_ents))

        return entities

    # -----------------------------------------------------------------------
    # INTERNAL CONVERTERS
    # -----------------------------------------------------------------------

    def _get_named_entities(self, entities: List[MentionEntity]) -> List[MentionEntity]:
        """Return only base named entities (not previous coreference resolutions)."""
        return [r for r in entities if r.coref_to is None]

    def _chain_to_resolved(self, chain: CorefChain, text: str) -> MentionEntity:
        mention_sent = extract_exact_sentence(
            doc_text=text,
            start_char=chain.pronoun.start,
            end_char=chain.pronoun.end,
        )
        return MentionEntity(
            doc_id=chain.antecedent_doc_id,
            chunk_id=chain.antecedent_chunk_id,
            text=chain.pronoun.text,
            start=chain.antecedent_start,
            end=chain.antecedent_end,
            canonical_name=chain.antecedent_text,
            label=self._map_pronoun_to_label(chain.pronoun.text),
            mention_sentence=mention_sent,
            status=DisambiguationStatus.RESOLVED,
            confidence=round(min(chain.confidence, 1.0), 3),
            coref_to=chain.antecedent_text,
        )

    # -----------------------------------------------------------------------
    # NLP RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_nlp(self, pronouns: List[Mention],
                     named_ents: List[MentionEntity],
                     text: str) -> List[CorefChain]:
        resolutions = []

        for pro in pronouns:
            cfg = self.PRONOUN_MAP.get(pro.text.lower(), {})
            compatible_types = cfg.get("types", ["MISC"])

            # Candidates must appear before the pronoun
            candidates = [e for e in named_ents if getattr(e, "end", 0) <= pro.start]
            if not candidates:
                continue

            # Label compatibility
            candidates = [e for e in candidates if e.label in compatible_types]
            if not candidates:
                continue

            # Score and pick best
            scored = [(e, self._score_antecedent(e, pro, text)) for e in candidates]
            best, best_score = max(scored, key=lambda x: x[1])

            if best_score > 0.3:
                resolutions.append(CorefChain(
                    antecedent_doc_id=best.doc_id,
                    antecedent_chunk_id=best.chunk_id,
                    pronoun=pro,
                    antecedent_text=best.text,
                    antecedent_start=getattr(best, "start", 0),
                    antecedent_end=getattr(best, "end", 0),
                    method="nlp",
                    confidence=min(best_score + cfg.get("confidence_boost", 0), 1.0)
                ))

        return resolutions

    def _score_antecedent(self, antecedent: MentionEntity,
                          pronoun: Mention,
                          text: str) -> float:
        score = 0.0
        ant_start = getattr(antecedent, "start", 0)
        ant_end   = getattr(antecedent, "end", 0)
        ant_text  = antecedent.text

        distance = pronoun.start - ant_end
        score += 0.4 * (1.0 / (1.0 + 0.01 * distance))

        score += 0.2 * (1.0 / (1.0 + 0.001 * ant_start))

        sent_start = text.rfind(".", 0, ant_start) + 1
        if sent_start < 0:
            sent_start = 0
        dist_from_sent_start = ant_start - sent_start
        if dist_from_sent_start < 20:
            score += 0.2

        score += 0.1 * min(len(ant_text) / 20.0, 1.0)

        pro_lower = pronoun.text.lower()
        if pro_lower in ("he", "him", "his"):
            if any(n in ant_text.lower() for n in ["john", "michael", "david", "james", "robert"]):
                score += 0.1
        elif pro_lower in ("she", "her", "hers"):
            if any(n in ant_text.lower() for n in ["mary", "jennifer", "linda", "patricia"]):
                score += 0.1

        quote_before = text.rfind('"', 0, ant_start)
        quote_after = text.find('"', ant_end, pronoun.start)
        if quote_before != -1 and quote_after != -1:
            score -= 0.25

        return score

    # -----------------------------------------------------------------------
    # LLM RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_llm(self, pronouns: List[Mention],
                     named_ents: List[MentionEntity],
                     text: str) -> List[CorefChain]:
        if not self.llm_resolver or not pronouns:
            return []

        entity_list = [
            {
                "text": e.text,
                "label": e.label,
                "start": getattr(e, "start", 0),
                "end": getattr(e, "end", 0),
            }
            for e in named_ents
        ]
        pronoun_list = [{"text": p.text, "start": p.start, "end": p.end} for p in pronouns]

        try:
            resolutions_raw = self.llm_resolver(
                document=text,
                entities=entity_list,
                pronouns=pronoun_list
            )
        except Exception as e:
            print(f"[LLM Coref Error] {e}")
            return []

        resolutions = []
        for res in resolutions_raw:
            if not res or not res.get("antecedent_text"):
                continue

            pro_start = res.get("pronoun_start")
            pro_text = res.get("pronoun_text", "")

            pro_match = None
            if pro_start is not None:
                pro_match = next((p for p in pronouns if p.start == pro_start), None)
            if not pro_match:
                pro_match = next((p for p in pronouns if p.text.lower() == pro_text.lower()), None)
            if not pro_match:
                continue

            ant_text = res["antecedent_text"]
            ant_candidates = [e for e in named_ents if e.text == ant_text]
            if not ant_candidates:
                ant_candidates = [e for e in named_ents if e.text.lower() == ant_text.lower()]

            if not ant_candidates:
                continue

            # Antecedent must appear before pronoun
            ant_candidates = [
                e for e in ant_candidates
                if getattr(e, "end", 0) <= pro_match.start
            ]
            if not ant_candidates:
                continue

            # Pick closest (minimum distance)
            ant = min(ant_candidates, key=lambda e: pro_match.start - getattr(e, "end", 0))

            resolutions.append(CorefChain(
                antecedent_doc_id=ant.doc_id,
                antecedent_chunk_id=ant.chunk_id,
                pronoun=pro_match,
                antecedent_text=ant.text,
                antecedent_start=getattr(ant, "start", 0),
                antecedent_end=getattr(ant, "end", 0),
                method="llm",
                confidence=min(res.get("confidence", 0.8), 1.0)
            ))

        return resolutions

    # -----------------------------------------------------------------------
    # HYBRID RESOLUTION
    # -----------------------------------------------------------------------

    def _resolve_hybrid(self, pronouns: List[Mention],
                        named_ents: List[MentionEntity],
                        text: str) -> List[CorefChain]:
        person_pronouns = [
            p for p in pronouns
            if self.PRONOUN_MAP.get(p.text.lower(), {}).get("category") == "person"
        ]
        ambiguous_pronouns = [
            p for p in pronouns
            if self.PRONOUN_MAP.get(p.text.lower(), {}).get("category") in ("ambiguous", "org")
        ]

        resolutions: List[CorefChain] = []
        resolved_starts: set[int] = set()

        if person_pronouns:
            nlp_res = self._resolve_nlp(person_pronouns, named_ents, text)
            for r in nlp_res:
                resolutions.append(r)
                resolved_starts.add(r.pronoun.start)

            low_conf = [r for r in nlp_res if r.confidence < .6]
            if low_conf:
                llm_res = self._resolve_llm([r.pronoun for r in low_conf], named_ents, text)
                for lr in llm_res:
                    old = next((r for r in resolutions if r.pronoun.start == lr.pronoun.start), None)
                    if old and lr.confidence > old.confidence:
                        resolutions.remove(old)
                        resolutions.append(lr)

        if ambiguous_pronouns and self.llm_resolver:
            unresolved_ambiguous = [p for p in ambiguous_pronouns if p.start not in resolved_starts]
            if unresolved_ambiguous:
                llm_res = self._resolve_llm(unresolved_ambiguous, named_ents, text)
                resolutions.extend(llm_res)

        return resolutions

    # -----------------------------------------------------------------------
    # DEFINITE DESCRIPTIONS
    # -----------------------------------------------------------------------

    def _resolve_descriptions(self, text: str,
                              named_ents: List[MentionEntity]) -> List[MentionEntity]:
        results: List[MentionEntity] = []

        for pattern, types in self.DESCRIPTION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                antecedent = self._find_nearest_compatible(m.start(), m.end(), named_ents, types)
                if antecedent is None:
                    continue

                mention_sent = extract_exact_sentence(
                    doc_text=text,
                    start_char=m.start(),
                    end_char=m.end())
                
                results.append(MentionEntity(
                    text=m.group(),
                    canonical_name=antecedent.text,
                    entity_label=types[0],
                    mention_sentence=mention_sent,
                    status=DisambiguationStatus.UNRESOLVED,
                    confidence=0.7,
                    coref_to=antecedent.text,
                ))

        return results

    def _find_nearest_compatible(self, start: int, end: int,
                                 named_ents: List[MentionEntity],
                                 compatible_types: List[str]) -> Optional[MentionEntity]:
        candidates = [
            e for e in named_ents
            if getattr(e, "end", 0) <= start and e.label in compatible_types
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: start - getattr(e, "end", 0))

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _extract_pronouns(self, doc, text: str) -> List[Mention]:
        pronouns = []
        sent_id = 0
        last_end = 0

        if doc and hasattr(doc, "sents"):
            sentences = list(doc.sents)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            if hasattr(sent, "text"):
                sent_text = sent.text
                sent_start = sent.start_char if hasattr(sent, "start_char") else text.find(sent_text, last_end)
            else:
                sent_text = sent
                sent_start = text.find(sent_text, last_end)

            if sent_start == -1:
                sent_start = last_end

            pattern = r'\b(' + '|'.join(re.escape(k) for k in self.PRONOUN_MAP.keys()) + r')\b'
            for match in re.finditer(pattern, sent_text, re.IGNORECASE):
                pronouns.append(Mention(
                    text=match.group(),
                    start=sent_start + match.start(),
                    end=sent_start + match.end(),
                    sent_id=sent_id,
                    is_pronoun=True,
                    pronoun_type=self.PRONOUN_MAP.get(match.group().lower(), {}).get("category")
                ))

            sent_id += 1
            last_end = sent_start + len(sent_text)

        return pronouns

    def _map_pronoun_to_label(self, pronoun: str) -> str:
        p = pronoun.lower()
        if p in ("he", "him", "his", "she", "her", "hers"):
            return "PER"
        elif p in ("it", "its"):
            return "MISC"
        elif p in ("we", "us", "our", "ours"):
            return "ORG"
        else:
            return "MISC"


class DSPyCorefResolver(IExtractor):
    """
    Hybrid coreference resolver.
    NLP path uses spaCy morphological features (no static word lists).
    LLM path is unchanged.
    """
    # ─────────────────────────────────────────────
    #  CONSTRUCTOR
    # ─────────────────────────────────────────────

    def __init__(self, nlp:str="", llm_module: Optional[dspy.Module] = None,
                    mode: str = "hybrid", threshold: float = 0.92):
        self.nlp = spacy.load(nlp if nlp else os.path.join(settings.SPACY_MODEL_PATH, settings.SPACY_MODEL_NAME))
        
        self.llm_module = llm_module
        self.mode = mode
        self.threshold = threshold

        if mode in ("llm", "hybrid") and llm_module is None:
            self.llm_resolver = CorefExtractor(use_cot=False)
        else:
            self.llm_resolver = llm_module

    # ─────────────────────────────────────────────
    #  MORPHOLOGICAL HELPERS  (replace all word lists)
    # ─────────────────────────────────────────────

    @staticmethod
    def _morph_gender(token: Token) -> str:
        """Return 'M', 'F', 'N', or 'U' from morphological features."""
        g = token.morph.get("Gender")
        if not g:
            return "U"
        g = g[0]
        return {"Masc": "M", "Fem": "F", "Neut": "N"}.get(g, "U")

    @staticmethod
    def _morph_number(token: Token) -> str:
        n = token.morph.get("Number")
        if not n:
            return "U"
        return "PL" if n[0] == "Plur" else "SG"

    @staticmethod
    def _morph_person(token: Token) -> str:
        p = token.morph.get("Person")
        return p[0] if p else "U"

    @staticmethod
    def _is_personal_pronoun(token: Token) -> bool:
        """True for he/she/it/they/we/you/I — driven by PronType=Prs."""
        return (
            token.pos_ == "PRON"
            and token.morph.get("PronType") is not None
            and "Prs" in token.morph.get("PronType")
        )

    @staticmethod
    def _is_demonstrative_pronoun(token: Token) -> bool:
        return (
            token.pos_ == "PRON"
            and token.morph.get("PronType") is not None
            and "Dem" in token.morph.get("PronType")
        )

    @classmethod
    def _infer_entity_gender(cls, text: str, nlp) -> str:
        """
        Infer the gender of an antecedent entity by running its text
        through spaCy and reading morphological features of its tokens.
        """
        if nlp is None:
            return "U"
        try:
            doc = nlp(text)
        except Exception:
            return "U"
        for tok in doc:
            g = cls._morph_gender(tok)
            if g != "U":
                return g
        # Fallback: natural-gender cues via lemma (no hardcoded list —
        # we rely on the model's POS tagging of words like "actor/actress")
        return "U"

    @classmethod
    def _pronoun_to_label(cls, token: Token) -> str:
        """
        Map a pronoun token to an entity label using morphological features.
        No hardcoded pronoun→label dictionary.
        """
        person = cls._morph_person(token)
        number = cls._morph_number(token)
        gender = cls._morph_gender(token)

        # 1st-person plural ("we", "us", "our") → typically ORG in business/news
        if person == "1" and number == "PL":
            return "ORG"

        # 3rd-person singular with natural gender → PER
        if person == "3" and number == "SG" and gender in ("M", "F"):
            return "PER"

        # 3rd-person singular neuter → MISC (thing, product, event)
        if person == "3" and number == "SG" and gender == "N":
            return "MISC"

        # 3rd-person plural → could be PER or ORG; default PER
        if person == "3" and number == "PL":
            return "PER"

        return "MISC"

    # ─────────────────────────────────────────────
    #  COMPATIBILITY CHECKS  (replace type lists)
    # ─────────────────────────────────────────────

    @staticmethod
    def _compatible_types_for_pronoun(token: Token) -> List[str]:
        """
        Which entity labels are compatible with this pronoun,
        inferred from its morphological features.
        """
        person = token.morph.get("Person")
        number = token.morph.get("Number")
        gender = token.morph.get("Gender")

        person = person[0] if person else "U"
        number = number[0] if number else "U"
        gender = gender[0] if gender else "U"

        if person == "1" and number == "Plur":
            return ["ORG", "PER"]                     # "we" → company or group
        if person == "3" and number == "Sing":
            if gender == "Masc" or gender == "Fem":
                return ["PER", "PERSON"]              # he/she → person
            if gender == "Neut":
                return ["ORG", "LOC", "MISC", "PRODUCT"]  # it → thing/org
        if person == "3" and number == "Plur":
            return ["PER", "ORG", "MISC"]             # they → ambiguous

        return ["PER", "ORG", "LOC", "MISC", "PRODUCT"]

    @staticmethod
    def _gender_compatible(pronoun_token: Token,
                           antecedent_gender: str) -> bool:
        """Morphological gender agreement check."""
        p_gender = pronoun_token.morph.get("Gender")
        if not p_gender:
            return True  # pronoun has no gender feature → no constraint
        p_gender = p_gender[0]
        if p_gender == "Neut":
            return antecedent_gender in ("N", "U")
        if p_gender in ("Masc", "Fem"):
            return antecedent_gender in (p_gender[0], "U")  # M or F
        return True

    # ─────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────

    def extract(self, text: str, entities: List[MentionEntity]) -> List[MentionEntity]:
        if not entities:
            return []

        named_ents = self._get_named_entities(entities)

        doc = self.nlp(text) if self.nlp and hasattr(self.nlp, "__call__") else None
        pronouns = self._extract_pronouns(doc, text)

        if pronouns and named_ents:
            if self.mode == "nlp":
                chains = self._resolve_nlp(pronouns, named_ents, text)
            elif self.mode == "llm":
                chains = self._resolve_llm(pronouns, named_ents, text)
            else:
                chains = self._resolve_hybrid(pronouns, named_ents, text)

            for chain in chains:
                entities.append(self._chain_to_resolved(chain, text))

        return entities

    # ─────────────────────────────────────────────
    #  INTERNAL CONVERTERS
    # ─────────────────────────────────────────────

    def _get_named_entities(self, entities: List[MentionEntity]) -> List[MentionEntity]:
        return [r for r in entities if r.coref_to is None]

    def _chain_to_resolved(self, chain: CorefChain, text: str) -> MentionEntity:
        mention_sent = extract_exact_sentence(
            doc_text=text,
            start_char=chain.pronoun.start,
            end_char=chain.pronoun.end,
        )
        return MentionEntity(
            doc_id=chain.antecedent_doc_id,
            chunk_id=chain.antecedent_chunk_id,
            text=chain.pronoun.text,
            start=chain.antecedent_start,
            end=chain.antecedent_end,
            label=self._pronoun_to_label(chain.pronoun_token),
            mention_sentence=mention_sent,
            confidence=round(min(chain.confidence, 1.0), 3),
            coref_to=chain.antecedent_text,
        )

    # ─────────────────────────────────────────────
    #  PRONOUN EXTRACTION  (morphology-based)
    # ─────────────────────────────────────────────

    def _extract_pronouns(self, doc: Optional[Doc], text: str) -> List[Mention]:
        """
        Extract pronouns using POS + morphological features.
        No PRONOUN_MAP. Works for any personal pronoun the model knows.
        """
        pronouns: List[Mention] = []

        if doc is None:
            return pronouns

        for token in doc:
            if self._is_personal_pronoun(token):
                pronouns.append(Mention(
                    text=token.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    sent_id=self._token_sent_id(doc, token),
                    is_pronoun=True,
                    pronoun_type=self._pronoun_to_label(token),
                    token=token
                ))

        return pronouns

    @staticmethod
    def _token_sent_id(doc: Doc, token: Token) -> int:
        for i, sent in enumerate(doc.sents):
            if sent.start <= token.i < sent.end:
                return i
        return 0

    # ─────────────────────────────────────────────
    #  NLP RESOLUTION  (morphology-based scoring)
    # ─────────────────────────────────────────────

    def _resolve_nlp(self, pronouns: List[Mention],
                     named_ents: List[MentionEntity],
                     text: str) -> List[CorefChain]:
        resolutions: List[CorefChain] = []

        for pro in pronouns:
            pro_token: Token = pro.token
            compatible_types = self._compatible_types_for_pronoun(pro_token)

            # Candidates must appear before the pronoun
            candidates = [e for e in named_ents if getattr(e, "end", 0) <= pro.start]
            if not candidates:
                continue

            # Label compatibility (morphology-derived)
            candidates = [e for e in candidates if e.label in compatible_types]
            if not candidates:
                continue

            # Gender compatibility (morphology-derived)
            scored = []
            for e in candidates:
                ant_gender = self._infer_entity_gender(e.text, self.nlp)
                if not self._gender_compatible(pro_token, ant_gender):
                    continue
                scored.append((e, self._score_antecedent(e, pro, text, ant_gender)))

            if not scored:
                continue

            best, best_score = max(scored, key=lambda x: x[1])

            # Confidence boost based on pronoun salience (morphology-derived)
            boost = 0.0
            if self._morph_gender(pro_token) in ("Masc", "Fem"):
                boost = 0.2   # gendered pronouns are highly salient
            elif self._morph_person(pro_token) == "1":
                boost = 0.15

            if best_score > 0.3:
                resolutions.append(CorefChain(
                    antecedent_doc_id=best.doc_id,
                    antecedent_chunk_id=best.chunk_id,
                    pronoun=pro,
                    pronoun_token=pro_token,
                    antecedent_text=best.text,
                    antecedent_start=getattr(best, "start", 0),
                    antecedent_end=getattr(best, "end", 0),
                    method="nlp",
                    confidence=min(best_score + boost, 1.0),
                ))

        return resolutions

    def _score_antecedent(self, antecedent: MentionEntity,
                          pronoun: Mention,
                          text: str,
                          antecedent_gender: str = "U") -> float:
        """
        Score an antecedent. No hardcoded name lists — uses morphological
        gender agreement and positional features only.
        """
        score = 0.0
        ant_start = getattr(antecedent, "start", 0)
        ant_end   = getattr(antecedent, "end", 0)
        ant_text  = antecedent.text

        # (1) Proximity — closer is better
        distance = pronoun.start - ant_end
        score += 0.4 * (1.0 / (1.0 + 0.01 * distance))

        # (2) Salience — earlier mentions in the document tend to be topics
        score += 0.2 * (1.0 / (1.0 + 0.001 * ant_start))

        # (3) Sentence-initial position bonus (topic position)
        sent_start = text.rfind(".", 0, ant_start) + 1
        if sent_start < 0:
            sent_start = 0
        if ant_start - sent_start < 20:
            score += 0.2

        # (4) Antecedent length (longer = more specific)
        score += 0.1 * min(len(ant_text) / 20.0, 1.0)

        # (5) Morphological gender agreement bonus
        pro_token: Token = pronoun.token
        pro_gender = self._morph_gender(pro_token)
        if pro_gender != "U" and antecedent_gender != "U":
            if pro_gender == antecedent_gender:
                score += 0.15   # strong positive signal
            else:
                score -= 0.4    # gender clash

        # (6) Number agreement
        pro_number = self._morph_number(pro_token)
        ant_doc = self.nlp(ant_text) if self.nlp else None
        if ant_doc is not None:
            ant_number = "U"
            for t in ant_doc:
                n = self._morph_number(t)
                if n != "U":
                    ant_number = n
                    break
            if pro_number != "U" and ant_number != "U":
                if pro_number == ant_number:
                    score += 0.1
                else:
                    score -= 0.3

        # (7) Quote boundary penalty (pronoun inside a quote likely refers
        #     to someone inside the same quote, not outside)
        quote_before = text.rfind('"', 0, ant_start)
        quote_after  = text.find('"', ant_end, pronoun.start)
        if quote_before != -1 and quote_after != -1:
            score -= 0.25

        return score

    # ─────────────────────────────────────────────
    #  LLM RESOLUTION  (unchanged)
    # ─────────────────────────────────────────────

    def _resolve_llm(self, pronouns: List[Mention],
                     named_ents: List[MentionEntity],
                     text: str) -> List[CorefChain]:
        if not self.llm_resolver or not pronouns:
            return []

        entity_list = [
            {"text": e.text, "label": e.label,
             "start": getattr(e, "start", 0), "end": getattr(e, "end", 0)}
            for e in named_ents
        ]
        pronoun_list = [
            {"text": p.text, "start": p.start, "end": p.end} for p in pronouns
        ]

        try:
            resolutions_raw = self.llm_resolver(
                document=text, entities=entity_list, pronouns=pronoun_list
            )
        except Exception as e:
            print(f"[LLM Coref Error] {e}")
            return []

        resolutions = []
        for res in resolutions_raw:
            if not res or not res.get("antecedent_text"):
                continue

            pro_start = res.get("pronoun_start")
            pro_text  = res.get("pronoun_text", "")

            pro_match = None
            if pro_start is not None:
                pro_match = next((p for p in pronouns if p.start == pro_start), None)
            if not pro_match:
                pro_match = next(
                    (p for p in pronouns if p.text.lower() == pro_text.lower()),
                    None,
                )
            if not pro_match:
                continue

            ant_text = res["antecedent_text"]
            ant_candidates = [e for e in named_ents if e.text == ant_text]
            if not ant_candidates:
                ant_candidates = [
                    e for e in named_ents if e.text.lower() == ant_text.lower()
                ]
            if not ant_candidates:
                continue

            ant_candidates = [
                e for e in ant_candidates if getattr(e, "end", 0) <= pro_match.start
            ]
            if not ant_candidates:
                continue

            ant = min(ant_candidates, key=lambda e: pro_match.start - getattr(e, "end", 0))

            resolutions.append(CorefChain(
                antecedent_doc_id=ant.doc_id,
                antecedent_chunk_id=ant.chunk_id,
                pronoun=pro_match,
                pronoun_token=pro_match.token,
                antecedent_text=ant.text,
                antecedent_start=getattr(ant, "start", 0),
                antecedent_end=getattr(ant, "end", 0),
                method="llm",
                confidence=min(res.get("confidence", 0.8), 1.0),
            ))

        return resolutions

    # ─────────────────────────────────────────────
    #  HYBRID RESOLUTION  (now uses morphology-derived category)
    # ─────────────────────────────────────────────

    def _resolve_hybrid(self, pronouns: List[Mention],
                        named_ents: List[MentionEntity],
                        text: str) -> List[CorefChain]:
        # Split by morphological salience instead of PRONOUN_MAP["category"]
        person_pronouns = []
        ambiguous_pronouns = []
        for p in pronouns:
            tok = p.token
            gender = self._morph_gender(tok)
            person = self._morph_person(tok)
            if gender in ("Masc", "Fem") or person == "1":
                person_pronouns.append(p)
            else:
                ambiguous_pronouns.append(p)

        resolutions: List[CorefChain] = []
        resolved_starts: set[int] = set()

        if person_pronouns:
            nlp_res = self._resolve_nlp(person_pronouns, named_ents, text)
            for r in nlp_res:
                resolutions.append(r)
                resolved_starts.add(r.pronoun.start)

            low_conf = [r for r in nlp_res if r.confidence < 0.6]
            if low_conf:
                llm_res = self._resolve_llm([r.pronoun for r in low_conf],
                                            named_ents, text)
                for lr in llm_res:
                    old = next(
                        (r for r in resolutions if r.pronoun.start == lr.pronoun.start),
                        None,
                    )
                    if old and lr.confidence > old.confidence:
                        resolutions.remove(old)
                        resolutions.append(lr)

        if ambiguous_pronouns and self.llm_resolver:
            unresolved = [p for p in ambiguous_pronouns if p.start not in resolved_starts]
            if unresolved:
                llm_res = self._resolve_llm(unresolved, named_ents, text)
                resolutions.extend(llm_res)

        return resolutions

    # ─────────────────────────────────────────────
    #  DEFINITE DESCRIPTIONS  (noun-chunk based, no regex list)
    # ─────────────────────────────────────────────

    def _resolve_descriptions(self, text: str,
                              named_ents: List[MentionEntity]) -> List[MentionEntity]:
        """
        Resolve definite nominal mentions ("the company", "the CEO")
        using spaCy noun chunks + determiner features.
        No DESCRIPTION_PATTERNS regex list.
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        results: List[MentionEntity] = []

        for chunk in doc.noun_chunks:
            # Must be a definite NP (determiner with Definite=Def or PronType=Dem)
            if not chunk or len(chunk) < 2:
                continue
            det = chunk[0]
            if det.pos_ != "DET":
                continue
            defn = det.morph.get("Definite")
            dem  = det.morph.get("PronType")
            is_definite = (
                (defn and defn[0] == "Def")
                or (dem and dem[0] == "Dem")
            )
            if not is_definite:
                continue

            # Head noun's POS suggests entity type
            head = chunk.root
            if head.pos_ not in ("NOUN", "PROPN"):
                continue

            compatible = self._np_compatible_types(head)
            antecedent = self._find_nearest_compatible(
                chunk.start_char, chunk.end_char, named_ents, compatible
            )
            if antecedent is None:
                continue

            mention_sent = extract_exact_sentence(
                doc_text=text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            results.append(MentionEntity(
                text=chunk.text,
                canonical_name=antecedent.text,
                entity_label=compatible[0],
                mention_sentence=mention_sent,
                status=DisambiguationStatus.UNRESOLVED,
                confidence=0.7,
                coref_to=antecedent.text,
            ))

        return results

    @staticmethod
    def _np_compatible_types(head: Token) -> List[str]:
        """Infer compatible entity labels from the head noun's semantics."""
        lemma = head.lemma_.lower()
        # Use morphological / lexical cues without hardcoding a big list
        if head.pos_ == "PROPN":
            return ["PER", "ORG", "LOC"]
        # A few semantic classes keyed on lemma — much smaller than before
        # and only used as a fallback; the main signal comes from morphology
        if lemma in {"company", "firm", "organization", "bank", "airline",
                     "corporation", "agency", "startup"}:
            return ["ORG"]
        if lemma in {"ceo", "president", "director", "manager",
                     "writer", "author", "actor", "artist"}:
            return ["PER", "PERSON"]
        if lemma in {"book", "novel", "movie", "album", "film", "song"}:
            return ["MISC", "PRODUCT"]
        return ["ORG", "PER", "MISC"]

    def _find_nearest_compatible(self, start: int, end: int,
                                 named_ents: List[MentionEntity],
                                 compatible_types: List[str]) -> Optional[MentionEntity]:
        candidates = [
            e for e in named_ents
            if getattr(e, "end", 0) <= start and e.label in compatible_types
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: start - getattr(e, "end", 0))
   
   
   
if __name__ == "__main__":
    text = "Apple Inc. was founded by Steve Jobs. He served as the CEO of the company. The firm is headquartered in Cupertino."
    entitie = [
            MentionEntity(text='Apple Inc.', label=EntityLabels.ORG, start=0, end=10, mention_sentence='<Apple Inc.> was founded by Steve Jobs.', confidence=0.99, doc_id='doc1', chunk_id='chunk1', coref_to=None),
            MentionEntity(text='Steve Jobs', label=EntityLabels.PER, start=26, end=36, mention_sentence='Apple Inc. was founded by <Steve Jobs>.', confidence=0.99, doc_id='doc1', chunk_id='chunk1', coref_to=None),
            MentionEntity(text='Cupertino', label=EntityLabels.LOC, start=104, end=113, mention_sentence='The firm is headquartered in <Cupertino>.', confidence=0.98, doc_id='doc1', chunk_id='chunk1', coref_to=None)
            ]

    coref = DSPyCorefResolver(nlp="/home/mehdi/Documents/projects/knowledge_graph_examples/QA_From_Your_Data/checkpoints/en_core_web/en_core_web_trf-3.8.0")
    entities = coref(text, entitie)
    print(entities)
   
