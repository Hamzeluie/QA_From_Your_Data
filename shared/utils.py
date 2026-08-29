from typing import List, Dict, Optional, Tuple
import os
import re
import requests
import json
from datetime import datetime, timedelta
from shared.data_classes import ResolvedEntity, DisambiguationStatus, EntityLabels
from sentence_transformers import SentenceTransformer
import spacy
from config.settings import settings
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from dateutil import parser as date_parser
try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

NON_LINKABLE_TYPES = {"DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "ORDINAL", "QUANTITY", "NUMBER", "NUM"}

def jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def levenshtein_ratio(s1: str, s2: str) -> float:
    def _dist(a, b):
        if len(a) < len(b): return _dist(b, a)
        if len(b) == 0: return len(a)
        prev = range(len(b) + 1)
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca!=cb)))
            prev = curr
        return prev[-1]
    d = _dist(s1, s2)
    m = max(len(s1), len(s2))
    return 1.0 - (d / m) if m > 0 else 1.0

def format_data(df):
    my_dict = {row["original_text"]: {"canonical_name": row["canonical_name"], "entity_label": row["entity_label"], "aliases": [], "context_indicators": [], "related_to": [], "fetch_wiki": True, "wiki_summary": row["wiki_summary"], "wiki_url": row["wiki_url"]} for _, row in df.iterrows()}
    return json.dumps(my_dict, indent=4)

def semantic_sentence_chunk(text, model:SentenceTransformer, nlp:spacy=None, use_nlp:bool=False, threshold:float=0.1):
    # Split sentences (handles ".He" in raw text)
    
    if use_nlp and nlp is not None:
        try:
            doc = nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        except:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\.(?=[A-Z])', text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\.(?=[A-Z])', text)
    
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = sk_cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]

        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_exact_sentence(
    doc_text: str,
    start_char: int,
    end_char: int,
    use_nlp: bool = True,
    nlp: Optional[spacy.language.Language] = None,
    marker_start: str = "<",
    marker_end: str = ">",
) -> str:
    """
    Extract the sentence containing the entity [start_char:end_char] and
    wrap the entity with marker_start / marker_end.
    """
    span = _find_sentence_span(doc_text, start_char, end_char, use_nlp, nlp)

    if span is None:
        # Ultimate fallback: treat whole text as the sentence
        sent_start, sentence = 0, doc_text
    else:
        sent_start, sent_end = span
        sentence = doc_text[sent_start:sent_end]

    # Entity offsets relative to the extracted sentence
    rel_start = max(0, start_char - sent_start)
    rel_end = max(rel_start, end_char - sent_start)

    highlighted = (
        sentence[:rel_start]
        + marker_start
        + sentence[rel_start:rel_end]
        + marker_end
        + sentence[rel_end:]
    )
    return highlighted.strip()


def _find_sentence_span(
    doc_text: str,
    start_char: int,
    end_char: int,
    use_nlp: bool,
    nlp: Optional[spacy.language.Language],
) -> Optional[Tuple[int, int]]:
    """Return the (start, end) char span of the sentence containing the entity."""

    # 1. spaCy sentence segmentation
    if use_nlp and nlp is not None:
        try:
            doc = nlp(doc_text)
            for sent in doc.sents:
                if sent.start_char <= start_char and sent.end_char >= end_char:
                    return sent.start_char, sent.end_char
        except Exception:
            pass

    # 2. Regex fallback with position tracking
    return _find_sentence_span_regex(doc_text, start_char, end_char)


def _find_sentence_span_regex(
    doc_text: str, start_char: int, end_char: int
) -> Optional[Tuple[int, int]]:
    """Split into sentences using regex, but keep char offsets."""
    boundary = r'(?<=[.!?])\s+(?=[A-Z])|\.(?=[A-Z])'

    # Build sentence spans from boundary positions
    spans = []
    prev = 0
    for m in re.finditer(boundary, doc_text):
        spans.append((prev, m.start()))
        prev = m.end()
    spans.append((prev, len(doc_text)))

    # Exact containment
    for s, e in spans:
        if s <= start_char and e >= end_char:
            return s, e

    # Overlap fallback
    for s, e in spans:
        if e > start_char and s < end_char:
            return s, e

    return None


def _locate_mention(sentence: str, text: str, used_spans: set) -> tuple[int, int]:
    """Return the first (start, end) occurrence of `text` not already used."""
    for occ_start in _find_all_occurrences(sentence, text):
        occ_end = occ_start + len(text)
        if (occ_start, occ_end) not in used_spans:
            used_spans.add((occ_start, occ_end))
            return occ_start, occ_end
    return -1, -1


def _find_all_occurrences(text: str, substring: str) -> list[int]:
    """Return start indices of all non-overlapping occurrences."""
    if not substring:
        return []
    occurrences = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + len(substring)  # non-overlapping
    return occurrences

class WikipediaEntitySummarizer:
    LABEL_SIGNATURES = {
        EntityLabels.PER.value: [
            "born", "politician", "businessman", "businesswoman", "actor", "actress",
            "author", "scientist", "footballer", "musician", "singer", "CEO",
            "entrepreneur", "engineer", "inventor", "philanthropist", "artist",
            "is a ", "was a ", "is an ", "was an "
        ],
        EntityLabels.ORG.value: [
            "company", "corporation", "inc.", "ltd", "organization", "firm",
            "multinational", "headquartered in", "founded in", "subsidiary of",
            "publicly traded", "listed on", "stock exchange", "enterprise"
        ],
        EntityLabels.LOC.value: [
            "country", "city", "state", "capital", "republic", "kingdom",
            "province", "county", "municipality", "located in", "population of",
            "island", "continent", "territory"
        ],
        EntityLabels.FAC.value: [
            "airport", "bridge", "highway", "building", "station", "hospital",
            "university", "museum", "stadium", "located in", "built in"
        ],
        EntityLabels.PRODUCT.value: [
            "software", "device", "car", "phone", "game", "console", "product",
            "launched in", "released by", "developed by", "chatbot", "model",
            "series", "platform", "application", "app"
        ],
        EntityLabels.TECHNOLOGY.value: [
            "technology", "artificial intelligence", "machine learning",
            "deep learning", "neural network", "algorithm", "computational",
            "software framework", "model", "system", "platform", "architecture",
            "is a field of", "is a branch of", "is a type of"
        ],
        EntityLabels.EVENT.value: [
            "war", "battle", "conference", "festival", "olympics", "tournament",
            "held in", "took place", "anniversary", "celebration"
        ],
        EntityLabels.WORK_OF_ART.value: [
            "novel", "book", "film", "movie", "song", "album", "painting",
            "written by", "directed by", "composed by", "published in"
        ],
        EntityLabels.LAW.value: [
            "act", "treaty", "constitution", "amendment", "law", "bill",
            "signed into law", "ratified", "legal"
        ],
        EntityLabels.LANGUAGE.value: [
            "language", "dialect", "spoken in", "official language", "lingua franca"
        ],
        EntityLabels.DATE.value: [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ],
        EntityLabels.TIME.value: [
            "morning", "afternoon", "evening", "night", "midnight", "noon",
            "a.m.", "p.m.", "o'clock", "hour", "minute", "second"
        ],
        EntityLabels.MONEY.value: [
            "dollar", "euro", "pound", "yen", "usd", "eur", "gbp",
            "million", "billion", "trillion", "budget", "revenue", "cost"
        ],
        EntityLabels.PERCENT.value: [
            "percent", "percentage", "%", "proportion", "rate", "share"
        ],
        EntityLabels.QUANTITY.value: [
            "meter", "kilometer", "mile", "kilogram", "ton", "liter",
            "degree", "celsius", "fahrenheit", "inch", "foot", "pound"
        ],
        EntityLabels.CARDINAL.value: [
            "one", "two", "three", "hundred", "thousand", "million"
        ],
        EntityLabels.ORDINAL.value: [
            "first", "second", "third", "fourth", "fifth", "last"
        ],
        EntityLabels.NORP.value: [
            "american", "european", "asian", "african", "christian", "muslim",
            "jewish", "buddhist", "hindu", "democrat", "republican", "conservative",
            "liberal", "socialist", "nationality", "ethnic"
        ],
        EntityLabels.MISC.value: [
            "award", "honor", "title", "degree", "religion", "ideology",
            "culture", "tradition", "custom", "mythology", "legend"
        ],
        EntityLabels.NUM.value: [
            "number", "amount", "total", "sum", "count", "quantity"
        ],
    }

    def __init__(self, embedding:SentenceTransformer=None):
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "EntityLinkerBot/1.0"}
        if embedding:
            self.embedder = embedding
        else:
            if os.path.isdir(settings.EMBEDDING_MODEL_PATH):
                self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
            else:
                self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                self.embedder.save(settings.EMBEDDING_MODEL_PATH)
        

    def search_wikipedia(self, entity: str, limit: int = 5) -> List[Dict]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": entity,
            "srlimit": limit,
            "format": "json"
        }
        try:
            response = requests.get(self.wiki_api, params=params,
                                   headers=self.headers, timeout=10)
            data = response.json()
        except Exception:
            return []

        candidates = []
        for result in data.get("query", {}).get("search", []):
            title = result["title"]
            snippet = self._clean_html(result.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            candidates.append({
                "title": title,
                "pageid": result["pageid"],
                "snippet": snippet,
                "url": url
            })
        return candidates

    def _clean_html(self, text: str) -> str:
        text = re.sub(r'<span class="searchmatch">(.*?)</span>', r'\1', text)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def _rank_candidates(self,
                         entity: str,
                         context: str,
                         candidates: List[Dict],
                         ner_label: Optional[str] = None) -> Optional[Dict]:
        if not candidates:
            return None

        label_hints = {
            "ORG": ["Inc.", "Company", "Corporation", "Ltd", "Group", "Airlines", "Bank"],
            "PERSON": ["born", "politician", "actor", "author", "scientist", "footballer"],
            "GPE": ["country", "city", "state", "capital", "republic", "kingdom"],
            "PRODUCT": ["software", "device", "car", "phone", "game", "console"],
            "TECHNOLOGY": ["software", "algorithm", "intelligence", "learning", "network", "model"],
        }

        context_vec = None
        if self.embedder and context:
            context_vec = self.embedder.encode([context], convert_to_numpy=True)

        scored = []
        for cand in candidates:
            score = 0.0
            snippet = cand["snippet"]
            title = cand["title"]

            if context_vec is not None and snippet:
                snippet_vec = self.embedder.encode([snippet], convert_to_numpy=True)
                sim = float(sk_cosine_similarity(context_vec, snippet_vec)[0][0])
                score += sim * 0.6

            context_words = set(context.lower().split())
            candidate_text = (title + " " + snippet).lower()
            overlap = len(context_words & set(candidate_text.split()))
            score += (overlap / max(len(context_words), 1)) * 0.3

            if ner_label and ner_label in label_hints:
                if any(h.lower() in candidate_text for h in label_hints[ner_label]):
                    score += 0.1

            if "disambiguation" in title.lower():
                score -= 0.5

            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def extract_summary(self, title: str, max_sentences: int = 2) -> str:
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "exintro": True,
            "exsentences": max_sentences,
            "explaintext": True,
            "format": "json"
        }
        try:
            response = requests.get(self.wiki_api, params=params,
                                   headers=self.headers, timeout=10)
            data = response.json()
        except Exception:
            return ""

        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            extract = page_data.get("extract", "")
            if extract:
                return self._normalize_summary(extract)
        return ""

    def _normalize_summary(self, text: str) -> str:
        text = re.sub(r'\s*\([^)]*(disambiguation|company|fruit|disambiguation page)[^)]*\)', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'For (other uses|the company|the fruit)[,.].*?(?=\n|$)', '', text)
        text = " ".join(text.split())
        return text.strip()

    def classify_from_summary(self, summary: str, fallback_label: Optional[str] = None) -> str:
        """
        Scan the Wikipedia summary for label-indicative keywords.
        Returns the best-matching label or the fallback.
        """
        if not summary:
            return fallback_label or "UNKNOWN"

        summary_lower = summary.lower()
        scores = {}

        for ent_label, phrases in self.LABEL_SIGNATURES.items():
            score = sum(1 for phrase in phrases if phrase.lower() in summary_lower)
            if score:
                scores[ent_label] = score

        if scores:
            return max(scores, key=scores.get)

        # Heuristic: if the summary mentions years of birth/death, it's likely a person
        if re.search(r'\b\d{4}\s*–\s*\d{4}\b', summary) or "born" in summary_lower:
            return "PERSON"

        return fallback_label or "UNKNOWN"

    def summarize(self,
                  entity: str,
                  context: str,
                  ner_label: Optional[str] = None) -> Optional[ResolvedEntity]:
        candidates = self.search_wikipedia(entity)
        if not candidates:
            return None

        best = self._rank_candidates(entity, context, candidates, ner_label)
        if not best:
            return None

        summary = self.extract_summary(best["title"])

        # ── INFER LABEL FROM SUMMARY (override spaCy guess if confident) ──
        wiki_inferred_label = self.classify_from_summary(summary, fallback_label=ner_label)

        confidence = 0.5
        if self.embedder and summary:
            ctx_vec = self.embedder.encode([context], convert_to_numpy=True)
            sum_vec = self.embedder.encode([summary], convert_to_numpy=True)
            confidence = float(sk_cosine_similarity(ctx_vec, sum_vec)[0][0])

        if confidence >= 0.5:
            status = DisambiguationStatus.NEW_ENTITY
        else:
            status = DisambiguationStatus.UNKNOWN

        return ResolvedEntity(
            original_text=entity,
            canonical_name=best["title"],
            entity_label=wiki_inferred_label,
            mention_sentence=context,
            confidence=confidence,
            status=status,
            source="wikipedia",
            summary=summary,
            context_clues=[
                f"Wikipedia match: {best['title']}",
                f"Label inferred from summary: {wiki_inferred_label}"
            ],
            needs_review=True,
        )
        
    def _summarize(self,
                  entity: str,
                  context: str,
                  ner_label: Optional[str] = None) -> Optional[ResolvedEntity]:
        candidates = self.search_wikipedia(entity)
        if not candidates:
            return None

        best = self._rank_candidates(entity, context, candidates, ner_label)
        if not best:
            return None

        summary = self.extract_summary(best["title"])

        confidence = 0.5
        if self.embedder and summary:
            ctx_vec = self.embedder.encode([context], convert_to_numpy=True)
            sum_vec = self.embedder.encode([summary], convert_to_numpy=True)
            confidence = float(sk_cosine_similarity(ctx_vec, sum_vec)[0][0])
        
        if confidence >= 0.7:
            status = DisambiguationStatus.RESOLVED
            needs_review = False
        else:
            status = DisambiguationStatus.UNKNOWN 
            needs_review = True

            
        return ResolvedEntity(
            original_text=entity,
            canonical_name=best["title"],
            entity_label=ner_label or "UNKNOWN",
            mention_sentence=context,
            confidence=confidence,
            status=status,
            source="wikipedia",
            wiki_summary=summary,
            wiki_url=best["url"],
            context_clues=[f"Wikipedia match: {best['title']}"],
            needs_review=needs_review,
        )


class ValueNormalizer:
    RELATIVE_DATES = {
        "today": datetime.now().strftime("%Y-%m-%d"),
        "yesterday": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "tomorrow": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
    }

    CURRENCY_SYMBOLS = {
        '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
        '₹': 'INR', '₩': 'KRW', '₽': 'RUB', 'A$': 'AUD',
        'C$': 'CAD', 'CHF': 'CHF', 'kr': 'SEK', 'zł': 'PLN'
    }

    CURRENCY_NAMES = {
        'dollar': 'USD', 'dollars': 'USD', 'euro': 'EUR', 'euros': 'EUR',
        'pound': 'GBP', 'pounds': 'GBP', 'yen': 'JPY', 'rupee': 'INR',
        'rupees': 'INR', 'won': 'KRW', 'franc': 'CHF', 'krona': 'SEK',
        'zloty': 'PLN', 'rial': 'IRR', 'rials': 'IRR'
    }

    MULTIPLIERS = {
        'thousand': 1e3, 'million': 1e6, 'billion': 1e9,
        'trillion': 1e12, 'k': 1e3, 'm': 1e6, 'b': 1e9
    }

    @staticmethod
    def normalize_date(text: str, doc_date: str = None) -> dict:
        text_clean = text.strip().lower()

        if text_clean in ValueNormalizer.RELATIVE_DATES:
            return {
                "canonical": ValueNormalizer.RELATIVE_DATES[text_clean],
                "granularity": "day",
                "type": "DATE"
            }

        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match and len(text.strip()) <= 6:
            return {
                "canonical": year_match.group(0),
                "granularity": "year",
                "type": "DATE"
            }

        if DATEUTIL_AVAILABLE:
            try:
                parsed = date_parser.parse(text, fuzzy=True, default=datetime(2000, 1, 1))
                return {
                    "canonical": parsed.strftime("%Y-%m-%d"),
                    "granularity": "day",
                    "type": "DATE"
                }
            except Exception:
                pass

        decade_match = re.search(r'\b(19|20)\d0s\b', text)
        if decade_match:
            return {
                "canonical": decade_match.group(0),
                "granularity": "decade",
                "type": "DATE"
            }

        return {"canonical": text, "granularity": "unknown", "type": "DATE"}

    @staticmethod
    def normalize_money(text: str) -> dict:
        text_clean = text.strip().replace(',', '')
        currency = None

        for sym, code in ValueNormalizer.CURRENCY_SYMBOLS.items():
            if text_clean.startswith(sym):
                currency = code
                text_clean = text_clean[len(sym):].strip()
                break

        if not currency:
            words = text_clean.lower().split()
            for word in words:
                if word in ValueNormalizer.CURRENCY_NAMES:
                    currency = ValueNormalizer.CURRENCY_NAMES[word]
                    text_clean = text_clean.lower().replace(word, '').strip()
                    break

        if not currency:
            currency = "UNKNOWN"

        number_match = re.search(r'[\d\.]+', text_clean)
        if not number_match:
            return {"currency": currency, "value": None, "canonical": text}

        value = float(number_match.group())
        text_lower = text_clean.lower()
        for word, mult in ValueNormalizer.MULTIPLIERS.items():
            if word in text_lower:
                value *= mult
                break

        return {
            "currency": currency,
            "value": value,
            "canonical": f"{currency} {value:.0f}"
        }


