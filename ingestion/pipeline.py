import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from typing import List, Tuple, Optional
import pandas as pd

from storage.factory import StorageFactory
from ingestion.unified_resolver import UnifiedEntityResolver
from ingestion.relation_pipeline import RelationPipeline
from shared.data_classes import Chunk
from ingestion.models.base import IExtractor


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    High-level orchestrator for the full ingestion flow:
    NER → Entity Resolution → Relation Extraction → Chunk Indexing.
    Stateless; all side effects go through StorageFactory.
    """

    def __init__(
        self,
        ner_model:IExtractor,
        coref_model:IExtractor,
        relation_extractor:IExtractor,
        factory: Optional[StorageFactory] = None,
        embedder=None,
    ):
        self.factory = factory or StorageFactory.from_env()
        self.embedder = embedder or self._load_embedder()

        # Ensure backends are ready
        self.factory.init_all()

        self.resolver = UnifiedEntityResolver(
            ner_model=ner_model,
            coref_model=coref_model,
            factory=self.factory,
            embedder=self.embedder,
        )
        self.relation_pipeline = RelationPipeline(
            extractor=relation_extractor,
            factory=self.factory,
        )

    def run(
        self,
        doc_id: str,
        raw_text: str,
        owner_id: str,
    ) -> dict:
        """
        Synchronous pipeline run. Returns result manifest.
        For async production use, call this from a Celery task.
        """
        # 1. Document state: uploaded
        existing = self.factory.postgres.get_document_state(doc_id)
        if existing and existing.get("status") == "indexed":
            logger.info(f"Document {doc_id} already indexed; skipping.")
            return {"status": "skipped", "reason": "already_indexed", "doc_id": doc_id}

        if not existing:
            self.factory.postgres.create_document(doc_id, owner_id)

        # 2. Entity Resolution (NER + Coref + ER + Cross-doc + Merge)
        chunk_info, clean_df, review_df = self.resolver.process_document(
            doc_id, raw_text, owner_id
        )

        # 3. Relation Extraction (only on resolved)
        rel_df = pd.DataFrame()
        if not clean_df.empty:
            rel_df = self.relation_pipeline.extract_and_store(clean_df)

        # 4. Index chunks for hybrid search (ES + Qdrant)
        self._index_chunks(chunk_info)

        # 5. Final state
        self.factory.postgres.transition_state(doc_id, "re_done", "indexed")

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_indexed": len(chunk_info),
            "resolved_count": len(clean_df),
            "review_count": len(review_df),
            "relations_count": len(rel_df),
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        from config.settings import settings
        model_path = getattr(settings, "EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_path)

    def _index_chunks(self, chunks: List[Chunk]) -> None:
        """Dense (Qdrant) + sparse (ES) indexing of text chunks."""
        if not chunks:
            return

        texts = [c.sentence for c in chunks]
        try:
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.error(f"Chunk embedding failed: {exc}")
            return

        for chunk, vec in zip(chunks, embeddings):
            # Qdrant dense vector
            try:
                self.factory.qdrant.upsert_chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=vec.tolist(),
                    text=chunk.sentence,
                    owner_id=chunk.owner_id,
                )
            except Exception as exc:
                logger.warning(f"Qdrant chunk index failed {chunk.chunk_id}: {exc}")

            # Elasticsearch sparse index
            try:
                self.factory.es.index_chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.sentence,
                    owner_id=chunk.owner_id,
                )
            except Exception as exc:
                logger.warning(f"ES chunk index failed {chunk.chunk_id}: {exc}")
                
if __name__ == "__main__":
    from config.settings import settings
    from ingestion.models.factory import get_ner_extractor, get_coref_resolver, get_relation_extractor
    is_demo = True
    if is_demo:
        ner = None
        coref = None
        relation_extractor = None
    else:
        ner = get_ner_extractor(backend="bert", model_name=settings.BERT_NER_MODEL_NAME, local_dir=str(Path(settings.BERT_MODEL_PATH) / "ner"), use_onnx=True, onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "ner"))
        coref = get_coref_resolver(backend="bert", nlp=settings.SPACY_MODEL_PATH)
        relation_extractor = get_relation_extractor("bert", model_name="distilbert-base-uncased", local_dir=str(Path(settings.BERT_MODEL_PATH) / "best_re"), use_onnx=False, onnx_dir=str(Path(settings.BERT_ONNX_MODEL_PATH) / "onnx_models" / "best_onnx_re"), threshold=0)
        
    resolver = UnifiedEntityResolver(ner_model=ner, coref_model=coref, run_demo=is_demo)  # Assuming UnifiedResolver is the class name
    # resolver = UnifiedEntityResolver(ner_model=None, coref_model=None, run_demo=True)  # Assuming UnifiedResolver is the class name
    doc_id = "doc1"
    record = {"doc_id": 6, "document": "Ezri Dax ( ) is a fictional character who appears in the of the American science fiction TV series .Portrayed by Nicole de Boer , she is a counselor aboard the Bajoran space station Deep Space Nine .The character is a member of the Trill species , and is formed of both a host and a symbiont – referred to as Dax .Ezri was introduced to the series following the death of the previous Dax host , Jadzia ( Terry Farrell ) at the end of .It had been the producers ' intention to introduce a new female character bearing the symbiont in order to ensure that Nana Visitor as Kira Nerys was not the only female member of the main cast .There were difficulties in casting initially , and the character changed from one who was intended to be \" spooky \" to one that was struggling to deal with all her previous personalities as a result of unexpectedly taking on the Dax symbiont .De Boer was not considered for the part until co - producer Hans Beimler suggested that she should submit an audition tape , which resulted in her invitation to meet with the producers in Los Angeles and in her gaining the role .The character made her first appearance in the first episode of the seventh season , \" Image in the Sand \" .The character continued to appear throughout the final season of the series , with her final appearance in the series finale \" What You Leave Behind \" .Her character stepped into the void left by Jadzia amongst the crew , but found that she had to redevelop those previous relationships and learn to get along with Jadzia 's widower , Worf ( Michael Dorn ) .During the course of the season , Ezri becomes less nervous of her role over time and learns from the Dax symbiont and becomes involved romantically with Dr. Julian Bashir ( Alexander Siddig ) .The fan reaction to the character was reported as positive , but several of the Ezri - centric episodes came in for criticism , with producer Ira Steven Behr apologising to de Boer for \" \" – an episode described as \" just a mess \" by writer Ronald D. Moore .The relationship between Ezri and both Worf and Bashir was described as one of five \" great geek TV love triangles \" .The inclusion of the character was criticised on the internet , with Ezri being referred to as both an \" ill - conceived idea \" and a \" replacement Dax \" .", "sentence": [["Ezri", "Dax", "(", ")", "is", "a", "fictional", "character", "who", "appears", "in", "the", "of", "the", "American", "science", "fiction", "TV", "series", "."], ["Portrayed", "by", "Nicole", "de", "Boer", ",", "she", "is", "a", "counselor", "aboard", "the", "Bajoran", "space", "station", "Deep", "Space", "Nine", "."], ["The", "character", "is", "a", "member", "of", "the", "Trill", "species", ",", "and", "is", "formed", "of", "both", "a", "host", "and", "a", "symbiont", "–", "referred", "to", "as", "Dax", "."], ["Ezri", "was", "introduced", "to", "the", "series", "following", "the", "death", "of", "the", "previous", "Dax", "host", ",", "Jadzia", "(", "Terry", "Farrell", ")", "at", "the", "end", "of", "."], ["It", "had", "been", "the", "producers", "'", "intention", "to", "introduce", "a", "new", "female", "character", "bearing", "the", "symbiont", "in", "order", "to", "ensure", "that", "Nana", "Visitor", "as", "Kira", "Nerys", "was", "not", "the", "only", "female", "member", "of", "the", "main", "cast", "."], ["There", "were", "difficulties", "in", "casting", "initially", ",", "and", "the", "character", "changed", "from", "one", "who", "was", "intended", "to", "be", "\"", "spooky", "\"", "to", "one", "that", "was", "struggling", "to", "deal", "with", "all", "her", "previous", "personalities", "as", "a", "result", "of", "unexpectedly", "taking", "on", "the", "Dax", "symbiont", "."], ["De", "Boer", "was", "not", "considered", "for", "the", "part", "until", "co", "-", "producer", "Hans", "Beimler", "suggested", "that", "she", "should", "submit", "an", "audition", "tape", ",", "which", "resulted", "in", "her", "invitation", "to", "meet", "with", "the", "producers", "in", "Los", "Angeles", "and", "in", "her", "gaining", "the", "role", "."], ["The", "character", "made", "her", "first", "appearance", "in", "the", "first", "episode", "of", "the", "seventh", "season", ",", "\"", "Image", "in", "the", "Sand", "\"", "."], ["The", "character", "continued", "to", "appear", "throughout", "the", "final", "season", "of", "the", "series", ",", "with", "her", "final", "appearance", "in", "the", "series", "finale", "\"", "What", "You", "Leave", "Behind", "\"", "."], ["Her", "character", "stepped", "into", "the", "void", "left", "by", "Jadzia", "amongst", "the", "crew", ",", "but", "found", "that", "she", "had", "to", "redevelop", "those", "previous", "relationships", "and", "learn", "to", "get", "along", "with", "Jadzia", "'s", "widower", ",", "Worf", "(", "Michael", "Dorn", ")", "."], ["During", "the", "course", "of", "the", "season", ",", "Ezri", "becomes", "less", "nervous", "of", "her", "role", "over", "time", "and", "learns", "from", "the", "Dax", "symbiont", "and", "becomes", "involved", "romantically", "with", "Dr.", "Julian", "Bashir", "(", "Alexander", "Siddig", ")", "."], ["The", "fan", "reaction", "to", "the", "character", "was", "reported", "as", "positive", ",", "but", "several", "of", "the", "Ezri", "-", "centric", "episodes", "came", "in", "for", "criticism", ",", "with", "producer", "Ira", "Steven", "Behr", "apologising", "to", "de", "Boer", "for", "\"", "\"", "–", "an", "episode", "described", "as", "\"", "just", "a", "mess", "\"", "by", "writer", "Ronald", "D.", "Moore", "."], ["The", "relationship", "between", "Ezri", "and", "both", "Worf", "and", "Bashir", "was", "described", "as", "one", "of", "five", "\"", "great", "geek", "TV", "love", "triangles", "\"", "."], ["The", "inclusion", "of", "the", "character", "was", "criticised", "on", "the", "internet", ",", "with", "Ezri", "being", "referred", "to", "as", "both", "an", "\"", "ill", "-", "conceived", "idea", "\"", "and", "a", "\"", "replacement", "Dax", "\"", "."]], "label_sents": [[{"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}, {"name": "Ezri", "sent_id": 3, "pos": [0, 1], "type": "PER"}, {"name": "Ezri", "sent_id": 13, "pos": [12, 13], "type": "PER"}, {"name": "Ezri", "sent_id": 12, "pos": [3, 4], "type": "PER"}, {"name": "Ezri", "sent_id": 11, "pos": [15, 16], "type": "PER"}, {"name": "Ezri", "sent_id": 10, "pos": [7, 8], "type": "PER"}, {"name": "Ezri Dax", "sent_id": 0, "pos": [0, 2], "type": "PER"}], [{"name": "American", "sent_id": 0, "pos": [14, 15], "type": "LOC"}], [{"name": "Nicole de Boer", "sent_id": 1, "pos": [2, 5], "type": "PER"}], [{"name": "Bajoran", "sent_id": 1, "pos": [12, 13], "type": "LOC"}], [{"name": "Deep Space Nine", "sent_id": 1, "pos": [15, 18], "type": "MISC"}], [{"name": "Trill", "sent_id": 2, "pos": [7, 8], "type": "MISC"}], [{"name": "Dax", "sent_id": 10, "pos": [20, 21], "type": "MISC"}, {"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "MISC"}, {"name": "Dax", "sent_id": 5, "pos": [41, 42], "type": "MISC"}], [{"name": "Dax", "sent_id": 2, "pos": [24, 25], "type": "PER"}, {"name": "Dax", "sent_id": 3, "pos": [12, 13], "type": "PER"}, {"name": "Dax", "sent_id": 13, "pos": [29, 30], "type": "PER"}], [{"name": "Terry Farrell", "sent_id": 3, "pos": [17, 19], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [29, 30], "type": "PER"}, {"name": "Jadzia", "sent_id": 9, "pos": [8, 9], "type": "PER"}, {"name": "Jadzia", "sent_id": 3, "pos": [15, 16], "type": "PER"}], [{"name": "Nana Visitor", "sent_id": 4, "pos": [21, 23], "type": "PER"}], [{"name": "Kira Nerys", "sent_id": 4, "pos": [24, 26], "type": "PER"}], [{"name": "De Boer", "sent_id": 6, "pos": [0, 2], "type": "PER"}, {"name": "de Boer", "sent_id": 11, "pos": [31, 33], "type": "PER"}], [{"name": "Hans Beimler", "sent_id": 6, "pos": [12, 14], "type": "PER"}], [{"name": "Los Angeles", "sent_id": 6, "pos": [34, 36], "type": "LOC"}], [{"name": "Image in the Sand", "sent_id": 7, "pos": [16, 20], "type": "MISC"}], [{"name": "What You Leave Behind", "sent_id": 8, "pos": [22, 26], "type": "MISC"}], [{"name": "Worf", "sent_id": 12, "pos": [6, 7], "type": "PER"}, {"name": "Worf", "sent_id": 9, "pos": [33, 34], "type": "PER"}], [{"name": "Michael Dorn", "sent_id": 9, "pos": [35, 37], "type": "PER"}], [{"name": "Julian Bashir", "sent_id": 10, "pos": [28, 30], "type": "PER"}], [{"name": "Alexander Siddig", "sent_id": 10, "pos": [31, 33], "type": "PER"}], [{"name": "Ira Steven Behr", "sent_id": 11, "pos": [26, 29], "type": "PER"}], [{"name": "Ronald D. Moore", "sent_id": 11, "pos": [48, 51], "type": "PER"}], [{"name": "Bashir", "sent_id": 12, "pos": [8, 9], "type": "PER"}], [{"name": "five", "sent_id": 12, "pos": [14, 15], "type": "NUM"}]], "label_doc": [{"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "Ezri", "label": "PER", "start": 314, "end": 318, "sents_id": 3}, {"text": "Ezri", "label": "PER", "start": 2207, "end": 2211, "sents_id": 13}, {"text": "Ezri", "label": "PER", "start": 2045, "end": 2049, "sents_id": 12}, {"text": "Ezri", "label": "PER", "start": 1842, "end": 1846, "sents_id": 11}, {"text": "Ezri", "label": "PER", "start": 1602, "end": 1606, "sents_id": 10}, {"text": "Ezri Dax", "label": "PER", "start": 0, "end": 8, "sents_id": 0}, {"text": "American", "label": "LOC", "start": 64, "end": 72, "sents_id": 0}, {"text": "Nicole de Boer", "label": "PER", "start": 113, "end": 127, "sents_id": 1}, {"text": "Bajoran", "label": "LOC", "start": 160, "end": 167, "sents_id": 1}, {"text": "Deep Space Nine", "label": "MISC", "start": 182, "end": 197, "sents_id": 1}, {"text": "Trill", "label": "MISC", "start": 232, "end": 237, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 1670, "end": 1673, "sents_id": 10}, {"text": "Dax", "label": "MISC", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "MISC", "start": 859, "end": 862, "sents_id": 5}, {"text": "Dax", "label": "PER", "start": 309, "end": 312, "sents_id": 2}, {"text": "Dax", "label": "PER", "start": 384, "end": 387, "sents_id": 3}, {"text": "Dax", "label": "PER", "start": 2286, "end": 2289, "sents_id": 13}, {"text": "Terry Farrell", "label": "PER", "start": 404, "end": 417, "sents_id": 3}, {"text": "Jadzia", "label": "PER", "start": 1525, "end": 1531, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 1406, "end": 1412, "sents_id": 9}, {"text": "Jadzia", "label": "PER", "start": 395, "end": 401, "sents_id": 3}, {"text": "Nana Visitor", "label": "PER", "start": 554, "end": 566, "sents_id": 4}, {"text": "Kira Nerys", "label": "PER", "start": 570, "end": 580, "sents_id": 4}, {"text": "De Boer", "label": "PER", "start": 873, "end": 880, "sents_id": 6}, {"text": "de Boer", "label": "PER", "start": 1935, "end": 1942, "sents_id": 11}, {"text": "Hans Beimler", "label": "PER", "start": 933, "end": 945, "sents_id": 6}, {"text": "Los Angeles", "label": "LOC", "start": 1061, "end": 1072, "sents_id": 6}, {"text": "Image in the Sand", "label": "MISC", "start": 1189, "end": 1206, "sents_id": 7}, {"text": "What You Leave Behind", "label": "MISC", "start": 1337, "end": 1358, "sents_id": 8}, {"text": "Worf", "label": "PER", "start": 2059, "end": 2063, "sents_id": 12}, {"text": "Worf", "label": "PER", "start": 1545, "end": 1549, "sents_id": 9}, {"text": "Michael Dorn", "label": "PER", "start": 1552, "end": 1564, "sents_id": 9}, {"text": "Julian Bashir", "label": "PER", "start": 1726, "end": 1739, "sents_id": 10}, {"text": "Alexander Siddig", "label": "PER", "start": 1742, "end": 1758, "sents_id": 10}, {"text": "Ira Steven Behr", "label": "PER", "start": 1904, "end": 1919, "sents_id": 11}, {"text": "Ronald D. Moore", "label": "PER", "start": 2003, "end": 2018, "sents_id": 11}, {"text": "Bashir", "label": "PER", "start": 2068, "end": 2074, "sents_id": 12}, {"text": "five", "label": "NUM", "start": 2099, "end": 2103, "sents_id": 12}]}

    document = record["document"]
    chunks, entities_df = resolver.process_document(doc_id=doc_id, document=document, owner_id="system")
    print(chunks)
    print(entities_df)