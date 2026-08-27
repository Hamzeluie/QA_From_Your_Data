import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from storage.base import AbstractTextSearch

logger = logging.getLogger(__name__)


class ElasticsearchEntitySearch(AbstractTextSearch):
    INDEX_NAME = "entities"
    CHUNKS_INDEX = "chunks"


    def __init__(self, hosts: List[str], username: Optional[str] = None, password: Optional[str] = None):
        if username and password:
            self.client = Elasticsearch(hosts, basic_auth=(username, password))
        else:
            self.client = Elasticsearch(hosts)

    def init_index(self) -> None:
        mapping = {
            "mappings": {
                "properties": {
                    "canonical": {"type": "keyword"},
                    "aliases": {"type": "text", "analyzer": "standard"},
                    "aliases_keyword": {"type": "keyword"},  # for exact term filters
                    "summary": {"type": "text", "analyzer": "standard"},
                    "context_indicators": {"type": "text", "analyzer": "standard"},
                    "related_to": {"type": "keyword"},
                    "label": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "updated_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }
        if not self.client.indices.exists(index=self.INDEX_NAME):
            self.client.indices.create(index=self.INDEX_NAME, body=mapping)
            logger.info(f"Created ES index: {self.INDEX_NAME}")

    def init_chunks_index(self) -> None:
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "owner_id": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                }
            },
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        }
        if not self.client.indices.exists(index=self.CHUNKS_INDEX):
            self.client.indices.create(index=self.CHUNKS_INDEX, body=mapping)
            logger.info(f"Created ES index: {self.CHUNKS_INDEX}")
            
    def index_entity(
        self,
        canonical: str,
        aliases: List[str],
        summary: str,
        context_indicators: List[str],
        related_to: List[str],
        label: str,
    ) -> None:
        doc = {
            "canonical": canonical,
            "aliases": aliases,
            "aliases_keyword": [a.lower() for a in aliases],
            "summary": summary,
            "context_indicators": context_indicators,
            "related_to": related_to,
            "label": label,
            "source": "catalog",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.client.index(index=self.INDEX_NAME, id=canonical, document=doc)
    
    def index_chunk(self, doc_id: str, chunk_id: str, text: str, owner_id: str) -> None:
        self.client.index(
            index=self.CHUNKS_INDEX,
            id=f"{doc_id}_{chunk_id}",
            document={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "owner_id": owner_id,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        )   

    def search_aliases(self, query: str, top_k: int = 10) -> List[Dict]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "aliases^3",
                        "aliases.keyword^5",
                        "canonical^2",
                        "summary",
                        "context_indicators^2",
                    ],
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                }
            },
        }
        resp = self.client.search(index=self.INDEX_NAME, body=body)
        hits = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            hits.append({
                "canonical": src["canonical"],
                "label": src.get("label", "UNKNOWN"),
                "aliases": src.get("aliases", []),
                "summary": src.get("summary", ""),
                "context_indicators": src.get("context_indicators", []),
                "related_to": src.get("related_to", []),
                "score": hit["_score"],
            })
        return hits

    def remove_entity(self, canonical: str) -> None:
        try:
            self.client.delete(index=self.INDEX_NAME, id=canonical)
        except Exception as e:
            logger.warning(f"ES delete failed for {canonical}: {e}")