import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
from dotenv import load_dotenv

load_dotenv(os.getenv("QA_FROM_YOUR_DATA_ENV_FILE_NAME", os.path.join(PROJECT_ROOT,".env")), override=True)


class Settings:
   EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME","all-MiniLM-L6-v2")
   EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH","../model/all-MiniLM-L6-v2")
   
   SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME","en_core_web_sm")
   SPACY_MODEL_PATH = os.getenv("SPACY_MODEL_PATH","../model/en_core_web_sm")
   
   ELASTIC_SEARCH_HOST = os.getenv("ELASTIC_SEARCH_HOST", "http://localhost:9200")
   VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
   
   GRAPH_DB_URL = os.getenv("GRAPH_DB_URL", "bolt://localhost:7687")
   GRAPH_DB_USER= os.getenv("GRAPH_DB_USER", "neo4j")
   GRAPH_DB_PASSWORD= os.getenv("GRAPH_DB_PASSWORD", "")
   
   CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
   CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", 8123)
   
   OUTBOX_PATH = os.getenv("OUTBOX_PATH", "kg_outbox.sqlite3")
   
   DB_DATA_DIR = os.getenv("DB_DATA_DIR", "./checkpoints/db")
   
   LLM_BASE_URL = os.getenv("LLM_BASE_URL","")
   LLM_API_KEY = os.getenv("LLM_API_KEY","") 
   LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME","") 
   
   DATASET_PATH = os.getenv("DATASET_PATH", "./checkpoints/ner_dataset.jsonl")
   CHECKPOINTS = os.getenv("CHECKPOINTS", "./checkpoints")
   ENTITY_RESOLUTION_STATE_PATH = os.getenv("ENTITY_RESOLUTION_STATE_PATH", "./checkpoints/entity_resolution_full_state")


settings = Settings()