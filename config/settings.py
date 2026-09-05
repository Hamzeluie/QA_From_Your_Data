import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
from dotenv import load_dotenv

load_dotenv(os.getenv("QA_FROM_YOUR_DATA_ENV_FILE_NAME", os.path.join(PROJECT_ROOT,".env")), override=True)


class Settings:
   # ========= Embedding Config 
   EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME","all-MiniLM-L6-v2")
   EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH","../model/all-MiniLM-L6-v2")
   # ========= Spacy Config 
   SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME","en_core_web_sm")
   SPACY_MODEL_PATH = os.getenv("SPACY_MODEL_PATH","../model/en_core_web_sm")
   # ========= DataBases Config
   ELASTIC_SEARCH_HOST = os.getenv("ELASTIC_SEARCH_HOST", "http://localhost:9200")
   
   VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "http://localhost")
   VECTOR_DB_PORT = os.getenv("VECTOR_DB_PORT", "6333")
   
   GRAPH_DB_URL = os.getenv("GRAPH_DB_HOST", "bolt://localhost:7687")
   GRAPH_DB_USER= os.getenv("GRAPH_DB_USER", "neo4j")
   GRAPH_DB_PASSWORD= os.getenv("GRAPH_DB_PASSWORD", "")
   
   POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
   POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
   POSTGRES_USER = os.getenv("POSTGRES_USER", "kg_user")
   POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "kg_pass")
   POSTGRES_DB = os.getenv("POSTGRES_DB", "knowledge_graph")
   
   DB_DATA_DIR = os.getenv("DB_DATA_DIR", "./checkpoints/db")
   DATASET_PATH = os.getenv("DATASET_PATH", "./checkpoints/ner_dataset.jsonl")
   # ======== OutBox Config
   OUTBOX_PATH = os.getenv("OUTBOX_PATH", "kg_outbox.sqlite3")
   # ======== LLM Config
   LLM_BASE_URL = os.getenv("LLM_BASE_URL","")
   LLM_API_KEY = os.getenv("LLM_API_KEY","") 
   LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME","") 
   # ======== CheckPoint Config
   CHECKPOINTS = os.getenv("CHECKPOINTS", "./checkpoints")
   ENTITY_RESOLUTION_STATE_PATH = os.getenv("ENTITY_RESOLUTION_STATE_PATH", "./checkpoints/entity_resolution_full_state")
   BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH", "./checkpoints/bert_models")
   BERT_NER_MODEL_NAME = os.getenv("BERT_NER_MODEL_NAME", "Davlan/distilbert-base-multilingual-cased-ner-hrl")
   BERT_RE_MODEL_NAME = os.getenv("BERT_RE_MODEL_NAME", "Babelscape/rebel-large")
   BERT_ONNX_MODEL_PATH = os.getenv("BERT_ONNX_MODEL_PATH", "./checkpoints/bert_models/onnx_models")


settings = Settings()