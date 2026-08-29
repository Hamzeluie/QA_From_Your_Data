import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
from config.settings import settings

# Load environment variables from the .env file
load_dotenv()

def test_neo4j():
    print("🔹 Testing Neo4j connection...", end=" ")
    try:
        from neo4j import GraphDatabase
        url = os.getenv("GRAPH_DB_URL").strip()
        user = os.getenv("GRAPH_DB_USER").strip()
        password = os.getenv("GRAPH_DB_PASSWORD").strip()
        
        auth = (user, password) if password and password.lower() != "not_needed" else None
        
        driver = GraphDatabase.driver(url, auth=auth)
        driver.verify_connectivity()
        driver.close()
        print("✅ SUCCESS")
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")

def test_qdrant():
    print("🔹 Testing Qdrant connection...", end=" ")
    try:
        from qdrant_client import QdrantClient
        url = os.getenv("VECTOR_DB_URL").strip()
        client = QdrantClient(url=url)
        client.get_collections()
        print("✅ SUCCESS")
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")

def test_elasticsearch():
    print("🔹 Testing Elasticsearch connection...", end=" ")
    try:
        from elasticsearch import Elasticsearch
        # .strip() removes any hidden trailing spaces or newlines from the .env file
        url = os.getenv("ELASTIC_SEARCH_HOST", settings.ELASTIC_SEARCH_HOST).strip()
        
        client = Elasticsearch(
            url, 
            verify_certs=False,
            ssl_show_warn=False  # Suppresses v8 warnings for local HTTP
        )
        
        # client.info() is more reliable than client.ping() in v8.x
        info = client.info()
        print(f"✅ SUCCESS (v{info['version']['number']})")
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")

def test_postgres():
    print("🔹 Testing PostgreSQL connection...", end=" ")
    try:
        import psycopg2
        host = os.getenv("POSTGRES_HOST", "localhost").strip()
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("POSTGRES_USER", "kg_user").strip()
        password = os.getenv("POSTGRES_PASSWORD", "kg_pass").strip()
        database = os.getenv("POSTGRES_DB", "knowledge_graph").strip()

        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=database
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        print("✅ SUCCESS")
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")

def test_redis():
    print("🔹 Testing Redis connection...", end=" ")
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        print("✅ SUCCESS")
    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Testing Database Connections")
    print("=" * 60)
    
    test_neo4j()
    test_qdrant()
    test_elasticsearch()
    test_postgres()
    test_redis()
    
    print("=" * 60)
    print("🏁 Connection tests completed.")
    print("=" * 60)