# QA_From_Your_Data

**Question and Answering Your Data** — A production-grade, multi-database Retrieval-Augmented Generation (RAG) and Knowledge Graph system. 

`QA_From_Your_Data` enables you to ingest unstructured and structured data into a hybrid storage architecture and query it using advanced LLM-driven retrieval. It operates in two primary modes: **Ingestion Mode** (for data processing, entity extraction, and graph construction) and **Query Mode** (for multi-modal retrieval and question answering).

> 🚧 **Status:** *This repository is under active development and is currently being engineered to reach production-level standards. Core infrastructure and database integrations are fully operational.*

---

## ✨ Key Features

- **Hybrid Ingestion Pipeline**: Extracts entities, resolves them, and stores them across Vector, Graph, Document, and OLAP databases simultaneously.
- **Multi-Modal Query Engine**: Combines semantic search (Qdrant), full-text search (Elasticsearch), graph traversal (Neo4j), and analytical queries (ClickHouse) for comprehensive answers.
- **Entity Resolution**: Advanced state-tracking to ensure entities are correctly merged and deduplicated across the knowledge graph.
- **Persistent Local Development**: Fully containerized infrastructure with persistent local storage for seamless development and debugging.
- **LLM Integration**: Powered by state-of-the-art models (via NVIDIA API) for extraction and generation.

---

## 🏗️ Architecture & Tech Stack

This system utilizes a "best-of-breed" database approach, routing specific types of data and queries to the most optimized engine:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Graph DB** | `Neo4j 5` | Stores entities, relationships, and graph topology for traversal. |
| **Vector DB** | `Qdrant` | Stores dense embeddings for semantic similarity search. |
| **Search DB** | `Elasticsearch 8` | Handles full-text search, keyword matching, and document storage. |
| **OLAP DB** | `ClickHouse` | Manages analytical queries, metrics, and high-speed aggregations. |
| **Cache/State**| `Redis` | Handles session management, caching, and temporary state tracking. |
| **LLM** | `NVIDIA API` | Runs `meta/llama-3.1-70b-instruct` for extraction and QA generation. |
| **Embeddings**| `all-MiniLM-L6-v2` | Generates dense vector representations for text. |
| **NLP / NER** | `spaCy` | Handles initial linguistic parsing and Named Entity Recognition. |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Poetry (for dependency management)
- An active NVIDIA API Key for LLM inference

### 2. Installation
```bash
# Clone the repository
git clone [YOUR_REPO_URL]
cd QA_From_Your_Data

# Install Python dependencies
poetry install

# Activate the virtual environment
poetry shell