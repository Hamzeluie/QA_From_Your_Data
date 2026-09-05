# postgres_store.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from storage.base import AbstractStateStore
from shared.data_classes import Entity
logger = logging.getLogger(__name__)


class PostgresStateStore(AbstractStateStore):
    def __init__(
        self,
        host: str,
        port: int = 5432,
        username: str = "kg_user",
        password: str = "kg_pass",
        database: str = "knowledge_graph",
    ):
        self.dsn = (
            f"dbname={database} user={username} password={password} "
            f"host={host} port={port}"
        )
        self.database = database
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    # ── Schema ──────────────────────────────────────────────────────────────

    def init_tables(self) -> None:
        ddls = [
            """
            CREATE TABLE IF NOT EXISTS document_state (
                doc_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unresolved_queue (
                resolution_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                mention_sentence TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                confidence REAL NOT NULL,
                candidates_json TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                assigned_to TEXT,
                resolved_canonical TEXT,
                resolved_by TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mention_log (
                id SERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_id TEXT,
                canonical_name TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                mention_sentence TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                confidence REAL NOT NULL,
                extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relations (
                id SERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                relation_id TEXT NOT NULL UNIQUE,
                subject TEXT NOT NULL,
                subject_label TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                object_label TEXT NOT NULL,
                mention_sentence TEXT,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                provisional BOOLEAN NOT NULL DEFAULT FALSE,
                chunk_id TEXT,
                extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_doc_state_status ON document_state(status)",
            "CREATE INDEX IF NOT EXISTS idx_unresolved_doc ON unresolved_queue(doc_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_mention_canonical ON mention_log(canonical_name, extracted_at)",
            "CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations(doc_id, predicate, extracted_at)",
        ]

        conn = self._get_conn()
        with conn.cursor() as cur:
            for ddl in ddls:
                try:
                    cur.execute(ddl)
                except Exception as e:
                    logger.warning(f"Postgres init warning: {e}")
            conn.commit()

    # ── Document State Machine ──────────────────────────────────────────────

    def create_document(self, doc_id: str, owner_id: str) -> None:
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_state (doc_id, owner_id, status, version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO NOTHING
                """,
                (doc_id, owner_id, "uploaded", 1, now, now),
            )
            conn.commit()

    def transition_state(
        self,
        doc_id: str,
        expected: Optional[str],
        to_state: str,
        error_message: Optional[str] = None,
    ) -> bool:
        conn = self._get_conn()
        with conn.cursor() as cur:
            if expected:
                cur.execute(
                    """
                    UPDATE document_state
                    SET status = %s, version = version + 1, updated_at = NOW(), error_message = %s
                    WHERE doc_id = %s AND status = %s
                    RETURNING version
                    """,
                    (to_state, error_message, doc_id, expected),
                )
            else:
                cur.execute(
                    """
                    UPDATE document_state
                    SET status = %s, version = version + 1, updated_at = NOW(), error_message = %s
                    WHERE doc_id = %s
                    RETURNING version
                    """,
                    (to_state, error_message, doc_id),
                )
            ok = cur.fetchone() is not None
            conn.commit()
            return ok

    def get_documents_by_state(self, status: str, limit: int = 100) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, owner_id, status, version, created_at, updated_at
                FROM document_state
                WHERE status = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (status, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_document_state(self, doc_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM document_state WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ── Unresolved Queue ────────────────────────────────────────────────────

    def enqueue_unresolved(
        self,
        resolution_id: str,
        doc_id: str,
        entity_row: Any,
        candidates_json: str,
    ) -> None:
        now = datetime.now(timezone.utc)

        if hasattr(entity_row, "to_dict"):
            d = entity_row.to_dict()
        elif hasattr(entity_row, "__dataclass_fields__"):
            d = entity_row.__dict__
        else:
            d = dict(entity_row)

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO unresolved_queue
                (resolution_id, doc_id, text, label, mention_sentence,
                 start_pos, end_pos, confidence, candidates_json, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resolution_id) DO NOTHING
                """,
                (
                    resolution_id,
                    doc_id,
                    d.get("text", ""),
                    d.get("label", "UNKNOWN"),
                    d.get("mention_sentence", ""),
                    int(d.get("start", 0)),
                    int(d.get("end", 0)),
                    float(d.get("confidence", 0.0)),
                    candidates_json,
                    "pending",
                    now,
                ),
            )
            conn.commit()

    def resolve_unresolved(
        self, resolution_id: str, canonical: str, user_id: str
    ) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE unresolved_queue
                SET status = 'resolved', resolved_canonical = %s,
                    resolved_by = %s, resolved_at = NOW()
                WHERE resolution_id = %s
                """,
                (canonical, user_id, resolution_id),
            )
            conn.commit()

    # ── Mention Log ─────────────────────────────────────────────────────────

    def log_mention(
        self,
        entity:Entity,
    ) -> None:
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mention_log
                (doc_id, chunk_id, canonical_name, text, label,
                 mention_sentence, start_pos, end_pos, confidence, extracted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entity.doc_id,
                    entity.chunk_id,
                    entity.canonical_name,
                    entity.text,
                    entity.label,
                    entity.mention_sentence,
                    entity.start,
                    entity.end,
                    entity.confidence,
                    now,
                ),
            )
            conn.commit()

    # ── Relations ───────────────────────────────────────────────────────────

    def insert_relations(self, relations: List[Any]) -> None:
        if not relations:
            return

        now = datetime.now(timezone.utc)
        rows = []
        for r in relations:
            if hasattr(r, "to_dict"):
                d = r.to_dict()
            else:
                d = dict(r)
            rows.append(
                (
                    d.get("doc_id", ""),
                    d.get("relation_id", ""),
                    d.get("subject", ""),
                    d.get("subject_label", "UNKNOWN"),
                    d.get("predicate", ""),
                    d.get("object", ""),
                    d.get("object_label", "UNKNOWN"),
                    d.get("mention_sentence", ""),
                    float(d.get("confidence", 0.0)),
                    d.get("source", "dspy"),
                    bool(d.get("provisional", False)),
                    d.get("chunk_id"),
                    now,
                )
            )

        conn = self._get_conn()
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO relations
                (doc_id, relation_id, subject, subject_label, predicate, object,
                 object_label, mention_sentence, confidence, source, provisional, chunk_id, extracted_at)
                VALUES %s
                ON CONFLICT (relation_id) DO NOTHING
                """,
                rows,
                page_size=1000,
            )
            conn.commit()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            
            
            