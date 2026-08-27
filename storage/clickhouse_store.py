import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import clickhouse_connect
from clickhouse_connect.driver.client import Client

from storage.base import AbstractStateStore

logger = logging.getLogger(__name__)


class ClickHouseStateStore(AbstractStateStore):
    def __init__(self, host: str, port: int = 8123, username: str = "default",
                 password: str = "", database: str = "default"):
        self.client: Client = clickhouse_connect.get_client(host=host, port=port, username=username, password=password, database=database, compress=False)
        self.database = database

    def init_tables(self) -> None:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS document_state (
                doc_id String,
                owner_id String,
                status String,
                version UInt32,
                created_at DateTime64(3),
                updated_at DateTime64(3),
                error_message Nullable(String)
            ) ENGINE = ReplacingMergeTree(version)
            ORDER BY doc_id
            """,
            """
            CREATE TABLE IF NOT EXISTS unresolved_queue (
                resolution_id String,
                doc_id String,
                original_text String,
                entity_label String,
                mention_sentence String,
                start UInt32,
                end UInt32,
                confidence Float32,
                candidates_json String,
                status String,
                assigned_to Nullable(String),
                resolved_canonical Nullable(String),
                resolved_by Nullable(String),
                created_at DateTime64(3),
                resolved_at Nullable(DateTime64(3))
            ) ENGINE = MergeTree()
            ORDER BY (doc_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS mention_log (
                doc_id String,
                chunk_id Nullable(String),
                canonical_name String,
                original_text String,
                entity_label String,
                mention_sentence String,
                start UInt32,
                end UInt32,
                confidence Float32,
                source String,
                extracted_at DateTime64(3)
            ) ENGINE = MergeTree()
            ORDER BY (canonical_name, extracted_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS relations (
                doc_id String,
                relation_id String,
                subject String,
                subject_label String,
                predicate String,
                object String,
                object_label String,
                mention_sentence String,
                confidence Float32,
                source String,
                provisional UInt8,
                chunk_id Nullable(String),
                extracted_at DateTime64(3)
            ) ENGINE = MergeTree()
            ORDER BY (doc_id, predicate, extracted_at)
            """,
        ]
        for ddl in ddl_statements:
            try:
                self.client.command(ddl)
            except Exception as e:
                logger.warning(f"ClickHouse init warning: {e}")

    # ── Document State Machine ────────────────────────────────────────────────

    def create_document(self, doc_id: str, owner_id: str) -> None:
        now = datetime.now(timezone.utc)
        self.client.insert(
            "document_state",
            [[doc_id, owner_id, "uploaded", 1, now, now, None]],
            column_names=["doc_id", "owner_id", "status", "version", "created_at", "updated_at", "error_message"],
        )

    def transition_state(
        self, doc_id: str, expected: Optional[str], to_state: str, error_message: Optional[str] = None
    ) -> bool:
        """
        Optimistic concurrency using ReplacingMergeTree.
        We read current version, then insert new row with version+1.
        """
        current = self.client.query(
            "SELECT status, version FROM document_state WHERE doc_id = {doc_id:String} ORDER BY version DESC LIMIT 1",
            parameters={"doc_id": doc_id},
        )
        if not current.result_rows:
            return False

        cur_status, cur_version = current.result_rows[0]
        if expected and cur_status != expected:
            return False

        now = datetime.now(timezone.utc)
        self.client.insert(
            "document_state",
            [[doc_id, "", to_state, cur_version + 1, now, now, error_message or None]],
            column_names=["doc_id", "owner_id", "status", "version", "created_at", "updated_at", "error_message"],
        )
        return True

    def get_documents_by_state(self, status: str, limit: int = 100) -> List[Dict]:
        rows = self.client.query(
            """
            SELECT doc_id, owner_id, status, version, created_at, updated_at
            FROM document_state
            WHERE status = {status:String}
            ORDER BY updated_at DESC
            LIMIT {limit:UInt32}
            """,
            parameters={"status": status, "limit": limit},
        )
        cols = ["doc_id", "owner_id", "status", "version", "created_at", "updated_at"]
        return [dict(zip(cols, r)) for r in rows.result_rows]

    def get_document_state(self, doc_id: str) -> Optional[Dict]:
        result = self.client.query(
            "SELECT * FROM document_state WHERE doc_id = {doc_id:String} ORDER BY version DESC LIMIT 1",
            parameters={"doc_id": doc_id},
        )
        if not result.result_rows:
            return None
        return dict(zip(result.column_names, result.result_rows[0]))
    
    # ── Unresolved Queue ──────────────────────────────────────────────────────

    def enqueue_unresolved(
        self,
        resolution_id: str,
        doc_id: str,
        entity_row: Any,
        candidates_json: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        # Handle both dataclass and dict
        if hasattr(entity_row, "to_dict"):
            d = entity_row.to_dict()
        elif hasattr(entity_row, "__dataclass_fields__"):
            d = entity_row.__dict__
        else:
            d = dict(entity_row)

        self.client.insert(
            "unresolved_queue",
            [[
                resolution_id,
                doc_id,
                d.get("original_text", ""),
                d.get("entity_label", "UNKNOWN"),
                d.get("mention_sentence", ""),
                int(d.get("start", 0)),
                int(d.get("end", 0)),
                float(d.get("confidence", 0.0)),
                candidates_json,
                "pending",
                None,
                None,
                None,
                now,
                None,
            ]],
            column_names=[
                "resolution_id", "doc_id", "original_text", "entity_label",
                "mention_sentence", "start", "end", "confidence",
                "candidates_json", "status", "assigned_to", "resolved_canonical",
                "resolved_by", "created_at", "resolved_at",
            ],
        )

    def resolve_unresolved(
        self, resolution_id: str, canonical: str, user_id: str
    ) -> None:
        now = datetime.now(timezone.utc)
        self.client.command(
            """
            ALTER TABLE unresolved_queue
            UPDATE status = 'resolved', resolved_canonical = {canonical:String},
                   resolved_by = {user_id:String}, resolved_at = {now:DateTime64(3)}
            WHERE resolution_id = {rid:String}
            """,
            parameters={
                "canonical": canonical,
                "user_id": user_id,
                "now": now,
                "rid": resolution_id,
            },
        )

    # ── Mention Log ───────────────────────────────────────────────────────────

    def log_mention(
        self,
        doc_id: str,
        chunk_id: Optional[str],
        canonical_name: str,
        original_text: str,
        entity_label: str,
        mention_sentence: str,
        start: int,
        end: int,
        confidence: float,
        source: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.client.insert(
            "mention_log",
            [[
                doc_id, chunk_id or None, canonical_name, original_text,
                entity_label, mention_sentence, start, end,
                confidence, source, now,
            ]],
            column_names=[
                "doc_id", "chunk_id", "canonical_name", "original_text",
                "entity_label", "mention_sentence", "start", "end",
                "confidence", "source", "extracted_at",
            ],
        )

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
            rows.append([
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
                1 if d.get("provisional", False) else 0,
                d.get("chunk_id"),
                now,
            ])

        self.client.insert(
            "relations",
            rows,
            column_names=[
                "doc_id", "relation_id", "subject", "subject_label", "predicate",
                "object", "object_label", "mention_sentence", "confidence",
                "source", "provisional", "chunk_id", "extracted_at",
            ],
        )