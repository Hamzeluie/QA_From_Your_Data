# postgres_store.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from storage.base import AbstractStateStore
from storage.data_classes import MentionEntity, Relation, DisambiguationStatus
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
            CREATE TABLE IF NOT EXISTS resolved_entities (
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
                resolved_by TEXT,
                resolved_at TIMESTAMP WITH TIME ZONE,
                extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unresolved_entities (
                resolution_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_id TEXT,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                mention_sentence TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                confidence REAL NOT NULL,
                candidates_json TEXT,
                assigned_to TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS resolved_relations (
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
                chunk_id TEXT NOT NULL,
                evidence TEXT NOT NULL,
                resolved_by TEXT,
                resolved_at TIMESTAMP WITH TIME ZONE,
                extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unresolved_relations (
                relation_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                subject_label TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                object_label TEXT NOT NULL,
                mention_sentence TEXT,
                confidence REAL NOT NULL,
                chunk_id TEXT NOT NULL,
                evidence TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_doc_state_status ON document_state(status)",
            "CREATE INDEX IF NOT EXISTS idx_resolved_entities_canonical ON resolved_entities(canonical_name, extracted_at)",
            "CREATE INDEX IF NOT EXISTS idx_unresolved_entities_doc ON unresolved_entities(doc_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_resolved_relations_doc ON resolved_relations(doc_id, predicate, extracted_at)",
            "CREATE INDEX IF NOT EXISTS idx_unresolved_relations_doc ON unresolved_relations(doc_id, created_at)",
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

    # ── Unresolved Entities ─────────────────────────────────────────────────

    def insert_unresolved(
        self,
        resolution_id: str,
        mentionentity:MentionEntity,
    ) -> None:
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO unresolved_entities
                (resolution_id, doc_id, chunk_id, text, label, mention_sentence,
                 start_pos, end_pos, confidence, candidates_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resolution_id) DO NOTHING
                """,
                (
                    resolution_id,
                    mentionentity.doc_id,
                    mentionentity.chunk_id,
                    mentionentity.text,
                    mentionentity.label,
                    mentionentity.mention_sentence,
                    mentionentity.start,
                    mentionentity.end,
                    mentionentity.confidence,
                    mentionentity.kg_candidates,
                    now,
                ),
            )
            conn.commit()
    
    def resolve_unresolved(self, resolution_id: str, canonical: str, user_id: str) -> None:
        """Moves an MentionEntity from unresolved_entities to resolved_entities."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Fetch the unresolved MentionEntity
            cur.execute(
                """
                SELECT doc_id, chunk_id, text, label, mention_sentence, start_pos, end_pos, confidence
                FROM unresolved_entities
                WHERE resolution_id = %s
                """,
                (resolution_id,)
            )
            row = cur.fetchone()
            if not row:
                logger.warning(f"Unresolved MentionEntity with resolution_id {resolution_id} not found.")
                return
            
            # Insert into resolved_entities
            cur.execute(
                """
                INSERT INTO resolved_entities
                (doc_id, chunk_id, canonical_name, text, label, mention_sentence, 
                 start_pos, end_pos, confidence, resolved_by, resolved_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    row["doc_id"],
                    row["chunk_id"],
                    canonical,
                    row["text"],
                    row["label"],
                    row["mention_sentence"],
                    row["start_pos"],
                    row["end_pos"],
                    row["confidence"],
                    user_id,
                )
            )
            
            # Delete from unresolved_entities
            cur.execute(
                "DELETE FROM unresolved_entities WHERE resolution_id = %s",
                (resolution_id,)
            )
            conn.commit()

    def get_unresolved_entities(self) -> List[MentionEntity]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT resolution_id, doc_id, chunk_id, text, label, mention_sentence,
                       start_pos, end_pos, confidence, candidates_json
                FROM unresolved_entities
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
            entities = []
            for row in rows:
                candidates = []
                if row["candidates_json"]:
                    try:
                        candidates = json.loads(row["candidates_json"])
                    except json.JSONDecodeError:
                        pass  # Fallback to empty list if JSON is malformed
                
                mentionentity = MentionEntity(
                    text=row["text"],
                    label=row["label"],
                    start=int(row["start_pos"]),
                    end=int(row["end_pos"]),
                    mention_sentence=row["mention_sentence"],
                    confidence=float(row["confidence"]),
                    canonical_name="",  # Unresolved
                    status=DisambiguationStatus.UNRESOLVED,
                    doc_id=row["doc_id"],
                    chunk_id=row.get("chunk_id"),
                    kg_candidates=candidates,
                )
                entities.append()
            return entities

    # ── Resolved Entities (formerly Mentions) ───────────────────────────────
    
    def insert_resolved_MentionEntity(self, MentionEntity: MentionEntity) -> None:
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO resolved_entities
                (doc_id, chunk_id, canonical_name, text, label,
                 mention_sentence, start_pos, end_pos, confidence, extracted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    MentionEntity.doc_id,
                    MentionEntity.chunk_id,
                    MentionEntity.canonical_name,
                    MentionEntity.text,
                    MentionEntity.label,
                    MentionEntity.mention_sentence,
                    MentionEntity.start,
                    MentionEntity.end,
                    MentionEntity.confidence,
                    now,
                ),
            )
            conn.commit() 

    def get_resolved_entities_by_canonical(self, canonical_name: str) -> List[MentionEntity]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, canonical_name, text, label,
                       mention_sentence, start_pos, end_pos, confidence
                FROM resolved_entities
                WHERE canonical_name = %s
                ORDER BY extracted_at DESC
                """,
                (canonical_name,),
            )
            return [self._row_to_MentionEntity(row) for row in cur.fetchall()]

    def get_resolved_entities_by_text(self, text: str) -> List[MentionEntity]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, canonical_name, text, label,
                       mention_sentence, start_pos, end_pos, confidence
                FROM resolved_entities
                WHERE text ILIKE %s
                ORDER BY extracted_at DESC
                """,
                (f"%{text}%",),
            )
            return [self._row_to_MentionEntity(row) for row in cur.fetchall()]

    def get_resolved_entities_by_doc(self, doc_id: str) -> List[MentionEntity]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, canonical_name, text, label,
                       mention_sentence, start_pos, end_pos, confidence
                FROM resolved_entities
                WHERE doc_id = %s
                ORDER BY extracted_at DESC
                """,
                (doc_id,),
            )
            return [self._row_to_MentionEntity(row) for row in cur.fetchall()]

    def _row_to_MentionEntity(self, row: Dict[str, Any]) -> MentionEntity:
        """Helper to map a database row to an MentionEntity dataclass."""
        return MentionEntity(
            text=row["text"],
            label=row["label"],
            start=int(row["start_pos"]),
            end=int(row["end_pos"]),
            mention_sentence=row["mention_sentence"],
            confidence=float(row["confidence"]),
            canonical_name=row["canonical_name"],
            status=DisambiguationStatus.RESOLVED,
            doc_id=row.get("doc_id"),
            chunk_id=row.get("chunk_id"),
        )

    # ── Relations ───────────────────────────────────────────────────────────

    def insert_resolved_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return

        now = datetime.now(timezone.utc)
        rows = []
        for r in relations:
            evidence_str = json.dumps(r.evidence) if isinstance(r.evidence, list) else (str(r.evidence) if r.evidence else "[]")
            rows.append(
                (
                    r.doc_id,
                    r.relation_id,
                    r.subject,
                    r.subject_label,
                    r.predicate,
                    r.object,
                    r.object_label,
                    r.mention_sentence,
                    r.confidence,
                    r.chunk_id,
                    evidence_str,
                    now,
                )
            )

        conn = self._get_conn()
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO resolved_relations
                (doc_id, relation_id, subject, subject_label, predicate, object,
                 object_label, mention_sentence, confidence, chunk_id, evidence, extracted_at)
                VALUES %s
                ON CONFLICT (relation_id) DO NOTHING
                """,
                rows,
                page_size=1000,
            )
            conn.commit()
    
    def insert_unresolved_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return

        now = datetime.now(timezone.utc)
        rows = []
        for r in relations:
            evidence_str = json.dumps(r.evidence) if isinstance(r.evidence, list) else (str(r.evidence) if r.evidence else "[]")
            rows.append(
                (
                    r.relation_id,
                    r.doc_id,
                    r.subject,
                    r.subject_label,
                    r.predicate,
                    r.object,
                    r.object_label,
                    r.mention_sentence,
                    r.confidence,
                    r.chunk_id,
                    evidence_str,
                    now,
                )
            )

        conn = self._get_conn()
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO unresolved_relations
                (relation_id, doc_id, subject, subject_label, predicate, object,
                 object_label, mention_sentence, confidence, chunk_id, evidence, created_at)
                VALUES %s
                ON CONFLICT (relation_id) DO NOTHING
                """,
                rows,
                page_size=1000,
            )
            conn.commit()

    def resolve_unresolved_relation(self, relation_id: str, user_id: str) -> None:
        """Moves a relation from unresolved_relations to resolved_relations."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, subject, subject_label, predicate, object, object_label,
                       mention_sentence, confidence, chunk_id, evidence
                FROM unresolved_relations
                WHERE relation_id = %s
                """,
                (relation_id,)
            )
            row = cur.fetchone()
            if not row:
                logger.warning(f"Unresolved relation with relation_id {relation_id} not found.")
                return
            
            cur.execute(
                """
                INSERT INTO resolved_relations
                (doc_id, relation_id, subject, subject_label, predicate, object, object_label,
                 mention_sentence, confidence, chunk_id, evidence, resolved_by, resolved_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    row["doc_id"], relation_id, row["subject"], row["subject_label"],
                    row["predicate"], row["object"], row["object_label"],
                    row["mention_sentence"], row["confidence"], row["chunk_id"],
                    row["evidence"], user_id
                )
            )
            
            cur.execute(
                "DELETE FROM unresolved_relations WHERE relation_id = %s",
                (relation_id,)
            )
            conn.commit()

    def get_resolved_relations_by_MentionEntity(self, canonical_name: str, limit: int = 200) -> List[Relation]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, relation_id, subject, subject_label, predicate, object,
                       object_label, mention_sentence, confidence, chunk_id, evidence
                FROM resolved_relations
                WHERE subject = %s OR object = %s
                ORDER BY extracted_at DESC
                LIMIT %s
                """,
                (canonical_name, canonical_name, limit),
            )
            return [self._row_to_relation(row) for row in cur.fetchall()]

    def get_resolved_relations_by_predicate(self, doc_id: str, predicate: str, limit: int = 200) -> List[Relation]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, relation_id, subject, subject_label, predicate, object,
                       object_label, mention_sentence, confidence, chunk_id, evidence
                FROM resolved_relations
                WHERE doc_id = %s AND predicate = %s
                ORDER BY extracted_at DESC
                LIMIT %s
                """,
                (doc_id, predicate, limit),
            )
            return [self._row_to_relation(row) for row in cur.fetchall()]

    def get_relations_by_predicate(self, doc_id: str, predicate: str, limit: int = 200) -> List[Relation]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, relation_id, subject, subject_label, predicate, object,
                       object_label, mention_sentence, confidence, provisional, chunk_id, evidence
                FROM relations
                WHERE doc_id = %s AND predicate = %s
                ORDER BY extracted_at DESC
                LIMIT %s
                """,
                (doc_id, predicate, limit),
            )
            return [self._row_to_relation(row) for row in cur.fetchall()]

    def get_unresolved_relations(self, limit: int = 200) -> List[Relation]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, relation_id, subject, subject_label, predicate, object,
                       object_label, mention_sentence, confidence, chunk_id, evidence
                FROM unresolved_relations
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            relations = []
            for row in cur.fetchall():
                rel = self._row_to_relation(row)
                rel.provisional = True  # Mark as provisional since it's from the unresolved table
                relations.append(rel)
            return relations

    def _row_to_relation(self, row: Dict[str, Any]) -> Relation:
        """Helper to map a database row to a Relation dataclass."""
        evidence = row.get("evidence")
        
        # Safely parse evidence from TEXT/JSON column back to List[str]
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = [evidence] if evidence else []
        elif evidence is None:
            evidence = []
            
        return Relation(
            doc_id=row["doc_id"],
            relation_id=row.get("relation_id"),
            subject=row["subject"],
            subject_label=row["subject_label"],
            predicate=row["predicate"],
            object=row["object"],
            object_label=row["object_label"],
            mention_sentence=row.get("mention_sentence", ""),
            confidence=float(row.get("confidence", 0.0)),
            provisional=bool(row.get("provisional", False)),
            chunk_id=row.get("chunk_id"),
            evidence=evidence if isinstance(evidence, list) else [],
        )
    
    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            
            
            