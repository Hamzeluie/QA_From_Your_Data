from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
import datetime
import psycopg2
import psycopg2.extras
from shared.identical_id import Ids
from domain import (
    CanonicalEntity, MentionEntity, CandidateResult, CandidateSource,
    CandidateMatchMethod, EntityLabels, Relation, OutboxDocument,
    OutboxEvent, EventType, ProcessSteps, DocumentStatus, EventPayload
)


class PostgresStateStore:
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
        self._conn = None

    # ─────────────────────────── connection ───────────────────────────

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    @contextmanager
    def _cursor(self):
        """One transaction per `with` block: commits on success, rolls back on error."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ─────────────────────────── helpers ──────────────────────────────

    def _fetchall(self, sql: str, params: tuple = ()) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    @staticmethod
    def _iso(dt) -> Optional[str]:
        return dt.isoformat() if dt else None

    @staticmethod
    def _mention_id(doc_id: str, chunk_id: Optional[str], start: int, end: int) -> str:
        return f"{doc_id}:{chunk_id or ''}:{start}:{end}"

    # ─────────────────────────── schema ───────────────────────────────

    def init_schema(self):
        with self._cursor() as cur:
            # pg_trgm is a "trusted" extension: DB owner can create it on PG >= 13.
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS canonicals (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    summary     TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_canonicals_name_trgm
                    ON canonicals USING gin (name gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS idx_canonicals_label
                    ON canonicals (label);

                CREATE TABLE IF NOT EXISTS alias (
                    id            TEXT PRIMARY KEY,
                    canonical_id  TEXT NOT NULL REFERENCES canonicals (id) ON DELETE CASCADE,
                    name          TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (canonical_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_alias_name_trgm
                    ON alias USING gin (name gin_trgm_ops);

                CREATE TABLE IF NOT EXISTS mentions (
                    id               TEXT PRIMARY KEY,      -- doc:chunk:start:end, idempotent upserts
                    name             TEXT NOT NULL,
                    label            TEXT NOT NULL,
                    start_offset     INTEGER NOT NULL,      -- 'end' is reserved in PG
                    end_offset       INTEGER NOT NULL,
                    mention_sentence TEXT,
                    confidence       REAL NOT NULL DEFAULT 0.0,
                    coref_to         TEXT,
                    doc_id           TEXT NOT NULL,
                    chunk_id         TEXT,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_mentions_doc   ON mentions (doc_id, chunk_id);
                CREATE INDEX IF NOT EXISTS idx_mentions_name_trgm
                    ON mentions USING gin (name gin_trgm_ops);

                CREATE TABLE IF NOT EXISTS relation (
                    id               TEXT PRIMARY KEY,
                    subject          TEXT NOT NULL,
                    subject_label    TEXT,
                    predicate        TEXT NOT NULL,
                    object           TEXT NOT NULL,
                    object_label     TEXT,
                    mention_sentence TEXT,
                    confidence       REAL NOT NULL DEFAULT 0.0,
                    needs_review     BOOLEAN NOT NULL DEFAULT FALSE,
                    evidence         JSONB NOT NULL DEFAULT '[]'::jsonb,
                    doc_id           TEXT,
                    chunk_id         TEXT,
                    provisional      BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                -- the "unresolved queue" is just this partial index:
                CREATE INDEX IF NOT EXISTS idx_relation_unresolved
                    ON relation (doc_id) WHERE needs_review;
                CREATE INDEX IF NOT EXISTS idx_relation_doc       ON relation (doc_id);
                CREATE INDEX IF NOT EXISTS idx_relation_predicate ON relation (predicate);

                CREATE TABLE IF NOT EXISTS outbox_document (
                    doc_id                   TEXT PRIMARY KEY,
                    owner_id                 TEXT,
                    status                   TEXT NOT NULL DEFAULT 'UPLOADED',
                    pending_unresolved_count INTEGER NOT NULL DEFAULT 0,
                    error_message            TEXT,
                    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_doc_owner
                    ON outbox_document (owner_id, status);
                    
                CREATE TABLE IF NOT EXISTS outbox_event (
                    event_id      TEXT PRIMARY KEY,
                    doc_id        TEXT NOT NULL REFERENCES outbox_document (doc_id) ON DELETE CASCADE,
                    event_type    TEXT NOT NULL,
                    processed_steps TEXT[]  NOT NULL DEFAULT '{}',
                    payload       JSONB   NOT NULL DEFAULT '{}'::jsonb,
                    processed     BOOLEAN NOT NULL DEFAULT FALSE,
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    max_attempts  INTEGER NOT NULL DEFAULT 3,
                    error_message TEXT,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    processed_at  TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_event_pending
                    ON outbox_event (created_at) WHERE processed = FALSE;
            """)

    # ══════════════════════════ CANONICALS ════════════════════════════

    def create_canonical(self, entity: CanonicalEntity) -> CanonicalEntity:
        """Upsert canonical + replace its alias set, atomically."""
        canonical_id = entity.canonical_id or Ids.canonical(entity.name, entity.label)

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO canonicals (id, name, label, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    label = EXCLUDED.label,
                    summary = EXCLUDED.summary,
                    updated_at = now()
                """,
                (canonical_id, entity.name, entity.label, entity.summary),
            )
            cur.execute("DELETE FROM alias WHERE canonical_id = %s", (canonical_id,))
            if entity.aliases:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO alias (id, canonical_id, name)
                    VALUES %s
                    ON CONFLICT (canonical_id, name) DO NOTHING
                    """,
                    [(Ids.alias(canonical_id, a), canonical_id, a) for a in entity.aliases],
                )
        entity.canonical_id = canonical_id
        return entity

    def create_canonicals(self, entities: List[CanonicalEntity]) -> List[CanonicalEntity]:
        return [self.create_canonical(e) for e in entities]
    
    def ensure_canonical(self, entity: CanonicalEntity) -> CanonicalEntity:
        """
        Outbox-safe create-or-merge (never destructive):

        1. Resolve the real id: by entity.canonical_id if that row exists,
            else by case-insensitive name match, else mint a new one
            (covers "found a new canonical, not fetched from db").
        2. Canonical missing -> INSERT it; entity.aliases (incl. mention text)
            go in with it.
        3. Canonical exists  -> name/label/existing aliases are LEFT ALONE;
            only aliases not yet present are added (ON CONFLICT DO NOTHING),
            so we never fetch the full alias set.
        """
        canonical_id = entity.canonical_id
        if canonical_id:
            row = self._fetchone("SELECT id FROM canonicals WHERE id = %s", (canonical_id,))
            if not row:
                canonical_id = None                    # fresh/stale id — fall through
        if not canonical_id:
            row = self._fetchone(
                "SELECT id FROM canonicals WHERE lower(name) = lower(%s) LIMIT 1",
                (entity.name,),
            )
            canonical_id = row["id"] if row else Ids.canonical(entity.name, entity.label)

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO canonicals (id, name, label, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,                                   # no name/label overwrite on conflict
                (canonical_id, entity.name, entity.label, entity.summary),
            )
            if entity.aliases:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO alias (id, canonical_id, name)
                    VALUES %s
                    ON CONFLICT (canonical_id, name) DO NOTHING
                    """,
                    [(Ids.alias(canonical_id, a), canonical_id, a) for a in entity.aliases],
                    page_size=500,
                )
        entity.canonical_id = canonical_id
        return entity

    def ensure_canonicals(self, entities: List[CanonicalEntity]) -> List[CanonicalEntity]:
        return [self.ensure_canonical(e) for e in entities]

    def get_canonical(self, canonical_id: str) -> Optional[CanonicalEntity]:
        row = self._fetchone(
            """
            SELECT c.*, COALESCE(a.aliases, '{}') AS aliases
            FROM canonicals c
            LEFT JOIN LATERAL (
                SELECT array_agg(a2.name ORDER BY a2.name) AS aliases
                FROM alias a2 WHERE a2.canonical_id = c.id
            ) a ON TRUE
            WHERE c.id = %s
            """,
            (canonical_id,),
        )
        return self._row_to_canonical(row) if row else None

    def get_canonical_by_name(self, name: str) -> Optional[CanonicalEntity]:
        row = self._fetchone(
            "SELECT id FROM canonicals WHERE lower(name) = lower(%s) LIMIT 1", (name,)
        )
        return self.get_canonical(row["id"]) if row else None

    def get_canonical_by_alias(self, alias: str) -> Optional[CanonicalEntity]:
        row = self._fetchone(
            """
            SELECT c.id FROM alias a
            JOIN canonicals c ON c.id = a.canonical_id
            WHERE lower(a.name) = lower(%s) LIMIT 1
            """,
            (alias,),
        )
        return self.get_canonical(row["id"]) if row else None

    def update_canonical(
        self,
        canonical_id: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Optional[CanonicalEntity]:
        sets, params = ["updated_at = now()"], []
        for col, val in (("name", name), ("label", label), ("summary", summary)):
            if val is not None:
                sets.append(f"{col} = %s")
                params.append(val)
        if len(sets) == 1:
            return self.get_canonical(canonical_id)
        params.append(canonical_id)
        updated = self._execute(
            f"UPDATE canonicals SET {', '.join(sets)} WHERE id = %s", tuple(params)
        )
        return self.get_canonical(canonical_id) if updated else None

    def delete_canonical(self, canonical_id: str) -> bool:
        """Aliases are removed automatically (ON DELETE CASCADE)."""
        return self._execute("DELETE FROM canonicals WHERE id = %s", (canonical_id,)) > 0

    def search_canonicals(
        self,
        query: str,
        label: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.30,
    ) -> List[CandidateResult]:
        """
        pg_trgm fuzzy search over names + aliases.
        The `%` operator hits the GIN trgm indexes; its cutoff is
        pg_trgm.similarity_threshold (set per-transaction below).
        """
        if not query or not query.strip():
            return []
        label_val = label.value if isinstance(label, EntityLabels) else label

        sql = """
            SELECT
                c.id, c.name, c.label, c.summary, c.updated_at,
                COALESCE(agg.aliases, '{}')      AS aliases,
                similarity(c.name, %(q)s)        AS name_sim,
                COALESCE(agg.max_alias_sim, 0.0) AS alias_sim,
                GREATEST(similarity(c.name, %(q)s),
                         COALESCE(agg.max_alias_sim, 0.0)) AS score
            FROM canonicals c
            LEFT JOIN LATERAL (
                SELECT array_agg(a.name ORDER BY a.name) AS aliases,
                       max(similarity(a.name, %(q)s))    AS max_alias_sim
                FROM alias a
                WHERE a.canonical_id = c.id
            ) agg ON TRUE
            WHERE (c.name %% %(q)s
                   OR EXISTS (SELECT 1 FROM alias a2
                              WHERE a2.canonical_id = c.id AND a2.name %% %(q)s))
              AND (%(label)s::text IS NULL OR c.label = %(label)s)
            ORDER BY score DESC
            LIMIT %(limit)s
        """
        with self._cursor() as cur:
            cur.execute("SET LOCAL pg_trgm.similarity_threshold = %s", (threshold,))
            cur.execute(sql, {"q": query, "label": label_val, "limit": limit})
            rows = [dict(r) for r in cur.fetchall()]

        q = query.strip().lower()
        results: List[CandidateResult] = []
        for r in rows:
            if r["name"].lower() == q:
                method = CandidateMatchMethod.EXACT
            elif r["alias_sim"] >= r["name_sim"] and r["alias_sim"] > 0:
                method = CandidateMatchMethod.ALIAS
            else:
                method = CandidateMatchMethod.FUZZY
            results.append(
                CandidateResult(
                    canonical_id=r["id"],
                    canonical_name=r["name"],
                    label=r["label"],
                    aliases=list(r["aliases"] or []),
                    summary=r["summary"],
                    source=CandidateSource.POSTGRES,
                    match_method=method,
                    match_score=float(r["score"]),
                    label_match=bool(label_val and r["label"] == label_val),
                )
            )
        return results

    # ─────────────────────────────── ALIASES ──────────────────────────

    def add_alias(self, canonical_id: str, name: str) -> str:
        alias_id = Ids.alias(canonical_id, name)
        
        self._execute(
            """
            INSERT INTO alias (id, canonical_id, name)
            VALUES (%s, %s, %s)
            ON CONFLICT (canonical_id, name) DO NOTHING
            """,
            (alias_id, canonical_id, name),
        )
        return alias_id

    def update_alias(self, alias_id: str, name: str) -> bool:
        return self._execute(
            "UPDATE alias SET name = %s, updated_at = now() WHERE id = %s",
            (name, alias_id),
        ) > 0

    def delete_alias(self, alias_id: str) -> bool:
        return self._execute("DELETE FROM alias WHERE id = %s", (alias_id,)) > 0

    def list_aliases(self, canonical_id: str) -> List[Dict[str, str]]:
        return self._fetchall(
            "SELECT id, name FROM alias WHERE canonical_id = %s ORDER BY name",
            (canonical_id,),
        )

    # ═══════════════════════════ MENTIONS ═════════════════════════════

    def create_mention(self, m: MentionEntity) -> MentionEntity:
        return self.create_mentions([m])[0]

    def create_mentions(self, mentions: List[MentionEntity]) -> List[MentionEntity]:
        """Bulk upsert — re-ingesting a chunk overwrites instead of duplicating."""
        if not mentions:
            return []
        rows = [
            (
                self._mention_id(m.doc_id, m.chunk_id, m.start, m.end),
                m.text, m.label, m.start, m.end, m.mention_sentence,
                m.confidence, m.coref_to, m.doc_id, m.chunk_id,
            )
            for m in mentions
        ]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO mentions (id, name, label, start_offset, end_offset,
                                      mention_sentence, confidence, coref_to, doc_id, chunk_id)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    label = EXCLUDED.label,
                    mention_sentence = EXCLUDED.mention_sentence,
                    confidence = EXCLUDED.confidence,
                    coref_to = EXCLUDED.coref_to,
                    updated_at = now()
                """,
                rows,
                page_size=500,
            )
        return mentions

    def update_mention(
        self,
        doc_id: str,
        chunk_id: Optional[str],
        start: int,
        end: int,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        mention_sentence: Optional[str] = None,
        coref_to: Optional[str] = None,
    ) -> bool:
        sets, params = ["updated_at = now()"], []
        for col, val in (
            ("label", label), ("confidence", confidence),
            ("mention_sentence", mention_sentence), ("coref_to", coref_to),
        ):
            if val is not None:
                sets.append(f"{col} = %s")
                params.append(val)
        if len(sets) == 1:
            return True
        params.extend([doc_id, chunk_id, start, end])
        return self._execute(
            f"""
            UPDATE mentions SET {', '.join(sets)}
            WHERE doc_id = %s AND chunk_id IS NOT DISTINCT FROM %s
              AND start_offset = %s AND end_offset = %s
            """,
            tuple(params),
        ) > 0

    def delete_mention(self, doc_id: str, chunk_id: Optional[str], start: int, end: int) -> bool:
        return self._execute(
            """
            DELETE FROM mentions
            WHERE doc_id = %s AND chunk_id IS NOT DISTINCT FROM %s
              AND start_offset = %s AND end_offset = %s
            """,
            (doc_id, chunk_id, start, end),
        ) > 0

    def delete_mentions(self, mentions: List[MentionEntity]) -> int:
        """Bulk delete using the same composite-id rule as create_mentions."""
        if not mentions:
            return 0
        ids = [self._mention_id(m.doc_id, m.chunk_id, m.start, m.end) for m in mentions]
        return self._execute("DELETE FROM mentions WHERE id = ANY(%s)", (ids,))
    
    def delete_mentions_for_doc(self, doc_id: str) -> int:
        return self._execute("DELETE FROM mentions WHERE doc_id = %s", (doc_id,))

    def search_mentions(
        self,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        label: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 500,
    ) -> List[MentionEntity]:
        clauses, params = [], []
        if doc_id is not None:
            clauses.append("doc_id = %s"); params.append(doc_id)
        if chunk_id is not None:
            clauses.append("chunk_id = %s"); params.append(chunk_id)
        if label is not None:
            clauses.append("label = %s"); params.append(label)
        if name is not None:
            clauses.append("name = %s"); params.append(name)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM mentions {where}ORDER BY doc_id, start_offset LIMIT %s",
            tuple(params),
        )
        return [self._row_to_mention(r) for r in rows]

    # ══════════════════════════ OUTBOX DOCUMENT ═══════════════════════

    def create_document(self, d: OutboxDocument) -> OutboxDocument:
        status = d.status.value if isinstance(d.status, DocumentStatus) else d.status
        self._execute(
            """
            INSERT INTO outbox_document (doc_id, owner_id, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                owner_id = EXCLUDED.owner_id,
                status = EXCLUDED.status,
                updated_at = now()
            """,
            (d.doc_id, d.owner_id, status),
        )
        return d

    def get_document(self, doc_id: str) -> Optional[OutboxDocument]:
        row = self._fetchone("SELECT * FROM outbox_document WHERE doc_id = %s", (doc_id,))
        return self._row_to_document(row) if row else None

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        status_val = status.value if isinstance(status, DocumentStatus) else status
        return self._execute(
            """
            UPDATE outbox_document
            SET status = %s, error_message = %s, updated_at = now()
            WHERE doc_id = %s
            """,
            (status_val, error_message, doc_id),
        ) > 0

    def delete_document(self, doc_id: str) -> bool:
        """Events cascade-delete with the document."""
        return self._execute("DELETE FROM outbox_document WHERE doc_id = %s", (doc_id,)) > 0

    def list_documents(
        self,
        owner_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
    ) -> List[OutboxDocument]:
        clauses, params = [], []
        if owner_id is not None:
            clauses.append("owner_id = %s"); params.append(owner_id)
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value if isinstance(status, DocumentStatus) else status)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM outbox_document {where}ORDER BY created_at DESC LIMIT %s",
            tuple(params),
        )
        return [self._row_to_document(r) for r in rows]

    def refresh_unresolved_count(self, doc_id: str) -> int:
        """Recompute pending_unresolved_count from relation.needs_review (single source of truth)."""
        row = self._fetchone(
            """
            UPDATE outbox_document d
            SET pending_unresolved_count = sub.cnt, updated_at = now()
            FROM (
                SELECT count(*) AS cnt FROM relation
                WHERE doc_id = %s AND needs_review
            ) sub
            WHERE d.doc_id = %s
            RETURNING pending_unresolved_count
            """,
            (doc_id, doc_id),
        )
        return row["pending_unresolved_count"] if row else 0

    def get_documents_by_status(
        self,
        status: Union[DocumentStatus, str],
        owner_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[OutboxDocument]:
        """
        All documents currently in `status`, ordered so the doc that has been
        waiting in this status the longest comes first (FIFO for workers
        and review queues).

        Accepts the enum or its raw string:
            store.get_documents_by_status(DocumentStatus.NEEDS_REVIEW)
            store.get_documents_by_status("NEEDS_REVIEW")
        """
        status_val = status.value if isinstance(status, DocumentStatus) else str(status)

        sql = "SELECT * FROM outbox_document WHERE status = %s"
        params: List[Any] = [status_val]
        if owner_id is not None:
            sql += " AND owner_id = %s"
            params.append(owner_id)
        sql += " ORDER BY updated_at LIMIT %s"
        params.append(limit)

        return [self._row_to_document(r) for r in self._fetchall(sql, tuple(params))]

    # ══════════════════════════ OUTBOX EVENTS ═════════════════════════

    def create_event(self, e: OutboxEvent) -> OutboxEvent:
        return self.create_events([e])[0]

    def create_events(self, events: List[OutboxEvent]) -> List[OutboxEvent]:
        """NOTE: doc row must exist first (FK)."""
        if not events:
            return []
        rows = []
        for e in events:
            if not e.event_id:
                e.event_id = e.event_id or Ids.event(e.doc_id, e.event_type, stage_key=e.stage_key or "", payload=e.payload,)
            rows.append((
                e.event_id, 
                e.doc_id,
                e.event_type.value if isinstance(e.event_type, EventType) else e.event_type,
                [s.value if isinstance(s, ProcessSteps) else s for s in (e.processed_steps or [])],
                psycopg2.extras.Json(e.payload or {}),
                e.processed, e.attempts, e.max_attempts,
            ))
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO outbox_event (event_id, doc_id, event_type, processed_steps,
                                          payload, processed, attempts, max_attempts)
                VALUES %s
                ON CONFLICT (event_id) DO NOTHING
                """,
                rows,
                page_size=500,
            )
        return events

    def get_event(self, event_id: str) -> Optional[OutboxEvent]:
        row = self._fetchone("SELECT * FROM outbox_event WHERE event_id = %s", (event_id,))
        return self._row_to_event(row) if row else None

    def get_events_by(
        self,
        doc_id: Optional[str] = None,
        event_id: Optional[str] = None,
        event_type: Optional[Union[EventType, str]] = None,
        processed: Optional[bool] = None,
        limit: int = 500,
    ) -> List[OutboxEvent]:
        """
        Events filtered by doc_id and/or event_type — either, both, or neither:
            store.get_events_by(doc_id="d1")                              # one doc's history
            store.get_events_by(event_type=EventType.UPSERT_NEO4J)        # one type, all docs
            store.get_events_by(doc_id="d1", event_type=EventType.UPSERT_ES)

        `processed` is an optional extra: None = all, True = done, False = pending/failed.
        With no filters it returns the newest `limit` events (full scan — cap it via limit).
        Ordered newest-first for inspection/debugging.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if doc_id is not None:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        if event_id is not None:
            clauses.append("event_id = %s")
            params.append(event_id)
        if event_type is not None:
            clauses.append("event_type = %s")
            params.append(
                event_type.value if isinstance(event_type, EventType) else str(event_type)
            )
        if processed is not None:
            clauses.append("processed = %s")
            params.append(processed)

        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)

        rows = self._fetchall(
            f"SELECT * FROM outbox_event {where}ORDER BY created_at DESC LIMIT %s",
            tuple(params),
        )
        return [self._row_to_event(r) for r in rows]

    def delete_event(self, event_id: str) -> bool:
        return self._execute("DELETE FROM outbox_event WHERE event_id = %s", (event_id,)) > 0

    def update_event(
        self,
        event_id: str,
        doc_id: Optional[str] = None,
        event_type: Optional[Union[EventType, str]] = None,
        processed_steps: Optional[List[ProcessSteps]] = None,
        payload: Optional[Union[Dict[str, Any], EventPayload]] = None,
        processed: Optional[bool] = None,
        attempts: Optional[int] = None,
        max_attempts: Optional[int] = None,
        error_message: Optional[str] = None,
        processed_at: Optional[datetime] = None,
    ) -> Optional[OutboxEvent]:
        """
        Partial update of one outbox event by its primary key (event_id).
        Only non-None kwargs are written; omitted fields stay untouched.

        Returns the updated OutboxEvent, or None if no row has that event_id.

            store.update_event(eid, processed_steps=[ProcessSteps.SAVE_CANONICAL_POSTGRES])
            store.update_event(eid, processed=True)                # processed_at auto-set to now()
            store.update_event(eid, attempts=0, processed=False)   # requeue for retry
        """
        sets: List[str] = []
        params: List[Any] = []
        explicit_processed_at = processed_at is not None

        if doc_id is not None:
            sets.append("doc_id = %s"); params.append(doc_id)

        if event_type is not None:
            sets.append("event_type = %s")
            params.append(event_type.value if isinstance(event_type, EventType) else str(event_type))

        if processed_steps is not None:
            sets.append("processed_steps = %s")
            params.append([s.value if isinstance(s, ProcessSteps) else str(s)
                        for s in processed_steps])

        if payload is not None:
            if isinstance(payload, EventPayload):
                payload = payload.to_dict()
            sets.append("payload = %s")
            params.append(psycopg2.extras.Json(payload))

        if attempts is not None:
            sets.append("attempts = %s"); params.append(attempts)

        if max_attempts is not None:
            sets.append("max_attempts = %s"); params.append(max_attempts)

        if error_message is not None:
            sets.append("error_message = %s"); params.append(error_message)

        if processed_at is not None:
            sets.append("processed_at = %s"); params.append(processed_at)

        if processed is not None:
            sets.append("processed = %s"); params.append(processed)
            if processed and not explicit_processed_at:
                sets.append("processed_at = now()")     # stamp on success
            elif not processed and not explicit_processed_at:
                sets.append("processed_at = NULL")      # requeued

        if not sets:                                    # nothing to change
            return self.get_event(event_id)

        params.append(event_id)
        row = self._fetchone(
            f"UPDATE outbox_event SET {', '.join(sets)} WHERE event_id = %s RETURNING *",
            tuple(params),
        )
        return self._row_to_event(row) if row else None

    def update_event_obj(self, event: OutboxEvent) -> Optional[OutboxEvent]:
        """Full-row update from an OutboxEvent instance."""
        return self.update_event(
            event_id=event.event_id,
            doc_id=event.doc_id,
            event_type=event.event_type,
            processed_steps=event.processed_steps,
            payload=event.payload if isinstance(event.payload, dict) else event.payload.to_dict(),
            processed=event.processed,
            attempts=event.attempts,
            max_attempts=event.max_attempts,
            error_message=event.error_message,
            processed_at=event.processed_at,
        )
        
    def claim_pending_events(self, limit: int = 50) -> List[OutboxEvent]:
        """
        Poller-safe claim: FOR UPDATE SKIP LOCKED so concurrent pollers never
        grab the same rows. Increments `attempts` on claim.
        """
        events: List[OutboxEvent] = []
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM outbox_event
                WHERE processed = FALSE AND attempts < max_attempts
                ORDER BY created_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                cur.execute(
                    "UPDATE outbox_event SET attempts = attempts + 1 WHERE event_id IN %s",
                    ([r["event_id"] for r in rows],),
                )
                events = [self._row_to_event(r) for r in rows]
        return events

    def mark_event_processed(self, event_id: str) -> bool:
        return self._execute(
            """
            UPDATE outbox_event
            SET processed = TRUE, processed_at = now(), error_message = NULL
            WHERE event_id = %s
            """,
            (event_id,),
        ) > 0

    def mark_event_failed(self, event_id: str, error_message: str) -> bool:
        return self._execute(
            "UPDATE outbox_event SET error_message = %s WHERE event_id = %s",
            (error_message, event_id),
        ) > 0

    def list_dead_letter_events(self, limit: int = 100) -> List[OutboxEvent]:
        rows = self._fetchall(
            """
            SELECT * FROM outbox_event
            WHERE processed = FALSE AND attempts >= max_attempts
            ORDER BY created_at LIMIT %s
            """,
            (limit,),
        )
        return [self._row_to_event(r) for r in rows]

    def search_events(
        self,
        doc_id: Optional[str] = None,
        processed: Optional[bool] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[OutboxEvent]:
        clauses, params = [], []
        if doc_id is not None:
            clauses.append("doc_id = %s"); params.append(doc_id)
        if processed is not None:
            clauses.append("processed = %s"); params.append(processed)
        if event_type is not None:
            clauses.append("event_type = %s")
            params.append(event_type.value if isinstance(event_type, EventType) else event_type)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM outbox_event {where}ORDER BY created_at LIMIT %s",
            tuple(params),
        )
        return [self._row_to_event(r) for r in rows]

    # ═══════════════════════════ RELATIONS ════════════════════════════

    def create_relation(self, r: Relation, flag_below_confidence: Optional[float] = None) -> Relation:
        self.create_relations([r], flag_below_confidence=flag_below_confidence)
        return r

    def create_relations(
        self,
        relations: List[Relation],
        flag_below_confidence: Optional[float] = None,
    ) -> int:
        """
        Bulk upsert. If flag_below_confidence is set, relations below that
        confidence are automatically marked needs_review (the unresolved queue).
        """
        if not relations:
            return 0
        rows = []
        for r in relations:
            if not r.relation_id:
                r.relation_id = r.relation_id or Ids.relation(r.doc_id or "", r.chunk_id or "",r.subject, r.predicate, r.object, r.mention_sentence or "",)
            needs_review = r.needs_review or (
                flag_below_confidence is not None and r.confidence < flag_below_confidence
            )
            r.needs_review = needs_review
            rows.append((
                r.relation_id, r.subject, r.subject_label, r.predicate,
                r.object, r.object_label, r.mention_sentence, r.confidence,
                needs_review, psycopg2.extras.Json(r.evidence or []),
                r.doc_id, r.chunk_id, r.provisional,
            ))
        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO relation (id, subject, subject_label, predicate, object,
                                      object_label, mention_sentence, confidence,
                                      needs_review, evidence, doc_id, chunk_id, provisional)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    subject = EXCLUDED.subject, subject_label = EXCLUDED.subject_label,
                    predicate = EXCLUDED.predicate, object = EXCLUDED.object,
                    object_label = EXCLUDED.object_label,
                    mention_sentence = EXCLUDED.mention_sentence,
                    confidence = EXCLUDED.confidence,
                    needs_review = EXCLUDED.needs_review,
                    evidence = EXCLUDED.evidence, doc_id = EXCLUDED.doc_id,
                    chunk_id = EXCLUDED.chunk_id, provisional = EXCLUDED.provisional,
                    updated_at = now()
                """,
                rows,
                page_size=500,
            )
        return len(rows)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        row = self._fetchone("SELECT * FROM relation WHERE id = %s", (relation_id,))
        return self._row_to_relation(row) if row else None

    def update_relation(
        self,
        relation_id: str,
        subject: Optional[str] = None,
        subject_label: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        object_label: Optional[str] = None,
        confidence: Optional[float] = None,
        needs_review: Optional[bool] = None,
        provisional: Optional[bool] = None,
        evidence: Optional[List[str]] = None,
    ) -> bool:
        sets, params = ["updated_at = now()"], []
        for col, val in (
            ("subject", subject), ("subject_label", subject_label),
            ("predicate", predicate), ("object", object),
            ("object_label", object_label), ("confidence", confidence),
            ("needs_review", needs_review), ("provisional", provisional),
        ):
            if val is not None:
                sets.append(f"{col} = %s"); params.append(val)
        if evidence is not None:
            sets.append("evidence = %s")
            params.append(psycopg2.extras.Json(evidence))
        if len(sets) == 1:
            return True
        params.append(relation_id)
        return self._execute(
            f"UPDATE relation SET {', '.join(sets)} WHERE id = %s", tuple(params)
        ) > 0

    def resolve_relation(
        self,
        relation_id: str,
        subject: Optional[str] = None,
        subject_label: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        object_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Human-in-the-loop: apply corrections (if any) and clear the flags."""
        return self.update_relation(
            relation_id,
            subject=subject, subject_label=subject_label,
            predicate=predicate, object=object, object_label=object_label,
            confidence=confidence,
            needs_review=False, provisional=False,
        )

    def delete_relation(self, relation_id: str) -> bool:
        return self._execute("DELETE FROM relation WHERE id = %s", (relation_id,)) > 0

    def delete_relations_for_doc(self, doc_id: str) -> int:
        return self._execute("DELETE FROM relation WHERE doc_id = %s", (doc_id,))

    def search_relations(
        self,
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        needs_review: Optional[bool] = None,
        predicate: Optional[str] = None,
        subject: Optional[str] = None,
        object: Optional[str] = None,
        min_confidence: Optional[float] = None,
        provisional: Optional[bool] = None,
        limit: int = 200,
    ) -> List[Relation]:
        clauses, params = [], []
        if doc_id is not None:
            clauses.append("doc_id = %s"); params.append(doc_id)
        if chunk_id is not None:
            clauses.append("chunk_id = %s"); params.append(chunk_id)
        if needs_review is not None:
            clauses.append("needs_review = %s"); params.append(needs_review)
        if predicate is not None:
            clauses.append("predicate = %s"); params.append(predicate)
        if subject is not None:
            clauses.append("subject = %s"); params.append(subject)
        if object is not None:
            clauses.append("object = %s"); params.append(object)
        if min_confidence is not None:
            clauses.append("confidence >= %s"); params.append(min_confidence)
        if provisional is not None:
            clauses.append("provisional = %s"); params.append(provisional)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM relation {where}ORDER BY created_at DESC LIMIT %s",
            tuple(params),
        )
        return [self._row_to_relation(r) for r in rows]

    def get_unresolved_relations(
        self, doc_id: Optional[str] = None, limit: int = 100
    ) -> List[Relation]:
        """The human review queue — backed by the partial index."""
        return self.search_relations(doc_id=doc_id, needs_review=True, limit=limit)

    # ─────────────────────────── row mappers ──────────────────────────

    @staticmethod
    def _row_to_canonical(row: Dict) -> CanonicalEntity:
        return CanonicalEntity(
            canonical_id=row["id"],
            name=row["name"],
            label=row["label"],
            aliases=list(row.get("aliases") or []),
            summary=row.get("summary"),
            updated_at=PostgresStateStore._iso(row.get("updated_at")),
        )

    @staticmethod
    def _row_to_mention(row: Dict) -> MentionEntity:
        return MentionEntity(
            text=row["name"],
            label=row["label"],
            start=row["start_offset"],
            end=row["end_offset"],
            mention_sentence=row.get("mention_sentence"),
            confidence=row["confidence"],
            doc_id=row["doc_id"],
            chunk_id=row.get("chunk_id"),
            coref_to=row.get("coref_to"),
        )

    @staticmethod
    def _row_to_document(row: Dict) -> OutboxDocument:
        return OutboxDocument(
            doc_id=row["doc_id"],
            owner_id=row.get("owner_id"),
            status=DocumentStatus(row["status"]),
            pending_unresolved_count=row["pending_unresolved_count"],
            error_message=row.get("error_message"),
            created_at=PostgresStateStore._iso(row.get("created_at")),
            updated_at=PostgresStateStore._iso(row.get("updated_at")),
        )

    @staticmethod
    def _row_to_event(row: Dict) -> OutboxEvent:
        return OutboxEvent(
            doc_id=row["doc_id"],
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            processed_steps=[ProcessSteps(s) for s in (row["processed_steps"] or [])],
            payload=row.get("payload") or {},
            processed=row["processed"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            error_message=row.get("error_message"),
            created_at=row.get("created_at"),
            processed_at=row.get("processed_at"),
        )

    @staticmethod
    def _row_to_relation(row: Dict) -> Relation:
        return Relation(
            relation_id=row["id"],
            subject=row["subject"],
            subject_label=row.get("subject_label"),
            predicate=row["predicate"],
            object=row["object"],
            object_label=row.get("object_label"),
            mention_sentence=row.get("mention_sentence"),
            confidence=row["confidence"],
            needs_review=row["needs_review"],
            evidence=row.get("evidence") or [],
            doc_id=row.get("doc_id"),
            chunk_id=row.get("chunk_id"),
            provisional=row["provisional"],
        )
       
            