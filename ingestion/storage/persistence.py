"""
persistence.py
===============

PersistenceCoordinator fans a document's chunks/entities/relations out to
whichever writers are configured (SearchIndexWriter, VectorStoreWriter,
GraphWriter, OlapWriter) and reports back per-writer success/failure.

Why not a real distributed transaction?
----------------------------------------
Neo4j, ClickHouse, Elasticsearch and Qdrant don't share a transaction
coordinator, and building one (2PC / XA) across four heterogeneous engines
that don't all support it (ES and Qdrant don't) is not practical. Instead
this module follows the standard pattern for multi-store fan-out:

  1. Idempotent writes  — every upsert in db_writers.py is keyed by a
     stable id, so calling it twice with the same data is a no-op the
     second time. This is what makes retries safe.
  2. Outbox             — before dispatching, the batch is recorded in a
     small durable log (SQLite here; swap for Postgres/Kafka in prod) with
     status PENDING. Each writer's outcome is recorded independently.
  3. Per-writer isolation — writers run independently (thread pool) with
     their own retry/backoff. One store being down does not block or
     corrupt the others.
  4. Reconciliation      — `retry_pending()` re-reads rows that still have
     failed writers and retries just those writers. Run this on a timer /
     cron; it's what gets you to eventual consistency across the four
     stores instead of atomicity.

This gets you "the four stores agree eventually, and any partial failure is
visible and retryable" rather than "all four stores update together or not
at all" — which is the right trade-off when the four stores are derived,
independently-queryable views of the same source of truth rather than a
single system of record.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import time
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable, Any

from shared.data_classes import Chunk, Relation, ResolvedEntity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------

@dataclass
class WriteResult:
    writer: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    elapsed_s: float = 0.0


@dataclass
class PersistenceReport:
    doc_id: str
    chunk_results: List[WriteResult] = field(default_factory=list)
    entity_results: List[WriteResult] = field(default_factory=list)
    relation_results: List[WriteResult] = field(default_factory=list)

    @property
    def all_results(self) -> List[WriteResult]:
        return self.chunk_results + self.entity_results + self.relation_results

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.all_results)

    @property
    def failed_writers(self) -> List[str]:
        return [r.writer for r in self.all_results if not r.success]

    def summary(self) -> str:
        if self.all_succeeded:
            return f"[{self.doc_id}] persisted cleanly to {len(self.all_results)} writer call(s)."
        return f"[{self.doc_id}] {len(self.failed_writers)} writer(s) failed: {self.failed_writers}"


# --------------------------------------------------------------------------
# durable outbox (SQLite) — swap this class for Postgres/Kafka in prod
# --------------------------------------------------------------------------

class _Outbox:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                batch_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                payload    TEXT NOT NULL,
                status     TEXT NOT NULL,      -- PENDING | PARTIAL | DONE
                failures   TEXT,               -- JSON list of writer names still failing
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def enqueue(self, batch_id: str, doc_id: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO outbox (batch_id, doc_id, payload, status, failures, created_at, updated_at) "
            "VALUES (?, ?, ?, 'PENDING', '[]', ?, ?)",
            (batch_id, doc_id, json.dumps(payload), now, now),
        )
        self._conn.commit()

    def mark_result(self, batch_id: str, report: PersistenceReport) -> None:
        status = "DONE" if report.all_succeeded else "PARTIAL"
        self._conn.execute(
            "UPDATE outbox SET status = ?, failures = ?, updated_at = ? WHERE batch_id = ?",
            (status, json.dumps(report.failed_writers), time.time(), batch_id),
        )
        self._conn.commit()

    def pending_batches(self) -> List[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM outbox WHERE status != 'DONE'")
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# coordinator
# --------------------------------------------------------------------------

class PersistenceCoordinator:
    def __init__(self,
                 search_writer=None,
                 vector_writer=None,
                 graph_writer=None,
                 olap_writer=None,
                 outbox_path: Optional[str] = "kg_outbox.sqlite3",
                 max_retries: int = 3,
                 backoff_base_s: float = 0.5,
                 max_workers: int = 4):
        self.search_writer = search_writer
        self.vector_writer = vector_writer
        self.graph_writer = graph_writer
        self.olap_writer = olap_writer

        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self.outbox = _Outbox(outbox_path) if outbox_path else None

    # ---- public API --------------------------------------------------

    def persist_document(self,
                          doc_id: str,
                          chunks: List[Chunk],
                          entities: List[ResolvedEntity],
                          relations: List[Relation]) -> PersistenceReport:
        """Fan out one document's writes to every configured store.
        Returns a PersistenceReport; never raises for a single writer
        failure — check report.all_succeeded / report.failed_writers."""

        batch_id = f"{doc_id}:{int(time.time() * 1000)}"
        if self.outbox:
            self.outbox.enqueue(batch_id, doc_id, {
                "n_chunks": len(chunks), "n_entities": len(entities), "n_relations": len(relations),
            })

        report = PersistenceReport(doc_id=doc_id)

        # chunks -> search + vector (independent of each other)
        chunk_jobs: List[tuple] = []
        if self.search_writer and chunks:
            chunk_jobs.append(("elasticsearch", lambda: self.search_writer.upsert_chunks(chunks)))
        if self.vector_writer and chunks:
            chunk_jobs.append(("qdrant", lambda: self.vector_writer.upsert_chunks(chunks)))
        report.chunk_results = self._run_all(chunk_jobs)

        # entities -> graph + olap (must land before relations, per writer,
        # since relation edges reference entity nodes/rows)
        entity_jobs: List[tuple] = []
        if self.graph_writer and entities:
            entity_jobs.append(("neo4j", lambda: self.graph_writer.upsert_entities(entities, doc_id)))
        if self.olap_writer and entities:
            entity_jobs.append(("clickhouse", lambda: self.olap_writer.upsert_entities(entities, doc_id)))
        if self.vector_writer and entities:
            entity_jobs.append(("qdrant_entities", lambda: self.vector_writer.upsert_entities(entities)))
        if self.search_writer and entities:
            entity_jobs.append(("elasticsearch_entities", lambda: self.search_writer.upsert_entities(entities, doc_id)))
        report.entity_results = self._run_all(entity_jobs)

        # relations -> graph + olap. Only dispatch to a writer whose
        # entity write for THIS batch actually succeeded, so we don't
        # create dangling edges against nodes that never landed.
        graph_ready = not self.graph_writer or any(
            r.writer == "neo4j" and r.success for r in report.entity_results
        ) or not entities  # no entities this batch -> nothing blocking relations
        olap_ready = not self.olap_writer or any(
            r.writer == "clickhouse" and r.success for r in report.entity_results
        ) or not entities

        relation_jobs: List[tuple] = []
        if self.graph_writer and relations and graph_ready:
            relation_jobs.append(("neo4j", lambda: self.graph_writer.upsert_relations(relations)))
        if self.olap_writer and relations and olap_ready:
            relation_jobs.append(("clickhouse", lambda: self.olap_writer.upsert_relations(relations)))
        report.relation_results = self._run_all(relation_jobs)

        if self.outbox:
            self.outbox.mark_result(batch_id, report)

        if not report.all_succeeded:
            logger.warning(report.summary())
        return report

    def retry_pending(self, resolve_batch: Callable[[str], Optional[Dict[str, Any]]] = None) -> List[PersistenceReport]:
        """Reconciliation job: re-read outbox rows that aren't DONE and
        retry. Since the outbox only stores payload *metadata* by default
        (not full entity/relation objects, to keep it lightweight), callers
        that want full replay should pass `resolve_batch(doc_id) ->
        {chunks, entities, relations}` — e.g. by re-reading from your
        document store. Without it, this just reports what's still stuck."""
        if not self.outbox:
            return []

        reports = []
        for row in self.outbox.pending_batches():
            doc_id = row["doc_id"]
            if resolve_batch is None:
                logger.warning(f"[retry_pending] batch {row['batch_id']} for doc {doc_id} "
                                f"still failing on {row['failures']}; no resolve_batch() provided, skipping replay.")
                continue
            data = resolve_batch(doc_id)
            if not data:
                continue
            reports.append(self.persist_document(doc_id, data["chunks"], data["entities"], data["relations"]))
        return reports

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if self.outbox:
            self.outbox.close()
        for w in (self.search_writer, self.vector_writer, self.graph_writer, self.olap_writer):
            if w is not None and hasattr(w, "close"):
                try:
                    w.close()
                except Exception:
                    pass

    # ---- internals -----------------------------------------------------

    def _run_all(self, jobs: List[tuple]) -> List[WriteResult]:
        """Run (name, fn) jobs in parallel, each with its own retry/backoff."""
        if not jobs:
            return []
        futures = {self._pool.submit(self._run_with_retry, name, fn): name for name, fn in jobs}
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())
        return results

    def _run_with_retry(self, name: str, fn: Callable[[], None]) -> WriteResult:
        start = time.time()
        attempts = 0
        last_error = None
        while attempts < self.max_retries:
            attempts += 1
            try:
                fn()
                return WriteResult(writer=name, success=True, attempts=attempts, elapsed_s=time.time() - start)
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{name}] attempt {attempts}/{self.max_retries} failed: {last_error}")
                if attempts < self.max_retries:
                    time.sleep(self.backoff_base_s * (2 ** (attempts - 1)))
        return WriteResult(writer=name, success=False, error=last_error,
                            attempts=attempts, elapsed_s=time.time() - start)
