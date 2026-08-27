import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import uuid
import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase, Driver, Session

from storage.base import AbstractEntityStore
from shared.data_classes import CandidateResult, OutboxEvent

logger = logging.getLogger(__name__)



class Neo4jEntityStore(AbstractEntityStore):
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Schema ──────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT entity_canonical IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical IS UNIQUE",
            "CREATE CONSTRAINT alias_normalized IF NOT EXISTS FOR (a:Alias) REQUIRE a.normalized IS UNIQUE",
            "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR ()-[r:PREDICT]-() REQUIRE r.relation_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE INDEX entity_source_idx IF NOT EXISTS FOR (e:Entity) ON (e.source)",
            "CREATE INDEX entity_updated_idx IF NOT EXISTS FOR (e:Entity) ON (e.updated_at)",
            "CREATE INDEX doc_id_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_id)",
        ]
        with self._driver.session(database=self.database) as session:
            for cypher in constraints + indexes:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning(f"Schema init warning: {e}")

    # ── Write ─────────────────────────────────────────────────────────────────
    def upsert_entity(
        self,
        canonical: str,
        label: str,
        aliases: List[str],
        summary: Optional[str] = None,
        source: str = "unknown",
        related_to: Optional[List[str]] = None,
        context_indicators: Optional[List[str]] = None,
        embedding_id: Optional[str] = None,
    ) -> None:
        cypher = """
        MERGE (e:Entity {canonical: $canonical})
        ON CREATE SET e.created_at = datetime(), e.mentions = 0
        SET e.label = $label,
            e.summary = $summary,
            e.source = $source,
            e.embedding_id = $embedding_id,
            e.updated_at = datetime()
        WITH e
        CALL {
            WITH e
            UNWIND $aliases AS alias_name
            MERGE (a:Alias {normalized: toLower(trim(alias_name))})
            ON CREATE SET a.name = trim(alias_name)
            MERGE (a)-[r:ALIAS_OF]->(e)
            ON CREATE SET r.weight = 1.0
            RETURN count(*) AS _dummy1
        }
        WITH e
        CALL {
            WITH e
            UNWIND $context_indicators AS ctx
            MERGE (ci:ContextIndicator {text: toLower(trim(ctx))})
            MERGE (ci)-[:INDICATES]->(e)
            RETURN count(*) AS _dummy2
        }
        WITH e
        CALL {
            WITH e
            UNWIND $related_to AS rel_canon
            MATCH (target:Entity {canonical: rel_canon})
            MERGE (e)-[:RELATED_TO]->(target)
            RETURN count(*) AS _dummy3
        }
        WITH e
        CREATE (o:OutboxEvent {
            event_id: $event_id,
            event_type: 'entity_upserted',
            payload: $payload_json,
            targets: $targets,
            processed: false,
            attempts: 0,
            created_at: datetime()
        })
        CREATE (o)-[:FOR_ENTITY]->(e)
        """
        payload = {
            "canonical": canonical,
            "label": label,
            "aliases": aliases,
            "summary": summary,
            "source": source,
            "context_indicators": context_indicators,
            "related_to": related_to,
        }
        with self._driver.session(database=self.database) as session:
            session.run(
                cypher,
                canonical=canonical,
                label=label,
                aliases=list(set(aliases)),
                summary=summary or "",
                source=source,
                embedding_id=embedding_id,
                context_indicators=list(set(context_indicators or [])),
                related_to=list(set(related_to or [])),
                event_id=str(uuid.uuid4()),
                payload_json=json.dumps(payload, ensure_ascii=False),
                targets=["es", "qdrant", "redis"],
            )
            
    def create_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        doc_id: str,
        confidence: float,
        provisional: bool,
        relation_id: str,
        evidence: List[str],
        chunk_id: Optional[str] = None,
    ) -> None:
        cypher = """
        MATCH (sub:Entity {canonical: $subject})
        MATCH (ob:Entity {canonical: $obj})
        MERGE (sub)-[r:PREDICT {relation_id: $relation_id}]->(ob)
        SET r.predicate = $predicate,
            r.doc_id = $doc_id,
            r.confidence = $confidence,
            r.provisional = $provisional,
            r.evidence = $evidence,
            r.chunk_id = $chunk_id,
            r.created_at = datetime()
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                cypher,
                subject=subject,
                obj=obj,
                predicate=predicate,
                relation_id=relation_id,
                doc_id=doc_id,
                confidence=confidence,
                provisional=provisional,
                evidence=evidence,
                chunk_id=chunk_id,
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def find_by_alias(self, normalized_alias: str) -> Optional[Dict]:
        cypher = """
        MATCH (a:Alias {normalized: $normalized})-[:ALIAS_OF]->(e:Entity)
        RETURN e {
            .canonical, .label, .summary, .source, .embedding_id,
            aliases: [(a2:Alias)-[:ALIAS_OF]->(e) | a2.normalized],
            related: [(e)-[:RELATED_TO]->(t) | t.canonical]
        } AS entity
        """
        with self._driver.session(database=self.database) as session:
            rec = session.run(cypher, normalized=normalized_alias.lower().strip()).single()
            return rec["entity"] if rec else None

    def find_by_canonical(self, canonical: str) -> Optional[Dict]:
        cypher = """
        MATCH (e:Entity {canonical: $canonical})
        RETURN e {
            .canonical, .label, .summary, .source, .embedding_id,
            aliases: [(a:Alias)-[:ALIAS_OF]->(e) | a.normalized],
            related: [(e)-[:RELATED_TO]->(t) | t.canonical]
        } AS entity
        """
        with self._driver.session(database=self.database) as session:
            rec = session.run(cypher, canonical=canonical).single()
            return rec["entity"] if rec else None

    def get_related(self, canonical: str) -> List[str]:
        cypher = """
        MATCH (e:Entity {canonical: $canonical})-[:RELATED_TO]->(t:Entity)
        RETURN t.canonical AS related
        """
        with self._driver.session(database=self.database) as session:
            return [r["related"] for r in session.run(cypher, canonical=canonical)]

    def find_candidates_cypher(
        self, entity_text: str, entity_label: Optional[str] = None
    ) -> List[CandidateResult]:
        """
        Three-phase graph search:
        1. Exact alias
        2. Partial / substring
        3. Word overlap
        """
        entity_lower = entity_text.lower().strip()
        words = entity_lower.split()
        results: List[CandidateResult] = []
        seen = set()

        # 1. Exact
        exact_cypher = """
        MATCH (a:Alias {normalized: $entity_lower})-[:ALIAS_OF]->(e:Entity)
        RETURN e.canonical AS canonical, e.label AS label,
               e.summary AS summary, 1.0 AS score, 'exact_match' AS method,
               [(x:Alias)-[:ALIAS_OF]->(e) | x.normalized] AS aliases,
               [(ci:ContextIndicator)-[:INDICATES]->(e) | ci.text] AS indicators,
               [(e)-[:RELATED_TO]->(t) | t.canonical] AS related
        """
        with self._driver.session(database=self.database) as session:
            for rec in session.run(exact_cypher, entity_lower=entity_lower):
                if rec["canonical"] not in seen:
                    seen.add(rec["canonical"])
                    results.append(self._record_to_candidate(rec))

        # 2. Partial
        partial_cypher = """
        MATCH (a:Alias)-[:ALIAS_OF]->(e:Entity)
        WHERE a.normalized CONTAINS $entity_lower OR $entity_lower CONTAINS a.normalized
        RETURN DISTINCT e.canonical AS canonical, e.label AS label,
               e.summary AS summary,
               CASE WHEN a.normalized = $entity_lower THEN 1.0
                    ELSE size($entity_lower) * 1.0 / size(a.normalized)
               END AS score,
               'partial_match' AS method,
               [(x:Alias)-[:ALIAS_OF]->(e) | x.normalized] AS aliases,
               [(ci:ContextIndicator)-[:INDICATES]->(e) | ci.text] AS indicators,
               [(e)-[:RELATED_TO]->(t) | t.canonical] AS related
        """
        with self._driver.session(database=self.database) as session:
            for rec in session.run(partial_cypher, entity_lower=entity_lower):
                if rec["canonical"] not in seen:
                    seen.add(rec["canonical"])
                    results.append(self._record_to_candidate(rec))

        # 3. Word overlap
        if len(words) > 1:
            word_cypher = """
            MATCH (a:Alias)-[:ALIAS_OF]->(e:Entity)
            WHERE a.normalized IN $words
            WITH e, count(DISTINCT a) AS match_count
            RETURN e.canonical AS canonical, e.label AS label,
                   e.summary AS summary,
                   match_count * 1.0 / size($words) AS score,
                   'word_overlap' AS method,
                   [(x:Alias)-[:ALIAS_OF]->(e) | x.normalized] AS aliases,
                   [(ci:ContextIndicator)-[:INDICATES]->(e) | ci.text] AS indicators,
                   [(e)-[:RELATED_TO]->(t) | t.canonical] AS related
            """
            with self._driver.session(database=self.database) as session:
                for rec in session.run(word_cypher, words=words):
                    if rec["canonical"] not in seen:
                        seen.add(rec["canonical"])
                        results.append(self._record_to_candidate(rec))

        return results

    # ── Merge ─────────────────────────────────────────────────────────────────
    def merge_entities(self, absorb: str, keep: str) -> None:
        cypher = """
        MATCH (absorb:Entity {canonical: $absorb})
        MATCH (keep:Entity {canonical: $keep})

        CALL (absorb, keep) {
            OPTIONAL MATCH (a:Alias)-[r:ALIAS_OF]->(absorb)
            WITH keep, a, r WHERE a IS NOT NULL
            MERGE (a)-[:ALIAS_OF {weight: coalesce(r.weight, 1.0)}]->(keep)
            DELETE r
            RETURN count(*) AS _d1
        }

        CALL (absorb, keep) {
            OPTIONAL MATCH (ci:ContextIndicator)-[ri:INDICATES]->(absorb)
            WITH keep, ci, ri WHERE ci IS NOT NULL
            MERGE (ci)-[:INDICATES]->(keep)
            DELETE ri
            RETURN count(*) AS _d2
        }

        CALL (absorb, keep) {
            OPTIONAL MATCH (src)-[rin:PREDICT]->(absorb)
            WITH keep, src, rin WHERE src IS NOT NULL
            MERGE (src)-[rnew:PREDICT {
                relation_id: rin.relation_id,
                doc_id: rin.doc_id,
                predicate: rin.predicate
            }]->(keep)
            SET rnew += properties(rin)
            DELETE rin
            RETURN count(*) AS _d3
        }

        CALL (absorb, keep) {
            OPTIONAL MATCH (absorb)-[rout:PREDICT]->(dst)
            WITH keep, dst, rout WHERE dst IS NOT NULL
            MERGE (keep)-[rnew2:PREDICT {
                relation_id: rout.relation_id,
                doc_id: rout.doc_id,
                predicate: rout.predicate
            }]->(dst)
            SET rnew2 += properties(rout)
            DELETE rout
            RETURN count(*) AS _d4
        }

        WITH keep, absorb
        SET keep.mentions = coalesce(keep.mentions, 0) + coalesce(absorb.mentions, 0)
        SET keep.summary = CASE WHEN keep.summary IS NULL OR keep.summary = ''
                                THEN absorb.summary ELSE keep.summary END
        DETACH DELETE absorb
        WITH keep
        CREATE (o:OutboxEvent {
            event_id: $event_id,
            event_type: 'entity_merged',
            payload: $payload_json,
            targets: ['es', 'qdrant', 'redis'],
            processed: false,
            attempts: 0,
            created_at: datetime()
        })
        CREATE (o)-[:FOR_ENTITY]->(keep)
        """
        payload = {"absorb": absorb, "keep": keep}
        with self._driver.session(database=self.database) as session:
            session.run(
                cypher,
                absorb=absorb,
                keep=keep,
                event_id=str(uuid.uuid4()),
                payload_json=json.dumps(payload),
            )
    
    def create_outbox_event(self, event: OutboxEvent) -> None:
        # Used by external callers if they need custom events
        cypher = """
        MATCH (e:Entity {canonical: $canonical})
        CREATE (o:OutboxEvent {
            event_id: $event_id,
            event_type: $event_type,
            payload: $payload,
            targets: $targets,
            processed: false,
            attempts: 0,
            created_at: datetime()
        })
        CREATE (o)-[:FOR_ENTITY]->(e)
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                cypher,
                canonical=event.canonical,
                event_id=event.event_id,
                event_type=event.event_type,
                payload=event.payload_json,
                targets=event.target_stores,
            )

    def get_pending_outbox(self, limit: int = 100) -> List[OutboxEvent]:
        cypher = """
        MATCH (o:OutboxEvent {processed: false})
        RETURN o.event_id AS event_id,
               o.event_type AS event_type,
               o.payload AS payload_json,
               o.targets AS target_stores,
               o.attempts AS attempts,
               [(o)-[:FOR_ENTITY]->(e:Entity) | e.canonical][0] AS canonical
        ORDER BY o.created_at ASC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            records = []
            for r in session.run(cypher, limit=limit):
                records.append(OutboxEvent(
                    event_id=r["event_id"],
                    event_type=r["event_type"],
                    canonical=r["canonical"],
                    payload_json=r["payload_json"],
                    target_stores=r["target_stores"],
                    attempts=r["attempts"],
                ))
            return records

    def mark_outbox_processed(self, event_id: str) -> None:
        cypher = """
        MATCH (o:OutboxEvent {event_id: $event_id})
        SET o.processed = true,
            o.processed_at = datetime(),
            o.attempts = o.attempts + 1
        """
        with self._driver.session(database=self.database) as session:
            session.run(cypher, event_id=event_id)

    def increment_outbox_attempt(self, event_id: str, error: Optional[str] = None) -> None:
        cypher = """
        MATCH (o:OutboxEvent {event_id: $event_id})
        SET o.attempts = o.attempts + 1,
            o.error_message = $error
        """
        with self._driver.session(database=self.database) as session:
            session.run(cypher, event_id=event_id, error=error)

    @staticmethod
    def _record_to_candidate(rec) -> CandidateResult:
        return CandidateResult(
            canonical=rec["canonical"],
            label=rec["label"],
            aliases=rec["aliases"] or [],
            summary=(rec["summary"] or "")[:120] + "...",
            context_indicators=rec["indicators"] or [],
            related_to=rec["related"] or [],
            match_score=round(float(rec["score"]), 3),
            match_method=rec["method"],
            source="neo4j",
        )
    
    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_candidate(rec) -> CandidateResult:
        return CandidateResult(
            canonical=rec["canonical"],
            label=rec["label"],
            aliases=rec["aliases"] or [],
            summary=(rec["summary"] or "")[:120] + "...",
            context_indicators=rec["indicators"] or [],
            related_to=rec["related"] or [],
            match_score=round(float(rec["score"]), 3),
            match_method=rec["method"],
            source="neo4j",
        )