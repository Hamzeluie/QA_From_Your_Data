import json
import logging
import re
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from domain import (CanonicalEntity, Relation, CandidateResult, 
                    CandidateSource, CandidateMatchMethod)

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Graph projection, mirroring PostgresStateStore's naming:
      - Postgres: names, aliases, fuzzy text lookup, outbox, review workflow
      - Neo4j:    Entity nodes + PREDICT edges keyed by canonical_id / relation_id

    No Alias concept here at all: Postgres resolves text -> canonical_id,
    Neo4j only ever sees canonical_id.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.auth = (user, password)
        self.database = database
        self._driver: Optional[Driver] = None

    # ── connection ────────────────────────────────────────────────────────

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self._driver.verify_connectivity()
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read(self, cypher: str, **params) -> List[Dict[str, Any]]:
        with self._get_driver().session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, **params)]

    def _write(self, cypher: str, **params) -> List[Dict[str, Any]]:
        def _tx(tx):
            return [dict(r) for r in tx.run(cypher, **params)]
        with self._get_driver().session(database=self.database) as session:
            return session.execute_write(_tx)

    # ── schema ────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
            "CREATE CONSTRAINT relation_id IF NOT EXISTS "
            "FOR ()-[r:PREDICT]-() REQUIRE r.relation_id IS UNIQUE",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.summary]",
        ]
        for stmt in statements:
            try:
                self._write(stmt)
            except Exception as e:
                logger.warning("Schema init warning: %s", e)

    # ── helpers ───────────────────────────────────────────────────────────

    _PRIMITIVE = (str, int, float, bool)
    _LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/])')

    @classmethod
    def _sanitize_props(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Neo4j property values must be primitives or lists of primitives:
          - nested dict / list-of-dict -> JSON string
          - None -> dropped (writing null would DELETE the property)
        """
        if not metadata:
            return {}
        out: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, dict):
                out[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (list, tuple)):
                out[key] = (
                    list(value)
                    if all(isinstance(x, cls._PRIMITIVE) for x in value)
                    else json.dumps(value, ensure_ascii=False)
                )
            elif isinstance(value, cls._PRIMITIVE):
                out[key] = value
            else:
                out[key] = str(value)
        return out

    @classmethod
    def _escape_fts(cls, query: str) -> str:
        return cls._LUCENE_SPECIAL.sub(r"\\\1", query.strip())

    @staticmethod
    def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
        """neo4j DateTime -> ISO string so the dicts are JSON-safe."""
        return {
            k: (v.iso_format() if hasattr(v, "iso_format") else v)
            for k, v in record.items()
        }

    # ═══════════════════════════ CANONICALS ═══════════════════════════════
    def upsert_canonical(
        self,
        entity: CanonicalEntity,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create/update an Entity node. Document provenance lives in
        Postgres.mentions — the graph holds identities and edges only.
        """
        cypher = """
        MERGE (e:Entity {canonical_id: $canonical_id})
        ON CREATE SET e.created_at = datetime()
        SET e.name = $name,
            e.label = $label,
            e.summary = $summary,
            e.updated_at = datetime()
        SET e += $metadata
        RETURN e.canonical_id AS canonical_id
        """
        rows = self._write(
            cypher,
            canonical_id=entity.canonical_id,
            name=entity.name,
            label=entity.label,
            summary=entity.summary,
            metadata=self._sanitize_props(metadata),
        )
        return rows[0]["canonical_id"]
    
    def get_canonical(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        rows = self._read(
            "MATCH (e:Entity {canonical_id: $cid}) RETURN e {.*} AS entity",
            cid=canonical_id,
        )
        return self._normalize(rows[0]["entity"]) if rows else None

    def get_canonicals(self, canonical_ids: List[str]) -> List[Dict[str, Any]]:
        if not canonical_ids:
            return []
        rows = self._read(
            "MATCH (e:Entity) WHERE e.canonical_id IN $ids RETURN e {.*} AS entity",
            ids=canonical_ids,
        )
        return [self._normalize(r["entity"]) for r in rows]

    def update_canonical(
        self,
        canonical_id: str,
        name: Optional[str] = None,
        label: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Partial update — only supplied fields change."""
        props = {k: v for k, v in
                 {"name": name, "label": label, "summary": summary}.items()
                 if v is not None}
        props.update(self._sanitize_props(metadata))

        rows = self._write(
            "MATCH (e:Entity {canonical_id: $cid}) "
            "SET e += $props, e.updated_at = datetime() "
            "RETURN e.canonical_id AS id",
            cid=canonical_id, props=props,
        )
        return bool(rows)

    def delete_canonical(self, canonical_id: str) -> bool:
        rows = self._write(
            "MATCH (e:Entity {canonical_id: $cid}) "
            "DETACH DELETE e RETURN count(*) AS deleted",
            cid=canonical_id,
        )
        return bool(rows and rows[0]["deleted"] > 0)

    def merge_entities(self, absorb_id: str, keep_id: str) -> bool:
        """
        Merge absorb -> keep: retarget all PREDICT edges, fill empty summary,
        DETACH DELETE absorb.
        """
        cypher = """
        MATCH (absorb:Entity {canonical_id: $absorb})
        MATCH (keep:Entity {canonical_id: $keep})
        CALL (absorb, keep) {
            MATCH (src:Entity)-[r:PREDICT]->(absorb)
            WHERE src <> keep
            WITH src, keep, r, properties(r) AS props
            DELETE r
            CREATE (src)-[nr:PREDICT]->(keep)
            SET nr = props
            RETURN count(*) AS moved_in
        }
        CALL (absorb, keep) {
            MATCH (absorb)-[r:PREDICT]->(dst:Entity)
            WHERE dst <> keep
            WITH dst, keep, r, properties(r) AS props
            DELETE r
            CREATE (keep)-[nr:PREDICT]->(dst)
            SET nr = props
            RETURN count(*) AS moved_out
        }
        WITH keep, absorb
        SET keep.summary = CASE WHEN coalesce(keep.summary, '') = ''
                                THEN absorb.summary ELSE keep.summary END,
            keep.updated_at = datetime()
        DETACH DELETE absorb
        RETURN keep.canonical_id AS kept
        """
        return bool(self._write(cypher, absorb=absorb_id, keep=keep_id))
    
    # ── canonical search ──────────────────────────────────────────────────

    def search_canonicals(
        self,
        query: str,
        label: Optional[Any] = None,          # EntityLabels | str
        limit: int = 10,
    ) -> List[CandidateResult]:
        """
        Graph-local search over node names. For entity RESOLUTION prefer
        Postgres search_canonicals (pg_trgm over names + aliases); this is
        for exploration inside the graph.

        Phases: 1) exact name   2) fulltext (name + summary)
        """
        if not query or not query.strip():
            return []
        label_val = label.value if hasattr(label, "value") else label
        results: Dict[str, CandidateResult] = {}

        exact_rows = self._read(
            """
            MATCH (e:Entity)
            WHERE (e.name = $q OR toLower(e.name) = toLower($q))
              AND ($label IS NULL OR e.label = $label)
            RETURN e.canonical_id AS canonical_id, e.name AS name,
                   e.label AS label, e.summary AS summary,
                   CASE WHEN e.name = $q THEN 1.0 ELSE 0.95 END AS score
            LIMIT $limit
            """,
            q=query.strip(), label=label_val, limit=limit,
        )
        for r in exact_rows:
            results[r["canonical_id"]] = self._to_candidate(
                r, CandidateMatchMethod.EXACT if r["score"] == 1.0
                   else CandidateMatchMethod.ALIAS,
                r["score"], label_val,
            )

        if len(results) < limit:
            fts_rows = self._read(
                """
                CALL db.index.fulltext.queryNodes('entity_fts', $fts)
                YIELD node, score
                WHERE node:Entity AND ($label IS NULL OR node.label = $label)
                RETURN node.canonical_id AS canonical_id, node.name AS name,
                       node.label AS label, node.summary AS summary, score
                LIMIT $limit
                """,
                fts=self._escape_fts(query), label=label_val, limit=limit,
            )
            for r in fts_rows:
                if r["canonical_id"] not in results:
                    results[r["canonical_id"]] = self._to_candidate(
                        r, CandidateMatchMethod.FUZZY, float(r["score"]), label_val
                    )

        ranked = sorted(results.values(), key=lambda c: c.match_score, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _to_candidate(row, method: CandidateMatchMethod, score: float,
                      label_filter: Optional[str]) -> CandidateResult:
        return CandidateResult(
            canonical_id=row["canonical_id"],
            canonical_name=row["name"],
            label=row["label"],
            summary=row.get("summary"),
            source=CandidateSource.NEO4J,
            match_method=method,
            match_score=round(float(score), 3),
            label_match=bool(label_filter and row["label"] == label_filter),
        )

    # ═══════════════════════════ RELATIONS ════════════════════════════════

    def upsert_relation(
        self,
        relation: Relation,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create/update a PREDICT edge. Both endpoints must exist already
        (sync canonicals first). Returns False if an endpoint is missing.
        doc_id is appended to r.doc_ids provenance.
        """
        cypher = """
        MATCH (s:Entity {canonical_id: $subject})
        MATCH (o:Entity {canonical_id: $object})
        MERGE (s)-[r:PREDICT {relation_id: $relation_id}]->(o)
        ON CREATE SET r.created_at = datetime()
        SET r.predicate = $predicate,
            r.confidence = $confidence,
            r.needs_review = $needs_review,
            r.provisional = $provisional,
            r.evidence = $evidence,
            r.mention_sentence = $mention_sentence,
            r.subject_label = $subject_label,
            r.object_label = $object_label,
            r.updated_at = datetime(),
            r.doc_ids = CASE WHEN $doc_id IS NULL OR $doc_id IN coalesce(r.doc_ids, [])
                             THEN coalesce(r.doc_ids, [])
                             ELSE coalesce(r.doc_ids, []) + $doc_id END
        SET r += $metadata
        RETURN r.relation_id AS relation_id
        """
        rows = self._write(
            cypher,
            subject=relation.subject,
            object=relation.object,
            relation_id=relation.relation_id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            needs_review=relation.needs_review,
            provisional=relation.provisional,
            evidence=relation.evidence or [],
            mention_sentence=relation.mention_sentence,
            subject_label=relation.subject_label,
            object_label=relation.object_label,
            doc_id=relation.doc_id,
            metadata=self._sanitize_props(metadata),
        )
        if not rows:
            logger.warning("Relation %s skipped: endpoint missing (%s -> %s)",
                           relation.relation_id, relation.subject, relation.object)
        return bool(rows)

    def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        rows = self._read(
            """
            MATCH (s:Entity)-[r:PREDICT {relation_id: $rid}]->(o:Entity)
            RETURN s.canonical_id AS subject, s.name AS subject_name,
                   o.canonical_id AS object, o.name AS object_name,
                   properties(r) AS props
            """,
            rid=relation_id,
        )
        return self._relation_dict(rows[0]) if rows else None

    def get_relations(
        self,
        canonical_id: Optional[str] = None,   # matches subject OR object
        predicate: Optional[str] = None,
        doc_id: Optional[str] = None,
        needs_review: Optional[bool] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        rows = self._read(
            """
            MATCH (s:Entity)-[r:PREDICT]->(o:Entity)
            WHERE ($canonical_id IS NULL OR s.canonical_id = $canonical_id
                   OR o.canonical_id = $canonical_id)
              AND ($predicate IS NULL OR r.predicate = $predicate)
              AND ($doc_id IS NULL OR $doc_id IN coalesce(r.doc_ids, []))
              AND ($needs_review IS NULL OR r.needs_review = $needs_review)
            RETURN s.canonical_id AS subject, s.name AS subject_name,
                   o.canonical_id AS object, o.name AS object_name,
                   properties(r) AS props
            ORDER BY r.confidence DESC LIMIT $limit
            """,
            canonical_id=canonical_id, predicate=predicate,
            doc_id=doc_id, needs_review=needs_review, limit=limit,
        )
        return [self._relation_dict(r) for r in rows]

    def update_relation(
        self,
        relation_id: str,
        predicate: Optional[str] = None,
        confidence: Optional[float] = None,
        needs_review: Optional[bool] = None,
        provisional: Optional[bool] = None,
        evidence: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        props = {k: v for k, v in {
            "predicate": predicate, "confidence": confidence,
            "needs_review": needs_review, "provisional": provisional,
            "evidence": evidence,
        }.items() if v is not None}
        props.update(self._sanitize_props(metadata))
        rows = self._write(
            """
            MATCH ()-[r:PREDICT {relation_id: $rid}]->()
            SET r += $props, r.updated_at = datetime()
            RETURN count(*) AS matched
            """,
            rid=relation_id, props=props,
        )
        return bool(rows and rows[0]["matched"] > 0)

    def rebind_relation(
        self,
        relation_id: str,
        new_subject: Optional[str] = None,
        new_object: Optional[str] = None,
    ) -> bool:
        """
        Graph side of Postgres' resolve_relation: move the edge to corrected
        endpoint(s), keeping every property (relation_id, evidence, provenance).
        """
        rows = self._write(
            """
            MATCH (os:Entity)-[old:PREDICT {relation_id: $rid}]->(oe:Entity)
            WITH old, properties(old) AS props, os, oe
            DELETE old
            MATCH (s:Entity {canonical_id: coalesce($new_subject, os.canonical_id)})
            MATCH (o:Entity {canonical_id: coalesce($new_object, oe.canonical_id)})
            CREATE (s)-[r:PREDICT]->(o)
            SET r = props, r.updated_at = datetime()
            RETURN r.relation_id AS relation_id
            """,
            rid=relation_id, new_subject=new_subject, new_object=new_object,
        )
        return bool(rows)

    def delete_relation(self, relation_id: str) -> bool:
        rows = self._write(
            "MATCH ()-[r:PREDICT {relation_id: $rid}]->() "
            "DELETE r RETURN count(*) AS deleted",
            rid=relation_id,
        )
        return bool(rows and rows[0]["deleted"] > 0)

    def delete_relations_for_doc(self, doc_id: str) -> int:
        rows = self._write(
            """
            MATCH ()-[r:PREDICT]->()
            WHERE $doc_id IN coalesce(r.doc_ids, [])
            DELETE r RETURN count(*) AS deleted
            """,
            doc_id=doc_id,
        )
        return rows[0]["deleted"] if rows else 0

    # ── traversal / misc ──────────────────────────────────────────────────

    def get_neighbors(self, canonical_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._read(
            """
            MATCH (e:Entity {canonical_id: $cid})-[r:PREDICT]-(other:Entity)
            RETURN other.canonical_id AS canonical_id, other.name AS name,
                   other.label AS label, r.predicate AS predicate,
                   r.confidence AS confidence, r.needs_review AS needs_review,
                   CASE WHEN startNode(r).canonical_id = $cid
                        THEN 'outgoing' ELSE 'incoming' END AS direction
            ORDER BY r.confidence DESC LIMIT $limit
            """,
            cid=canonical_id, limit=limit,
        )

    def get_stats(self) -> Dict[str, int]:
        rows = self._read(
            """
            MATCH (e:Entity) WITH count(e) AS entities
            OPTIONAL MATCH ()-[r:PREDICT]->()
            RETURN entities, count(r) AS relations
            """
        )
        row = rows[0] if rows else {}
        return {"entities": row.get("entities", 0), "relations": row.get("relations", 0)}

    # ── row assembly ──────────────────────────────────────────────────────

    def _relation_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        props = dict(row["props"])
        meta_keys = props.pop("metadata_keys", [])
        props["metadata"] = {k: props.pop(k) for k in meta_keys if k in props}
        props["subject"], props["subject_name"] = row["subject"], row["subject_name"]
        props["object"], props["object_name"] = row["object"], row["object_name"]
        return self._normalize(props)     
    
    