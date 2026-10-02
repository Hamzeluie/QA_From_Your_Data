import logging
from typing import Optional, Union, List, Dict, Callable, Awaitable, Tuple
import asyncio
from sentence_transformers import SentenceTransformer
from shared.identical_id import Ids
from config.settings import settings
from storage.qdrant_store import QdrantIndexer
from .factory import StorageFactory
from domain import (MentionEntity, CanonicalEntity, Chunk, 
                    DocumentStatus, OutboxDocument, Relation, 
                    ProcessSteps, EventType, OutboxEvent, 
                    EventPayload,CandidateResult)


logger = logging.getLogger(__name__)
StepExecutor = Callable[[OutboxEvent], Awaitable[None]]

class EventStepMapper:
    """
    Single source of truth: EventType -> ordered ProcessSteps.
    Order matters: Postgres first (source of truth), then Neo4j, then vector/search stores.
    """

    _REGISTRY: Dict[EventType, tuple] = {
        EventType.RESOLVED_ENTITY: (
            ProcessSteps.SAVE_CANONICAL_POSTGRES,
            ProcessSteps.SAVE_MENTION_POSTGRES,
            ProcessSteps.SAVE_CANONICAL_NEO4J,
            ProcessSteps.SAVE_CANONICAL_QDRANT,
            ProcessSteps.SAVE_CHUNKS_QDRANT,
            ProcessSteps.SAVE_CHUNKS_ELASTICSEARCH,
        ),
        EventType.HUMAN_RESOLVED_MENTION: (
            ProcessSteps.SAVE_CANONICAL_POSTGRES,
            ProcessSteps.SAVE_CANONICAL_NEO4J,
            ProcessSteps.SAVE_CANONICAL_QDRANT,
            ProcessSteps.UPDATE_CHUNK_CANONICALS_QDRANT,
            ProcessSteps.SAVE_RESOLVED_MENTION_AS_CANONICAL_PAYLOAD,
            ProcessSteps.DELETE_RESOLVED_MENTION_POSTGRES,
            ProcessSteps.DELETE_MENTION_PAYLOAD,
        ),
        EventType.RELATION_EXTRACTION: (
            ProcessSteps.SAVE_RELATION_POSTGRES,
            ProcessSteps.SAVE_RELATION_NEO4J,   
        ),
        EventType.UPDATE_RELATION: (
            ProcessSteps.SAVE_RELATION_POSTGRES,
            ProcessSteps.SAVE_RELATION_NEO4J,
        ),
        EventType.UPDATE_CANONICAL: (
            ProcessSteps.SAVE_CANONICAL_POSTGRES,
            ProcessSteps.SAVE_CANONICAL_NEO4J,
            ProcessSteps.SAVE_CANONICAL_QDRANT,
        ),
        EventType.UPDATE_CHUNK_METADATA: (
            ProcessSteps.SAVE_CHUNKS_QDRANT,
            ProcessSteps.SAVE_CHUNKS_ELASTICSEARCH,
        ),
        EventType.DELETE_CANONICAL: (),  # wire up when you add DELETE_* steps
        EventType.DELETE_MENTION: (
            ProcessSteps.DELETE_RESOLVED_MENTION_POSTGRES,
            ProcessSteps.DELETE_MENTION_PAYLOAD,
            ),
        EventType.UPLOAD_RAW_TEXT: ()
    }

    # what DocumentStatus to set once all steps succeed
    _FINAL_STATUS: Dict[EventType, DocumentStatus] = {
        EventType.UPLOAD_RAW_TEXT:         DocumentStatus.UPLOADED,
        EventType.RESOLVED_ENTITY:         DocumentStatus.ENTITIES_RESOLVED,
        EventType.HUMAN_RESOLVED_MENTION:  DocumentStatus.ENTITIES_RESOLVED,
        EventType.RELATION_EXTRACTION:     DocumentStatus.RELATION_EXTRACTED,
    }

    @classmethod
    def steps_for(cls, event_type: EventType) -> List[ProcessSteps]:
        """Ordered steps for an event type. Returns a copy — safe to mutate."""
        steps = cls._REGISTRY.get(event_type)
        if steps is None:
            raise KeyError(
                f"No process steps registered for {event_type!r}. "
                f"Use EventStepMapper.register()."
            )
        return list(steps)

    @classmethod
    def pending_steps(cls, event: "OutboxEvent") -> List[ProcessSteps]:
        """Steps not yet completed — enables resuming a crashed event mid-pipeline."""
        steps = cls.steps_for(event.event_type)
        done = set(event.processed_steps or [])
        return [s for s in steps if s not in done]

    @classmethod
    def final_status_for(cls, event_type: EventType) -> Optional[DocumentStatus]:
        return cls._FINAL_STATUS.get(event_type)

    @classmethod
    def register(cls, event_type: EventType, steps: List[ProcessSteps]) -> None:
        """Add/override a pipeline at startup (e.g. for new event types)."""
        cls._REGISTRY[event_type] = tuple(steps)
        

class OutBoxFactory:
    def __init__(self, storage_factory:StorageFactory=None):
        self.storage_factory = storage_factory or StorageFactory.from_env()
        self.outbox_poller = SmartOutboxPoller(storage_factory=self.storage_factory)
        self.storage_factory.init_all()
    
    @staticmethod
    def _merge_canonicals(
        existing: List[CanonicalEntity],
        incoming: List[CanonicalEntity],
    ) -> List[CanonicalEntity]:
        """Append-only merge of the doc's canonical record. Dedupe by canonical_id
        when set, else by (name, label) case-insensitively — also repairs dupes
        already sitting in the payload from earlier rounds."""
        merged, seen_ids, seen_names = [], set(), set()
        for c in list(existing or []) + list(incoming or []):
            name_key = (c.name.strip().lower(), (c.label or "").strip().lower())
            if (c.canonical_id and c.canonical_id in seen_ids) or name_key in seen_names:
                continue
            if c.canonical_id:
                seen_ids.add(c.canonical_id)
            seen_names.add(name_key)
            merged.append(c)
        return merged
    
    # ======== OutBoxPoller Management
    async def run_poll(self):
        return self.outbox_poller.run()
    
    # ======== Document Managment
    async def get_doc(self, owner_id:str, source_name:str):
        doc_id = Ids.document(owner_id=owner_id, source_name=source_name)
        return self.storage_factory.postgres.get_document(doc_id=doc_id)
    
    def get_documents_by_status(
            self,
            status: Union[DocumentStatus, str],
            owner_id: Optional[str] = None,
            limit: int = 500,
        ) -> List[OutboxDocument]:
        return self.storage_factory.postgres.get_documents_by_status(status=status, owner_id=owner_id, limit=limit)
    
    def create_document(self, owner_id:str, source_name:str):
        doc_id = Ids.document(owner_id=owner_id, source_name=source_name)
        doc = OutboxDocument(doc_id=doc_id, owner_id=owner_id, status=DocumentStatus.UPLOADED)
        self.storage_factory.postgres.create_document(d=doc)
        return doc
        
    def transition_state(self, doc_id: str, change_to: DocumentStatus):
        return self.storage_factory.postgres.update_document_status(doc_id=doc_id, status=change_to)
    
    def delete_document(self, doc_id:str):
        return self.storage_factory.postgres.delete_document(doc_id=doc_id)
    
    def get_canonicals_and_chunks(self, doc_id:str):
        event = self.storage_factory.postgres.get_events_by(doc_id=doc_id, event_type=EventType.RESOLVED_ENTITY)[0]
        return [CanonicalEntity.from_dict(cann) for cann in event.payload["canonicals"]], [Chunk.from_dict(ch) for ch in event.payload["chunks"]], event
    
    async def wait_for_doc_status(self, doc_id: str, statuses: set, timeout: float = 300.0,
                                  poll_sec: float = 1.0) -> Optional[OutboxDocument]:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            doc = self.storage_factory.postgres.get_document(doc_id)
            if doc and doc.status in statuses:
                return doc
            await asyncio.sleep(poll_sec)
        return self.storage_factory.postgres.get_document(doc_id)
    
    # ======== Dependent Event Managment        
    def event_create_document(self, owner_id:str, source_name:str, raw_text:str):
        doc = self.create_document(owner_id=owner_id, source_name=source_name)
        
        event = OutboxEvent(
                    doc_id=doc.doc_id,
                    event_id=Ids.event(doc_id=doc.doc_id, event_type=EventType.UPLOAD_RAW_TEXT, stage_key=f"owner_id:{owner_id}"),
                    event_type=EventType.UPLOAD_RAW_TEXT,
                    payload=EventPayload(raw_text=raw_text),
                    processed=True
                    )
                
        self.storage_factory.postgres.create_event(event)
        return event
    
    def event_entity_resolver(self, 
                              event:OutboxEvent,
                              canonicals: List[CanonicalEntity], 
                              mentions: List[MentionEntity], 
                              chunks: List[Chunk]
                              )-> OutboxEvent:
        
        event.event_type = EventType.RESOLVED_ENTITY
        event.payload.canonicals = canonicals
        event.payload.mentions = mentions
        event.payload.chunks = chunks
        event.processed = False
        
        self.storage_factory.postgres.update_event_obj(event)
        self.transition_state(doc_id=event.doc_id, change_to=DocumentStatus.ENTITIES_RESOLVING)
        return event

    def event_relation_extraction(self, event:OutboxEvent, relations:List[Relation])-> OutboxEvent:
        event.event_type = EventType.RELATION_EXTRACTION
        event.payload.relations = relations
        event.processed = False
        
        self.storage_factory.postgres.update_event_obj(event)
        self.transition_state(doc_id=event.doc_id, change_to=DocumentStatus.EXTRACTING_RELATIONS)
        return event
    
    def event_human_resolved_mentions(
        self,
        event_id: str,
        resolved: List[Tuple[MentionEntity, CandidateResult]],
    ) -> OutboxEvent:
        """Partial resolution: ONLY the pairs passed here are resolved.
        Every other mention of the doc stays unresolved (table + payload)."""
        event = self.storage_factory.postgres.get_event(event_id)
        if event is None:
            raise ValueError(f"event {event_id} not found")

        p = event.typed_payload

        # the batch being resolved NOW — delete steps consume exactly this
        p.resolved_mentions = [m for m, _ in resolved]

        # save steps get only THIS batch (re-sending old ones would needlessly
        # re-run Neo4j upserts and Qdrant re-indexing)
        p.canonicals = self._merge_canonicals(
            p.canonicals or [],          # ← accumulated record, preserved
            [                            # ← this round's batch
                CanonicalEntity(
                    canonical_id=cr.canonical_id or None,
                    name=cr.canonical_name,
                    label=cr.label,
                    aliases=[m.name] if m.name and m.name != cr.canonical_name else [],
                    summary=cr.summary,
                )
                for m, cr in resolved
            ],
        )
        
        p.chunk_canonical_links = [
            {"chunk_id": m.chunk_id, "canonical_name": cr.canonical_name, "label": cr.label}
            for m, cr in resolved
            if m.chunk_id                    # mentions without a chunk can't be linked
        ]

        # p.mentions is LEFT ALONE — it still holds the unresolved set;
        # _step_delete_mention_payload subtracts the batch from it.

        event.event_type = EventType.HUMAN_RESOLVED_MENTION
        event.payload = p

        # ── re-queue for a fresh pipeline run ──
        event.processed = False
        self.storage_factory.postgres.update_event_obj(event)
        return event
                
    # ======== Inpendent Event Managment        
    def event_delete_canonicals(self, owner_id:str, canonical_ids:List[str])-> OutboxEvent:
        pass
        
    def event_update_canonicals(self, owner_id:str, canonicals: List[CanonicalEntity])-> OutboxEvent:
        pass
    
    def event_update_relations(self, owner_id:str, relations:List[Relation]):
        pass
    
    def event_delete_relations(self, owner_id:str, relations:List[Relation]):
        pass
    
    def event_delete_mention(self, owner_id:str, mentions:List[MentionEntity]):
        pass
    
    # ======== Storage Managment
    def find_pair_canonical(self, mention: MentionEntity):
        pass
    
    # ======== Relation Extraction
    

class SmartOutboxPoller:
    def __init__(self, storage_factory:StorageFactory):
        self.storage_factory = storage_factory
        model = SentenceTransformer(settings.INDEXING_EMBEDDER_PATH)
        embed = lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        self.qdrant_index = QdrantIndexer(store=self.storage_factory._qdrant, embed=embed)
        # ProcessSteps -> executor. The only wiring point for new steps.
        self._step_executors: Dict[ProcessSteps, StepExecutor] = {
            ProcessSteps.SAVE_CANONICAL_POSTGRES:            self._step_save_canonical_postgres,
            ProcessSteps.SAVE_MENTION_POSTGRES:              self._step_save_mention_postgres,
            ProcessSteps.SAVE_CANONICAL_NEO4J:               self._step_save_canonical_neo4j,
            ProcessSteps.SAVE_CANONICAL_QDRANT:              self._step_save_canonical_qdrant,
            ProcessSteps.SAVE_CHUNKS_QDRANT:                 self._step_save_chunks_qdrant,
            ProcessSteps.SAVE_CHUNKS_ELASTICSEARCH:          self._step_save_chunks_elasticsearch,
            ProcessSteps.SAVE_RELATION_POSTGRES:             self._step_save_relation_postgres,
            ProcessSteps.SAVE_RELATION_NEO4J:                self._step_save_relation_neo4j,
            ProcessSteps.SAVE_RESOLVED_MENTION_AS_CANONICAL_PAYLOAD: self._step_save_resolved_mention_as_canonical_payload,
            ProcessSteps.DELETE_RESOLVED_MENTION_POSTGRES:   self._step_delete_resolved_mention_postgres,
            ProcessSteps.DELETE_MENTION_PAYLOAD:             self._step_delete_mention_payload,
            ProcessSteps.UPDATE_CHUNK_CANONICALS_QDRANT:     self._step_update_chunk_canonicals_qdrant,
        }
    
    @staticmethod
    def _mention_key(m: MentionEntity) -> tuple:
        """Identity of a mention — same rule as PostgresStateStore._mention_id."""
        return (m.doc_id, m.chunk_id, m.start, m.end)

    def _post_event_status(self, event: OutboxEvent) -> Optional[DocumentStatus]:
        """
        The DocumentStatus an event's outcome should drive the doc to,
        or None = leave the doc's status alone (event still retrying,
        or event type has no status transition).

        This replaces the inline while-loops that used to live in
        IngestionPipeline.process_doc and _relation_extraction_loop.
        """
        # ── success: status depends on event type ──
        if event.processed:
            if event.event_type == EventType.RESOLVED_ENTITY:
                # your process_doc conditions:
                # mentions found  -> NEEDS_REVIEW, none -> ENTITIES_RESOLVED
                mentions = event.typed_payload.mentions or []
                return DocumentStatus.NEEDS_REVIEW if mentions else DocumentStatus.ENTITIES_RESOLVED

            if event.event_type == EventType.RELATION_EXTRACTION:
                # your _relation_extraction_loop condition:
                return DocumentStatus.RELATION_EXTRACTED

            if event.event_type == EventType.HUMAN_RESOLVED_MENTION:
                # re-check what's actually left, not what was in the payload
                remaining = self.storage_factory.postgres.search_mentions(doc_id=event.doc_id)
                return DocumentStatus.NEEDS_REVIEW if remaining else DocumentStatus.ENTITIES_RESOLVED

            return EventStepMapper.final_status_for(event.event_type)

        # ── failure: only give up when retries are exhausted ──
        if event.error_message and event.attempts >= event.max_attempts:
            return DocumentStatus.FAILD

        return None   # failed this round but retries remain → doc keeps its current status
    
    async def run(self):
        print("Outbox Poller started...")
        while True:
            try:
                # 1. Fetch a batch of events (e.g., 50 at a time)
                events = await self._fetch_batch()
                
                if events:
                    # 2. Process them
                    await self._process_events(events)
                    
                await asyncio.sleep(2) 
                    
            except Exception as e:
                print(f"Poller error: {e}")
                await asyncio.sleep(5) # Backoff on error

    async def _fetch_batch(self):
        async with self.db_pool.acquire() as conn:
            # CRITICAL: Use FOR UPDATE SKIP LOCKED to prevent race conditions 
            # if you run multiple poller workers!
            return self.storage_factory.postgres.get_events_by(processed=False)
    
    async def _step_save_canonical_postgres(self, event: OutboxEvent) -> bool:
        """
        Create-or-merge: new canonical -> created with mention text as alias;
        existing canonical -> kept, only missing aliases appended.
        Idempotent on retry; never wipes the alias table.
        """
        p = event.typed_payload
        if p.canonicals:
            self.storage_factory.postgres.ensure_canonicals(p.canonicals)
            event.payload = p
        return True

    async def _step_save_mention_postgres(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        if p.mentions:
            self.storage_factory.postgres.create_mentions(p.mentions)
        return True

    async def _step_save_canonical_neo4j(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        for c in (p.canonicals or []):          # return was INSIDE the loop before
            self.storage_factory.neo4j.upsert_canonical(c)
        return True

    async def _step_save_chunks_qdrant(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        if p.chunks:
            self.qdrant_index.index_chunks(p.chunks)
        return True

    async def _step_save_canonical_qdrant(self, event:OutboxEvent) -> bool:
        # registry order guarantees SAVE_CANONICAL_POSTGRES ran first,
        # so canonical_id is resolved on every payload entry by now
        p = event.typed_payload
        if p.canonicals:
            self.qdrant_index.index_canonicals(p.canonicals)
        return True

    async def _step_save_chunks_elasticsearch(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        if p.chunks:
            self.storage_factory.es.index_chunks(p.chunks)
        return True
    
    async def _step_save_relation_postgres(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        if p.relations:
            self.storage_factory.postgres.create_relations(p.relations)
        return True

    async def _step_save_relation_neo4j(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        for r in (p.relations or []):
            self.storage_factory.neo4j.upsert_relation(r)
        return True

    async def _step_save_resolved_mention_as_canonical_payload(self, event:OutboxEvent) -> bool:
        p = event.typed_payload
        if p.resolved_mentions:
            p.canonicals = (p.canonicals or []) + p.resolved_mentions
            event.payload = p
            self.storage_factory.postgres.update_event_obj(event=event)
        return True
    
    async def _step_delete_resolved_mention_postgres(self, event:OutboxEvent) -> bool:
        """Delete ONLY the mentions resolved in this batch. Unresolved stay."""
        p = event.typed_payload
        batch = p.resolved_mentions or []
        if batch:
            self.storage_factory.postgres.delete_mentions(batch)
        return True

    async def _step_delete_mention_payload(self, event:OutboxEvent) -> bool:
        """
        Subtract ONLY the resolved batch from payload.mentions — the
        still-unresolved mentions REMAIN in the payload. Then clear the
        consumed batch field. Runs after DELETE_RESOLVED_MENTION_POSTGRES per the
        registry order, so a crash in between retries cleanly (the pg delete
        is idempotent — already-deleted rows match nothing).
        """
        p = event.typed_payload
        batch = {self._mention_key(m) for m in (p.resolved_mentions or [])}
        if batch:
            p.mentions = [m for m in (p.mentions or [])
                          if self._mention_key(m) not in batch]
            p.resolved_mentions = []
            event.payload = p          # reassign → driver's update_event_obj persists it
            self.storage_factory.postgres.update_event_obj(event=event)
        return True
    
    async def _step_update_chunk_canonicals_qdrant(self, event:OutboxEvent) -> bool:
        """
        Link each human-resolved mention's chunk to the chosen canonical in
        Qdrant (merge mode). Runs AFTER SAVE_CANONICAL_POSTGRES in the
        registry, so payload canonicals carry their resolved canonical_id
        (stamped in-place by ensure_canonical and persisted by that step's
        `event.payload = p` write-back).
        """
        p = event.typed_payload
        links = p.chunk_canonical_links or []
        if not links:
            return True

        id_by_key = {
            ((c.name or "").strip().lower(), (c.label or "").strip().lower()): c.canonical_id
            for c in (p.canonicals or []) if c.canonical_id
        }

        by_chunk: Dict[str, List[str]] = {}
        for link in links:
            key = ((link.get("canonical_name") or "").strip().lower(),
                   (link.get("label") or "").strip().lower())
            cid = id_by_key.get(key)
            if not cid:   # fallback for a mid-pipeline crash-resume edge
                canon = self.storage_factory.postgres.get_canonical_by_name(link.get("canonical_name") or "")
                cid = canon.canonical_id if canon else None
            if not cid or not link.get("chunk_id"):
                continue
            bucket = by_chunk.setdefault(link["chunk_id"], [])
            if cid not in bucket:
                bucket.append(cid)

        for chunk_id, cids in by_chunk.items():
            if not self.qdrant_index.update_chunk_canonicals(chunk_id, cids):
                logger.warning("Qdrant chunk %s missing; canonical link skipped", chunk_id)
        return True

    # ── generic driver: mapper decides WHAT, executors decide HOW ──
    async def _process_event(self, event: OutboxEvent) -> bool:
        # ← the lookup you asked for: RESOLVED_ENTITY pulls its pipeline from the mapper
        event_steps = EventStepMapper.pending_steps(event)
        for step in event_steps:
            executor = self._step_executors.get(step)
            if executor is None:
                self.storage_factory.postgres.mark_event_failed(event_id=event.event_id, error_message=f"No executor registered for step {step!r}")
                return False           
            
            try:
                ok = await executor(event)
            except Exception as e:
                ok = False
                event.error_message = f"{type(e).__name__}: {e}"

            if ok:
                event.processed_steps.append(step)
                all_done = not EventStepMapper.pending_steps(event)
                self.storage_factory.postgres.update_event_obj(event)
                if all_done:
                    # LAST write, so nothing afterwards overwrites processed=TRUE
                    self.storage_factory.postgres.mark_event_processed(event.event_id)
            else:
                event.attempts += 1
                if event.attempts >= event.max_attempts:
                    self.storage_factory.postgres.mark_event_failed(
                        event_id=event.event_id,
                        error_message=event.error_message or f"failed after {event.attempts} attempts in step '{step}'")
                    return False
                self.storage_factory.postgres.update_event_obj(event)
                break
            
        return True

    async def _process_events(self, events: List[OutboxEvent]):
        for event in events:
            try:
                await self._process_event(event)

                # refresh from DB — _process_event persisted the final state there
                event = self.storage_factory.postgres.get_event(event_id=event.event_id)
                if event is None:          # doc was deleted → event row cascaded away
                    continue

                status = self._post_event_status(event)
                if status is not None:
                    self.storage_factory.postgres.update_document_status(
                        doc_id=event.doc_id, status=status)

            except Exception as e:
                # infra-level error (e.g. postgres down). Don't FAILD the doc here —
                # retries may remain; the next poll round's _post_event_status
                # will see the dead state and transition then.
                self.storage_factory.postgres.mark_event_failed(
                    event_id=event.event_id, error_message=str(e))
            
# Run it
# asyncio.run(SmartOutboxPoller(db_pool).run())