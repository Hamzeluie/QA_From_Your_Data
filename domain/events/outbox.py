from .enums import DocumentStatus, EventType, ProcessSteps
from .payloads import EventPayload
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class OutboxDocument:
    doc_id: str
    owner_id: str
    status: DocumentStatus
    pending_unresolved_count:int = 0
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class OutboxEvent:
    doc_id: str
    event_id: str
    event_type: EventType
    
    # Must be a list! A chunk needs to go to BOTH Qdrant and ES.
    processed_steps: List[ProcessSteps] = field(default_factory=list)  # list of steps whcih is processed
    
    # Generic payload. The poller will parse this based on event_type.
    # e.g., {"canonicals": [...], "mentions": [...]}
    payload: Union[Dict[str, Any], EventPayload] = field(default_factory=dict) 
    
    # Poller tracking fields
    processed: bool = False
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    @property
    def typed_payload(self) -> EventPayload:
        if isinstance(self.payload, EventPayload):
            return self.payload
        return EventPayload.from_dict(self.payload or {})
