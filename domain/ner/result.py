from ..documents import Chunk
from ..entities import MentionEntity
from typing import List
from dataclasses import dataclass, field


@dataclass
class NEResult:
    chunks: List[Chunk] = field(default_factory=list)
    entities: List[MentionEntity] = field(default_factory=list)
