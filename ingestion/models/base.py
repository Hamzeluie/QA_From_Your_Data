import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from abc import ABC, abstractmethod
from typing import List
from shared.data_classes import Entity


class IExtractor(ABC):
    @abstractmethod
    def extract(self, *args, **kwargs)->List[Entity]:
        pass

    def __call__(self, *args, **kwargs)->List[Entity]:
        # Delegate the call to the extract method
        return self.extract(*args, **kwargs)