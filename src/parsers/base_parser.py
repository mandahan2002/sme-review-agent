from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ParsedContent:
    title: str
    full_text: str
    sections: List[dict] = field(default_factory=list)
    format: str = "unknown"
    word_count: int = 0

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(self.full_text.split())


class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParsedContent:
        pass
