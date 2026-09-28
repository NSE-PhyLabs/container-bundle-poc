"""Finder 추상: Frame + Card → Candidate 목록 (인스턴스 단위 2D 마스크)."""
from abc import ABC, abstractmethod
from typing import List

from ..registry import Card
from ..types import Candidate, Frame


class Finder(ABC):
    name: str = "finder"

    @abstractmethod
    def find(self, frame: Frame, card: Card) -> List[Candidate]:
        """후보 목록. 순서 규약: 영상 위→아래(y0), 같은 높이면 왼→오른(x0)."""
