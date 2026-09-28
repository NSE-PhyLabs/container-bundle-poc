"""텍스트 프롬프트 Finder (SAM 3 서버) — 인터페이스 + NotAvailable 스텁. 서버 경로 구현은 승인 후(Step 5.5)."""
from typing import List

from ..errors import NotAvailable
from ..registry import Card
from ..types import Candidate, Frame
from .base import Finder


class Sam3TextFinder(Finder):
    name = "sam3_text"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def find(self, frame: Frame, card: Card) -> List[Candidate]:
        raise NotAvailable(f"sam3_text 서버 미구현 (text_prompt={card.find.get('text_prompt')!r})")
