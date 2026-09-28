"""참조 사진 특징 매칭 Finder (DINOv2-small + SAM) — Step 5.5 에서 구현. 지금은 NotAvailable 스텁 (인터페이스만)."""
from typing import List

from ..errors import NotAvailable
from ..registry import Card
from ..types import Candidate, Frame
from .base import Finder


class RefMatchFinder(Finder):
    name = "ref_match"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def find(self, frame: Frame, card: Card) -> List[Candidate]:
        raise NotAvailable("ref_match(DINOv2) 는 Step 5.5 에서 구현 — refs/ 특징 파일(refs.feat.npz) 필요")
