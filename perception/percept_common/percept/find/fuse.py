"""여러 Finder 의 Candidate 병합: IoU > iou_between_finders 면 같은 물체로 보고 점수 높은 쪽 유지.
prefer_split(기본 false, Step 7): 큰 마스크 1개 vs 그 안의 작은 마스크 2개 이상이 경쟁하면, 작은 것들이 각각 카드 치수 범위에 맞을 때 작은 것들을 채택."""
from typing import List, Sequence

import numpy as np

from ..types import Candidate


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return 0.0 if inter == 0 else inter / np.logical_or(a, b).sum()


def fuse(groups: Sequence[List[Candidate]], iou_thr: float, prefer_split: bool = False, size_ok=None) -> List[Candidate]:
    """groups: Finder 별 후보 목록. size_ok(cand) -> bool 은 prefer_split 판정용 (없으면 항상 True)."""
    if len(groups) == 1:
        return list(groups[0])
    pool = sorted((c for g in groups for c in g), key=lambda c: -c.score)
    keep: List[Candidate] = []
    for c in pool:
        dup = [k for k in keep if iou(c.mask, k.mask) > iou_thr]
        if not dup:
            keep.append(c)
    if prefer_split:
        size_ok = size_ok or (lambda c: True)
        out, consumed = [], set()                      # id 기반 (Candidate 는 eq=False)
        for k in keep:
            if id(k) in consumed:
                continue
            inside = [c for c in pool if c is not k and c.finder != k.finder
                      and np.logical_and(c.mask, k.mask).sum() >= 0.8 * c.mask.sum() and c.mask.sum() < 0.8 * k.mask.sum()]
            if len(inside) >= 2 and all(size_ok(c) for c in inside):
                for c in inside:
                    if id(c) not in consumed:
                        out.append(c); consumed.add(id(c))
                consumed.add(id(k))
            else:
                out.append(k); consumed.add(id(k))
        keep = out
    keep.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return keep
