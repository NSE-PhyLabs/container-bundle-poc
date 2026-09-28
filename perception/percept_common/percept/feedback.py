"""되먹임 저장소 (v3 §7.12). 제어/FSM 이 보낸 ContactFeedback 을 모아 (a) 여유 예측의 편차(bias), (b) 행동별 결과 비율을 낸다.
명세는 docs/contact_feedback.md."""
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .contract import jsonable
from .types import ContactFeedback

log = logging.getLogger(__name__)
CLEARANCE_KEYS = ("uphill", "below", "side_l", "side_r")


class FeedbackStore:
    """JSONL 하나에 쌓고 메모리에 인덱스를 둔다. path=None 이면 메모리에만(테스트)."""

    def __init__(self, path=None, recent_n: int = 5):
        self.path = Path(path) if path else None
        self.recent_n = int(recent_n)
        self.rows: List[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    # ---------- 쓰기 ----------
    def add(self, cb: ContactFeedback, predicted_clearance_mm: Optional[dict] = None, group: Optional[str] = None) -> dict:
        """predicted_clearance_mm 을 함께 주면 그 자리에서 (실측 − 예측) 편차가 계산된다. group 은 bias 를 묶는 단위(기본 scene_id)."""
        row = dict(ts=time.time(), group=group or cb.scene_id, predicted_clearance_mm=predicted_clearance_mm, **cb.to_dict())
        self.rows.append(jsonable(row))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        return row

    # ---------- 읽기 ----------
    def bias(self, group: str, key: str = "uphill") -> Optional[float]:
        """최근 recent_n 건의 (실측 − 예측) 평균 mm. 짝이 없으면 None. group 이 없으면 전체에서 찾는다."""
        vals = []
        for r in reversed(self.rows):
            if group is not None and r.get("group") != group and r.get("scene_id") != group:
                continue
            m, p = r.get("measured_clearance_mm"), r.get("predicted_clearance_mm")
            if m and p and m.get(key) is not None and p.get(key) is not None:
                vals.append(float(m[key]) - float(p[key]))
            if len(vals) >= self.recent_n:
                break
        return float(sum(vals) / len(vals)) if vals else None

    def outcome_rates(self, task: Optional[str] = None, action: Optional[str] = None, stage: Optional[str] = None) -> Dict[str, float]:
        """결과 비율 (Step 8 순위용). task·action 은 아직 기록에 없으면 무시된다."""
        sel = [r for r in self.rows if (stage is None or r.get("stage") == stage)
               and (task is None or r.get("task") in (None, task)) and (action is None or r.get("action") in (None, action))]
        if not sel:
            return {}
        c = Counter(r.get("outcome", "unknown") for r in sel)
        return {k: v / len(sel) for k, v in sorted(c.items())}

    def summary(self) -> dict:
        by_stage = defaultdict(Counter)
        for r in self.rows:
            by_stage[r.get("stage", "?")][r.get("outcome", "unknown")] += 1
        return dict(n=len(self.rows), groups=len({r.get("group") for r in self.rows}),
                    by_stage={k: dict(v) for k, v in sorted(by_stage.items())},
                    bias_uphill_mm=self.bias(None, "uphill"))
