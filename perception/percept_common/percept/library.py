"""도서관(ExperienceRecord, v3 §5.3): 실행 1회(프레임 1장) = library/records.jsonl 1줄. 되먹임은 feedback.jsonl 에 따로 쌓고 조회 시 join.
stats() 는 Step 4 (library_stats.py)."""
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from .contract import Affordances, Hypotheses, jsonable
from .registry import PKG_ROOT
from .types import ContactFeedback


def git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PKG_ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


class Library:
    def __init__(self, path):
        self.path = Path(path)
        self.feedback_path = self.path.with_name("feedback.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.version = git_version()

    def record(self, hyps: Hypotheses, affs: Optional[Affordances], card: str, gripper: Optional[dict], task: Optional[dict],
               frame_ref: str, chosen_candidate_id: Optional[str] = None, explore: bool = False) -> dict:
        h = hyps.header
        rec = {"ts": time.time(), "scene_id": h.get("scene_id"), "frame_ref": frame_ref, "card": card,
               "gripper": (gripper or {}).get("name"), "task": (task or {}).get("name"), "source": h.get("source"),
               "finder": sorted({o.extras.get("finder", "") for o in hyps.items} - {""}) or None,
               "backend": sorted({o.backend for o in hyps.items}), "refine_used": any(o.refine_used for o in hyps.items),
               "n_hyp": len(hyps.items), "n_valid": sum(o.pose_valid for o in hyps.items),
               "hypotheses_summary": [dict(id=o.id, confidence=round(o.confidence, 4), pose_valid=o.pose_valid, class_name=o.class_name) for o in hyps.items],
               "affordances_summary": [dict(candidate_id=c.candidate_id, score=c.score, rank=c.rank, clearance_mm=c.clearance_mm) for c in (affs.candidates if affs else [])],
               "chosen_candidate_id": chosen_candidate_id, "timing_ms": h.get("timing_ms", {}), "reason": h.get("reason"),
               "feedback": None, "outcome": "unknown", "explore": explore, "version": self.version}
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(jsonable(rec), ensure_ascii=False) + "\n")
        return rec

    def join(self, cb: ContactFeedback, predicted_clearance_mm=None, group=None) -> None:
        """되먹임을 feedback.jsonl 에 추가 (records.jsonl 은 재작성하지 않음; 조회 시 cb.scene_id+cb.candidate_id 로 join).
        **FeedbackStore.add 와 같은 평면 스키마**로 쓴다 — 한 파일에 두 형식이 섞이면 load 가 죽고 bias 도 못 낸다."""
        from .feedback import FeedbackStore
        FeedbackStore(self.feedback_path).add(cb, predicted_clearance_mm, group or cb.scene_id)

    def load(self, with_feedback: bool = True) -> list:
        recs = [json.loads(l) for l in open(self.path, encoding="utf-8")] if self.path.exists() else []
        if with_feedback and self.feedback_path.exists():
            fb = {}
            for l in open(self.feedback_path, encoding="utf-8"):
                d = json.loads(l)
                fb[(d.get("scene_id"), d.get("candidate_id"))] = d.get("feedback") or d   # 평면·중첩 스키마 모두 허용
            for r in recs:
                f = fb.get((r["scene_id"], r.get("chosen_candidate_id")))
                if f:
                    r["feedback"], r["outcome"] = f, f.get("outcome", "unknown")
        return recs
