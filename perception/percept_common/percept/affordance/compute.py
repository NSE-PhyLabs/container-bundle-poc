"""⑤ 진입점: `compute(scene, hyps, card, gripper, task, frame)` → Affordances (v3 §7.8~7.9).
흐름: 커널 → (타깃마다) 코스트맵 → 후보 생성(2.5D) → 여유 측정 → 선호 거리 가중 → gate.apply_affordance(clearance_short·되먹임 bias) → **그 뒤에** 순위.

후보 자세 규약 (README §6):
- `uv_theta` = [u mm, v mm, θ rad] — 삽입(놓기) 지점의 기준면 좌표와 커널 회전(+n 둘레, u→v 가 +).
- 삽입 지점 높이 = **기준면에서 −blade_zero_mm**(2.5D 판정이 가정한 날 윗면 위치). 놓기는 0(지지면). 관측 표면 높이가 아니다.
- `pose_base` = **접근 자세**: 삽입 지점 − approach.offset_mm·방향 (normal = 용기 안쪽 −n → 접근점은 +n 바깥, uphill = +u, down = base −z).
  회전: x = 커널 긴 축(θ = 0 에서 +v = 물체 축), y = n × x, z = +n(용기 바깥). 놓기도 x = 물체 x(발자국 긴 축)이다.
실패는 AffordanceError/ValueError(→ pipeline 'affordance_failed'). 후보 0 은 실패가 아니라 header.detail='no_feasible_candidate'."""
import logging
import time
from typing import List, Optional

import numpy as np

from ..contract import AffordanceCandidate, Affordances
from ..pose.base import make_T
from ..scene.plane import to_base, to_plane
from .clearance import measure
from .costmap import build, valley_partner
from .generate import generate
from .kernel import footprint_kernel, load_kernel, rotate
from .rank import rank

log = logging.getLogger(__name__)


def _approach_dir(plane, base_frame: bool, direction: str) -> np.ndarray:
    R = plane["T_base_plane"][:3, :3]
    if direction == "normal":
        return -R[:, 2]                                                        # 용기 안쪽(법선 반대)
    if direction == "uphill":
        return R[:, 0]
    return np.array([0.0, 0.0, -1.0]) if base_frame else -R[:, 2]            # down: base -z (카메라 프레임이면 법선 안쪽으로 대신)


def _preferred_weight(dist_m: Optional[float], task: dict) -> float:
    pref = task.get("preferred_distance_m")
    if not pref or dist_m is None or not np.isfinite(dist_m):
        return 1.0
    lo, hi = pref
    if lo <= dist_m <= hi:
        return 1.0
    d = (lo - dist_m) if dist_m < lo else (dist_m - hi)
    return float(max(0.5, 1.0 - d / max(hi - lo, 1e-6)))                     # 밖은 선형 감쇠, 최소 0.5


def exclude_ids(hyps, target, rules: dict, scene) -> List[int]:
    """disturbance_exclude: target · uphill_chain(같은 단에서 타깃보다 경사 위) · valley_partner(경사 아래 골 상대)."""
    if target is None:
        return []
    ex, kinds = set(), set(rules.get("disturbance_exclude", []))
    idx = {o.id: i for i, o in enumerate(hyps.items)}
    tier = lambda i: (int(scene.layers["tier"][scene.layers["instance_id"] == i + 1].max())
                      if (scene.layers["instance_id"] == i + 1).any() else -1)
    u_of = lambda o: float(to_plane(scene.plane, np.asarray(o.position, float)[None])[0][0])
    if "target" in kinds:
        ex.add(idx[target.id])
    if "valley_partner" in kinds:
        vp = valley_partner(hyps, target, scene.plane)
        if vp is not None:
            ex.add(idx[vp[0].id])
    if "uphill_chain" in kinds:
        tt, ut = tier(idx[target.id]), u_of(target)
        ex |= {idx[o.id] for o in hyps.items if o.pose_valid and o.id != target.id and tier(idx[o.id]) == tt and u_of(o) > ut}
    return sorted(ex)


def compute(scene, hyps, card, gripper: dict, task: dict, frame=None, feedback_store=None, group=None) -> Affordances:
    t0 = time.perf_counter()
    kernel = load_kernel(gripper)
    goal = task.get("goal", "extract")
    base_frame = frame is None or not np.allclose(frame.T_base_cam, np.eye(4))
    header = dict(stamp=(frame.stamp if frame is not None else time.time()), frame_id=(frame.frame_id if frame is not None else "base_link"),
                  scene_id=scene.scene_id, gripper=gripper["name"], task=task["name"], source=(frame.source if frame is not None else ""))
    res = float(scene.plane["res_mm"])
    if goal == "place":
        targets = [None]
        size = (card.geometry.get("params") or {}).get("size_mm")
        footprint = footprint_kernel([size[0], size[1]], res) if size else kernel["support"]
    else:
        targets = [o for o in hyps.items if o.pose_valid]
        footprint = None
    if not targets:
        return Affordances.empty(header, "gate_rejected")
    from ..scene.model_surface import fused_height_mm, model_height_mm                # 2.5D 판정면: 모델 있는 셀 = 모델, 없는 셀 = 관측
    scene.layers["height_model_mm"], scene.layers["model_owner"] = model_height_mm(scene.plane, scene.layers["object"].shape, hyps, card)
    scene.layers["height_2p5_mm"] = fused_height_mm(scene.layers["height_mm"], scene.layers["height_model_mm"])
    idx = {o.id: i for i, o in enumerate(hyps.items)}
    raw, cm_cache, support = [], {}, None
    for tgt in targets:
        cost, hard, cmeta = build(scene, task, tgt, hyps, card)
        support = cmeta.get("support", support)
        for c in generate(scene, kernel, task, tgt, cost, hard, card, idx.get(tgt.id) if tgt is not None else None, footprint, cm_cache,
                          cmeta.get("valley"), cmeta.get("slots")):
            raw.append(c)
    if not raw:
        a = Affordances.empty(header, "affordance_failed")
        a.header["detail"] = "no_feasible_candidate"                          # 실패가 아니라 '놓을/넣을 자리가 없음' (2.5D 충돌·hard·격자 밖)
        a.free_space = dict(scene.free_space)
        return a
    by_id = {o.id: o for o in hyps.items}
    R_plane = scene.plane["T_base_plane"][:3, :3]
    ap = kernel["meta"]["approach"]
    h_contact = 0.0 if goal == "place" else -kernel["meta"]["blade_zero_mm"] / 1000.0
    cands = []
    for c in raw:
        tgt = by_id.get(c["target_id"])
        K = rotate(footprint if footprint is not None else kernel["insert"], c["theta_deg"])
        cl = measure(scene, (c["row"], c["col"]), K, exclude_ids(hyps, tgt, task, scene), goal, support,
                     cm_cache.get(c["theta_deg"]) if footprint is None else None, c["blade_offsets"])
        reasons = list(cl.pop("flags"))
        contact = to_base(scene.plane, np.array([[c["u_mm"] / 1000.0, c["v_mm"] / 1000.0, h_contact]]))[0]
        th = np.radians(c["theta_deg"])
        M = np.array([[-np.sin(th), -np.cos(th), 0.0], [np.cos(th), -np.sin(th), 0.0], [0.0, 0.0, 1.0]])   # 열 = [x(긴 축), y = n×x, z = n] (기준면 좌표)
        pos = contact - float(ap.get("offset_mm", 0.0)) / 1000.0 * _approach_dir(scene.plane, base_frame, ap.get("direction", "normal"))
        w = _preferred_weight((tgt.geometry.get("params_fit") or {}).get("camera_distance_m") if tgt is not None else None, task)
        if w < 1.0:
            reasons.append("outside_preferred_distance")
        score = float(c["score"]) * w if c["score"] >= 0 else float(c["score"]) / w          # 음수 점수(비용)는 나눠서 악화
        cands.append(AffordanceCandidate(candidate_id="", target_id=(tgt.id if tgt is not None else ""), action=task["action"],
                                         pose_base=make_T(R_plane @ M, pos).tolist(), uv_theta=[float(c["u_mm"]), float(c["v_mm"]), float(th)],
                                         score=score, clearance_mm={k: float(cl[k]) for k in ("uphill", "below", "side_l", "side_r")},
                                         clearance_measured_mm=None, risk=0.5, rank=0, reasons=reasons))
        c["_obj"] = cands[-1]
    from ..gate import apply_affordance
    affs = apply_affordance(Affordances(header=header, candidates=cands, free_space=dict(scene.free_space)), kernel["meta"], feedback_store, group,
                            check_below=(goal != "place"))
    for c in raw:                                                               # 순위는 감점이 끝난 최종 점수로
        c["score"] = float(c["_obj"].score)
    ordered = rank(raw, hyps, scene, task)
    affs.candidates = []
    for i, c in enumerate(ordered):
        o = c["_obj"]
        o.candidate_id, o.rank, o.risk = f"c{i:02d}", int(c["rank"]), float(c["risk"])
        affs.candidates.append(o)
    affs.header["timing_ms"] = (time.perf_counter() - t0) * 1000.0
    return affs
