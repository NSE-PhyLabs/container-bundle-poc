"""⑥ 검사: 신뢰도 정규화·유효성·축소 모드·extras·neighbors·시간 → Hypotheses (v3 §7.9).

유효성(pose_valid=False + invalid_reason): too_few_points / fit_failed(사유) / 치수 범위 밖(card gate.fit_valid, 원본 verdict :132-137 문자열) /
관측 거리 밖 / partial box 요 신뢰 낮음(Step 5) / stable_frames 미달(연속 모드만). 축소 모드(deformable·none·backend none) = 포즈 항등, backend 'none',
bbox3d·마스크 중심만, invalid_reason 'degraded_mode'.
extras 는 카드 extras.compute 만: axis{dir, d_perp} · grooves(V골, grasp_point.py valley_dist :54-66) · row_level{level,row,n_in_level}(box_normal :135-154
+ select_target :157-172 의 tie_mm·id 규칙) · budget_uphill(SceneMap.free_space.direct_mm, Step 4). 예외: 축소 모드 mask_center_px, 연속 모드 stability.
neighbors: 같은 카드 인스턴스 중 중심 거리 < 2×반경(box: size 최대/2)+margin → displacement(base), contact = 거리 < 2×반경+contact_margin.
장면 단위 계산은 원본처럼 **카메라 좌표**(부호 전제: 영상 위 = 경사 위, 카메라에서 먼 쪽 = 바닥)에서 하고 결과만 base 로 돌린다.
"""
import time
from typing import List, Optional

import numpy as np

from .contract import Hypotheses, ObjectHypothesis, SceneMap
from .lift import m2mm, mm2m
from .pose.base import invert_T, quat_from_matrix, unit
from .registry import Card
from .types import Candidate, Frame, Instance, PoseResult


def confidence_from(score_raw: float, backend: str, card: Card, fit: Optional[dict] = None) -> float:
    """카드 confidence_map[backend] 규칙으로 0~1. {rms_mm_good, rms_mm_bad}: 작을수록 좋은 잔차(mm) / {score_good, score_bad}: 클수록 좋은 점수 / identity.
    길이 범위 이탈: gate.length_penalty.per_mm × length_outside_mm 만큼 confidence 감산."""
    base = backend.split("(")[0].split("+")[0]
    rule = card.gate["confidence_map"].get(base)
    if rule is None or score_raw is None or not np.isfinite(score_raw):
        return 0.0
    x = float(score_raw)
    if "rms_mm_good" in rule:
        good, bad = float(rule["rms_mm_good"]), float(rule["rms_mm_bad"])
        conf = (bad - x) / (bad - good)
    elif "score_good" in rule:
        good, bad = float(rule["score_good"]), float(rule["score_bad"])
        conf = (x - bad) / (good - bad)
    else:
        conf = x
    pen = card.gate.get("length_penalty") or {}
    if fit and pen.get("per_mm"):
        conf -= float(pen["per_mm"]) * float(fit.get("length_outside_mm", 0.0))
    return float(np.clip(conf, 0.0, 1.0))


def _fit_valid_reason(fit: dict, card: Card) -> Optional[str]:
    if fit.get("partial") and float(fit.get("top_coverage", 1.0)) < float((card.gate or {}).get("partial_min_coverage", 0.8)):
        return "partial_box"                                                 # 측면도 없고 상면도 다 안 보임 → 요·중심을 못 믿는다 [추측 0.8]
    names = {"length_mm": "길이", "cross_mm": "굵기"}
    for key, (lo, hi) in (card.gate.get("fit_valid") or {}).items():
        v = fit.get(key)
        if v is not None and not (lo <= v <= hi):
            return f"{names.get(key, key)} 이상({v:.0f}mm)"
    return None


class StabilityTracker:
    """연속 모드: 같은 물체(최근접 중심)가 stable_frames 연속 stable_tol 안에 머물면 안정(locked)."""

    def __init__(self):
        self.hist: List[List[np.ndarray]] = []

    def update(self, centers: List[np.ndarray], tol_m: float, need: int) -> List[int]:
        counts = []
        for c in centers:
            n = 1
            for prev in reversed(self.hist):
                if prev and min(np.linalg.norm(c - p) for p in prev) <= tol_m:
                    n += 1
                else:
                    break
            counts.append(n)
        self.hist.append(list(centers))
        self.hist = self.hist[-max(need, 1):]
        return counts


def _rows_with_ties(dot_cs: np.ndarray, members: List[int], ids: List[int], tie_mm: float) -> List[List[int]]:
    """단 안의 줄 순위: dot_cs 최대에서 tie_mm 이내를 한 묶음(동점)으로, 묶음 안은 id 오름차순. 남은 것으로 반복 (select_target :168-172 규칙)."""
    rest, groups = sorted(members, key=lambda k: -dot_cs[k]), []
    while rest:
        hi = dot_cs[rest[0]]
        tied = [k for k in rest if hi - dot_cs[k] <= tie_mm]
        groups.append(sorted(tied, key=lambda k: ids[k]))
        rest = [k for k in rest if k not in tied]
    return groups


def _scene_extras(objs: List[ObjectHypothesis], poses: List[Optional[PoseResult]], card: Card, frame: Frame) -> dict:
    """axis / grooves / row_level (장면 단위, 유효 포즈만). 반환: header 용 scene dict. extras 는 objs 에 직접 기록."""
    want = set(card.extras.get("compute", []))
    if not (want & {"grooves", "row_level", "axis"}):
        return {}
    T_cb = invert_T(frame.T_base_cam)
    R_cb, R_bc, t_bc = T_cb[:3, :3], frame.T_base_cam[:3, :3], frame.T_base_cam[:3, 3]
    idx = [i for i, (o, p) in enumerate(zip(objs, poses)) if p is not None and o.pose_valid]
    C = [R_cb @ np.asarray(objs[i].position) + T_cb[:3, 3] for i in idx]                 # 카메라 좌표 m
    A = []
    for i in idx:
        a = R_cb @ poses[i].T_base_obj[:3, 0]                                             # x 열 = 축 (카메라 좌표)
        A.append(-a if a[0] < 0 else a)                                                   # 부호 정규화 (grasp_point.py:187)
    for i, a in zip(idx, A):
        if "axis" in want:
            objs[i].extras["axis"] = dict(dir=[float(v) for v in (R_bc @ a)], d_perp=None)
    scene = dict(n_valid=len(idx), normal=None, s_uphill=None, reason="")
    if not (want & {"grooves", "row_level"}):
        return scene
    npar, vp, tie_mm = card.extras["normal"], card.extras["valley"], float(card.extras["tie_mm"])
    if len(idx) < npar["min_bundles"]:
        scene["reason"] = f"묶음 {len(idx)}개 (최소 {npar['min_bundles']}개)"
        return scene
    Cmm = m2mm(np.array(C))
    s = np.linalg.svd(Cmm - Cmm.mean(axis=0), full_matrices=False)[2][0]                  # box_normal (:135-154)
    if s[1] > 0:
        s = -s
    axis = unit(np.mean(A, axis=0))
    if abs(s @ axis) > npar["max_abs_dot_s_axis"]:
        scene["reason"] = f"줄 방향이 축과 평행 (|s·axis| {abs(s @ axis):.2f})"
        return scene
    n = unit(np.cross(axis, s))
    if n[2] < 0:
        n = -n
    scene.update(normal=[float(v) for v in (R_bc @ n)], s_uphill=[float(v) for v in (R_bc @ s)])
    for i, a in zip(idx, A):
        if "axis" in want:
            objs[i].extras["axis"]["d_perp"] = [float(v) for v in (R_bc @ unit(n - (n @ a) * a))]   # 4) 바닥 쪽 수직 방향
    if "grooves" in want:                                                                  # valley_dist (:54-66)
        for i in idx:
            objs[i].extras["grooves"] = []
        for k1 in range(len(idx)):
            for k2 in range(k1 + 1, len(idx)):
                d = Cmm[k2] - Cmm[k1]
                dist = float(np.linalg.norm(d))
                if not vp["dist_min_mm"] <= dist <= vp["dist_max_mm"]:
                    continue
                u = d / dist
                if abs(u @ unit(A[k1] + A[k2])) >= vp["max_abs_dot_u_axis"] or abs(u @ n) >= vp["max_abs_dot_u_n"]:
                    continue
                pt = [float(v) for v in (R_bc @ ((C[k1] + C[k2]) / 2) + t_bc)]
                i1, i2 = idx[k1], idx[k2]
                objs[i1].extras["grooves"].append(dict(with_id=objs[i2].id, point=pt, dist_mm=dist))
                objs[i2].extras["grooves"].append(dict(with_id=objs[i1].id, point=pt, dist_mm=dist))
    if "row_level" in want:                                                                # 단 = dot_cn 최소에서 tie_mm 이내, 줄 = dot_cs 최대(동점 id)
        dot_cn, dot_cs = Cmm @ n, Cmm @ s
        rest = sorted(range(len(idx)), key=lambda k: (dot_cn[k], idx[k]))
        lv = 0
        while rest:
            lo = dot_cn[rest[0]]
            members = [k for k in rest if dot_cn[k] - lo <= tie_mm]
            for row, group in enumerate(_rows_with_ties(dot_cs, members, idx, tie_mm)):
                for k in group:
                    objs[idx[k]].extras["row_level"] = dict(level=lv, row=row, n_in_level=len(members))
            rest = [k for k in rest if k not in members]
            lv += 1
    return scene


def _neighbors(objs: List[ObjectHypothesis], card: Card, cfg: dict) -> None:
    """같은 카드 인스턴스 사이의 이웃 관계 (유효 포즈 또는 위치가 있는 것). 반경 = 카드 대표 반경, 없으면(축소 모드) bbox3d 반 폭 최대."""
    ncfg = cfg.get("neighbors", {})
    margin, cmargin = float(mm2m(ncfg.get("margin_mm", 20))), float(mm2m(ncfg.get("contact_margin_mm", 5)))
    r0 = card.nominal_radius_m()
    have = [o for o in objs if o.position is not None and all(v is not None and np.isfinite(v) for v in o.position)]
    def radius(o):
        if np.isfinite(r0):
            return r0
        mn, mx = np.asarray(o.bbox3d["min"], float), np.asarray(o.bbox3d["max"], float)
        return float(np.nanmax(mx - mn) / 2) if np.isfinite(mn).all() else 0.0
    for o in have:
        o.neighbors = []
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            d = np.asarray(b.position) - np.asarray(a.position)
            dist = float(np.linalg.norm(d))
            rr = radius(a) + radius(b)
            if dist < rr + margin:
                contact = dist < rr + cmargin
                a.neighbors.append(dict(id=b.id, displacement=[float(v) for v in d], contact=contact))
                b.neighbors.append(dict(id=a.id, displacement=[float(v) for v in -d], contact=contact))


def gate(cands: List[Candidate], insts: List[Instance], poses: List[Optional[PoseResult]], reasons: List[str],
         card: Card, cfg: dict, frame: Frame, refine_used: List[bool], tracker: Optional[StabilityTracker] = None,
         scene: Optional[SceneMap] = None, task: Optional[dict] = None, now: Optional[float] = None) -> Hypotheses:
    """task 가 있으면 rules.observe_distance_m 로 관측 거리 검사(Q14). cfg gate.confidence_filter 가 켜져 있으면 confidence < min_confidence 도 무효."""
    now = time.time() if now is None else now
    age_ms = (now - frame.stamp) * 1000.0
    objs: List[ObjectHypothesis] = []
    cam_origin = frame.T_base_cam[:3, 3]
    obs = (task or {}).get("observe_distance_m")
    pref = (task or {}).get("preferred_distance_m")
    gcfg = cfg.get("gate", {})
    for i, (c, inst) in enumerate(zip(cands, insts)):
        oid = f"{card.class_name}_{i:02d}"
        has_pts = len(inst.points_base) > 0
        bbox = dict(min=[float(v) for v in inst.bbox3d[0]], max=[float(v) for v in inst.bbox3d[1]])
        ys, xs = np.nonzero(c.mask)
        mask_center = [float(xs.mean()), float(ys.mean())] if xs.size else [float("nan")] * 2
        common = dict(id=oid, class_name=card.class_name, rigidity=card.rigidity, frame_id=frame.frame_id,
                      stamp=frame.stamp, age_ms=age_ms, bbox3d=bbox)
        if card.reduced:                                                                  # 축소 모드
            pos = inst.points_base.mean(axis=0).tolist() if has_pts else [float("nan")] * 3
            objs.append(ObjectHypothesis(position=pos, orientation=[0.0, 0.0, 0.0, 1.0], pose_valid=False,
                                         geometry=dict(type=card.geometry["type"], params_fit=dict(n_points=int(len(inst.points_base)), n_valid=inst.n_valid)),
                                         confidence=float(np.clip(c.score, 0.0, 1.0)), score_raw=float(c.score), backend="none",
                                         refine_used=False, extras=dict(mask_center_px=mask_center),
                                         invalid_reason="degraded_mode" if has_pts else "too_few_points", **common))
            continue
        p = poses[i]
        if p is None:                                                                     # 점 부족·맞춤 실패
            objs.append(ObjectHypothesis(position=[float("nan")] * 3, orientation=[0.0, 0.0, 0.0, 1.0], pose_valid=False,
                                         geometry=dict(type=card.geometry["type"], params_fit=dict(n_points=int(len(inst.points_base)), n_valid=inst.n_valid, mask_center_px=mask_center)),
                                         confidence=0.0, score_raw=float("nan"), backend=card.pose["backend"], refine_used=False,
                                         extras={}, invalid_reason=reasons[i] or "fit_failed", **common))
            continue
        pos = p.T_base_obj[:3, 3]
        dist = float(np.linalg.norm(pos - cam_origin))
        reason = _fit_valid_reason(p.geometry_fit, card)
        if reason is None and obs and not (obs[0] <= dist <= obs[1]):
            reason = "observe_distance"                                   # 유효성 한계 밖 (Q21). 선호 범위는 유효성과 무관
        fit = dict(p.geometry_fit, mask_center_px=mask_center, n_valid=inst.n_valid, camera_distance_m=dist,
                   preferred_distance=bool(pref and pref[0] <= dist <= pref[1]) if pref else None)
        conf = confidence_from(p.score_raw, p.backend, card, p.geometry_fit)
        if reason is None and gcfg.get("confidence_filter", False) and conf < float(gcfg.get("min_confidence", 0.0)):
            reason = f"신뢰도 낮음({conf:.2f})"
        objs.append(ObjectHypothesis(position=[float(v) for v in pos], orientation=quat_from_matrix(p.T_base_obj[:3, :3]),
                                     pose_valid=reason is None, geometry=dict(type=card.geometry["type"], params_fit=fit),
                                     confidence=conf, score_raw=float(p.score_raw),
                                     backend=p.backend, refine_used=bool(refine_used[i]), extras={}, invalid_reason=reason, **common))
    if tracker is not None and not card.reduced:                                            # 연속 모드 안정성
        need, tol_m = int(card.gate["stable_frames"]), float(mm2m(card.gate["stable_tol_mm"]))
        valid = [o for o in objs if o.pose_valid]
        counts = tracker.update([np.asarray(o.position) for o in valid], tol_m, need)
        for o, k in zip(valid, counts):
            o.extras["stability"] = dict(frames=k, need=need, locked=k >= need)
            if k < need:
                o.pose_valid, o.invalid_reason = False, f"안정 프레임 미달({k}/{need})"
    scene_info = _scene_extras(objs, poses, card, frame)
    if "budget_uphill" in card.extras.get("compute", []) and scene is not None:              # Step 4: SceneMap.free_space.direct_mm
        for o in objs:
            if o.pose_valid:
                o.extras["budget_uphill"] = dict(direct_mm=scene.free_space.get("direct_mm"), manipulable_rigid_mm=scene.free_space.get("manipulable_rigid_mm"),
                                                 manipulable_measured_mm=scene.free_space.get("manipulable_measured_mm"))
    if card.scene.get("neighbors", False):
        _neighbors(objs, card, cfg)
    header = dict(stamp=frame.stamp, source=frame.source, frame_id=frame.frame_id, published=now, scene=scene_info)
    return Hypotheses(header=header, items=objs)


def apply_affordance(affs, kernel_meta: dict, feedback_store=None, group: Optional[str] = None, check_below: bool = True):
    """⑥ affordance 처리 (v3 §7.9): 후보 여유 < kernel.clearance_need_mm 이면 감점(score ≥ 0 은 ×0.5, 음수 score 는 ×2.0 — 어느 쪽이든 나빠지게) +
    reasons 'clearance_short:<축>'. feedback_store 에 같은 묶음(group)의 되먹임이 있으면 `clearance_measured_mm` = **예측 + 최근 bias(실측 − 예측 평균)** 를 채우고
    (실측값 자체가 아니라 보정된 예측이다) 그 값으로 판정한다. clearance_mm(원 예측)은 그대로 둔다."""
    need = kernel_meta.get("clearance_need_mm") or {}
    bias = {}
    if feedback_store is not None:
        for k in ("uphill", "below", "side_l", "side_r"):
            b = feedback_store.bias(group, k)
            if b is not None:
                bias[k] = float(b)
    for c in affs.candidates:
        cl = dict(c.clearance_mm)
        if bias:
            c.clearance_measured_mm = {k: float(cl[k] + bias[k]) for k in cl if k in bias}
            for k, v in c.clearance_measured_mm.items():
                cl[k] = v
        short = []
        for k, v in cl.items():
            if k == "below" and not check_below:                               # 놓기: 아래 여유는 받침 편평도라 필요치 검사 대상이 아님
                continue
            n = need.get("side" if k in ("side_l", "side_r") else k)
            if n is not None and v < float(n):
                short.append(k)
        if short:
            c.score = float(c.score) * 0.5 if c.score >= 0 else float(c.score) * 2.0
            c.reasons = list(c.reasons) + ["clearance_short:" + ",".join(short)]
    return affs
