"""채점기 (v3 §7.11). 정답(gt)이 있는 어댑터(synthetic, 또는 gt 를 실은 replay)로 지표를 낸다. 모든 물체가 같은 표 형식.

매칭: 추정↔GT 헝가리안(중심 거리). GT 는 가시성 ≥ `vis_thr`(기본 0.5) 인 것만. 정답 판정 문턱 θp = 카드 `gate.grasp_tolerance.pos_mm`.
화면 밖으로 잘린 GT(`in_frame_frac < in_frame_thr`)는 **무시 대상**이다 — 채점에서 빼고, 그 근처의 추정도 TP·FP 어느 쪽으로도 세지 않는다(BOP 관례).
지표: 위치 mm(평균·중앙·95%), 회전 deg(대칭 고려), 치수 mm, 파지 가능률, 개수 일치율, MDE(대칭 최소), AP·AR, 신뢰도 곡선,
신뢰도 보정 곡선·ECE, 프레임당 시간, 실패 분류. 결과는 dict + `evaluate/report_template.md` 로 md·csv·png.
"""
import logging
from collections import Counter
from typing import List, Optional

import numpy as np

from .lift import m2mm, mm2m
from .pose.base import make_T, matrix_from_quat, rot_axis
from .registry import Card

log = logging.getLogger(__name__)
FAILURE_KINDS = ("no_detection", "merged", "pose_error", "dim_error", "gate_rejected", "scene_failed", "ok")


# ---------- 대칭 ----------
def symmetry_transforms(card: Card, n_rot: int = 36) -> List[np.ndarray]:
    """물체 대칭군(물체 좌표 4x4). cylinder = 축(x) 둘레 연속 회전(이산화) × 양끝 뒤집기. box = 치수가 같은 축의 180° 회전."""
    gt, gp = card.geometry["type"], card.geometry.get("params") or {}
    out = [np.eye(4)]
    if gt == "cylinder":
        out = []
        for k in range(n_rot):
            th = 2 * np.pi * k / n_rot
            c, s = np.cos(th), np.sin(th)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            out.append(make_T(Rx, [0, 0, 0]))
            out.append(make_T(Rx @ np.diag([-1.0, 1.0, -1.0]), [0, 0, 0]))     # 축 뒤집기(끝 교환)
        return out
    if gt == "box":
        sz = np.asarray(gp["size_mm"], float)
        gen = []
        for ax in range(3):                                                     # 180° 회전 3개 (직육면체 기본 대칭)
            R = -np.eye(3)
            R[ax, ax] = 1.0
            gen.append(R)
        for ax, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):                  # 두 변이 같으면 그 축 둘레 90°
            if abs(sz[i] - sz[j]) <= 5.0:                                        # 근사 정사각(5 mm): fit_box 의 축 배정 허용오차와 같음
                R = np.eye(3)
                R[i, i] = R[j, j] = 0.0
                R[i, j], R[j, i] = -1.0, 1.0
                gen.append(R)
        G = [np.eye(3)]                                                          # 생성원으로 군을 닫는다(중복 제거)
        for _ in range(4):
            new = [g @ h for g in G for h in gen]
            for m in new:
                if not any(np.allclose(m, x, atol=1e-9) for x in G):
                    G.append(m)
        return [make_T(R, [0, 0, 0]) for R in G]
    return out


def model_points(card: Card, n: int = 200, length_m: Optional[float] = None) -> np.ndarray:
    """MDE 용 물체 표면 표본점 (물체 좌표, m). 결정론적."""
    gt, gp = card.geometry["type"], card.geometry.get("params") or {}
    rng = np.random.default_rng(0)
    if gt == "cylinder":
        r = mm2m(gp["radius_mm"])
        L = length_m if length_m is not None else mm2m(float(np.mean(gp["length_mm"])))
        t = rng.uniform(-L / 2, L / 2, n)
        a = rng.uniform(0, 2 * np.pi, n)
        return np.stack([t, r * np.cos(a), r * np.sin(a)], axis=1)
    if gt == "box":
        s = np.asarray(gp["size_mm"], float) / 2000.0
        p = rng.uniform(-1, 1, (n, 3))
        face = rng.integers(0, 3, n)
        p[np.arange(n), face] = np.sign(p[np.arange(n), face])
        return p * s
    return rng.uniform(-0.05, 0.05, (n, 3))


def _mssd(Pe, T_gt, pts, S) -> float:
    Tg = T_gt @ S
    return float(np.max(np.linalg.norm(Pe - (pts @ Tg[:3, :3].T + Tg[:3, 3]), axis=1)))


def mde_mm(T_est, T_gt, pts, syms, continuous_axis: bool = False) -> float:
    """대칭을 고려한 최대 표면 거리 (BOP MSSD). mm.
    원통처럼 축 둘레 대칭이 **연속**이면 이산 표본만으로는 가짜 오차가 남는다(36 단계 = 반지름 70 mm 에서 최대 6.1 mm).
    continuous_axis=True 면 이산 최적해 근처를 한 번 더 좁혀(±1 스텝, 41 분할) 잔차를 0.1 mm 아래로 낮춘다."""
    Pe = pts @ T_est[:3, :3].T + T_est[:3, 3]
    vals = [_mssd(Pe, T_gt, pts, S) for S in syms]
    k = int(np.argmin(vals))
    best = float(vals[k])
    if continuous_axis and len(syms) >= 2:
        step = 2 * np.pi / (len(syms) // 2)                                      # syms = 회전 n 개 × 뒤집기 2
        base = syms[k]
        for th in np.linspace(-step, step, 41):
            best = min(best, _mssd(Pe, T_gt, pts, base @ make_T(rot_axis([1, 0, 0], np.degrees(th)), [0, 0, 0])))
    return float(m2mm(best))


def rot_err_deg(T_est, T_gt, card: Card, syms) -> float:
    """cylinder = 축 각도(부호 무시), box = 대칭 최소 회전각."""
    if card.geometry["type"] == "cylinder":
        a, b = T_est[:3, 0], T_gt[:3, 0]
        return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))))
    best = 180.0
    for S in syms:
        R = T_est[:3, :3].T @ (T_gt @ S)[:3, :3]
        best = min(best, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    return best


# ---------- 프레임 채점 ----------
def match_frame(hyps, gt: dict, card: Card, vis_thr: float, pos_thr_mm: float, syms,
                in_frame_thr: float = 0.98) -> dict:
    """한 프레임: 헝가리안 매칭 → 물체별 오차·TP/FP/FN·실패 분류.
    GT 는 가시성(가림) ≥ vis_thr **이고** 잘림 없는(in_frame_frac ≥ in_frame_thr) 것만 — 화면 밖으로 잘린 물체를
    '다 보인다'로 채점하면 짧은 거리에서 표가 통째로 틀어진다."""
    from scipy.optimize import linear_sum_assignment
    inf = gt.get("in_frame_frac")
    vis_idx = [i for i, v in enumerate(gt["visibility"]) if v >= vis_thr]          # 보이는 것(가림 기준)
    gt_idx = [i for i in vis_idx if inf is None or inf[i] >= in_frame_thr]         # 채점 대상(잘리지 않은 것)
    ign_idx = [i for i in vis_idx if i not in gt_idx]                              # 잘린 것 = **무시**(TP 도 FP 도 아님)
    gt_T = [np.asarray(gt["poses"][i], float) for i in gt_idx]
    ign_T = [np.asarray(gt["poses"][i], float) for i in ign_idx]
    est = [o for o in hyps.items if o.pose_valid]
    rows, tp, fp, fn = [], 0, 0, 0
    gp = card.geometry.get("params") or {}
    half = float(max(gp.get("size_mm", [0])) if card.geometry["type"] == "box" else (gp.get("length_mm", [0])[-1] if gp.get("length_mm") else 0)) / 2
    ign_r = max(mm2m(pos_thr_mm) * 3, mm2m(half))                                  # 잘린 물체의 추정 중심은 반 길이까지 밀릴 수 있다(상자 510 → 255 mm)
    all_T = gt_T + ign_T                                                           # 헝가리안은 **무시 GT 도 포함**해 짝짓는다 — 안 그러면 잘린 상자의 추정이 545 mm 떨어진 채점 GT 와 짝지어진다
    if est and all_T:
        C = np.array([[np.linalg.norm(np.asarray(o.position) - T[:3, 3]) for T in all_T] for o in est])
        ri, ci = linear_sum_assignment(C)
    else:
        ri, ci = np.array([], int), np.array([], int)
    matched_gt, matched_est, ignored, paired = set(), set(), set(), set()
    for i, j in zip(ri, ci):
        if j >= len(gt_T):                                                         # 무시 GT 와 짝 → 가까우면 무시, 멀면 우연한 짝이라 FP
            if C[i, j] <= ign_r:
                ignored.add(i)
            continue
        o, T_gt = est[i], gt_T[j]
        T_est = make_T(matrix_from_quat(o.orientation), o.position)
        L = gt["dims"]["length_mm"][gt_idx[j]] if gt["dims"].get("length_mm") else None
        p = model_points(card, 200, mm2m(L) if L and np.isfinite(L) else None)
        cont = card.geometry["type"] == "cylinder"
        d = mde_mm(T_est, T_gt, p, syms, continuous_axis=cont)
        pos = float(m2mm(np.linalg.norm(np.asarray(o.position) - T_gt[:3, 3])))
        rot = rot_err_deg(T_est, T_gt, card, syms)
        dim, lm = None, o.geometry["params_fit"].get("length_mm")
        if cont and L and np.isfinite(L) and lm is not None:
            dim = float(abs(lm - L))
        ok = d <= pos_thr_mm
        paired.add(i)                                                              # 채점 GT 와 짝지어진 추정(TP 든 FP 든) — 아래에서 다시 세지 않는다
        rows.append(dict(id=o.id, gt=int(gt_idx[j]), pos_mm=pos, rot_deg=rot, mde_mm=d, dim_mm=dim,
                         confidence=float(o.confidence), rms_mm=float(o.geometry["params_fit"].get("rms_mm", np.nan)),
                         distance_m=float(o.geometry["params_fit"].get("camera_distance_m", np.nan)),
                         gt_distance_m=float((gt.get("object_distance_m") or [np.nan] * (gt_idx[j] + 1))[gt_idx[j]]),
                         visibility=float(gt["visibility"][gt_idx[j]]),
                         in_frame=float(inf[gt_idx[j]]) if inf else float("nan"),
                         preferred=o.geometry["params_fit"].get("preferred_distance"), tp=bool(ok)))
        if ok:
            tp += 1
            matched_gt.add(j)
            matched_est.add(i)
        else:
            fp += 1
    unmatched = [i for i in range(len(est)) if i not in paired and i not in ignored]        # 남은 추정: 잘린 GT 근처면 무시, 아니면 FP
    for i in unmatched:
        c = np.asarray(est[i].position)
        if ign_T and min(float(np.linalg.norm(c - T[:3, 3])) for T in ign_T) <= ign_r:
            ignored.add(i)
        else:
            fp += 1
    fn = len(gt_T) - len(matched_gt)
    for i, o in enumerate(est):
        if i in ignored:
            continue
        if i not in paired:
            rows.append(dict(id=o.id, gt=None, pos_mm=None, rot_deg=None, mde_mm=None, dim_mm=None,
                             confidence=float(o.confidence), rms_mm=float(o.geometry["params_fit"].get("rms_mm", np.nan)),
                             distance_m=float(o.geometry["params_fit"].get("camera_distance_m", np.nan)),
                             gt_distance_m=float("nan"), visibility=None, in_frame=float("nan"),
                             preferred=o.geometry["params_fit"].get("preferred_distance"), tp=False))
    return dict(rows=rows, tp=tp, fp=fp, fn=fn, n_gt=len(gt_T), n_ignored=len(ign_T), n_visible=len(vis_idx),
                n_est=len(est) - len(ignored), n_est_raw=len(est), n_items=len(hyps.items))


def classify_failure(frame_res: dict, reason: Optional[str]) -> str:
    if reason in ("no_target", "finder_unavailable", "camera_unavailable", "depth_invalid"):
        return "no_detection"
    if reason == "scene_failed":
        return "scene_failed"
    if reason in ("gate_rejected", "fit_failed", "backend_unavailable"):
        return "gate_rejected"
    if frame_res["n_est"] == 0 and frame_res["n_gt"] > 0:
        return "no_detection"
    if frame_res["n_est"] < frame_res["n_gt"] and frame_res["tp"] < frame_res["n_gt"]:
        return "merged"
    if frame_res["fp"] > 0 or frame_res["fn"] > 0:
        return "pose_error"
    return "ok"


# ---------- 곡선 ----------
def confidence_curve(rows: List[dict], step: float = 0.05, n_gt: Optional[int] = None) -> List[dict]:
    """신뢰도 문턱별 AP·AR·통과 비율. n_gt(보이는 GT 총수)를 주지 않으면 매칭된 행 수로 대신하므로 AR 이 낙관적이 된다."""
    out = []
    n_gt = int(n_gt) if n_gt is not None else sum(1 for r in rows if r["gt"] is not None)
    for t in np.arange(0.0, 1.0 + 1e-9, step):
        sel = [r for r in rows if r["confidence"] >= t - 1e-12]
        tp = sum(r["tp"] for r in sel)
        ap = tp / len(sel) if sel else float("nan")
        ar = tp / n_gt if n_gt else float("nan")
        out.append(dict(threshold=float(t), n=len(sel), ap=float(ap), ar=float(ar),
                        pass_ratio=float(len(sel) / len(rows)) if rows else 0.0))
    return out


def calibration_curve(rows: List[dict], bins: int = 10) -> tuple:
    """신뢰도 구간별 실제 TP 비율 + ECE."""
    edges = np.linspace(0, 1, bins + 1)
    conf = np.array([r["confidence"] for r in rows], float)
    tp = np.array([bool(r["tp"]) for r in rows])
    out, ece = [], 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        n = int(m.sum())
        if n == 0:
            out.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=0, mean_conf=float("nan"), tp_rate=float("nan")))
            continue
        mc, tr = float(conf[m].mean()), float(tp[m].mean())
        ece += n / len(rows) * abs(mc - tr)
        out.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=n, mean_conf=mc, tp_rate=tr))
    return out, float(ece)


# ---------- 본체 ----------
def affordance_metrics(affs, hyps, scene) -> dict:
    """프레임 하나의 affordance 지표 (v3 §7.12): top-1 존재, top-1 타깃이 맨 위 단인지, top-1 이 자기 골(경사 아래 상대와의 중점 ± 20 mm) 위인지,
    top-1 아래 여유(2.5D) mm, 후보 수, 사유."""
    out = dict(has_top=False, top_tier1=None, own_valley=None, below_mm=None, n_cand=len(affs.candidates), reason=affs.invalid_reason,
               detail=affs.header.get("detail"), timing_ms=affs.header.get("timing_ms"))
    if not affs.candidates or scene is None:
        return out
    from .scene.plane import to_plane
    top = affs.candidates[0]
    out["has_top"] = True
    out["below_mm"] = float(top.clearance_mm.get("below", np.nan))
    by = {o.id: (i, o) for i, o in enumerate(hyps.items)}
    if top.target_id in by:
        i, t = by[top.target_id]
        cells = scene.layers["instance_id"] == i + 1
        out["top_tier1"] = bool(cells.any() and int(scene.layers["tier"][cells].max()) == 1)
        ut = float(to_plane(scene.plane, np.asarray(t.position, float)[None])[0][0] * 1000)
        partners = [by[g["with_id"]][1] for g in (t.extras.get("grooves") or []) if g["with_id"] in by]
        down = [float(to_plane(scene.plane, np.asarray(p.position, float)[None])[0][0] * 1000) for p in partners]
        down = [u for u in down if u < ut]
        if down:
            out["own_valley"] = bool(abs(top.uv_theta[0] - (ut + max(down)) / 2) <= 20.0)
    return out


def affordance_summary(rows: List[dict]) -> dict:
    if not rows:
        return {}
    f = lambda k: [r[k] for r in rows if r.get(k) is not None]
    return dict(n_frames=len(rows), top1_rate=float(np.mean([r["has_top"] for r in rows])),
                top1_tier1_rate=float(np.mean(f("top_tier1"))) if f("top_tier1") else float("nan"),
                own_valley_rate=float(np.mean(f("own_valley"))) if f("own_valley") else float("nan"),
                below_mm_median=float(np.nanmedian(f("below_mm"))) if f("below_mm") else float("nan"),
                n_cand_mean=float(np.mean([r["n_cand"] for r in rows])),
                reasons=dict(Counter(str(r["reason"]) + (f"/{r['detail']}" if r.get("detail") else "") for r in rows)))


def evaluate(pipeline, card: Card, n: int, vis_thr: float = 0.5, task: Optional[dict] = None,
             progress=None, in_frame_thr: float = 0.98) -> dict:
    """pipeline 을 n 프레임 돌려 채점. pipeline.adapter 는 extra['gt'] 를 내야 한다.
    GT 선별은 가시성(가림)과 잘림(in_frame_frac)을 따로 본다 — 잘린 물체는 아예 GT 에서 뺀다."""
    tol = card.gate["grasp_tolerance"]
    pos_thr, axis_thr = float(tol["pos_mm"]), float(tol["axis_deg"])
    syms = symmetry_transforms(card)
    frames, rows, fails, times, percept_ms, count_ok = [], [], Counter(), [], [], 0
    aff_rows = []
    for k in range(n):
        try:
            hyps, affs = pipeline.step()
        except StopIteration:
            break
        if affs is not None:
            aff_rows.append(affordance_metrics(affs, hyps, pipeline.last.get("scene")))
        fobj = pipeline.last.get("frame")
        if fobj is not None and fobj.extra.get("gt") is None:                 # 어댑터를 잘못 골랐다 — 조용히 넘기면 안 됨
            raise ValueError("어댑터가 정답(extra['gt'])을 내지 않음 — synthetic 어댑터를 쓰세요")
        gt = (fobj.extra.get("gt") if fobj is not None else None)
        if gt is None:                                                        # 카메라 실패 등으로 프레임 자체가 없음
            frames.append(dict(rows=[], tp=0, fp=0, fn=0, n_gt=0, n_ignored=0, n_visible=0, n_est=0, n_est_raw=0, n_items=0, reason=hyps.header.get("reason"),
                               scene_id=hyps.header.get("scene_id"), distance_m=float("nan"),
                               timing_ms=hyps.header["timing_ms"], kind="no_detection"))
            fails["no_detection"] += 1
            times.append(hyps.header["timing_ms"]["total"])
            continue
        fr = match_frame(hyps, gt, card, vis_thr, pos_thr, syms, in_frame_thr)
        fr["reason"] = hyps.header.get("reason")
        fr["scene_id"] = hyps.header.get("scene_id")
        fr["distance_m"] = float(gt.get("distance_m", np.nan))
        fr["timing_ms"] = hyps.header["timing_ms"]
        fr["kind"] = classify_failure(fr, fr["reason"])
        for r in fr["rows"]:
            r["scene_id"] = fr["scene_id"]
            r["frame_distance_m"] = fr["distance_m"]                          # 프레임 공칭(주시점) 거리 — 물체별 거리는 distance_m/gt_distance_m
            r["grasp_ok"] = bool(r["pos_mm"] is not None and r["pos_mm"] <= pos_thr and r["rot_deg"] is not None and r["rot_deg"] <= axis_thr)
        rows += fr["rows"]
        frames.append(fr)
        fails[fr["kind"]] += 1
        times.append(fr["timing_ms"]["total"])
        percept_ms.append(sum(fr["timing_ms"].get(k, 0.0) for k in ("find", "lift", "pose", "scene", "affordance", "gate")))
        count_ok += int(fr["n_est_raw"] == fr.get("n_visible", fr["n_gt"]))     # 검출 개수는 '보이는 것' 기준(잘림 포함)
        if progress:
            progress(k, fr)
    matched = [r for r in rows if r["gt"] is not None]
    pos = np.array([r["pos_mm"] for r in matched], float) if matched else np.array([])
    rot = np.array([r["rot_deg"] for r in matched], float) if matched else np.array([])
    dim = np.array([r["dim_mm"] for r in matched if r["dim_mm"] is not None and np.isfinite(r["dim_mm"])], float)
    mde = np.array([r["mde_mm"] for r in matched], float) if matched else np.array([])
    tp = sum(f["tp"] for f in frames)
    fp = sum(f["fp"] for f in frames)
    fn = sum(f["fn"] for f in frames)
    ap_f = [f["tp"] / (f["tp"] + f["fp"]) for f in frames if f["tp"] + f["fp"]]
    ar_f = [f["tp"] / f["n_gt"] for f in frames if f["n_gt"]]
    cal, ece = calibration_curve(rows) if rows else ([], float("nan"))
    st = lambda a: dict(mean=float(np.mean(a)), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(np.max(a))) if len(a) else {}
    return dict(
        summary=dict(n_frames=len(frames), n_gt=sum(f["n_gt"] for f in frames), n_est=sum(f["n_est"] for f in frames),
                     tp=tp, fp=fp, fn=fn, AP=float(np.mean(ap_f)) if ap_f else float("nan"),
                     AR=float(np.mean(ar_f)) if ar_f else float("nan"),
                     AP_micro=float(tp / (tp + fp)) if tp + fp else float("nan"),      # 프레임 평균(AP)은 검출 0 프레임을 빼므로 micro 도 함께 본다
                     AR_micro=float(tp / (tp + fn)) if tp + fn else float("nan"),
                     n_frames_no_est=int(sum(1 for f in frames if f["n_est"] == 0)),
                     n_ignored=int(sum(f.get("n_ignored", 0) for f in frames)),
                     n_visible=int(sum(f.get("n_visible", 0) for f in frames)),
                     grasp_feasible_rate=float(np.mean([r["grasp_ok"] for r in matched])) if matched else float("nan"),
                     count_match_rate=float(count_ok / len(frames)) if frames else float("nan"),
                     pos_err_mm=st(pos), rot_err_deg=st(rot), dim_err_mm=st(dim), mde_mm=st(mde),
                     time_ms=st(np.array(times)), percept_ms=st(np.array(percept_ms)), ece=ece, pos_thr_mm=pos_thr, axis_thr_deg=axis_thr, vis_thr=vis_thr,
                     card=card.class_name, geometry=card.geometry["type"], task=(task or {}).get("name")),
        rows=rows, frames=frames, failures=dict(fails), affordance=affordance_summary(aff_rows),
        confidence_curve=confidence_curve(rows, n_gt=sum(f["n_gt"] for f in frames)) if rows else [],
        calibration=cal)
