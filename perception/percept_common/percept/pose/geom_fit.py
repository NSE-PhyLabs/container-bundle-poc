"""A: 기하 맞춤 백엔드. cylinder = estimate_pose.py fit_cylinder/_fit_circle_fixed_r(:78-129) 이식 (m 단위, 상수는 카드 pose.geom_fit).
box = 평면 맞춤 (v3 §7.6, fit_box docstring 참조): 가장 큰 평면과 그에 수직인 평면을 찾아 '위'에 가까운 쪽 = 상면, 다른 쪽 = 측면. 측면 법선은 측면의 수평 폭으로
x 면/y 면을 가려 그 축에 붙이고(못 가리면 상면 PCA), x 부호는 forward(직각이면 left)로 정한다. 중심 = 상면 범위 중점 + 측면 스냅 − sz/2. 측면이 없으면 partial=True.
score_raw = **상자 표면 잔차 RMS mm**(모든 점 → 맞춘 상자 표면; 상면 내점 잔차는 문턱에 묶여 늘 3 mm 아래라 신뢰도에 못 씀). geometry_fit = {size_mm(카드),
size_fit_mm(상면 관측 범위 p0.5~p99.5, 보통 0.98·실치수), yaw_deg(base 프레임에서만 의미), partial, top_coverage, rms_mm, top_rms_mm}.
'위' = T_base_cam 이 있으면 base +z(상면 법선이 20° 안이어야 함), 항등이면 영상 위(−y, 중력 문턱 없음·카메라 숙임 < 45° 전제). 대칭(요 180°·근사 정사각 90°)은 evaluate 가 처리.
결과 T_base_obj(4x4, m), score_raw(잔차 RMS mm, 순수값), geometry_fit(맞춘 치수 m + _mm 병기, length_outside_mm).

좌표 규약: 부호·기저 선택(axis[0] >= 0, |axis_z| < 0.9)과 '카메라→물체' 시선은 원본처럼 **카메라 좌표**에서 정하고, 결과만 base 로 돌린다
(T_base_cam 이 항등이면 원본과 같은 수치). 쿼터니언은 publish_container_pose.py axis_to_quat 규약(x=축, up=카메라 -Y).
init_T(refine): PCA 축 부호를 init 축과 맞추고 원 중심 초기값을 init 중심에서 잡는다.
"""
from typing import Optional

import numpy as np

from ..errors import FitFailed
from ..lift import m2mm, mm2m
from ..registry import Card
from ..types import Frame, Instance, PoseResult
from .base import PoseBackend, invert_T, make_T, rotation_from_axis, unit


def fit_circle_fixed_r(pts2: np.ndarray, R: float, init: np.ndarray, max_iter: int, step_tol: float, damping: float) -> np.ndarray:
    """반지름 R 고정, 원 중심만 가우스-뉴턴 (:78-93). 단위는 입력과 같음."""
    c = init.copy()
    for _ in range(max_iter):
        d = pts2 - c
        r = np.linalg.norm(d, axis=1)
        r = np.maximum(r, 1e-9)                       # 원본 1e-6 mm (= 1e-9 m) 거리 하한
        resid = r - R
        J = -d / r[:, None]
        H = J.T @ J + damping * np.eye(2)
        g = J.T @ resid
        step = np.linalg.solve(H, -g)
        c = c + step
        if np.linalg.norm(step) < step_tol:
            break
    return c


def fit_cylinder(P_cam: np.ndarray, radius: float, gf: dict, init_axis=None, init_center=None):
    """점구름(카메라 좌표, m) → center, axis, length, cross, circle_rms, flags. (:96-129) init_* 는 refine 용(카메라 좌표)."""
    c_surf = P_cam.mean(axis=0)
    Q = P_cam - c_surf
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])
    if init_axis is not None:
        if axis @ init_axis < 0:
            axis = -axis
    elif gf.get("axis_sign", "x_positive") == "x_positive" and axis[0] < 0:
        axis = -axis
    t = Q @ axis
    lp = gf["length_percentiles"]
    length = float(np.percentile(t, lp[1]) - np.percentile(t, lp[0]))
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < gf["basis_switch_abs_z"] else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    perp = Q - np.outer(t, axis)
    cross = float(2 * np.percentile(np.linalg.norm(perp, axis=1), gf["cross_percentile"]))
    pts2 = np.stack([perp @ e1, perp @ e2], axis=1)
    if init_center is not None:
        init = np.array([(init_center - c_surf) @ e1, (init_center - c_surf) @ e2])
    else:
        view = c_surf / (np.linalg.norm(c_surf) + 1e-9)          # 카메라 원점 → 물체
        view2 = np.array([view @ e1, view @ e2])
        nv = np.linalg.norm(view2)
        view2 = view2 / nv if nv > 1e-6 else np.array([0.0, 1.0])
        init = view2 * (gf["init_frac_r"] * radius)
    flags = dict(diverged=False, linalg_error=False)
    try:
        c2 = fit_circle_fixed_r(pts2, radius, init, int(gf["max_iter"]), float(mm2m(gf["step_tol_mm"])), float(gf["damping"]))
        if np.linalg.norm(c2) > gf["diverge_frac_r"] * radius:
            c2, flags["diverged"] = init, True
    except np.linalg.LinAlgError:
        c2, flags["linalg_error"] = init, True
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts2 - c2, axis=1) - radius) ** 2)))
    center = c_surf + c2[0] * e1 + c2[1] * e2
    return center, axis, length, cross, rms, flags


def _ransac_plane(P: np.ndarray, rng, iters: int, thr: float, accept=None) -> tuple:
    """3점 평면 RANSAC. accept(n) 이 False 인 법선은 건너뛴다. 반환 (내점 수, 법선, 기준점) 또는 (0, None, None)."""
    best_k, best_n, best_q = 0, None, None
    N = len(P)
    if N < 3:
        return best_k, best_n, best_q
    for _ in range(iters):
        q = P[rng.choice(N, 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        l = float(np.linalg.norm(n))
        if l < 1e-12:
            continue
        n = n / l
        if accept is not None and not accept(n):
            continue
        k = int((np.abs((P - q[0]) @ n) < thr).sum())
        if k > best_k:
            best_k, best_n, best_q = k, n, q[0]
    return best_k, best_n, best_q


def _refit(P: np.ndarray, n, q, thr: float) -> tuple:
    """내점으로 법선 재추정. 반환 (법선, 중심, 내점 마스크)."""
    m = np.abs((P - q) @ n) < thr
    c = P[m].mean(axis=0)
    n2 = unit(np.linalg.svd(P[m] - c, full_matrices=False)[2][2])
    if n2 @ n < 0:
        n2 = -n2
    return n2, c, np.abs((P - c) @ n2) < thr


def _span(t: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple:
    """(범위, 중점). 백분위를 0.5/99.5 로 좁게 잡는다 — 먼 쪽은 점 밀도가 낮아 같은 개수 비율이 더 긴 구간을 잘라 중점이 카메라 쪽으로 밀린다(2/98 에서 −7~−10 mm)."""
    a, b = np.percentile(t, lo), np.percentile(t, hi)
    return float(b - a), float((a + b) / 2)


def box_surface_rms(P: np.ndarray, R: np.ndarray, c: np.ndarray, size: np.ndarray) -> float:
    """모든 점에서 맞춘 상자 표면까지 거리의 RMS (m). 상면 내점 잔차와 달리 문턱에 묶이지 않아 중심·축 오류가 드러난다."""
    q = np.abs((P - c) @ R)
    half = np.asarray(size, float) / 2
    out = np.linalg.norm(np.maximum(q - half, 0.0), axis=1)
    inside = np.min(half - q, axis=1)
    d = np.where((q <= half).all(axis=1), inside, out)
    return float(np.sqrt(np.mean(d ** 2)))


def fit_box(P: np.ndarray, size: np.ndarray, up, gf: dict, forward=None, left=None, gravity_gate: bool = True) -> dict:
    """점구름(m) → 직육면체 자세. size = [sx, sy, sz] m (sz = 상면 법선 방향 높이).
    ① 가장 큰 평면 A, ② A 밖의 점에서 A 와 수직(±side_perp_tol)인 평면 B → **둘 중 up 에 더 가까운 쪽이 상면**, 다른 쪽이 측면. 중력 문턱으로 후보를 거르지 않으므로
    앞면 점이 더 많아도, 카메라 프레임(up = 영상 위, 카메라가 35° 숙임)이어도 상면을 찾는다. gravity_gate(=base 프레임)면 상면 법선이 up 과 top_normal_max_angle_deg 안이어야 한다.
    ③ 상면보다 위에 평행한 평면이 또 있으면(마스크에 팔레트가 섞인 경우) 그쪽을 상면으로. ④ 축: 측면이 있으면 측면의 수평 폭이 sx·sy 중 어느 쪽인지로 면을 정하고,
    없으면 상면 2D PCA(긴 변 ↔ 긴 범위). ⑤ 중심 = 상면 범위 중점, 측면 축은 측면 평면 − size/2, 높이 −sz/2.
    반환 dict(R, c, rms(상자 표면 잔차), top_rms, ext, partial, top_coverage, n_top, n_side, yaw)."""
    rs = gf.get("ransac") or {}
    iters, thr = int(rs.get("iters", 200)), float(mm2m(rs.get("inlier_mm", 5.0)))
    minf = float(rs.get("min_inlier_frac", 0.12))
    rng = np.random.default_rng(int(rs.get("seed", 0)))
    top_max = np.radians(float(gf.get("top_normal_max_angle_deg", 20.0)))
    side_tol = np.radians(float(gf.get("side_perp_tol_deg", 10.0)))
    up = unit(np.asarray(up, float))
    size = np.asarray(size, float)
    N = len(P)
    if N < 30:
        raise FitFailed("too_few_points")
    need = max(30, minf * N)
    kA, nA, qA = _ransac_plane(P, rng, iters, thr)
    if nA is None or kA < need:
        raise FitFailed(f"box_plane: 내점 {kA}/{N}")
    nA, cA, mA = _refit(P, nA, qA, thr)
    planes, used = [(nA, cA, int(mA.sum()))], mA.copy()
    for _ in range(2):                                                          # 서로 수직인 평면을 최대 3개(상면 + 측면 둘)까지 — 세 면이 다 보이면 두 번째가 또 측면일 수 있다
        rest = P[~used]
        if len(rest) < 50:
            break
        found = [pl[0] for pl in planes]
        k2, n2, q2 = _ransac_plane(rest, rng, iters, thr, lambda n: all(abs(float(n @ f)) <= np.sin(side_tol) for f in found))
        if n2 is None or k2 < max(30, 0.1 * len(rest)):
            break
        n2, c2, _ = _refit(rest, n2, q2, thr)
        m2 = np.abs((P - c2) @ n2) < thr
        planes.append((n2, c2, int((m2 & ~used).sum())))
        used |= m2
    planes.sort(key=lambda t: -abs(float(t[0] @ up)))                          # up 에 가까운 쪽이 상면
    planes = [planes[0]] + sorted(planes[1:], key=lambda t: -t[2])              # 측면은 내점 많은 쪽
    n_top, c0, k_top = planes[0]
    if n_top @ up < 0:
        n_top = -n_top
    if abs(float(n_top @ up)) < np.cos(np.radians(60.0)):
        raise FitFailed("box_top_not_visible: 위를 향한 면이 없음")
    if gravity_gate and np.arccos(np.clip(float(n_top @ up), -1, 1)) > top_max:
        raise FitFailed(f"box_top_tilted: 상면 법선이 위와 {np.degrees(np.arccos(np.clip(float(n_top @ up), -1, 1))):.0f}°")
    dist = (P - c0) @ n_top
    above = P[dist > 2 * thr]                                                  # ③ 더 위에 평행한 면(진짜 상면)이 있는가
    if len(above) >= max(30, 0.15 * N):
        kU, nU, qU = _ransac_plane(above, rng, iters, thr, lambda n: abs(float(n @ n_top)) >= np.cos(side_tol))
        if nU is not None and kU >= max(30, 0.1 * N):
            nU, cU, _ = _refit(above, nU, qU, thr)
            n_top, c0 = (nU if nU @ up > 0 else -nU), cU
            dist = (P - c0) @ n_top
    top = np.abs(dist) < thr
    inl = P[top]
    top_rms = float(np.sqrt(np.mean(dist[top] ** 2)))
    side = None
    if len(planes) > 1:
        ns, cs, ks = planes[1]
        ns = unit(ns - float(ns @ n_top) * n_top)                                # 상면 법선에 수직으로
        if float((cs - c0) @ ns) < 0:                                            # 바깥 = 측면 중심이 상면 중심에서 벗어난 쪽
            ns = -ns
        S = P[(np.abs((P - cs) @ planes[1][0]) < thr) & (dist < -2 * thr)]
        hdir = unit(np.cross(n_top, ns))                                          # 측면 안의 수평 방향
        width = _span((S - cs) @ hdir)[0] if len(S) >= 30 else 0.0
        side = (ns, cs, ks, width)
    # ④ 축 배정
    Q = inl - c0
    Q = Q - np.outer(Q @ n_top, n_top)
    e1 = unit(np.linalg.svd(Q, full_matrices=False)[2][0])
    e1 = unit(e1 - float(e1 @ n_top) * n_top)
    e2 = unit(np.cross(n_top, e1))
    square = abs(float(size[0] - size[1])) <= mm2m(5.0)                          # 근사 정사각: 축 배정을 PCA 1축으로 고정(채점기 대칭 허용오차와 같은 5 mm)
    ex, ey = (e1, e2) if (size[0] >= size[1] or square) else (e2, e1)
    axis_side = None
    if side is not None:
        ns, _, _, width = side
        dx, dy = abs(width - 0.98 * size[0]), abs(width - 0.98 * size[1])
        if not square and min(dx, dy) < 0.15 * min(size[0], size[1]) and abs(dx - dy) > mm2m(20.0):
            axis_side = 1 if dx < dy else 0                                      # 측면 폭 ≈ sx → 그 면은 ±y 면(법선 = y 축)
        else:
            axis_side = 0 if abs(float(ns @ ex)) >= abs(float(ns @ ey)) else 1   # 폭으로 못 가리면 PCA 축에 가까운 쪽
        if axis_side == 0:
            ex = ns
        else:
            ex = unit(np.cross(ns, n_top))
    ref = forward if forward is not None and abs(float(ex @ np.asarray(forward, float))) >= 0.2 else left
    if ref is not None and float(ex @ np.asarray(ref, float)) < 0:              # x 축 부호 결정론: forward, 직각이면 left (180° 대칭은 채점기가 처리)
        ex = -ex
    ey = unit(np.cross(n_top, ex))
    R = np.stack([ex, ey, n_top], axis=1)
    tx, ty = (inl - c0) @ ex, (inl - c0) @ ey
    (ext_x, mid_x), (ext_y, mid_y) = _span(tx), _span(ty)
    c_top = c0 + ex * mid_x + ey * mid_y
    if side is not None:
        a = ex if axis_side == 0 else ey
        outward = a if float(a @ side[0]) > 0 else -a
        c_top = c_top - (float((c_top - side[1]) @ outward) + float(size[axis_side]) / 2) * outward
    c = c_top - n_top * float(size[2]) / 2
    return dict(R=R, c=c, rms=box_surface_rms(P, R, c, size), top_rms=top_rms, ext=[ext_x, ext_y], partial=side is None,
                top_coverage=float(min(ext_x / size[0], ext_y / size[1])), n_top=int(top.sum()), n_side=int(side[2]) if side else 0,
                yaw=float(np.arctan2(ex[1], ex[0])))


class GeomFit(PoseBackend):
    name = "geom_fit"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.up_cam = tuple(cfg.get("frames", {}).get("camera_up", (0.0, -1.0, 0.0)))

    def estimate(self, inst: Instance, card: Card, frame: Frame, init_T: Optional[np.ndarray] = None) -> PoseResult:
        gt = card.geometry["type"]
        if gt == "cylinder":
            return self._cylinder(inst, card, frame, init_T)
        if gt == "box":
            return self._box(inst, card, frame)
        raise FitFailed(f"geom_fit: geometry.type={gt!r} 는 맞춤 대상이 아님")

    def _box(self, inst: Instance, card: Card, frame: Frame) -> PoseResult:
        if len(inst.points_base) == 0:
            raise FitFailed("too_few_points")
        gf, gp = card.pose["geom_fit"], card.geometry["params"]
        size = np.asarray(gp["size_mm"], float) / 1000.0
        in_base = not np.allclose(frame.T_base_cam, np.eye(4))                   # 캘리브가 있으면 점은 base 좌표 (frame_id 문자열에 기대지 않는다)
        if in_base:
            up, forward, left = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        else:                                                                     # 카메라 좌표: 영상 위·광축·영상 왼쪽. 중력 문턱은 못 건다(카메라 숙임 각을 모름) — 숙임 < 45° 전제
            T = frame.T_base_cam
            up, forward, left = -T[:3, 1], T[:3, 2], -T[:3, 0]
        r = fit_box(inst.points_base, size, up, gf, forward, left, gravity_gate=in_base)
        fit = dict(size_mm=[float(v) for v in gp["size_mm"]], size_fit_mm=[float(m2mm(v)) for v in r["ext"]],     # size_fit = 상면 관측 범위(p0.5~p99.5)
                   yaw_deg=float(np.degrees(r["yaw"])), partial=bool(r["partial"]), top_coverage=r["top_coverage"],
                   rms_mm=float(m2mm(r["rms"])), top_rms_mm=float(m2mm(r["top_rms"])), n_inliers=r["n_top"], n_side=r["n_side"],
                   top_normal=[float(v) for v in r["R"][:, 2]])
        return PoseResult(T_base_obj=make_T(r["R"], r["c"]), score_raw=float(m2mm(r["rms"])), backend="geom_fit",
                          geometry_fit=fit, partial=bool(r["partial"]))

    def _cylinder(self, inst: Instance, card: Card, frame: Frame, init_T: Optional[np.ndarray]) -> PoseResult:
        if len(inst.points_base) == 0:
            raise FitFailed("too_few_points")
        gf, gp = card.pose["geom_fit"], card.geometry["params"]
        radius = float(mm2m(gp["radius_mm"]))
        T_cb = invert_T(frame.T_base_cam)
        P_cam = inst.points_base @ T_cb[:3, :3].T + T_cb[:3, 3]        # 원본과 같은 카메라 좌표에서 맞춤
        init_axis = init_center = None
        if init_T is not None:
            init_axis = T_cb[:3, :3] @ init_T[:3, 0]
            init_center = T_cb[:3, :3] @ init_T[:3, 3] + T_cb[:3, 3]
        center_c, axis_c, length, cross, rms, flags = fit_cylinder(P_cam, radius, gf, init_axis, init_center)
        length_mm, cross_mm = float(m2mm(length)), float(m2mm(cross))
        lo, hi = (float(v) for v in gp["length_mm"])                    # registry 가 [lo, hi] 로 정규화
        outside = max(0.0, lo - length_mm, length_mm - hi)
        R_cam_obj = rotation_from_axis(axis_c, self.up_cam)
        R_bc, t_bc = frame.T_base_cam[:3, :3], frame.T_base_cam[:3, 3]
        T = make_T(R_bc @ R_cam_obj, R_bc @ center_c + t_bc)
        fit = dict(radius_m=radius, radius_mm=float(gp["radius_mm"]), length_m=length, length_mm=length_mm, cross_m=cross, cross_mm=cross_mm,
                   axis=[float(v) for v in (R_bc @ axis_c)], axis_cam=[float(v) for v in axis_c], rms_mm=float(m2mm(rms)),
                   n_inliers=int(len(inst.points_base)), length_outside_mm=outside, **flags)
        return PoseResult(T_base_obj=T, score_raw=float(m2mm(rms)), backend=self.name, geometry_fit=fit, partial=False)
