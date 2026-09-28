"""④-1 기준면 추정 (v3 §7.7). 카드 `scene.reference_plane` 이 방법을 고른다.

- `instance_centers` (기울어진 박스): **중심점들만으로 평면을 맞추지 않는다.** 관찰 자세에서는 맨 위 단 한 줄만 보여 중심이 거의 한 직선 위에 놓이므로
  SVD 평면이 축 둘레로 자유롭게 돌아간다(퇴화). 대신 원본이 검증한 방법을 쓴다: 줄 방향 s = 중심점들의 1주성분, 축 a = 물체 축 평균 →
  법선 n = unit(a × s) (grasp_point.py box_normal :135-154). 원통이면 n 방향으로 반지름만큼 밀어 **표면 접평면**을 기준면으로 삼는다.
- `ransac_up` (팔레트·컨베이어): 물체가 아닌 점 중 법선이 base +z 에 가까운 최대 평면을 RANSAC 으로.
- `calib_file`: calib/plane_<type>.npz {T(4x4), extent_mm}.

반환 plane dict: {T_base_plane(4x4: 열 = [u, v, n], 원점 = 격자 (0,0) 의 base 좌표), res_mm, extent_mm [w, h], origin_px [0,0], method, rms_mm, n_used}
u 축 = 기울어진 박스는 **경사 위(uphill)**, 평평한 면은 base +x. v = n × u.
"""
import logging
from pathlib import Path

import numpy as np

from ..errors import SceneError
from ..lift import m2mm, mm2m
from ..pose.base import invert_T, make_T, unit

log = logging.getLogger(__name__)


def _frame_from_normal(n, u_hint) -> tuple:
    """법선 n 과 u 축 힌트 → 정규직교 (u, v, n)."""
    n = unit(n)
    u = np.asarray(u_hint, float) - float(np.asarray(u_hint, float) @ n) * n
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, [1.0, 0.0, 0.0])
    u = unit(u)
    return u, unit(np.cross(n, u)), n


def _pack(u, v, n, span_base, res_mm: float, margin_mm: float, method: str, extra: dict, datum_base=None) -> dict:
    """평면 좌표계와 격자 범위. 격자 범위(u, v)는 span_base(관측 점구름)로, **법선 방향 영점(h=0)은 datum_base**(원통 표면 접평면)로 따로 잡는다.
    둘을 섞으면 h=0 이 접평면이 아니게 되어 free 층의 절대 높이 문턱이 어긋난다."""
    R = np.stack([u, v, n], axis=1)
    p = np.asarray(span_base, float)
    c = p.mean(axis=0)
    uv = (p - c) @ R[:, :2]
    lo = uv.min(axis=0) - mm2m(margin_mm)
    hi = uv.max(axis=0) + mm2m(margin_mm)
    origin = c + R[:, 0] * lo[0] + R[:, 1] * lo[1]
    if datum_base is not None:                                               # 원점의 법선 성분을 접평면에 맞춤
        origin = origin - float((origin - np.asarray(datum_base, float).mean(axis=0)) @ n) * n
    extent_mm = [float(m2mm(hi[0] - lo[0])), float(m2mm(hi[1] - lo[1]))]
    return dict(T_base_plane=make_T(R, origin), res_mm=float(res_mm), extent_mm=extent_mm, origin_px=[0, 0],
                method=method, **extra)


def from_instance_centers(hyps, poses, card, res_mm: float, margin_mm: float, instances=None, T_base_cam=None) -> dict:
    """줄 방향 × 축 → 법선. 원통은 반지름만큼 밀어 표면 접평면.
    **부호는 프레임에 기대지 않고 카메라로 정한다** — base +z 로 판정하면 T_base_cam 이 항등인 실사 경로(좌표가 카메라 광학계)에서 뒤집힌다.
      - 법선 n: 카메라 위치 쪽(= 용기 바깥·개구부 쪽). gate 의 `scene.normal` 은 반대(용기 안쪽)이니 혼동하지 말 것.
      - u 축(경사 위): 카메라의 '위' 축 `-T_base_cam[:3, 1]` 에 투영해 양수인 쪽. 항등이면 카메라 -y(영상 위) = grasp_point 의 s_uphill 규약과 같고,
        진짜 base 프레임이면 대략 +z 가 된다."""
    idx = [i for i, (o, p) in enumerate(zip(hyps.items, poses)) if o.pose_valid and p is not None]
    if len(idx) < 2:
        raise SceneError(f"plane_fit_failed: 유효 물체 {len(idx)}개 (최소 2)")
    C = np.array([hyps.items[i].position for i in idx], float)
    A = []
    for i in idx:
        a = poses[i].T_base_obj[:3, 0]
        A.append(-a if a @ A[0] < 0 else a) if A else A.append(a)            # 축 부호 정렬
    axis = unit(np.mean(A, axis=0))
    if len(idx) == 2:
        s = unit(C[1] - C[0])
    else:
        s = unit(np.linalg.svd(C - C.mean(axis=0), full_matrices=False)[2][0])
    if abs(float(s @ axis)) > 0.9:
        raise SceneError(f"plane_fit_failed: 줄 방향이 축과 평행 (|s·axis| {abs(float(s @ axis)):.2f})")
    n = unit(np.cross(axis, s))
    c0 = C.mean(axis=0)
    up_hint = np.array([0.0, 0.0, 1.0])
    if T_base_cam is not None:
        if float(n @ (T_base_cam[:3, 3] - c0)) < 0:                          # 카메라는 언제나 개구부 쪽에 있다
            n = -n
        up_hint = unit(-T_base_cam[:3, 1])                                   # 카메라의 '위' 축 (광학 -y)
    elif n[2] < 0:
        n = -n
    s_uphill = s if float(s @ up_hint) > 0 else -s                           # u 축 = 경사 위
    r_m = float(mm2m((card.geometry.get("params") or {}).get("radius_mm", 0.0)))
    pts = C + n * r_m                                                        # 원통 표면 접평면 (h = 0 기준)
    spread = float(m2mm(np.std((pts - pts.mean(axis=0)) @ n)))               # 중심점 산포(평면 잔차가 아님 — 법선이 s 에 수직이라 축 오차엔 둔감)
    u, v, n = _frame_from_normal(n, s_uphill)
    # 격자 범위는 중심점이 아니라 **관측된 점구름 범위**로 잡는다 (중심만 쓰면 축 방향으로 물체 길이만큼 모자람)
    span = pts
    if instances:
        corners = [np.array(np.meshgrid(*zip(i.bbox3d[0], i.bbox3d[1]))).reshape(3, -1).T
                   for i in instances if len(i.points_base)]
        if corners:
            span = np.vstack([pts] + corners)
    surf_rms = float("nan")
    if instances:                                                            # 진짜 평면 품질: 관측점의 법선 방향 잔차(표면이라 반지름만큼 아래로 퍼짐)
        P = np.vstack([i.points_base for i in instances if len(i.points_base)]) if any(len(i.points_base) for i in instances) else None
        if P is not None:
            h = (P - pts.mean(axis=0)) @ n
            surf_rms = float(m2mm(np.sqrt(np.mean(np.minimum(h, 0.0) ** 2))))
    return _pack(u, v, n, span, res_mm, margin_mm, "instance_centers",
                 dict(centers_spread_mm=spread, surface_rms_mm=surf_rms, n_used=len(idx)), datum_base=pts)


def from_ransac_up(frame, instances, res_mm: float, margin_mm: float, thr_mm: float, max_angle_deg: float = 30.0,
                   iters: int = 200, seed: int = 0) -> dict:
    """물체가 아닌 점에서 법선이 base +z 에 가까운 최대 평면."""
    ys, xs = np.nonzero(np.isfinite(frame.depth))
    if len(ys) < 100:
        raise SceneError("plane_fit_failed: 유효 depth 점 부족")
    obj = np.zeros(frame.depth.shape, bool)
    for inst in instances:
        obj |= inst.candidate.mask
    keep = ~obj[ys, xs]
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 100:
        raise SceneError("plane_fit_failed: 물체 밖 점 부족")
    step = max(1, len(ys) // 20000)
    ys, xs = ys[::step], xs[::step]
    z = frame.depth[ys, xs].astype(float)
    K = frame.K
    P = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
    P = P @ frame.T_base_cam[:3, :3].T + frame.T_base_cam[:3, 3]
    rng = np.random.default_rng(seed)
    thr, best = float(mm2m(thr_mm)), (0, None, None)
    for _ in range(iters):
        q = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = unit(n)
        if n[2] < 0:
            n = -n
        if np.degrees(np.arccos(np.clip(n[2], -1, 1))) > max_angle_deg:
            continue
        d = np.abs((P - q[0]) @ n)
        k = int((d < thr).sum())
        if k > best[0]:
            best = (k, n, q[0])
    if best[1] is None or best[0] < 0.1 * len(P):
        raise SceneError(f"plane_fit_failed: RANSAC 내점 {best[0]}/{len(P)}")
    n, p0 = best[1], best[2]
    inl = P[np.abs((P - p0) @ n) < thr]
    c = inl.mean(axis=0)
    n = unit(np.linalg.svd(inl - c, full_matrices=False)[2][2])              # 내점으로 법선 재추정
    if n[2] < 0:
        n = -n
    rms = float(m2mm(np.sqrt(np.mean(((inl - c) @ n) ** 2))))
    u, v, n = _frame_from_normal(n, [1.0, 0.0, 0.0])
    return _pack(u, v, n, inl, res_mm, margin_mm, "ransac_up",
                 dict(centers_spread_mm=float("nan"), surface_rms_mm=rms, n_used=len(inl)), datum_base=inl)


def from_calib_file(path, res_mm: float) -> dict:
    p = Path(path)
    if not p.exists():
        raise SceneError(f"plane_fit_failed: {p} 없음")
    z = np.load(p, allow_pickle=False)
    return dict(T_base_plane=np.asarray(z["T"], float), res_mm=float(res_mm),
                extent_mm=[float(v) for v in z["extent_mm"]], origin_px=[0, 0], method="calib_file",
                centers_spread_mm=0.0, surface_rms_mm=0.0, n_used=0)


def estimate_plane(frame, instances, hyps, poses, card, cfg) -> dict:
    """카드 scene.reference_plane 대로 기준면. 실패는 SceneError('plane_fit_failed: 사유')."""
    rp = (card.scene or {}).get("reference_plane") or {}
    kind, how = rp.get("type", "none"), rp.get("plane_from")
    if kind == "none":
        raise SceneError("plane_fit_failed: 카드 scene.reference_plane.type = none")
    sc = cfg["scene"]
    res_mm, margin = float(sc["res_mm"]), float(sc["margin_mm"])
    if how == "instance_centers":
        return from_instance_centers(hyps, poses, card, res_mm, margin, instances, frame.T_base_cam)
    if how == "ransac_up":
        return from_ransac_up(frame, instances, res_mm, margin, float(sc["ransac_thr_mm"]), seed=int(sc.get("seed", 0)))
    if how == "calib_file":
        return from_calib_file(card.path / f"plane_{kind}.npz", res_mm)
    raise SceneError(f"plane_fit_failed: plane_from={how!r}")


def to_plane(plane: dict, P_base: np.ndarray) -> np.ndarray:
    """base 점 → 평면 좌표 (u, v, h) m. h = 평면 위 높이(법선 방향)."""
    T = invert_T(plane["T_base_plane"])
    return np.asarray(P_base, float) @ T[:3, :3].T + T[:3, 3]


def to_base(plane: dict, P_uvh: np.ndarray) -> np.ndarray:
    T = plane["T_base_plane"]
    return np.asarray(P_uvh, float) @ T[:3, :3].T + T[:3, 3]
