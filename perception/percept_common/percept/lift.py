"""마스크 × depth → 카메라 점구름 → T_base_cam → base 점구름(m). 이상치 제거. (v3 §7.5)

**mm→m 변환은 이 파일에서만 한다** (mm2m/m2mm/depth_m_from_disparity). 다른 모듈은 이 함수들을 import 해 쓴다.
원본 estimate_pose.py points_3d(:63-75) 이식: 유효 = mask & depth 유효, 점 < min_points 면 n_valid 만 채운 빈 Instance('too_few_points'),
Z 상하위 백분위 제거(카드 lift.outlier) 또는 knn 통계(configs/default.yaml lift.knn).
인스턴스 분리는 Finder(color_rule 의 시차 기반 분할)가 담당하므로 Candidate 1개 = Instance 1개 (ASSUMPTIONS 13).
"""
from typing import List

import numpy as np

from .pose.base import transform_points
from .registry import Card
from .types import Candidate, Frame, Instance

MM_PER_M = 1000.0


def mm2m(x):
    """mm → m. 이 줄이 percept 안의 유일한 mm→m 변환 지점 (지시서 §8 · 사용자 지시 9/18)."""
    return np.asarray(x, dtype=np.float64) / MM_PER_M


def m2mm(x):
    """m → mm (보고서·회귀 비교·카드 mm 값 대조용)."""
    return np.asarray(x, dtype=np.float64) * MM_PER_M


def depth_m_from_disparity(disp_px: np.ndarray, fx: float, baseline_mm: float, invalid_max: float = 0.0) -> np.ndarray:
    """시차(px, float32) → 깊이 m (float32, 무효 NaN). 원본 식 Z_mm = fx*baseline/d (estimate_pose.py:69, float32) 를 그대로 계산한 뒤 m 로."""
    disp = np.asarray(disp_px, dtype=np.float32)
    valid = disp > invalid_max
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mm = np.where(valid, np.float32(fx * baseline_mm) / disp, np.float32(np.nan)).astype(np.float32)
    return (z_mm / np.float32(MM_PER_M)).astype(np.float32)     # mm → m (depth_m_from_disparity 는 mm2m 와 함께 유일한 변환 지점)


def _outlier_keep(P_cam: np.ndarray, cfg: dict, knn_cfg: dict) -> np.ndarray:
    """이상치 제거 마스크(bool). z_percentile = 원본(:74-75), knn = k-최근접 평균거리 통계(k, std_ratio), none."""
    method = (cfg or {}).get("method", "none")
    if method == "z_percentile":
        lo, hi = np.percentile(P_cam[:, 2], cfg.get("percentiles", [5, 95]))
        return (P_cam[:, 2] >= lo) & (P_cam[:, 2] <= hi)
    if method == "knn":
        from scipy.spatial import cKDTree
        k, ratio = int(cfg.get("k", knn_cfg["k"])), float(cfg.get("std_ratio", knn_cfg["std_ratio"]))
        if len(P_cam) <= 2:
            return np.ones(len(P_cam), dtype=bool)
        d, _ = cKDTree(P_cam).query(P_cam, k=min(k + 1, len(P_cam)))
        md = d[:, 1:].mean(axis=1)
        return md <= md.mean() + ratio * md.std()
    return np.ones(len(P_cam), dtype=bool)


def _empty(c: Candidate, n_valid: int) -> Instance:
    return Instance(candidate=c, points_base=np.zeros((0, 3)), colors=np.zeros((0, 3), np.uint8),
                    bbox3d=(np.full(3, np.nan), np.full(3, np.nan)), n_valid=n_valid)


def lift(frame: Frame, cands: List[Candidate], card: Card, cfg: dict) -> List[Instance]:
    """Candidate 마다 Instance. 점이 min_points(card.gate.min_points) 미만이면 points_base 가 (0,3) 인 Instance (pipeline 이 'too_few_points')."""
    fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
    depth = frame.depth
    min_pts = int(card.gate.get("min_points", cfg["lift"]["min_points_default"]))
    ocfg = card.lift.get("outlier") or cfg["lift"]["outlier"]
    disp = frame.extra.get("disparity")
    baseline_mm = frame.extra.get("baseline_mm")
    out = []
    for c in cands:
        ys, xs = np.nonzero(c.mask & np.isfinite(depth))
        n_raw = int(ys.size)
        if n_raw < min_pts:
            out.append(_empty(c, n_raw))
            continue
        if disp is not None and baseline_mm is not None:
            # [분기 1/2 — 회귀 충실도 목적, 출력은 항상 m] 스테레오 프레임: 원본 estimate_pose.py:69 와 비트 동일한 float32 mm 깊이(fx·B/d)를
            # 계산한 뒤 m 로 바꾼다. depth(float32 m) 왕복의 1e-5 mm 섭동이 조건 나쁜 SVD 축을 0.03° 흔드는 것을 막는다(obj_multi/015). ASSUMPTIONS 17
            Z = mm2m(np.float32(fx * baseline_mm) / disp[ys, xs].astype(np.float32))
        else:
            # [분기 2/2] depth(m) 만 주는 어댑터(synthetic·realsense·isaac): m 직접. Step 4 합성에서 두 경로 차이 1회 측정
            Z = depth[ys, xs].astype(np.float64)
        X = (xs - cx) * Z / fx                               # 원본 :70-71 과 같은 식 (m)
        Y = (ys - cy) * Z / fy
        P_cam = np.stack([X, Y, Z], axis=1)
        keep = _outlier_keep(P_cam, ocfg, cfg["lift"]["knn"])
        if not keep.any():
            out.append(_empty(c, n_raw))
            continue
        P_cam, ys, xs = P_cam[keep], ys[keep], xs[keep]
        P = transform_points(frame.T_base_cam, P_cam)         # base 좌표 (항등이면 카메라 좌표 그대로)
        out.append(Instance(candidate=c, points_base=P, colors=frame.rgb[ys, xs], bbox3d=(P.min(axis=0), P.max(axis=0)), n_valid=n_raw))
    return out
