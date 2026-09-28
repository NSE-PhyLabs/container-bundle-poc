"""⑤-4 여유 측정 (v3 §7.8 + Q23 2.5D).

`measure(scene, cell, K, ...)` → {uphill, below, side_l, side_r, flags}. cell = 커널 중심 (행, 열), K = 회전된 발자국(bool, 홀수 크기).
- uphill = +u(열 +), **side_l = +v(행 +) = 로봇 왼쪽(base +y)**, side_r = −v. (v = n × u: 기울어진 박스·팔레트·카메라 프레임 모두 +v 가 로봇 왼쪽이다.)
  커널 외곽에서 그 방향으로 **장애물**까지 ray-march (mm). 장애물 = 통과 가능 셀(free 또는 지지면)도 아니고 제외 인스턴스(타깃·경사 위 사슬·골 상대)도 아닌 셀.
  미관측에 막히면 flags 에 'uphill_unobserved' 등. 날이 둘(two_point)이면 두 위치 각각 재서 방향별 최솟값.
- below (2.5D): clearance_map = D_surface − (blade_zero + blade_depth), D_surface = −높이(아래 +), 미관측·격자 밖 = 0(기준면, 보수적). 날 셀의 최솟값, < 0 이면 충돌.
  generate 가 θ별로 만든 지도를 cm 으로 받아 다시 계산하지 않는다. 날이 없으면 기준면 − 커널 아래 최고 표면(골 깊이)로 근사.
  goal='place' 면 below = −(발자국 아래 |h| 99 %) = 받침 편평도(필요치 검사 없음).
"""
from typing import Iterable, Optional, Sequence

import numpy as np


def blade_clearance_map(height_mm: np.ndarray, blade: np.ndarray, blade_zero_mm: float) -> np.ndarray:
    """모든 셀 p 에 대해 min_k (D_surface[p + k] − zero − blade[k]) (날 셀만, k = 날 중심 기준 오프셋)."""
    from scipy.ndimage import grey_dilation
    D = -np.nan_to_num(np.asarray(height_mm, np.float32), nan=0.0)          # 아래 +
    S = np.flip(np.asarray(blade, np.float32), (0, 1)).copy()                 # p + k ↔ grey_dilation 의 p − k
    fp = S > 0
    if not fp.any():
        return np.full(D.shape, np.inf, np.float32)
    mx = grey_dilation(-D, structure=np.where(fp, S, -np.inf), footprint=fp, mode="constant", cval=0.0)   # max_k (−D[p+k] + blade[k])
    return (-mx - float(blade_zero_mm)).astype(np.float32)


def place_cells(K: np.ndarray, cell, shape):
    """커널 중심을 cell=(행, 열) 에 놓았을 때 점유 셀 인덱스(격자 안만)."""
    h, w = K.shape
    rr, cc = np.nonzero(K)
    rr, cc = rr + int(cell[0]) - h // 2, cc + int(cell[1]) - w // 2
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[ok], cc[ok]


def _march(obstacle: np.ndarray, rows, cols, direction: str, res: float) -> float:
    """커널 외곽 바로 다음 셀부터 direction 으로 장애물 전까지 셀 수 × res."""
    H, W = obstacle.shape
    best = None
    if direction == "+u":
        edge = cols.max()
        for l in np.unique(rows[cols == edge]):
            seg = obstacle[l, edge + 1:]
            d = int(np.argmax(seg)) if seg.any() else len(seg)
            best = d if best is None else min(best, d)
    else:
        edge = rows.max() if direction == "+v" else rows.min()
        for l in np.unique(cols[rows == edge]):
            seg = obstacle[edge + 1:, l] if direction == "+v" else obstacle[:edge, l][::-1]
            d = int(np.argmax(seg)) if seg.any() else len(seg)
            best = d if best is None else min(best, d)
    return float((best or 0) * res)


def measure(scene, cell, K: np.ndarray, exclude_ids: Iterable[int] = (), goal: str = "extract", support: Optional[np.ndarray] = None,
            cm: Optional[np.ndarray] = None, blade_offsets: Sequence = ((0, 0),)) -> dict:
    import cv2
    L = scene.layers
    res = float(scene.plane["res_mm"])
    shape = L["object"].shape
    excl = np.zeros(shape, bool)
    owner = L.get("model_owner")
    for i in exclude_ids:
        excl |= L["instance_id"] == int(i) + 1
        if owner is not None:
            excl |= owner == int(i) + 1                                         # 모델 footprint(가려진 셀 포함)도 제외 인스턴스로
    if excl.any():                                                              # 제외 인스턴스 사이의 골 바닥 셀(모델 밖·미관측)을 잇는다 [추측 6 mm]
        r = max(1, int(round(6.0 / res)))
        excl = cv2.dilate(excl.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    passable = (support if (goal == "place" and support is not None) else L["free"]) | excl
    obstacle, unob = ~passable, L["unobserved"]
    out = dict(flags=[])
    vals = dict(uphill=[], side_l=[], side_r=[])
    under = []
    for dr, dc in blade_offsets:
        rows, cols = place_cells(K, (cell[0] + dr, cell[1] + dc), shape)
        if rows.size == 0:
            return dict(uphill=0.0, below=0.0, side_l=0.0, side_r=0.0, flags=["kernel_outside"])
        under.append((rows, cols))
        for name, d in (("uphill", "+u"), ("side_l", "+v"), ("side_r", "-v")):
            m = _march(obstacle, rows, cols, d, res)
            vals[name].append(m)
            if _march(obstacle & ~unob, rows, cols, d, res) > m + 1e-6 and f"{name}_unobserved" not in out["flags"]:
                out["flags"].append(f"{name}_unobserved")                       # 미관측을 장애물로 안 보면 더 멀리 간다 = 미관측에 막힌 것
    for k, v in vals.items():
        out[k] = float(min(v))
    rows = np.concatenate([u[0] for u in under]); cols = np.concatenate([u[1] for u in under])
    hk = L["height_mm"][rows, cols]
    if goal == "place":
        out["below"] = float(-np.nanpercentile(np.abs(hk), 99)) if np.isfinite(hk).any() else 0.0   # 받침 편평도(|h| 99 %). 최댓값은 고립 셀 1개에 끌려간다
    elif cm is not None:
        H, W = shape
        out["below"] = float(min(cm[min(max(cell[0] + dr, 0), H - 1), min(max(cell[1] + dc, 0), W - 1)] for dr, dc in blade_offsets))
        if unob[rows, cols].any():
            out["flags"].append("unobserved_under_blade")
    else:                                                                       # 날 프로파일 없음: 골 깊이 근사(기준면 − 커널 아래 최고 표면)
        out["below"] = float(-np.nanmax(hk)) if np.isfinite(hk).any() else 0.0
    return out
