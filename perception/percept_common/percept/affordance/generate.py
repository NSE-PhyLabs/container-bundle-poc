"""⑤-3 후보 생성 (v3 §7.8 + Q23).

`generate(scene, kernel, rules, target, cost, hard, ...)` → [candidate dict]. 대상 인스턴스마다(놓기는 전체 지도):
θ ∈ rules.rotations_deg: K_θ = rotate(발자국). allow = 커널이 격자 안에 다 들어가고 hard 와 한 셀도 안 겹침(filter2D 상관, 홀수 커널이라 중심 = 앵커).
score = −(커널 아래 평균 비용). 2.5D: 날 프로파일이 있으면 clearance_map ≥ 0 인 곳만(충돌 없음). θ별 지도는 cm_cache 로 타깃 사이에 재사용한다.
ROI = 골 상대가 있으면 **자기 골 띠**(u_valley ± band, 타깃 v 범위), 없으면 타깃 셀 ± (반경×2 또는 size 최대). 피크 → **정사각** NMS(한 변 2·nms_mm) →
θ 를 합쳐 점수순 top_k(타깃당). two_point: 커널 긴 축 방향 ±(L/2 − end_inset) 두 위치(θ 만큼 회전)가 모두 허용이어야 하고 점수는 합, 여유는 둘 중 작은 쪽.
pattern.yaml 슬롯이 있으면 후보에 가장 가까운 슬롯 번호(60 mm 안)를 'slot' 으로 붙인다(rank 의 pattern_order 키).
"""
from typing import List, Optional

import numpy as np

from .clearance import blade_clearance_map
from .kernel import axis_offset_cells, rotate, two_point_offsets_mm


def _overlap(mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """각 셀에 커널 중심을 놓았을 때 mask 와 겹치는 셀 수 (상관 = 커널을 그대로 놓음. 격자 밖은 0)."""
    import cv2
    return cv2.filter2D(mask.astype(np.float32), -1, K.astype(np.float32), borderType=cv2.BORDER_CONSTANT)


def _shift(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc] (격자 밖은 fill) — np.roll 과 달리 반대쪽 끝이 섞이지 않는다."""
    out = np.full(a.shape, fill, a.dtype)
    H, W = a.shape
    r0, r1, c0, c1 = max(0, -dr), min(H, H - dr), max(0, -dc), min(W, W - dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def score_maps(cost: np.ndarray, hard: np.ndarray, K: np.ndarray, height_mm=None, blade=None, blade_zero_mm: float = 0.0, cm=None):
    """반환 (score, allow, clearance_map|None). cm 을 주면 다시 계산하지 않는다(타깃과 무관한 지도)."""
    n = float(K.sum())
    if n <= 0:
        raise ValueError("빈 커널")
    allow = (_overlap(np.ones(hard.shape, bool), K) >= n - 0.5) & (_overlap(hard, K) < 0.5)
    score = (-_overlap(cost, K) / n).astype(np.float32)
    if blade is not None and height_mm is not None:
        if cm is None:
            cm = blade_clearance_map(height_mm, blade, blade_zero_mm)
        allow &= cm >= 0.0
    else:
        cm = None
    return score, allow, cm


def nms_peaks(score: np.ndarray, allow: np.ndarray, roi, nms_cells: int, top_k: int) -> List[tuple]:
    """ROI 안 허용 셀에서 점수 높은 순으로 (2·nms_cells+1) 정사각 안을 억제하며 top_k 개. 동점은 (행, 열) 오름차순(결정론)."""
    r0, r1, c0, c1 = roi
    sub = np.where(allow[r0:r1, c0:c1], score[r0:r1, c0:c1], -np.inf)
    if not np.isfinite(sub).any():
        return []
    picked, taken = [], np.zeros(sub.shape, bool)
    for f in np.argsort(-sub, axis=None, kind="stable"):
        r, c = np.unravel_index(f, sub.shape)
        if not np.isfinite(sub[r, c]):
            break
        if taken[r, c]:
            continue
        picked.append((int(r + r0), int(c + c0), float(sub[r, c])))
        if len(picked) >= top_k:
            break
        taken[max(0, r - nms_cells):r + nms_cells + 1, max(0, c - nms_cells):c + nms_cells + 1] = True
    return picked


def target_roi(scene, tidx: Optional[int], card, res: float, shape, valley=None, band_mm: float = 40.0):
    if tidx is None:
        return (0, shape[0], 0, shape[1])
    rows, cols = np.nonzero(scene.layers["instance_id"] == tidx + 1)
    if rows.size == 0:
        return None
    if valley is not None:                                                     # 자기 골 띠 — 자기 골이 막혔다고 남의 골에 후보를 내지 않는다
        c, b = int(valley["u_mm"] / res), int(round(band_mm / res))
        return (max(0, rows.min()), min(shape[0], rows.max() + 1), max(0, c - b), min(shape[1], c + b + 1))
    gp = card.geometry.get("params") or {}
    pad_mm = 2 * float(gp.get("radius_mm", 0.0)) if card.geometry["type"] == "cylinder" else float(max(gp.get("size_mm", [100])))
    pad = int(round(pad_mm / res))
    return (max(0, rows.min() - pad), min(shape[0], rows.max() + pad + 1), max(0, cols.min() - pad), min(shape[1], cols.max() + pad + 1))


def generate(scene, kernel: dict, rules: dict, target, cost: np.ndarray, hard: np.ndarray, card=None, tidx: Optional[int] = None,
             footprint: Optional[np.ndarray] = None, cm_cache: Optional[dict] = None, valley=None, slots=None) -> List[dict]:
    L = scene.layers
    res = float(scene.plane["res_mm"])
    shape = L["object"].shape
    roi = target_roi(scene, tidx if target is not None else None, card, res, shape, valley)
    if roi is None:
        return []
    nms_cells = max(1, int(round(float(rules.get("nms_mm", 40.0)) / res)))
    top_k = int(rules.get("top_k", 5))
    base_k = footprint if footprint is not None else kernel["insert"]
    blade = kernel.get("blade") if footprint is None else None
    two = bool(kernel["meta"].get("two_point")) and target is not None and footprint is None
    out = []
    for th in rules.get("rotations_deg", [0]):
        th = float(th)
        K = rotate(base_k, th)
        B = rotate(blade, th) if blade is not None else None
        score, allow, cm = score_maps(cost, hard, K, L.get("height_2p5_mm", L["height_mm"]), B, kernel["meta"].get("blade_zero_mm", 0.0),
                                      cm_cache.get(th) if cm_cache is not None else None)
        if cm_cache is not None and cm is not None:
            cm_cache[th] = cm
        offs = [(0, 0)]
        if two:
            Lmm = float((target.geometry.get("params_fit") or {}).get("length_mm") or np.mean(card.geometry["params"]["length_mm"]))
            offs = [axis_offset_cells(d, th, res) for d in two_point_offsets_mm(Lmm, kernel["meta"]["end_inset_mm"])]
            score = sum(_shift(score, dr, dc, -np.inf) for dr, dc in offs)
            allow = np.logical_and.reduce([_shift(allow, dr, dc, False) for dr, dc in offs])
            score = np.where(allow, score, -np.inf).astype(np.float32)
        for r, c, s in nms_peaks(score, allow, roi, nms_cells, top_k):
            cand = dict(target_id=(target.id if target is not None else None), row=r, col=c, u_mm=(c + 0.5) * res, v_mm=(r + 0.5) * res,
                        theta_deg=th, score=float(s), blade_offsets=offs,
                        blade_clearance_mm=(float(min(cm[min(max(r + dr, 0), shape[0] - 1), min(max(c + dc, 0), shape[1] - 1)] for dr, dc in offs))
                                            if cm is not None else None))
            if slots:
                d = [np.hypot(cand["u_mm"] - s_["u_mm"], cand["v_mm"] - s_["v_mm"]) for s_ in slots]
                if min(d) <= 60.0:
                    cand["slot"] = int(np.argmin(d))
            out.append(cand)
    out.sort(key=lambda c: (-c["score"], c["row"], c["col"], c["theta_deg"]))   # θ 를 합쳐 타깃당 top_k
    return out[:top_k]
