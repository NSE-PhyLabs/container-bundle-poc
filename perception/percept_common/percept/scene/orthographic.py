"""④-2 정사영 층 만들기 (v3 §7.7). 기준면 위에서 내려다본 격자(res_mm)로 관측을 다시 그린다.

`instance_id` 값 = **Hypotheses.items 색인 + 1** (0 = 물체 없음). Candidate.instance_id 와는 1 만큼 차이난다.
벽·라이너 층은 scene/walls.py (카드 치수 + hull|marker 원점, 라이너는 빈 층 `liner_source='none'`).
층: `height_mm`(셀에서 가장 카메라 쪽 높이, 관측 없으면 NaN) · `instance_id`(0=없음) · `object` · `free`(관측됐고 물체 아니고 낮음) ·
`unobserved` · `tier`(물체 셀의 단, 0=개구부 쪽) · `wall` · `liner`. 벽·라이너는 카드 `scene.walls` 가 있으면 사각 테두리로 그리고,
없으면 빈 층 + `meta.walls_source='none'` (색 기반 검출은 나중).
"""
import logging
from typing import Dict

import numpy as np

from ..lift import m2mm, mm2m
from .plane import to_plane

log = logging.getLogger(__name__)


def _grid_shape(plane: dict):
    res = plane["res_mm"]
    return int(round(plane["extent_mm"][1] / res)), int(round(plane["extent_mm"][0] / res))   # (행 = v, 열 = u)


def _cells(plane: dict, P_base: np.ndarray, shape):
    """base 점 → (행, 열, 높이 m, 격자 안 여부)."""
    uvh = to_plane(plane, P_base)
    res_m = mm2m(plane["res_mm"])
    col = np.floor(uvh[:, 0] / res_m).astype(int)
    row = np.floor(uvh[:, 1] / res_m).astype(int)
    ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return row, col, uvh[:, 2], ok


def reproject(frame, instances, plane: dict, card, cfg) -> Dict[str, np.ndarray]:
    """프레임의 모든 유효 depth 픽셀을 기준면 격자에 올린다. 같은 셀은 **높이가 큰(카메라 쪽) 점**이 이긴다."""
    shape = _grid_shape(plane)
    H, W = shape
    height = np.full(shape, np.nan, np.float32)
    inst_id = np.zeros(shape, np.int32)

    ys, xs = np.nonzero(np.isfinite(frame.depth))
    if len(ys) == 0:
        raise ValueError("유효 depth 없음")
    z = frame.depth[ys, xs].astype(float)
    K = frame.K
    P = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=1)
    P = P @ frame.T_base_cam[:3, :3].T + frame.T_base_cam[:3, 3]
    owner = np.zeros(len(ys), np.int32)                                    # 픽셀이 속한 인스턴스 (1부터)
    for k, inst in enumerate(instances):
        owner[inst.candidate.mask[ys, xs]] = k + 1

    row, col, h, ok = _cells(plane, P, shape)
    row, col, h, owner = row[ok], col[ok], h[ok], owner[ok]
    order = np.argsort(h)                                                  # 낮은 것부터 → 높은 것이 덮음
    height[row[order], col[order]] = h[order].astype(np.float32)
    inst_id[row[order], col[order]] = owner[order]

    observed = np.isfinite(height)
    free_h = mm2m(float((card.scene or {}).get("free_height_mm", cfg["scene"]["free_height_mm"])))
    obj = inst_id > 0
    layers = dict(height_mm=m2mm(height).astype(np.float32), instance_id=inst_id, object=obj,
                  free=observed & ~obj & (np.nan_to_num(height, nan=1e9) < free_h), unobserved=~observed)

    # 단(tier): **인스턴스 단위**로 묶는다. 셀 높이로 끊으면 지름 140 mm 원통 하나가 혼자 여러 단으로 쪼개진다.
    tier = np.zeros(shape, np.int32)
    tc = float((card.scene or {}).get("tier_cluster_mm", 30.0))
    if obj.any():
        ids = [k for k in range(1, int(inst_id.max()) + 1) if (inst_id == k).any()]
        rep_h = {k: float(m2mm(np.nanmedian(height[inst_id == k]))) for k in ids}      # 인스턴스 대표 높이
        rest, lvl = sorted(ids, key=lambda k: -rep_h[k]), 1
        while rest:                                                          # 가장 높은 것에서 tier_cluster_mm 안은 같은 단 (gate.row_level 과 같은 규칙)
            top = rep_h[rest[0]]
            same = [k for k in rest if top - rep_h[k] <= tc]
            for k in same:
                tier[inst_id == k] = lvl
            rest = [k for k in rest if k not in same]
            lvl += 1
    layers["tier"] = tier

    from .walls import walls_layer                                           # 벽 = 카드 치수(hull|marker), 라이너 = 빈 층 (Q24)
    wall, src, wmeta = walls_layer(frame, plane, shape, card, obj)
    layers["wall"] = wall
    layers["liner"] = np.zeros(shape, bool)
    layers["_walls_source"] = src
    layers["_walls_meta"] = wmeta
    return layers
