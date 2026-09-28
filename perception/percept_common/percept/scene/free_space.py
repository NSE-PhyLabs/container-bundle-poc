"""④-3 여유 공간 (v3 §7.7).

- `direct_mm`: free 층에서 축 방향으로 **커널 폭 이상 연속으로** 비어 있는 최대 길이. 축은 기준면 u 축(경사 위)이다.
  각 행의 연속 free 구간을 누적합으로 구하고, 커널 폭만큼 이웃한 행들이 모두 겹치는 구간의 길이를 최대화한다.
- `manipulable_rigid_mm`: 축 방향 전체 길이 − Σ(인스턴스의 축 방향 투영 폭). 강체를 밀어 만들 수 있는 최대 여유.
- `manipulable_measured_mm`: FeedbackStore 의 (실측 − 예측) 편차를 direct_mm 에 더한 값. 되먹임이 없으면 None.
"""
from typing import Optional

import numpy as np


def _max_run_with_width(free: np.ndarray, width_cells: int) -> int:
    """free (H, W) 에서, 세로로 width_cells 행이 모두 비어 있는 가로 구간의 최대 길이(셀)."""
    if width_cells > free.shape[0]:
        return 0                                                            # 커널이 지도보다 넓으면 들어갈 자리가 없다(자르지 말 것)
    if width_cells <= 1:
        band = free
    else:
        k = width_cells
        c = np.cumsum(np.vstack([np.zeros((1, free.shape[1]), int), free.astype(int)]), axis=0)
        band = (c[k:] - c[:-k]) == k                                       # 연속 k 행이 모두 free 인 시작 행들
        if band.size == 0:
            return 0
    best = 0
    for r in band:
        run = 0
        for x in r:
            run = run + 1 if x else 0
            best = max(best, run)
    return int(best)


def free_space(layers: dict, plane: dict, axis: str = "uphill", kernel_width_mm: float = 60.0,
               bias_mm: Optional[float] = None, container_extent_mm: Optional[float] = None) -> dict:
    """반환 {direct_mm, manipulable_rigid_mm, manipulable_measured_mm, axis, kernel_width_mm, extent_mm}.
    `manipulable_rigid_mm` 은 **용기 내부 길이**(container_extent_mm, 카드 scene.walls.inner_mm)가 있어야 뜻이 있다 —
    격자 폭은 그림 여백(cfg scene.margin_mm)을 포함하므로 물리량이 아니다. 없으면 None."""
    res = float(plane["res_mm"])
    free = np.asarray(layers["free"], bool)
    obj = np.asarray(layers["object"], bool)
    if axis in ("uphill", "x", "u"):                                        # 기준면 u 축 = 격자 가로
        free_a, obj_a = free, obj
    else:                                                                    # v 축으로 재면 전치
        free_a, obj_a = free.T, obj.T
    width_cells = max(1, int(round(kernel_width_mm / res)))
    direct = _max_run_with_width(free_a, width_cells) * res
    total = free_a.shape[1] * res
    occupied = float(obj_a.any(axis=0).sum()) * res                          # 축 방향으로 물체가 차지한 폭
    rigid = float(max(0.0, container_extent_mm - occupied)) if container_extent_mm else None
    return dict(direct_mm=float(direct), manipulable_rigid_mm=rigid, occupied_mm=float(occupied),
                manipulable_measured_mm=(float(max(0.0, direct + bias_mm)) if bias_mm is not None else None),   # 음수 여유는 없다(0 으로 자름)
                axis=axis, kernel_width_mm=float(kernel_width_mm), extent_mm=float(total))
