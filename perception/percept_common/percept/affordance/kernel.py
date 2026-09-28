"""⑤-1 그리퍼 커널 (v3 §7.8 + Q23 2.5D).

`load_kernel(gripper)` → {'insert': bool HxW, 'support': bool HxW, 'blade': float32 HxW mm | None, 'meta': {...}}.
배열 규약 = 정사영 격자와 같다: **행 = v(축 방향, 로봇 왼쪽 +), 열 = u(경사 위 +)**. θ = 0 이면 커널의 긴 변(물체 축 방향)이 행을 따라 놓인다.
모든 커널은 **홀수 크기**로 맞춘다(짝수면 0 한 줄을 덧댐) — 그래야 중심 셀이 (h//2, w//2) 하나로 정해져 filter2D·grey_dilation·place_cells 의 앵커가 일치한다.
`rotate(arr, θ)`: θ > 0 = 기준면 법선(+n) 둘레 회전(u → v). 배열에서는 행이 아래로 증가하므로 화면상 시계 방향 = cv2 각도 −θ. 중심을 고정하고 대칭으로 자른다.
blade: 픽셀값 = 그 칸에서 날이 차지하는 **아래 방향 깊이 mm**(0 = 날 없음). meta.blade_zero_mm = 기준면에서 날 윗면까지의 깊이(아래 +).
회전 후보(rotations_deg)는 작업 rules.yaml 이 정한다(그리퍼 쪽 필드는 없앰 — 한 곳에서만 정한다).
"""
import numpy as np


def _odd(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    return np.pad(arr, ((0, 1 - h % 2), (0, 1 - w % 2)))


def load_kernel(gripper: dict) -> dict:
    k = gripper["kernels"]
    blade = k.get("blade")
    meta = dict(name=gripper["name"], res_mm=float(gripper["res_mm"]), approach=dict(gripper["approach"]),
                insert_depth_mm=float(gripper["insert_depth_mm"]), clearance_need_mm=dict(gripper["clearance_need_mm"]),
                two_point=bool(gripper.get("two_point", False)), end_inset_mm=float(gripper.get("end_inset_mm", 0.0)),
                blade_zero_mm=float(gripper.get("blade_zero_mm", 0.0)))
    return dict(insert=_odd(np.asarray(k["insert"], bool)), support=_odd(np.asarray(k["support"], bool)),
                blade=None if blade is None else _odd(np.asarray(blade, np.float32)), meta=meta)


def rotate(arr: np.ndarray, theta_deg: float) -> np.ndarray:
    """홀수 크기 입력 → 홀수 크기 출력, 중심 셀 고정(최근접 보간)."""
    arr = _odd(arr)
    if abs(theta_deg) < 1e-9:
        return arr.copy()
    import cv2
    h, w = arr.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2
    diag += 1 - diag % 2
    c = diag // 2
    canvas = np.zeros((diag, diag), np.float32)
    canvas[c - h // 2:c + h // 2 + 1, c - w // 2:c + w // 2 + 1] = arr.astype(np.float32)
    M = cv2.getRotationMatrix2D((float(c), float(c)), -float(theta_deg), 1.0)       # +n 둘레(u→v) = 화면 시계 방향 = cv2 음수 각
    out = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.nonzero(out > 0)
    if ys.size == 0:
        return np.zeros((1, 1), arr.dtype)
    hr, hc = int(np.abs(ys - c).max()), int(np.abs(xs - c).max())                   # 중심 대칭으로 자른다(비대칭 형상도 중심이 안 움직임)
    out = out[c - hr:c + hr + 1, c - hc:c + hc + 1]
    return (out > 0.5) if arr.dtype == bool else out.astype(arr.dtype)


def footprint_kernel(size_mm, res_mm: float) -> np.ndarray:
    """물체 발자국(놓기): size_mm = [x(물체 긴 축 → v 행), y(→ u 열)] → bool 홀수 크기."""
    h, w = max(1, int(round(float(size_mm[0]) / res_mm))), max(1, int(round(float(size_mm[1]) / res_mm)))
    return _odd(np.ones((h, w), bool))


def two_point_offsets_mm(length_mm: float, end_inset_mm: float) -> tuple:
    """양끝 파지: 커널 긴 축 방향 ±(L/2 − end_inset) mm."""
    d = max(0.0, float(length_mm) / 2.0 - float(end_inset_mm))
    return (-d, d)


def axis_offset_cells(d_mm: float, theta_deg: float, res_mm: float) -> tuple:
    """커널 긴 축(θ = 0 에서 +v) 방향으로 d_mm 떨어진 점의 (행, 열) 셀 오프셋. θ 만큼 +n 둘레로 돌면 v̂ → (−sinθ·û, cosθ·v̂)."""
    th = np.radians(theta_deg)
    return int(round(d_mm * np.cos(th) / res_mm)), int(round(-d_mm * np.sin(th) / res_mm))
