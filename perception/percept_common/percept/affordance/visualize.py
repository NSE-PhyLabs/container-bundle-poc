"""⑤ 시각화: 코스트맵(열지도) + hard(빨강) + 후보(커널 외곽·순위·여유) → png. scene.visualize 의 층 그림과 나란히."""
import numpy as np

from ..scene.multimask import visualize as scene_visualize
from .kernel import rotate


def visualize(scene, affs, cost=None, hard=None, kernel=None, path=None, title: str = "") -> np.ndarray:
    import cv2
    left = scene_visualize(scene, None, show_height=False)
    H, W = scene.layers["object"].shape
    if cost is not None:
        c = cost.astype(np.float32)
        v = np.isfinite(c)
        cn = np.zeros_like(c)
        if v.any():
            cn[v] = (c[v] - c[v].min()) / max(float(np.ptp(c[v])), 1e-6)
        heat = cv2.applyColorMap((255 * cn).astype(np.uint8), cv2.COLORMAP_JET)
        if hard is not None:
            heat[hard] = (0, 0, 255)
    else:
        heat = np.zeros((H, W, 3), np.uint8)
    res = float(scene.plane["res_mm"])
    for c in (affs.candidates if affs is not None else []):
        u, v, th = c.uv_theta
        col, row = int(u / res), int(v / res)
        if kernel is not None:
            K = rotate(kernel["insert"], np.degrees(th))
            h, w = K.shape
            r0, c0 = row - h // 2, col - w // 2                                  # 홀수 커널: 중심 셀 = (h//2, w//2)
            color = (0, 255, 0) if c.rank == 1 else (255, 255, 0)
            cv2.rectangle(heat, (c0, r0), (c0 + w - 1, r0 + h - 1), color, 1)
            cv2.rectangle(left, (c0, r0), (c0 + w - 1, r0 + h - 1), color, 1)
        cv2.putText(heat, f"{c.rank}", (col + 3, row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.hstack([left[:H], np.full((H, 4, 3), 255, np.uint8), heat])
    lines = [title or f"{scene.scene_id} {affs.header.get('gripper') if affs else ''}/{affs.header.get('task') if affs else ''}"]
    for c in (affs.candidates[:3] if affs is not None else []):
        cl = c.clearance_mm
        lines.append(f"#{c.rank} {c.target_id} score {c.score:+.2f} up {cl['uphill']:.0f} below {cl['below']:.0f} side {cl['side_l']:.0f}/{cl['side_r']:.0f} {','.join(c.reasons)[:40]}")
    if affs is not None and affs.invalid_reason:
        lines.append(f"invalid: {affs.invalid_reason}")
    pad = np.zeros((14 * len(lines) + 6, out.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(pad, t, (4, 12 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([out, pad])
    if path is not None:
        cv2.imwrite(str(path), out)
    return out
