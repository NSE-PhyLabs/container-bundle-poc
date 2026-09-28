"""④-5 모델 표면 높이 (2.5D 판정용). 맞춘 기하(원통·상자)를 기준면 격자에 그려 각 셀의 **모델 표면 높이 mm** 를 만든다.
관측 높이 지도는 골 바닥처럼 가려진 곳을 못 보므로(카메라가 법선에서 27° 만 기울어도 V골은 10 mm 깊이까지만 보인다), 2.5D 삽입 판정은
**모델이 있는 셀은 모델, 없는 셀만 관측**(`fused_height_mm`)을 쓴다. max(관측, 모델)로 두면 실루엣 근처 관측(셀별 최대값 + 깊이 잡음 + splat 번짐)이
모델보다 8~13 mm 높아 골이 막힌다(9/21 합성 확인). 모델 footprint 안의 미모델 장애물은 놓친다 [추측 — Step 8 실사로 검증]. 둘 다 없으면 NaN(기준면 취급, 보수적).
`model_height_mm` 은 (높이, 소유 인스턴스 id 층: Hypotheses.items 색인 + 1) 을 돌려준다."""
import numpy as np

from ..pose.base import matrix_from_quat
from .plane import to_plane


def model_height_mm(plane: dict, shape, hyps, card):
    H, W = shape
    res = float(plane["res_mm"])
    out = np.full((H, W), np.nan, np.float32)
    owner = np.zeros((H, W), np.int32)
    R_p = plane["T_base_plane"][:3, :3]
    gp = card.geometry.get("params") or {}
    uu = (np.arange(W) + 0.5) * res
    vv = (np.arange(H) + 0.5) * res
    for k, o in enumerate(hyps.items):
        if not o.pose_valid:
            continue
        c = to_plane(plane, np.asarray(o.position, float)[None])[0] * 1000.0     # (u, v, h) mm
        R_o = matrix_from_quat(o.orientation)
        if card.geometry["type"] == "cylinder":
            r = float(gp["radius_mm"])
            L = float((o.geometry.get("params_fit") or {}).get("length_mm") or np.mean(gp["length_mm"]))
            a = R_p.T @ R_o[:, 0]                                             # 축을 기준면 좌표로
            a2 = a[:2] / max(float(np.linalg.norm(a[:2])), 1e-9)
            c0, c1 = max(0, int((c[0] - L / 2 - r) / res)), min(W, int((c[0] + L / 2 + r) / res) + 1)
            r0, r1 = max(0, int((c[1] - L / 2 - r) / res)), min(H, int((c[1] + L / 2 + r) / res) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            du = uu[c0:c1][None, :] - c[0]
            dv = vv[r0:r1][:, None] - c[1]
            t = du * a2[0] + dv * a2[1]
            perp = -du * a2[1] + dv * a2[0]
            inside = (np.abs(t) <= L / 2) & (np.abs(perp) <= r)
            h = c[2] + np.sqrt(np.maximum(r ** 2 - perp ** 2, 0.0))
            sub = out[r0:r1, c0:c1]
            win = inside & (h > np.nan_to_num(sub, nan=-np.inf))
            out[r0:r1, c0:c1] = np.where(win, h, sub).astype(np.float32)
            owner[r0:r1, c0:c1] = np.where(win, k + 1, owner[r0:r1, c0:c1])
        elif card.geometry["type"] == "box":
            sx, sy, sz = (float(v) for v in gp["size_mm"])
            ex, ey = R_p.T @ R_o[:, 0], R_p.T @ R_o[:, 1]
            ex2, ey2 = ex[:2] / max(float(np.linalg.norm(ex[:2])), 1e-9), ey[:2] / max(float(np.linalg.norm(ey[:2])), 1e-9)
            rad = float(np.hypot(sx, sy)) / 2
            c0, c1 = max(0, int((c[0] - rad) / res)), min(W, int((c[0] + rad) / res) + 1)
            r0, r1 = max(0, int((c[1] - rad) / res)), min(H, int((c[1] + rad) / res) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            du = uu[c0:c1][None, :] - c[0]
            dv = vv[r0:r1][:, None] - c[1]
            tx, ty = du * ex2[0] + dv * ex2[1], du * ey2[0] + dv * ey2[1]
            inside = (np.abs(tx) <= sx / 2) & (np.abs(ty) <= sy / 2)
            h_top = c[2] + sz / 2 * float(abs(R_p[:, 2] @ R_o[:, 2]))
            sub = out[r0:r1, c0:c1]
            win = inside & (h_top > np.nan_to_num(sub, nan=-np.inf))
            out[r0:r1, c0:c1] = np.where(win, h_top, sub).astype(np.float32)
            owner[r0:r1, c0:c1] = np.where(win, k + 1, owner[r0:r1, c0:c1])
    return out, owner


def fused_height_mm(height_mm: np.ndarray, model_mm: np.ndarray) -> np.ndarray:
    """모델이 있는 셀 = 모델, 없는 셀 = 관측. 둘 다 없으면 NaN."""
    m = np.asarray(model_mm, np.float32)
    return np.where(np.isfinite(m), m, np.asarray(height_mm, np.float32)).astype(np.float32)
