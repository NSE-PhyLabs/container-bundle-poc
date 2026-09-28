"""④-4 SceneMap 조립·저장·시각화 (v3 §7.7)."""
import logging

import numpy as np

from ..contract import SceneMap
from .free_space import free_space
from .orthographic import reproject
from .plane import estimate_plane

log = logging.getLogger(__name__)
LAYER_COLORS = dict(unobserved=(40, 40, 40), free=(70, 120, 70), object=(200, 140, 60), wall=(60, 60, 200), liner=(180, 100, 200))


def build_scene(frame, instances, hyps, poses, card, cfg, task=None, gripper=None, bias_mm=None) -> SceneMap:
    """grab→find→lift→pose 뒤의 ④: 기준면 → 정사영 층 → 여유 공간. 실패는 SceneError."""
    plane = estimate_plane(frame, instances, hyps, poses, card, cfg)
    layers = reproject(frame, instances, plane, card, cfg)
    walls_source = layers.pop("_walls_source", "none")
    walls_meta = layers.pop("_walls_meta", {})
    axis = (task or {}).get("axis", "uphill")
    kw = 60.0
    if gripper is not None:                                                  # 커널 폭은 **축과 수직인 방향**(free_space 가 세로 k 행으로 쓰는 축)의 치수다.
        k = gripper["kernels"]["insert"]                                     # 삽입 날 기준(발자국 전체가 아니라 들어갈 부분, QUESTIONS 23)
        kw = float(k.shape[0] if axis in ("uphill", "x", "u") else k.shape[1]) * float(gripper["res_mm"])
    walls = (card.scene or {}).get("walls") or {}
    inner = walls.get("inner_mm")                                            # [축 방향 길이(v), 경사 방향 폭(u), 깊이]
    container = float(inner[1] if axis in ("uphill", "x", "u") else inner[0]) if inner else None
    fs = free_space(layers, plane, axis, kw, bias_mm, container)
    rect = walls_meta.get("rect")
    if rect and layers["object"].any():                                      # 벽까지의 실제 틈(카드 벽 − 물체 외곽): 정사영이 벽면을 보느라 free 로 못 잡는 여유
        rows, cols = np.nonzero(layers["object"])
        res = float(plane["res_mm"])
        if axis in ("uphill", "x", "u"):
            fs["wall_gap_uphill_mm"] = float(max(0.0, rect["u1"] - (cols.max() + 1) * res))
            fs["wall_gap_downhill_mm"] = float(max(0.0, cols.min() * res - rect["u0"]))
        else:
            fs["wall_gap_uphill_mm"] = float(max(0.0, rect["v1"] - (rows.max() + 1) * res))
            fs["wall_gap_downhill_mm"] = float(max(0.0, rows.min() * res - rect["v0"]))
    meta = dict(reference_plane_type=(card.scene or {}).get("reference_plane", {}).get("type"),
                plane_from=(card.scene or {}).get("reference_plane", {}).get("plane_from"),
                n_instances=int(len(instances)), walls_source=walls_source, liner_source="none", walls=walls_meta,
                centers_spread_mm=plane.get("centers_spread_mm"), surface_rms_mm=plane.get("surface_rms_mm"),
                plane_n_used=plane["n_used"], task=(task or {}).get("name"), gripper=(gripper or {}).get("name"),
                free_empty=bool(not layers["free"].any()))
    return SceneMap(scene_id=frame.extra.get("scene_id", ""), plane=plane, layers=layers, free_space=fs, meta=meta)


def visualize(scene: SceneMap, path=None, show_height: bool = True) -> np.ndarray:
    """층을 한 장으로: 회색=미관측, 초록=free, 주황=물체(단 밝기), 파랑=벽. 오른쪽에 높이 지도."""
    import cv2
    L = scene.layers
    H, W = L["free"].shape
    img = np.zeros((H, W, 3), np.uint8)
    img[L["unobserved"]] = LAYER_COLORS["unobserved"]
    img[L["free"]] = LAYER_COLORS["free"]
    obj = L["object"]
    if obj.any():
        tier = L["tier"]
        for t in range(1, int(tier.max()) + 1):
            f = max(0.35, 1.0 - 0.25 * (t - 1))
            img[obj & (tier == t)] = np.clip(np.asarray(LAYER_COLORS["object"], float) * f, 0, 255).astype(np.uint8)
    img[L["liner"]] = LAYER_COLORS["liner"]
    img[L["wall"]] = LAYER_COLORS["wall"]
    out = img
    if show_height:
        h = L["height_mm"].copy()
        v = np.isfinite(h)
        hn = np.zeros_like(h)
        if v.any():
            hn[v] = (h[v] - h[v].min()) / max(float(np.ptp(h[v])), 1e-6)
        hm = cv2.applyColorMap((255 * hn).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        hm[~v] = LAYER_COLORS["unobserved"]
        out = np.hstack([img, np.full((H, 4, 3), 255, np.uint8), hm])
    fs = scene.free_space
    rigid = fs.get("manipulable_rigid_mm")
    txt = (f"{scene.scene_id}  res {scene.plane['res_mm']:.0f}mm  free {fs['direct_mm']:.0f}mm  "
           f"rigid {'–' if rigid is None else f'{rigid:.0f}mm'}  plane {scene.meta.get('plane_from')} "
           f"spread {scene.meta.get('centers_spread_mm', float('nan')):.1f}mm")
    out = np.vstack([out, np.zeros((22, out.shape[1], 3), np.uint8)])
    cv2.putText(out, txt, (4, out.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
    return out
