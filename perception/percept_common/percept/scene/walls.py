"""④ 벽·라이너 층 (Q24 답 9/21): **카드 치수로 그린다**, 색 검출은 하지 않는다(조명 의존).

카드 `scene.walls: {inner_mm: [L(축 방향, v), W(경사 방향, u), D(깊이)], lip_mm, thickness_mm, origin: hull|marker, marker: {id, size_mm, dict, T_box_marker}}`.
박스 안쪽 좌표계 B: 원점 = (축− 끝벽, 경사 위 벽, 바닥) 안쪽 구석, x = 축(+ 방향은 v 와 같음), y = 경사 아래(+), z = 개구부(+).
- origin=hull: 물체 셀의 외곽에서 역산. 전제 "줄은 경사 아래 벽에 붙어 있다"(PROJECT_NOTES) → 경사 아래 벽 안쪽 면 = 물체 셀의 u 최소, 경사 위 벽 = 거기서 +W;
  끝벽은 물체 셀의 v 범위 중심 ± L/2. **첫 단만 보이면 부정확**(경고 1회) — 벽 위치는 마커가 정확하다.
- origin=marker: 프레임 rgb 에서 ArUco(id, size) 검출 → T_base_marker = T_base_cam·T_cam_marker → T_base_box = T_base_marker·inv(T_box_marker) →
  안쪽 사각형 네 모서리(개구부 높이 z = D)를 기준면 (u, v) 로 투영해 그린다. 검출 실패·T_box_marker 미기입이면 hull 로 폴백 + 경고.
라이너 층은 비운다(`liner_source='none'`) — 검출 방법이 없는 것을 있는 척하지 않는다.
"""
import logging
from typing import Optional, Tuple

import numpy as np

from ..lift import mm2m
from ..pose.base import invert_T, make_T, rpy_to_R
from .plane import to_plane

log = logging.getLogger(__name__)
_warned = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _draw_ring(shape, u0: float, u1: float, v0: float, v1: float, res: float, t_mm: float) -> np.ndarray:
    """안쪽 사각형 [u0,u1]×[v0,v1] (mm) 바깥으로 두께 t 의 테두리를 격자(행 = v, 열 = u)에 그린다."""
    H, W = shape
    wall = np.zeros(shape, bool)
    c0, c1 = int(np.floor(u0 / res)), int(np.ceil(u1 / res))
    r0, r1 = int(np.floor(v0 / res)), int(np.ceil(v1 / res))
    t = max(1, int(round(t_mm / res)))
    def fill(ra, rb, ca, cb):
        ra, rb, ca, cb = max(0, ra), min(H, rb), max(0, ca), min(W, cb)
        if rb > ra and cb > ca:
            wall[ra:rb, ca:cb] = True
    fill(r0 - t, r1 + t, c0 - t, c0)
    fill(r0 - t, r1 + t, c1, c1 + t)
    fill(r0 - t, r0, c0 - t, c1 + t)
    fill(r1, r1 + t, c0 - t, c1 + t)
    return wall


def walls_from_hull(shape, obj: np.ndarray, res: float, wl: dict) -> Tuple[np.ndarray, dict]:
    L, W, D = (float(v) for v in wl["inner_mm"])
    rows, cols = np.nonzero(obj)
    if rows.size == 0:
        raise ValueError("물체 셀 없음")
    u_down = float(cols.min()) * res                                        # 경사 아래 벽 안쪽 면 = 물체의 u 최소 (줄이 아래 벽에 붙음)
    vc = (float(rows.min()) + float(rows.max()) + 1) / 2 * res
    rect = dict(u0=u_down, u1=u_down + W, v0=vc - L / 2, v1=vc + L / 2)
    return _draw_ring(shape, rect["u0"], rect["u1"], rect["v0"], rect["v1"], res, float(wl.get("thickness_mm", 10.0))), rect


def walls_from_marker(shape, frame, plane: dict, res: float, wl: dict) -> Tuple[Optional[np.ndarray], Optional[dict], str]:
    """박스 ArUco 로 벽. 반환 (wall|None, rect|None, 사유). 프레임 rgb 는 정렬된 왼쪽 영상, K = frame.K."""
    mk = wl.get("marker") or {}
    if mk.get("T_box_marker") is None:
        return None, None, "T_box_marker 미기입(실측 후 카드에 기입)"
    from ..camera.aruco import detect_marker
    det = detect_marker(frame.rgb, frame.K, mm2m(float(mk["size_mm"])), int(mk["id"]), mk.get("dict", "DICT_4X4_50"))
    if det["T"] is None:
        return None, None, f"마커 {det['reason']}"
    tbm = mk["T_box_marker"]
    T_box_marker = (make_T(rpy_to_R(*(float(a) for a in tbm[3:])), mm2m(np.asarray(tbm[:3], float)))      # [x y z mm, rx ry rz deg]
                    if len(tbm) == 6 else np.asarray(tbm, float))
    T_base_box = frame.T_base_cam @ det["T"] @ invert_T(T_box_marker)
    import cv2
    L, W, D = (mm2m(float(v)) for v in wl["inner_mm"])
    t = mm2m(float(wl.get("thickness_mm", 10.0)))
    def uv_px(corners_B):                                                        # 박스 좌표 모서리 → 기준면 (u, v) → 격자 (열, 행)
        uv = to_plane(plane, np.asarray(corners_B, float) @ T_base_box[:3, :3].T + T_base_box[:3, 3])[:, :2] * 1000.0
        return uv, np.round(uv / res).astype(np.int32)
    uv_in, px_in = uv_px([[0, 0, D], [L, 0, D], [L, W, D], [0, W, D]])          # 개구부 높이의 안쪽 네 모서리
    _, px_out = uv_px([[-t, -t, D], [L + t, -t, D], [L + t, W + t, D], [-t, W + t, D]])
    ring = np.zeros(shape, np.uint8)                                             # 기준면과 박스가 조금 틀어져 있어도(±1°) 부풀지 않게 **회전된 다각형**으로 그린다
    cv2.fillPoly(ring, [px_out.reshape(-1, 1, 2)], 1)
    cv2.fillPoly(ring, [px_in.reshape(-1, 1, 2)], 0)
    rect = dict(u0=float(uv_in[:, 0].min()), u1=float(uv_in[:, 0].max()), v0=float(uv_in[:, 1].min()), v1=float(uv_in[:, 1].max()),
                corners_uv_mm=uv_in.tolist(), T_base_box=T_base_box.tolist(), marker_flags=det["flags"])   # u0..v1 = 안쪽 다각형의 축 정렬 범위(벽 틈 계산용)
    return ring > 0, rect, "marker"


def walls_layer(frame, plane: dict, shape, card, obj: np.ndarray) -> Tuple[np.ndarray, str, dict]:
    """반환 (wall bool HxW, walls_source, meta{rect, lip_mm, inner_mm})."""
    wl = (card.scene or {}).get("walls") or {}
    empty = np.zeros(shape, bool)
    if not wl:
        return empty, "none", {}
    res = float(plane["res_mm"])
    meta = dict(inner_mm=list(wl["inner_mm"]), lip_mm=float(wl.get("lip_mm", 0.0)), origin=wl.get("origin", "hull"))
    if wl.get("origin") == "marker":
        wall, rect, why = walls_from_marker(shape, frame, plane, res, wl)
        if wall is not None:
            meta["rect"] = rect
            return wall, "card.scene.walls/marker", meta
        _warn_once("walls_marker", f"벽 마커 실패({why}) → hull 로 폴백")
        meta["marker_fallback"] = why
    try:
        wall, rect = walls_from_hull(shape, obj, res, wl)
    except ValueError as e:
        _warn_once("walls_hull", f"벽 hull 실패({e}) → 빈 층")
        return empty, "none", meta
    _warn_once("walls_hull_accuracy", "벽 위치를 인스턴스 외곽(hull)에서 역산 — 맨 위 단만 보이면 경사 위 여유가 과대/과소일 수 있다. 정확한 위치는 박스 마커(origin=marker)")
    meta["rect"] = rect
    return wall, "card.scene.walls/hull", meta
