"""ArUco 검출 공용 모듈 (hand-eye 솔버·박스 마커 벽 층이 함께 쓴다). OpenCV 4.13 tutorial_aruco_detection 원문 대조(9/21):
- ArucoDetector + 코너 정밀화(기본 apriltag). 튜토리얼은 "자세 추정이면 정밀화를 켜라"고만 하고 방법은 지정하지 않는다. APRILTAG 는 OpenCV 소스
  (aruco_detector.cpp "STEP 2.a Detect marker candidates :: using AprilTag")에서 **후보 검출 단계 자체**를 바꾸므로 흰 여백·두꺼운 테두리가 만드는 중첩 후보 함정을 피한다.
- objPoints = 튜토리얼 detect_markers.cpp 와 같은 **마커 중앙 원점** (-L/2, L/2, 0), (L/2, L/2, 0), (L/2, -L/2, 0), (-L/2, -L/2, 0) — 오프셋은 모두 마커 중앙 기준.
- solvePnP 는 SOLVEPNP_IPPE_SQUARE(마커 전용 특수 경우, 같은 코너 순서 요구). 튜토리얼 기본 플래그와 합성 정확도 동일(0.25/0.45 mm).
- 코너가 화면 가장자리 EDGE_PX 안이면 'edge', 마커 영역 라플라시안 분산 < BLUR_THR 이면 'blur' 경고 플래그.
"""
from pathlib import Path

import cv2
import numpy as np

from ..pose.base import make_T

ARUCO_DICTS = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
               "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_6X6_250": cv2.aruco.DICT_6X6_250}
CORNER_REFINE = {"apriltag": cv2.aruco.CORNER_REFINE_APRILTAG, "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
                 "contour": cv2.aruco.CORNER_REFINE_CONTOUR, "none": cv2.aruco.CORNER_REFINE_NONE}
EDGE_PX = 10.0                                                              # 검출 코너가 화면 가장자리 이 안이면 경고(잘림·왜곡 잔여 의심)
BLUR_THR = 30.0                                                             # 마커 영역 라플라시안 분산이 이 아래면 '흐림' 경고 [추측]


def marker_objp(marker_m: float) -> np.ndarray:
    """마커 중앙 원점, 한 변 marker_m, z = 0 (검출 코너 순서와 같음)."""
    h = marker_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def detect_marker(bgr, K, marker_m: float, marker_id, dict_name: str = "DICT_4X4_50", refine: str = "apriltag") -> dict:
    """정렬된 영상(왜곡 0)에서 마커 → dict(T=T_cam_marker|None, seen=[id...], corners(4x2)|None, flags=[...], all=(corners, ids), reason)."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = CORNER_REFINE[refine]
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids, _ = detector.detectMarkers(gray)
    out = dict(T=None, seen=None, corners=None, flags=[], all=(corners, ids), reason="")
    if ids is None or not len(ids):
        out["reason"] = "미검출"
        return out
    ids_l = [int(v) for v in ids.ravel()]
    out["seen"] = sorted(ids_l)
    k = 0 if marker_id is None else next((j for j, v in enumerate(ids_l) if v == marker_id), None)
    if k is None:
        out["reason"] = f"id {marker_id} 없음(보인 id {out['seen']})"
        return out
    c = corners[k].reshape(4, 2).astype(np.float32)
    out["corners"] = c
    H, W = gray.shape[:2]
    if (c[:, 0] < EDGE_PX).any() or (c[:, 0] > W - 1 - EDGE_PX).any() or (c[:, 1] < EDGE_PX).any() or (c[:, 1] > H - 1 - EDGE_PX).any():
        out["flags"].append("edge")                                         # 가장자리 10 px 안 — 잔차 계산에 쓰되 경고
    x0, y0 = np.floor(c.min(axis=0)).astype(int)
    x1, y1 = np.ceil(c.max(axis=0)).astype(int)
    roi = gray[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)]
    blur = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
    out["blur_var"] = blur
    if blur < BLUR_THR:
        out["flags"].append("blur")
    ok, rvec, tvec = cv2.solvePnP(marker_objp(marker_m), c, np.asarray(K, float), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        out["reason"] = "PnP 실패"
        return out
    out["T"] = make_T(cv2.Rodrigues(rvec)[0], tvec)
    out["rvec"], out["tvec"] = rvec, tvec
    return out


def save_debug_image(path, bgr, det: dict, K, marker_m: float, label: str) -> None:
    """검출 결과 그림: drawDetectedMarkers + 축(1.5·L, 튜토리얼과 같음) + 상태 문구. 실패 자세는 사유."""
    path = Path(path)
    vis = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    corners, ids = det.get("all", (None, None))
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    if det.get("T") is not None:
        cv2.drawFrameAxes(vis, np.asarray(K, float), np.zeros(5), det["rvec"], det["tvec"], marker_m * 1.5, 2)
    color = (0, 200, 0) if det.get("T") is not None and not det.get("flags") else ((0, 200, 255) if det.get("T") is not None else (0, 0, 255))
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
