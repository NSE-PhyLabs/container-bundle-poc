"""실물 머리 스테레오 카메라(teleimager 좌우 붙임 1280x480) — 기존 코드 함수 단위 이식 (v3 §7.3).

detect_container.py:34-56 / estimate_pose.py:34-60 (load_calib, make_matcher, rectify_pair), make_depth.py:64-73 (시차),
capture_object.py:34-46, 69-74 (ZMQ SUB 수신·디코딩). 상수는 configs/camera_stereo_head.yaml.
소스: 'zmq' = 로봇 teleimager PUB 수신(하드웨어 필요), 'dir' = 폴더의 PNG 순회(replay 와 같은 처리, source 이름만 'stereo_head').
T_base_cam: calib/T_base_cam.npz {T 4x4 (P_base = T·P_cam, 병진 m), frame_id, date[, unit]} 가 있으면 frame_id 그대로,
없으면 항등 + frame_id 'camera' + 경고 1회 (hand-eye 는 하드웨어 작업, README §5).
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from ..errors import NotAvailable
from ..lift import depth_m_from_disparity
from ..registry import PKG_ROOT, load_config
from ..types import Frame
from .base import CameraAdapter

log = logging.getLogger(__name__)
_WARNED = set()
JPEG_SOI = b"\xff\xd8\xff"


@dataclass
class StereoCalib:
    maps: tuple                 # (mLx, mLy, mRx, mRy)
    K: np.ndarray               # 3x3 정렬 후 왼쪽 (P1 기준)
    baseline_mm: float
    image_size: Tuple[int, int]   # (가로, 세로) 한쪽 영상


def load_camera_config(path=None) -> dict:
    return load_config(Path(path) if path else PKG_ROOT / "configs" / "camera_stereo_head.yaml")


def load_calib(npz_path, cam_cfg: Optional[dict] = None) -> StereoCalib:
    """npz → 정렬맵 + 내부 파라미터(P1). cam_cfg 가 있으면 yaml 값과 대조해 tolerance 넘게 다르면 경고."""
    c = np.load(npz_path)
    size = tuple(int(x) for x in c["image_size"])
    mLx, mLy = cv2.initUndistortRectifyMap(c["KL"], c["DL"], c["R1"], c["P1"], size, cv2.CV_32FC1)
    mRx, mRy = cv2.initUndistortRectifyMap(c["KR"], c["DR"], c["R2"], c["P2"], size, cv2.CV_32FC1)
    P1 = c["P1"]
    K = np.array([[float(P1[0, 0]), 0.0, float(P1[0, 2])], [0.0, float(P1[1, 1]), float(P1[1, 2])], [0.0, 0.0, 1.0]])
    calib = StereoCalib((mLx, mLy, mRx, mRy), K, float(c["baseline_mm"]), size)
    if cam_cfg:
        got = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], baseline_mm=calib.baseline_mm)
        exp = dict(cam_cfg["rectified_left"], baseline_mm=cam_cfg["baseline_mm"])
        tol = cam_cfg.get("tolerance", {})
        for k in exp:
            if abs(got[k] - exp[k]) > tol.get(k, 0.01):
                log.warning("calib npz %s=%.4f 가 %s 의 %s 와 다름", k, got[k], cam_cfg.get("_path"), exp[k])
        if list(size) != list(cam_cfg["image_size"]):
            log.warning("calib image_size %s != yaml %s", size, cam_cfg["image_size"])
    return calib


def make_matcher(sgbm: dict):
    """StereoSGBM (원본 값: bs 7, numDisp 160, P1/P2 = 8·32×3×bs², 3WAY)."""
    bs = int(sgbm["block_size"])
    mode = {"SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY, "SGBM": cv2.STEREO_SGBM_MODE_SGBM, "HH": cv2.STEREO_SGBM_MODE_HH}[sgbm["mode"]]
    return cv2.StereoSGBM_create(
        minDisparity=int(sgbm["min_disparity"]), numDisparities=int(sgbm["num_disparities"]), blockSize=bs,
        P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs,
        disp12MaxDiff=int(sgbm["disp12_max_diff"]), uniquenessRatio=int(sgbm["uniqueness_ratio"]),
        speckleWindowSize=int(sgbm["speckle_window_size"]), speckleRange=int(sgbm["speckle_range"]),
        preFilterCap=int(sgbm["pre_filter_cap"]), mode=mode)


def rectify_pair(frame_bgr: np.ndarray, calib: StereoCalib):
    """좌우 붙은 프레임 → (L, R) 정렬 영상. 폭이 2×image_size[0] 이 아니면 오류 (remap 은 크기가 달라도 조용히 틀리므로 검사)."""
    half = frame_bgr.shape[1] // 2
    if half != calib.image_size[0] or frame_bgr.shape[0] != calib.image_size[1]:
        raise ValueError(f"프레임 {frame_bgr.shape[1]}x{frame_bgr.shape[0]} 이 캘리브 한쪽 크기 {calib.image_size} 의 좌우 붙임이 아님")
    m = calib.maps
    L = cv2.remap(frame_bgr[:, :half], m[0], m[1], cv2.INTER_LINEAR)
    R = cv2.remap(frame_bgr[:, half:half * 2], m[2], m[3], cv2.INTER_LINEAR)
    return L, R


def disparity(L: np.ndarray, R: np.ndarray, matcher, scale: float = 16.0) -> np.ndarray:
    """SGBM int16 고정소수 → float32 px (무효 = -1.0)."""
    return matcher.compute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)).astype(np.float32) / np.float32(scale)


def load_T_base_cam(path, camera_frame_id: str, base_frame_id: str, tol: float = 1e-3):
    """calib/T_base_cam.npz → (T 4x4, frame_id). 없으면 항등 + camera_frame_id + 경고(1회).
    npz 계약: T (4x4, P_base = T @ P_cam, 병진 **m**), frame_id (str, 예 'base_link'), date (str), unit ('m', 옵션).
    검사: 마지막 행 [0,0,0,1], R 직교성·det = 1 ± tol, |t| < 10 m (mm 로 저장한 실수 방지). 어긋나면 ValueError."""
    p = Path(path)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "T" not in z.files:
            raise ValueError(f"{p}: 키 'T' 없음 (키: {z.files})")
        T = np.asarray(z["T"], dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
            raise ValueError(f"{p}: T 는 4x4 동차 행렬이어야 함 (마지막 행 [0,0,0,1])")
        R, t = T[:3, :3], T[:3, 3]
        if not np.allclose(R @ R.T, np.eye(3), atol=tol) or abs(np.linalg.det(R) - 1.0) > tol:
            raise ValueError(f"{p}: R 이 회전행렬이 아님 (직교성·det=1±{tol} 검사 실패)")
        if "unit" in z.files and str(z["unit"]) != "m":
            raise ValueError(f"{p}: unit={z['unit']!r}, 'm' 이어야 함")
        if np.linalg.norm(t) > 10.0:
            raise ValueError(f"{p}: |t|={np.linalg.norm(t):.1f} — m 가 아니라 mm 로 저장된 듯")
        fid = str(z["frame_id"]) if "frame_id" in z.files else base_frame_id
        log.info("T_base_cam 로드 %s (frame_id=%s, date=%s)", p, fid, z["date"] if "date" in z.files else "?")
        return T, fid
    if str(p) not in _WARNED:
        _WARNED.add(str(p))
        log.warning("T_base_cam 없음 (%s) → 항등 변환, frame_id='%s' (hand-eye 캘리브 필요, README §5)", p, camera_frame_id)
    return np.eye(4), camera_frame_id


def decode_frame(buf: bytes, raw_shapes: Iterable[Tuple[int, int]]) -> Optional[np.ndarray]:
    """capture_object.py recv_frame(:34-46) 이식: JPEG(SOI 탐색, 앞 헤더 허용) → 실패 시 raw BGR 후보 크기(+8B 헤더) 순서대로 → None."""
    soi = buf.find(JPEG_SOI)
    if soi >= 0:
        img = cv2.imdecode(np.frombuffer(buf[soi:], np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    for h, w in raw_shapes:
        n = h * w * 3
        if len(buf) == n:
            return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        if len(buf) == n + 8:
            return np.frombuffer(buf[8:], np.uint8).reshape(h, w, 3).copy()
    return None


class ZmqStereoSource:
    """teleimager PUB 수신 (capture_object.py:69-74): SUB, SUBSCRIBE b'', CONFLATE 1(최신 1장), RCVTIMEO. 타임아웃·디코딩 실패 → NotAvailable."""

    def __init__(self, host: str, port: int, recv_timeout_ms: int, raw_shapes, conflate: bool = True):
        import zmq
        self.raw_shapes = [tuple(s) for s in raw_shapes]
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        if conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVTIMEO, int(recv_timeout_ms))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://{host}:{port}"
        self.sock.connect(self.endpoint)

    def recv(self) -> np.ndarray:
        import zmq
        try:
            buf = self.sock.recv()
        except zmq.Again:
            raise NotAvailable(f"카메라 프레임 없음 ({self.endpoint}, 타임아웃)")
        img = decode_frame(buf, self.raw_shapes)
        if img is None:
            raise NotAvailable(f"프레임 디코딩 실패 ({len(buf)} B; JPEG 도 raw {self.raw_shapes} 도 아님)")
        return img

    def close(self):
        self.sock.close(0)


class StereoProcessor:
    """정렬·시차·깊이 계산 묶음 (replay 와 live 가 공유)."""

    def __init__(self, cfg: dict, cam_cfg: Optional[dict] = None):
        self.cfg = cfg
        self.cam = cam_cfg or load_camera_config()
        self.calib = load_calib(cfg["paths"]["calib_npz"], self.cam)
        self.matcher = make_matcher(self.cam["sgbm"])
        self.T_base_cam, self.frame_id = load_T_base_cam(cfg["paths"]["T_base_cam_npz"], cfg["frames"]["camera"], cfg["frames"]["base"])

    def build_frame(self, frame_bgr: np.ndarray, source: str, stamp: Optional[float] = None, extra: Optional[dict] = None) -> Frame:
        L, R = rectify_pair(frame_bgr, self.calib)
        disp = disparity(L, R, self.matcher, float(self.cam["disparity_scale"]))
        depth = depth_m_from_disparity(disp, self.calib.K[0, 0], self.calib.baseline_mm, float(self.cam["invalid_disparity_max"]))
        ex = dict(right=R, disparity=disp, baseline_mm=self.calib.baseline_mm, camera_frame_id=self.cam.get("frame_id_camera"))
        ex.update(extra or {})
        return Frame(rgb=L, depth=depth, K=self.calib.K.copy(), T_base_cam=self.T_base_cam.copy(), frame_id=self.frame_id,
                     stamp=time.time() if stamp is None else stamp, source=source, extra=ex)


class StereoHeadAdapter(CameraAdapter):
    """실물 머리 카메라 어댑터. source='zmq'(하드웨어 필요) 또는 'dir'(PNG 폴더/목록, 인터페이스 검증용)."""
    name = "stereo_head"

    def __init__(self, cfg: dict, cam_cfg: Optional[dict] = None, source: str = "zmq", items: Optional[Iterable] = None):
        self.proc = StereoProcessor(cfg, cam_cfg)
        self.source_kind = source
        self.n = 0
        if source == "zmq":
            g = self.proc.cam["grab"]
            self.src = ZmqStereoSource(g["zmq_host"], int(g["zmq_port"]), int(g["recv_timeout_ms"]), g["raw_shapes"], bool(g.get("conflate", True)))
        elif source == "dir":
            self.items = []
            for it in (items or []):
                p = Path(it)
                self.items += sorted(p.glob("*.png")) if p.is_dir() else [p]
            if not self.items:
                raise NotAvailable("source=dir 인데 PNG 가 없음")
        else:
            raise ValueError(f"source 는 zmq | dir, 지금 {source!r}")

    def grab(self) -> Frame:
        if self.source_kind == "zmq":
            bgr = self.src.recv()                                   # NotAvailable → pipeline 이 camera_unavailable
            path = None
        else:
            if self.n >= len(self.items):
                raise StopIteration
            path = self.items[self.n]
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise IOError(f"읽기 실패: {path}")
        self.n += 1
        extra = dict(scene_id=f"live/{self.n:06d}", image_key=f"stereo_head:{id(self)}:{self.n}")
        if path is not None:
            extra.update(path=str(path), stem=path.stem)
        return self.proc.build_frame(bgr, self.name, extra=extra)

    def close(self) -> None:
        if self.source_kind == "zmq":
            self.src.close()
