"""RealSense D435i 어댑터 (v3 §7.3). pyrealsense2 가 있으면 컬러에 정렬된 depth 를 Frame 으로, 없으면 NotAvailable.
하드웨어 없이는 인터페이스만 검증된다. T_base_cam 은 머리 스테레오와 다른 카메라이므로 별도 파일 `calib/T_base_cam_realsense.npz`(없으면 항등 + 'camera').
Step 5 실기 촬영 목표 거리 1.0~1.1 m (Q22)."""
import logging
import time
from typing import Optional

import numpy as np

from ..errors import NotAvailable
from ..types import Frame
from .base import CameraAdapter
from .stereo_head import load_T_base_cam

log = logging.getLogger(__name__)


class RealSenseAdapter(CameraAdapter):
    name = "realsense"

    def __init__(self, cfg: dict, width: int = 640, height: int = 480, fps: int = 30, T_base_cam_npz: Optional[str] = None,
                 warmup: int = 10):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise NotAvailable(f"pyrealsense2 없음 ({e}) — percept_common/env 에 설치해야 D435i 를 쓸 수 있다(승인 후)")
        self.rs = rs
        self.cfg = cfg
        self.pipe = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        conf.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipe.start(conf)
        except RuntimeError as e:
            raise NotAvailable(f"RealSense 시작 실패: {e}")
        self.align = rs.align(rs.stream.color)                                # depth → color 정렬
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        path = T_base_cam_npz or str(cfg["paths"].get("T_base_cam_realsense_npz", "calib/T_base_cam_realsense.npz"))
        self.T_base_cam, self.frame_id = load_T_base_cam(path, cfg["frames"]["camera"], cfg["frames"]["base"])
        for _ in range(int(warmup)):                                          # 자동 노출 안정
            self.pipe.wait_for_frames()
        self.i = 0

    def grab(self) -> Frame:
        frames = self.align.process(self.pipe.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            raise IOError("RealSense 프레임 없음")
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale     # m
        depth[depth <= 0] = np.nan
        rgb = np.asanyarray(c.get_data()).copy()
        self.i += 1
        return Frame(rgb=rgb, depth=depth, K=self.K.copy(), T_base_cam=self.T_base_cam, frame_id=self.frame_id, stamp=time.time(),
                     source=self.name, extra=dict(scene_id=f"realsense/{self.i:06d}", image_key=f"realsense:{self.i}"))

    def close(self) -> None:
        try:
            self.pipe.stop()
        except Exception:                                                    # noqa: BLE001
            pass
