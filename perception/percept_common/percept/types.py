"""파이프라인 단계 사이의 공통 데이터 타입 (v3 §5.1). 길이 단위는 전부 m, 각도 rad. 4x4 는 float64."""
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Frame:
    """카메라 어댑터가 내는 한 장. rgb 는 정렬(rectify)된 왼쪽 영상."""
    rgb: np.ndarray            # HxWx3 uint8, **BGR** (이름은 계약대로 rgb, 채널 순서는 OpenCV BGR. RGB 를 주는 어댑터는 변환, TRAPS.md)
    depth: np.ndarray          # HxW float32, 단위 m, 무효 = NaN
    K: np.ndarray              # 3x3 내부 파라미터 (정렬 후 왼쪽 P1 기준)
    T_base_cam: np.ndarray     # 4x4, P_base = T @ P_cam. 없으면 항등 + frame_id='camera'
    frame_id: str              # 'base_link' | 'camera'
    stamp: float               # time.time()
    source: str                # 'stereo_head' | 'replay' | 'synthetic' | 'realsense' | 'isaac'
    extra: dict = field(default_factory=dict)   # 'right' 우측 영상, 'disparity'(px), 'gt'(synthetic), 'scene_id', 'baseline_mm', 'path', 'image_key'


@dataclass(eq=False)              # mask ndarray 의 == 비교를 막음 (fuse 가 id 로 구분)
class Candidate:
    """Finder 가 낸 2D 후보 하나 (= 인스턴스 하나)."""
    mask: np.ndarray           # HxW bool (정렬 왼쪽 영상)
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    score: float               # Finder 원점수 (SAM 예측 점수 등)
    finder: str                # 'color_rule' | 'ref_match' | 'sam3_text'
    instance_id: int = -1      # 프레임 안 순번 (영상 위→아래, 왼→오른)


@dataclass
class Instance:
    """lift 가 만든 3D 인스턴스 (base 좌표, m)."""
    candidate: Candidate
    points_base: np.ndarray    # Nx3 float64 (이상치 제거 뒤). 점 부족이면 (0,3)
    colors: np.ndarray         # Nx3 uint8 BGR
    bbox3d: Tuple[np.ndarray, np.ndarray]   # (min3, max3) m
    n_valid: int               # 마스크∧depth 유효 픽셀 수 (이상치 제거 전)


@dataclass
class PoseResult:
    """PoseBackend 결과."""
    T_base_obj: np.ndarray     # 4x4, m
    score_raw: float           # 부품 원점수 (geom_fit: 잔차 RMS mm, mesh_pose: 점수)
    backend: str               # 'geom_fit' | 'mesh_pose' | 'ref_pose' | 'geom_fit(fallback)' | 'mesh_pose+geom_fit'
    geometry_fit: dict         # 실제 맞춘 값 (m + _mm 병기)
    partial: bool = False      # box: 측면 없이 요만 잡음 등


@dataclass
class ContactFeedback:
    """FSM → 인지 되먹임 (접촉 실측). JSON 왕복은 to_dict/from_dict."""
    scene_id: str
    candidate_id: str
    stage: str                 # 'descend' | 'insert' | 'lift' | 'extract' | 'place'
    measured_clearance_mm: Optional[dict] = None   # {'uphill':..,'below':..,'side_l':..,'side_r':..}
    reached_depth_mm: Optional[float] = None
    peak_force_N: Optional[float] = None
    duration_s: Optional[float] = None
    outcome: str = "unknown"   # 'success' | 'unproductive' | 'drop' | 'damage' | 'abort' | 'unknown'
    cycle_time_s: Optional[float] = None
    note: str = ""

    STAGES = ("descend", "insert", "lift", "extract", "place")
    OUTCOMES = ("success", "unproductive", "drop", "damage", "abort", "unknown")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContactFeedback":
        cb = cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})
        if cb.stage not in cls.STAGES or cb.outcome not in cls.OUTCOMES:
            raise ValueError(f"ContactFeedback: stage={cb.stage!r} outcome={cb.outcome!r}")
        return cb
