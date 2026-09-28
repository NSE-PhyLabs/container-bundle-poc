"""PoseBackend / Refiner 추상 + 자세 행렬 도우미 (publish_container_pose.py axis_to_quat 이식)."""
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..registry import Card
from ..types import Frame, Instance, PoseResult


class PoseBackend(ABC):
    name: str = "backend"

    @abstractmethod
    def estimate(self, inst: Instance, card: Card, frame: Frame, init_T: Optional[np.ndarray] = None) -> PoseResult:
        """실패는 FitFailed, 부품을 쓸 수 없으면 NotAvailable. init_T(4x4 base) 가 있으면 초기값으로 (refine 용).
        v3 시그니처 estimate(instance, card, init_T) 에 frame 을 더한 이유: 시선 방향·부호 규약이 카메라 자세(T_base_cam)를 필요로 함 (ASSUMPTIONS 20)."""


class Refiner(ABC):
    name: str = "refiner"

    @abstractmethod
    def refine(self, inst: Instance, card: Card, frame: Frame, T_init: np.ndarray) -> PoseResult:
        """B/C 결과를 초기값으로 국소 재실행."""


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def rotation_from_axis(axis, up=(0.0, -1.0, 0.0), degenerate=0.95) -> np.ndarray:
    """x축 = axis, 축 둘레 롤은 up 기준으로 고정 (publish_container_pose.py:49-58 그대로).
    up 은 axis 와 같은 좌표계(기본: 카메라 광학 -Y = 영상 위쪽). |x·up| > 0.95 면 up=(0,0,-1)."""
    x = unit(axis)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(x @ up)) > degenerate:
        up = np.array([0.0, 0.0, -1.0])
    z = unit(np.cross(x, up))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)          # 열 = 기저벡터


def quat_from_matrix(R) -> list:
    """3x3 → [qx, qy, qz, qw] (publish_container_pose.py:59-84 와 같은 분기)."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(qx), float(qy), float(qz), float(w)]


def matrix_from_quat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).ravel()
    return T


def invert_T(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, P) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def rot_axis(axis, deg) -> np.ndarray:
    """축 둘레 회전(도) → 3x3 (로드리게스)."""
    a = unit(axis)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rpy_to_R(rx, ry, rz) -> np.ndarray:
    """고정축 XYZ 회전(도) → 3x3 (R = Rz·Ry·Rx)."""
    return rot_axis([0, 0, 1], rz) @ rot_axis([0, 1, 0], ry) @ rot_axis([1, 0, 0], rx)


def rot_angle_deg(R) -> float:
    """회전행렬의 회전각(도)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(R, float)[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
