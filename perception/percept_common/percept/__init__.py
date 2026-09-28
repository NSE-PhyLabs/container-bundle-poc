"""percept — 공통 인지 패키지 (ROS·Isaac import 금지. numpy/opencv/torch 만).

grab(camera) → find → lift → estimate(pose) → refine → gate → contract(ObjectHypothesis). 진입점은 percept.pipeline.Pipeline.
"""
from . import _bootstrap  # noqa: F401  ← 반드시 첫 줄 (ssl·torch 를 rclpy 보다 먼저, TRAPS.md)

__version__ = "0.1.0"
