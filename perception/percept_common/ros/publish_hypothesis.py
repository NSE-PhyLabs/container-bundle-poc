#!/usr/bin/env python
"""ROS2 Humble 발행 노드 (Step 3). percept_common/env(sam6d 복제)에서 rclpy 가 직접 도므로 ZMQ 브리지가 없다.

발행 (configs/default.yaml ros.topics)
  /object_hypotheses  std_msgs/String        Hypotheses.to_json()  — 전체 계약
  /object_poses       geometry_msgs/PoseArray  유효 물체만, header.frame_id = base_link(또는 camera)
  /affordances        std_msgs/String        Affordances.to_json() — gripper·task 를 줬을 때만
  /container_pose     geometry_msgs/PoseStamped  기존 호환: 첫 유효 물체
  /container_poses    geometry_msgs/PoseArray    기존 호환: /object_poses 와 같은 내용
구독
  /percept/contact_feedback  std_msgs/String  ContactFeedback JSON (docs/contact_feedback.md) → Pipeline.feedback()

실행: ros/launch_percept.sh 또는
  python ros/publish_hypothesis.py --ros-args -p card:=objects/ramen_bundle -p adapter:=replay -p rate_hz:=5.0
확인: ros2 topic echo /object_hypotheses --once / ros2 topic hz /object_poses
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import percept  # noqa: F401,E402  ← rclpy 보다 먼저! percept._bootstrap 이 ssl·torch 를 적재해야 시스템 libcrypto 충돌을 피한다 (TRAPS.md)
from percept.camera.replay import ReplayAdapter            # noqa: E402
from percept.camera.stereo_head import StereoHeadAdapter    # noqa: E402
from percept.library import Library                        # noqa: E402
from percept.pipeline import build_default                 # noqa: E402
from percept.registry import PKG_ROOT, load_card, load_config, load_gripper, load_task   # noqa: E402
from percept.types import ContactFeedback                  # noqa: E402

import rclpy  # noqa: E402  ← 반드시 마지막
from geometry_msgs.msg import Pose, PoseArray, PoseStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402


def to_pose(o) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in o.position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in o.orientation)
    return p


class PerceptPublisher(Node):
    def __init__(self):
        super().__init__("percept_publisher")
        for name, default in (("card", "objects/ramen_bundle"), ("gripper", ""), ("task", ""), ("adapter", "replay"),
                              ("source", "dir"), ("dirs", [""]), ("config", ""), ("library", ""),
                              ("rate_hz", 0.0), ("max_frames", 0), ("record", True)):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value          # noqa: E731
        self.cfg = load_config(g("config") or None)
        rcfg, topics = self.cfg["ros"], self.cfg["ros"]["topics"]
        card = load_card(self._resolve(g("card")), self.cfg)
        gripper = load_gripper(self._resolve(g("gripper")), self.cfg["scene"]["res_mm"]) if g("gripper") else None
        task = load_task(self._resolve(g("task"))) if g("task") else None
        dirs = [d for d in (g("dirs") or []) if d]
        if g("adapter") == "replay":
            adapter = ReplayAdapter(self.cfg, dirs or None)
        else:
            adapter = StereoHeadAdapter(self.cfg, source=g("source"), items=dirs)
        lib = Library(g("library")) if g("library") else None
        self.pipe = build_default(self.cfg, card, adapter, gripper, task, continuous=True, record=bool(g("record")))
        if lib is not None:
            self.pipe.library = lib
        self.max_frames, self.n, self.done = int(g("max_frames")), 0, False

        qos = QoSProfile(depth=int(rcfg.get("qos_depth", 1)))
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub_hyp = self.create_publisher(String, topics["hypotheses"], qos)
        self.pub_poses = self.create_publisher(PoseArray, topics["poses"], qos)
        self.pub_aff = self.create_publisher(String, topics["affordances"], qos)
        self.pub_legacy_one = self.create_publisher(PoseStamped, topics["legacy_pose"], qos)
        self.pub_legacy_all = self.create_publisher(PoseArray, topics["legacy_poses"], qos)
        self.sub_fb = self.create_subscription(String, topics["feedback"], self.on_feedback, qos)

        self.backoff0, self.backoff_max = (float(v) for v in rcfg.get("camera_backoff_s", [0.5, 10.0]))
        self.backoff, self.next_t = self.backoff0, 0.0
        self.period = 1.0 / float(g("rate_hz") or rcfg["rate_hz"])
        self.timer = self.create_timer(self.period, self.tick)
        self.get_logger().info(
            f"card={card.class_name} gripper={(gripper or {}).get('name')} task={(task or {}).get('name')} "
            f"adapter={g('adapter')} rate={1.0 / self.period:.1f}Hz → {topics['hypotheses']}, {topics['poses']}, "
            f"{topics['affordances']}, {topics['legacy_pose']}, {topics['legacy_poses']} / 구독 {topics['feedback']}")

    @staticmethod
    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PKG_ROOT / p

    def on_feedback(self, msg: String) -> None:
        """FSM → 인지. 실패해도 노드를 죽이지 않는다(외부 입력)."""
        try:
            cb = ContactFeedback.from_dict(json.loads(msg.data))
        except Exception as e:                                  # noqa: BLE001  외부 JSON
            self.get_logger().warning(f"contact_feedback 파싱 실패: {type(e).__name__}: {e}")
            return
        self.pipe.feedback(cb)
        self.get_logger().info(f"feedback scene={cb.scene_id} cand={cb.candidate_id} stage={cb.stage} outcome={cb.outcome}")

    def tick(self) -> None:
        if self.done or time.time() < self.next_t:
            return
        try:
            hyps, affs = self.pipe.step()
        except StopIteration:
            self.get_logger().info(f"프레임 끝 ({self.n} 장 발행)")
            self.done = True
            return
        except IOError as e:
            self.get_logger().warning(f"프레임 건너뜀: {e}")
            return
        if hyps.header.get("reason") == "camera_unavailable":   # 재시도 간격을 2배씩 (상한 backoff_max)
            self.get_logger().warning(f"camera_unavailable — {self.backoff:.1f}s 후 재시도")
            self.next_t = time.time() + self.backoff
            self.backoff = min(self.backoff * 2, self.backoff_max)
            return
        self.backoff = self.backoff0
        self.publish(hyps, affs)
        self.n += 1
        if self.max_frames and self.n >= self.max_frames:
            self.get_logger().info(f"max_frames {self.max_frames} 도달")
            self.done = True

    def publish(self, hyps, affs) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = hyps.header.get("frame_id", "camera")
        self.pub_hyp.publish(String(data=hyps.to_json()))
        arr = PoseArray()
        arr.header.stamp, arr.header.frame_id = stamp, frame_id
        arr.poses = [to_pose(o) for o in hyps.valid]
        self.pub_poses.publish(arr)
        self.pub_legacy_all.publish(arr)                        # 기존 호환: 같은 내용
        if arr.poses:                                           # 기존 호환: 첫 유효 물체
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id, ps.pose = stamp, frame_id, arr.poses[0]
            self.pub_legacy_one.publish(ps)
        if affs is not None:
            self.pub_aff.publish(String(data=affs.to_json()))


def main() -> int:
    rclpy.init()
    node = PerceptPublisher()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipe.adapter.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
