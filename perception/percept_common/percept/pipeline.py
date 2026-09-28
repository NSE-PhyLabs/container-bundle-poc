"""Pipeline (v3 §7.10): grab → find(+fuse) → lift → estimate(+refine) → scene(④, Step 4) → affordance(⑤, Step 5) → gate → library.record.
step() → (Hypotheses, Affordances|None). 단계별 timing_ms{find,lift,pose,scene,affordance,gate,total}.
어느 단계에서 실패해도 예외로 죽지 않고 빈 결과 + 사유 코드(errors.FAILURE_CODES): camera_unavailable / no_target / finder_unavailable /
depth_invalid / fit_failed / backend_unavailable / scene_failed / affordance_failed / gate_rejected."""
import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .contract import Affordances, Hypotheses
from .errors import AffordanceError, FAILURE_CODES, FitFailed, NotAvailable, SceneError
from .find.base import Finder
from .find.fuse import fuse
from .gate import StabilityTracker, gate
from .library import Library
from .lift import lift
from .pose.base import PoseBackend, Refiner
from .registry import Card, PKG_ROOT
from .types import ContactFeedback, Frame

log = logging.getLogger(__name__)
TIMING = ("find", "lift", "pose", "scene", "affordance", "gate", "total")


class Pipeline:
    def __init__(self, cfg: dict, card: Card, adapter, gripper: Optional[dict] = None, task: Optional[dict] = None,
                 finders: Optional[Sequence[Finder]] = None, backend: Optional[PoseBackend] = None, refiner: Optional[Refiner] = None,
                 library: Optional[Library] = None, feedback_store=None, continuous: bool = False, record: bool = True,
                 container_id: Optional[str] = None):
        """record=False 면 library 기록 생략 (회귀·테스트·run_once 가 경험 기록을 오염시키지 않게). gripper·task 둘 다 있어야 ⑤ 실행."""
        self.cfg, self.card, self.adapter = cfg, card, adapter
        self.gripper, self.task = gripper, task
        self.finders, self.backend, self.refiner = list(finders or []), backend, refiner
        self.library = (library if library is not None else Library(PKG_ROOT / cfg["paths"]["library"])) if record else None
        self.feedback_store = feedback_store
        self.container_id = container_id        # 되먹임 bias 를 묶는 지속 단위(박스·팔레트 id). None 이면 전체 평균
        self.tracker = StabilityTracker() if continuous else None
        self._fallback = None
        self.n_steps = 0
        self.last = {}                      # 디버그: frame, cands, insts, poses, reasons, scene

    def _fallback_backend(self) -> PoseBackend:
        if self._fallback is None:
            from .pose.geom_fit import GeomFit
            self._fallback = GeomFit(self.cfg)
        return self._fallback

    def step(self) -> Tuple[Hypotheses, Optional[Affordances]]:
        """다음 프레임 1장 처리. 프레임이 없으면 StopIteration."""
        t0 = time.perf_counter()
        timing = {k: 0.0 for k in TIMING}
        try:
            frame: Frame = self.adapter.grab()
        except NotAvailable as e:
            log.error("camera_unavailable: %s", e)
            self.last = {}
            h = Hypotheses.empty(dict(stamp=time.time(), source=getattr(self.adapter, "name", "?"), frame_id="camera", scene_id="",
                                      timing_ms=timing), "camera_unavailable")
            return h, self._aff_placeholder(h.header, "camera_unavailable")
        self.n_steps += 1
        frame.extra.setdefault("image_key", f"{frame.source}:{id(self)}:{self.n_steps}:{frame.stamp}")   # 프로세스 안에서 유일 (SAM set_image 캐시 키)
        scene_id = frame.extra.get("scene_id") or f"{frame.source}/{self.n_steps}"
        header = dict(stamp=frame.stamp, source=frame.source, frame_id=frame.frame_id, scene_id=scene_id)
        frame_ref = frame.extra.get("path") or scene_id

        def finish(hyps: Hypotheses, reason: Optional[str] = None, scene=None, affs=None) -> Tuple[Hypotheses, Optional[Affordances]]:
            timing["total"] = (time.perf_counter() - t0) * 1000
            hyps.header.update(header)
            hyps.header["timing_ms"] = {k: round(v, 2) for k, v in timing.items()}
            if reason:
                assert reason in FAILURE_CODES, reason
                hyps.header["reason"] = reason
            if scene is not None:
                hyps.header["free_space"] = scene.free_space
            if affs is None:
                affs = self._aff_placeholder(hyps.header, "no_scene" if scene is None else "affordance_failed")
            elif affs.header.get("stamp") is None:
                affs.header["stamp"] = hyps.header.get("stamp")
            if affs is not None and scene is not None:
                affs.free_space = scene.free_space
            if self.library is not None:
                self.library.record(hyps, affs, self.card.class_name, self.gripper, self.task, frame_ref)
            return hyps, affs

        if not np.isfinite(frame.depth).any():
            self.last = dict(frame=frame, cands=[], insts=[], poses=[], reasons=[], scene=None)
            return finish(Hypotheses.empty({}, "depth_invalid"), "depth_invalid")

        t = time.perf_counter()
        try:
            groups = [f.find(frame, self.card) for f in self.finders]
        except NotAvailable as e:                       # 예: MobileSAM 체크포인트·DINOv2 없음 → 찾기 부품 불가
            log.error("finder_unavailable: %s", e)
            self.last = dict(frame=frame, cands=[], insts=[], poses=[], reasons=[], scene=None)
            return finish(Hypotheses.empty({}, "finder_unavailable"), "finder_unavailable")
        fz = self.cfg["fuse"]
        cands = fuse(groups, float(fz["iou_between_finders"]), bool(fz.get("prefer_split", False)))   # size_ok(카드 치수 검사)는 Step 5.5
        for k, c in enumerate(cands):
            c.instance_id = k
        timing["find"] = (time.perf_counter() - t) * 1000
        if not cands:
            self.last = dict(frame=frame, cands=[], insts=[], poses=[], reasons=[], scene=None)
            return finish(Hypotheses.empty({}, "no_target"), "no_target")

        t = time.perf_counter()
        insts = lift(frame, cands, self.card, self.cfg)
        timing["lift"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        poses, reasons, refined = [], [], []
        fallback_used = False
        for inst in insts:
            p, why, r = None, "", False
            if self.card.reduced:
                why = "degraded_mode"
            elif len(inst.points_base) == 0:
                why = "too_few_points"
            else:
                try:
                    p = self.backend.estimate(inst, self.card, frame)
                except NotAvailable as e:
                    log.warning("backend %s 사용 불가(%s) → geom_fit fallback", self.backend.name, e)
                    fallback_used = True
                    try:
                        p = self._fallback_backend().estimate(inst, self.card, frame)
                        p.backend = "geom_fit(fallback)"
                    except (NotAvailable, FitFailed) as e2:
                        why = f"backend_unavailable: {e2}"
                except FitFailed as e:
                    why = str(e) or "fit_failed"
            if p is not None and self.refiner is not None:
                try:
                    p, r = self.refiner.refine(inst, self.card, frame, p.T_base_obj), True
                except (NotAvailable, FitFailed) as e:
                    log.warning("refine 실패(%s) → 초기값 유지", e)
            poses.append(p); reasons.append(why); refined.append(r)
        timing["pose"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()                                 # gate 는 한 번만 부른다(두 번 부르면 기준면을 맞춘 집합과 발행 집합이 달라진다)
        hyps = gate(cands, insts, poses, reasons, self.card, self.cfg, frame, refined, self.tracker, None, self.task)
        timing["gate"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        scene, scene_reason = None, None                        # ④ 기준면 → 정사영 → 여유 공간
        if (self.card.scene or {}).get("reference_plane", {}).get("type", "none") != "none":
            try:
                from .scene.multimask import build_scene
                # bias 묶음 단위는 프레임이 아니라 지속 단위(박스·팔레트). container_id 가 없으면 전체 평균(None)을 쓴다
                bias = self.feedback_store.bias(self.container_id, "uphill") if self.feedback_store is not None else None
                scene = build_scene(frame, insts, hyps, poses, self.card, self.cfg, self.task, self.gripper, bias)
                for o in hyps.items:                            # scene 의존 extras 를 뒤늦게 채움
                    if o.pose_valid and "budget_uphill" in self.card.extras.get("compute", []):
                        o.extras["budget_uphill"] = dict(direct_mm=scene.free_space.get("direct_mm"),
                                                         manipulable_rigid_mm=scene.free_space.get("manipulable_rigid_mm"),
                                                         manipulable_measured_mm=scene.free_space.get("manipulable_measured_mm"))
            except (SceneError, ValueError) as e:
                scene_reason = "scene_failed"
                log.warning("scene_failed: %s", e)
        timing["scene"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()                                 # ⑤ gripper·task 둘 다 있을 때 (실패 → affordance_failed)
        affs = None
        if self.gripper is not None and self.task is not None and scene is not None:
            try:
                from .affordance.compute import compute
                affs = compute(scene, hyps, self.card, self.gripper, self.task, frame, self.feedback_store, self.container_id)
            except (AffordanceError, ValueError, KeyError, IndexError) as e:
                log.warning("affordance_failed: %s", e)
                affs = self._aff_placeholder(hyps.header, "affordance_failed")
        timing["affordance"] = (time.perf_counter() - t) * 1000
        self.last = dict(frame=frame, cands=cands, insts=insts, poses=poses, reasons=reasons, scene=scene, affs=affs)
        if fallback_used:
            hyps.header["backend_fallback"] = True
        reason = None
        if not self.card.reduced:
            if all(p is None for p in poses):
                reason = "backend_unavailable" if fallback_used else "fit_failed"
            elif not any(o.pose_valid for o in hyps.items):
                reason = "gate_rejected"
        if reason is None and scene_reason:
            reason = scene_reason
        return finish(hyps, reason, scene, affs)

    def _aff_placeholder(self, header: dict, reason: str) -> Optional[Affordances]:
        """⑤ 가 켜져 있으면(gripper·task 둘 다) 빈 Affordances(invalid_reason), 아니면 None."""
        if self.gripper is None or self.task is None:
            return None
        h = dict(stamp=header.get("stamp"), frame_id=header.get("frame_id"), scene_id=header.get("scene_id"),
                 gripper=self.gripper.get("name"), task=self.task.get("name"), source=header.get("source"))
        return Affordances.empty(h, reason)

    def feedback(self, cb: ContactFeedback) -> None:
        """FSM 되먹임: feedback_store.add + library.join (Step 4 에서 feedback.py 구현). 저장소가 없으면 library 만."""
        if self.feedback_store is not None:
            self.feedback_store.add(cb)
        if self.library is not None:
            self.library.join(cb)

    def run(self, n: Optional[int] = None) -> List[Tuple[Hypotheses, Optional[Affordances]]]:
        """프레임이 끝날 때까지(또는 n 장). 읽기 실패(IOError)는 경고 후 건너뜀, camera_unavailable 이면 중단."""
        out = []
        while n is None or len(out) < n:
            try:
                res = self.step()
            except StopIteration:
                break
            except IOError as e:
                log.warning("프레임 건너뜀: %s", e)
                continue
            out.append(res)
            if res[0].header.get("reason") == "camera_unavailable":
                break
        return out


def build_default(cfg: dict, card: Card, adapter, gripper: Optional[dict] = None, task: Optional[dict] = None,
                  continuous: bool = False, predictor=None, record: bool = True, feedback_store=None) -> Pipeline:
    """카드의 find.methods / pose.backend / pose.refine 대로 부품을 골라 Pipeline 구성 (선택 규칙은 README §8)."""
    from .find.color_rule import ColorRuleFinder
    from .pose.geom_fit import GeomFit
    finders: List[Finder] = []
    for m in card.find["methods"]:
        if m == "color_rule":
            finders.append(ColorRuleFinder(cfg, predictor))
        elif m == "ref_match":
            from .find.ref_match import RefMatchFinder      # Step 5.5
            finders.append(RefMatchFinder(cfg))
        elif m == "sam3_text":
            from .find.sam3_text import Sam3TextFinder      # 승인 시
            finders.append(Sam3TextFinder(cfg))
    be = card.pose["backend"]
    backend: Optional[PoseBackend]
    if be == "geom_fit" or be == "none":
        backend = GeomFit(cfg)
    elif be == "mesh_pose":
        from .pose.mesh_pose import MeshPoseClient          # Step 6
        backend = MeshPoseClient(cfg)
    else:
        from .pose.ref_pose import RefPose                  # Step 6 스텁
        backend = RefPose(cfg)
    refiner = None
    if card.pose.get("refine") == "geom_fit" and be != "geom_fit":
        from .pose.refine import GeomRefiner                # Step 6
        refiner = GeomRefiner(cfg)
    return Pipeline(cfg, card, adapter, gripper, task, finders, backend, refiner, continuous=continuous, record=record, feedback_store=feedback_store)
