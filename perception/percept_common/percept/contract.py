"""출력 계약 (v3 §5.2). to_json/from_json 왕복 무손실(NaN/inf 는 null), 표준 JSON(allow_nan=False). 스키마 검증 함수 validate_*.

계약 두 가지: Hypotheses(무엇이 어디에 어떤 자세로) · Affordances(이 그리퍼로 어디에 무엇을 할 수 있나). SceneMap 은 ④ 출력(npz 저장).
§5.2 외 추가: Hypotheses.header.reason(빈 결과 사유 코드, errors.FAILURE_CODES) · header.published · header.scene(장면 법선 등).
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .errors import FAILURE_CODES

RIGIDITY = ("rigid", "semi_rigid", "deformable")
GEOMETRY_TYPES = ("cylinder", "box", "mesh", "none")
ACTIONS = ("insert", "scoop", "place", "push")
TIMING_KEYS = ("find", "lift", "pose", "scene", "affordance", "gate", "total")


def jsonable(x: Any) -> Any:
    """numpy·tuple 을 JSON 가능한 파이썬 기본형으로 (재귀). NaN/inf → None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _dumps(d: dict, indent=None) -> str:
    return json.dumps(jsonable(d), ensure_ascii=False, indent=indent, allow_nan=False)


# ---------- Hypotheses ----------
@dataclass
class ObjectHypothesis:
    id: str                    # 이번 관측 안 고유 id (예: ramen_bundle_00). 프레임 단위 임시 번호
    class_name: str
    rigidity: str              # RIGIDITY
    frame_id: str              # 'base_link' | 'camera'(T_base_cam 항등일 때, 경고)
    position: List[float]      # [x, y, z] m (무효면 null)
    orientation: List[float]   # [qx, qy, qz, qw]
    pose_valid: bool
    geometry: dict             # {type, params_fit: {... m 값 + _mm 병기}}
    confidence: float          # 0~1
    score_raw: float           # 부품 원점수 (디버그)
    backend: str
    refine_used: bool
    stamp: float               # 관측 시각 (Frame.stamp)
    age_ms: float              # 발행 시각 − 관측 시각
    bbox3d: dict               # {min: [x,y,z], max: [x,y,z]} m — 축소 모드에서도 항상
    neighbors: List[dict] = field(default_factory=list)   # [{id, displacement: [dx,dy,dz] m (base), contact: bool}]
    extras: dict = field(default_factory=dict)            # 카드 extras.compute 항목만 (+ 축소 모드 mask_center_px, 연속 모드 stability)
    invalid_reason: Optional[str] = None                  # pose_valid=False 사유. 유효면 None

    def to_dict(self) -> dict:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectHypothesis":
        return cls(**d)


@dataclass
class Hypotheses:
    """한 프레임의 결과. header: stamp, source, frame_id, scene_id, timing_ms{find,lift,pose,scene,affordance,gate,total} (+reason, published, scene)."""
    header: dict
    items: List[ObjectHypothesis]

    def to_json(self, indent=None) -> str:
        return _dumps({"header": self.header, "items": [o.to_dict() for o in self.items]}, indent)

    @classmethod
    def from_json(cls, s: str) -> "Hypotheses":
        d = json.loads(s)
        return cls(header=d["header"], items=[ObjectHypothesis.from_dict(o) for o in d["items"]])

    @classmethod
    def empty(cls, header: dict, reason: str) -> "Hypotheses":
        assert reason in FAILURE_CODES, reason
        return cls(header=dict(header, reason=reason), items=[])

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    @property
    def valid(self) -> List[ObjectHypothesis]:
        return [o for o in self.items if o.pose_valid]


# ---------- SceneMap (④) ----------
@dataclass
class SceneMap:
    """정사영 공간 지도. layers: 'free','object','wall','liner','height_mm','instance_id','tier','unobserved' (HxW).
    plane: {T_base_plane 4x4, res_mm, extent_mm [w,h], origin_px [u0,v0]}. free_space: {direct_mm, manipulable_rigid_mm, manipulable_measured_mm|None, axis}."""
    scene_id: str
    plane: dict
    layers: Dict[str, np.ndarray]
    free_space: dict
    meta: dict = field(default_factory=dict)   # reference_plane_type, plane_from, n_instances, walls_source

    def save(self, path) -> None:
        p = Path(path)
        np.savez_compressed(p, scene_id=self.scene_id, plane=json.dumps(jsonable(self.plane)), free_space=json.dumps(jsonable(self.free_space)),
                            meta=json.dumps(jsonable(self.meta)), layer_names=json.dumps(list(self.layers)),
                            **{f"layer_{k}": v for k, v in self.layers.items()})

    @classmethod
    def load(cls, path) -> "SceneMap":
        z = np.load(path, allow_pickle=False)
        names = json.loads(str(z["layer_names"]))
        plane = json.loads(str(z["plane"]))
        plane["T_base_plane"] = np.asarray(plane["T_base_plane"], float)      # JSON 왕복에서 list 로 돌아오면 to_plane/to_base 가 죽는다
        return cls(scene_id=str(z["scene_id"]), plane=plane, free_space=json.loads(str(z["free_space"])),
                   meta=json.loads(str(z["meta"])), layers={k: z[f"layer_{k}"] for k in names})

    def summary(self) -> dict:
        """JSON 용 요약 (층 배열 제외)."""
        return dict(scene_id=self.scene_id, plane=self.plane, free_space=self.free_space, meta=self.meta,
                    layers={k: list(v.shape) for k, v in self.layers.items()})


# ---------- Affordances (⑤) ----------
@dataclass
class AffordanceCandidate:
    candidate_id: str
    target_id: str
    action: str                # ACTIONS
    pose_base: List[List[float]]   # 4x4
    uv_theta: List[float]      # [u, v, theta] (plane 좌표 mm, rad)
    score: float
    clearance_mm: dict         # {uphill, below, side_l, side_r} 예측
    clearance_measured_mm: Optional[dict] = None
    risk: float = 0.0
    rank: int = 0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "AffordanceCandidate":
        return cls(**d)


@dataclass
class Affordances:
    """header: {stamp, frame_id, scene_id, gripper, task, source}. free_space: SceneMap.free_space 복사."""
    header: dict
    candidates: List[AffordanceCandidate]
    free_space: Optional[dict] = None
    invalid_reason: Optional[str] = None

    def to_json(self, indent=None) -> str:
        return _dumps({"header": self.header, "candidates": [c.to_dict() for c in self.candidates],
                       "free_space": self.free_space, "invalid_reason": self.invalid_reason}, indent)

    @classmethod
    def from_json(cls, s: str) -> "Affordances":
        d = json.loads(s)
        return cls(header=d["header"], candidates=[AffordanceCandidate.from_dict(c) for c in d["candidates"]],
                   free_space=d.get("free_space"), invalid_reason=d.get("invalid_reason"))

    @classmethod
    def empty(cls, header: dict, reason: str) -> "Affordances":
        return cls(header=header, candidates=[], free_space=None, invalid_reason=reason)


# ---------- 스키마 검증 (외부 소비자·테스트용) ----------
def _check(cond: bool, where: str, msg: str):
    if not cond:
        raise ValueError(f"{where}: {msg}")


def _vec(v, n, where, allow_null=False):
    _check(isinstance(v, list) and len(v) == n, where, f"길이 {n} 목록이어야 함")
    for x in v:
        _check((x is None and allow_null) or isinstance(x, (int, float)), where, "숫자(또는 null)")


def validate_hypotheses(d: dict) -> None:
    """Hypotheses.to_dict() 형태 검증. 어긋나면 ValueError(필드 경로)."""
    _check(isinstance(d, dict) and "header" in d and "items" in d, "hypotheses", "header/items 필요")
    h = d["header"]
    for k in ("stamp", "source", "frame_id", "scene_id", "timing_ms"):
        _check(k in h, f"header.{k}", "없음")
    for k in TIMING_KEYS:
        _check(k in h["timing_ms"], f"header.timing_ms.{k}", "없음")
    if h.get("reason") is not None:
        _check(h["reason"] in FAILURE_CODES, "header.reason", f"{FAILURE_CODES} 중 하나")
    for i, o in enumerate(d["items"]):
        w = f"items[{i}]"
        for k in ("id", "class_name", "rigidity", "frame_id", "position", "orientation", "pose_valid", "geometry", "confidence",
                  "score_raw", "backend", "refine_used", "stamp", "age_ms", "bbox3d", "neighbors", "extras", "invalid_reason"):
            _check(k in o, f"{w}.{k}", "없음")
        _check(o["rigidity"] in RIGIDITY, f"{w}.rigidity", f"{RIGIDITY}")
        _vec(o["position"], 3, f"{w}.position", allow_null=True)
        _vec(o["orientation"], 4, f"{w}.orientation")
        _check(o["geometry"].get("type") in GEOMETRY_TYPES, f"{w}.geometry.type", f"{GEOMETRY_TYPES}")
        _check(0.0 <= o["confidence"] <= 1.0, f"{w}.confidence", "0~1")
        _vec(o["bbox3d"]["min"], 3, f"{w}.bbox3d.min", allow_null=True)
        _vec(o["bbox3d"]["max"], 3, f"{w}.bbox3d.max", allow_null=True)
        _check(o["pose_valid"] == (o["invalid_reason"] is None), f"{w}.invalid_reason", "pose_valid 와 일치해야 함")
        for j, nb in enumerate(o["neighbors"]):
            _check({"id", "displacement", "contact"} <= set(nb), f"{w}.neighbors[{j}]", "id/displacement/contact")
            _vec(nb["displacement"], 3, f"{w}.neighbors[{j}].displacement")


def validate_affordances(d: dict) -> None:
    _check(isinstance(d, dict) and "header" in d and "candidates" in d, "affordances", "header/candidates 필요")
    for k in ("stamp", "frame_id", "scene_id", "gripper", "task", "source"):
        _check(k in d["header"], f"header.{k}", "없음")
    for i, c in enumerate(d["candidates"]):
        w = f"candidates[{i}]"
        for k in ("candidate_id", "target_id", "action", "pose_base", "uv_theta", "score", "clearance_mm", "risk", "rank", "reasons"):
            _check(k in c, f"{w}.{k}", "없음")
        _check(c["action"] in ACTIONS, f"{w}.action", f"{ACTIONS}")
        _check(isinstance(c["pose_base"], list) and len(c["pose_base"]) == 4 and all(len(r) == 4 for r in c["pose_base"]), f"{w}.pose_base", "4x4")
        _vec(c["uv_theta"], 3, f"{w}.uv_theta")
        _check({"uphill", "below", "side_l", "side_r"} <= set(c["clearance_mm"]), f"{w}.clearance_mm", "uphill/below/side_l/side_r")
