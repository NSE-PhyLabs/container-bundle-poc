"""등록 3종 로드·검증 (v3 §6): 물체 카드 objects/<name>/card.yaml, 그리퍼 grippers/<name>/kernel.yaml, 작업 tasks/<name>/rules.yaml,
그리고 configs/default.yaml. 검증은 런타임이 실제로 읽는 하위 키까지 본다 — 잘못된 파일은 실행 중 KeyError 가 아니라 로드 시 CardError(필드 경로 포함).
mesh 스케일 검증: mesh_units 로 m 변환 후 extents 와 카드 치수 비교, 3% 초과면 경고 + scale_correction 기록(자동 보정 기본 on)."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .errors import CardError

log = logging.getLogger(__name__)
PKG_ROOT = Path(__file__).resolve().parents[1]          # percept_common/
GEOMETRY_TYPES = ("cylinder", "box", "mesh", "none")
RIGIDITY = ("rigid", "semi_rigid", "deformable")
BACKENDS = ("geom_fit", "mesh_pose", "ref_pose", "none")
FIND_METHODS = ("color_rule", "ref_match", "sam3_text")
EXTRAS = ("axis", "grooves", "row_level", "budget_uphill")
OUTLIER_METHODS = ("z_percentile", "knn", "none")
PLANE_TYPES = ("box_slope", "pallet_top", "conveyor", "none")
PLANE_FROM = ("instance_centers", "ransac_up", "calib_file")
MESH_UNITS = {"mm": 0.001, "m": 1.0}
SPLIT_KEYS = ("min_height_px", "max_cuts", "probe_min_px", "probe_frac", "min_diam_px", "remain_factor", "window_frac",
              "lo_frac", "hi_margin_px", "min_band_px", "depth_min_samples", "fallback_length_mm", "profile")
CYL_FIT_KEYS = ("max_iter", "step_tol_mm", "damping", "init_frac_r", "diverge_frac_r", "length_percentiles", "cross_percentile", "basis_switch_abs_z")
APPROACH_DIRS = ("normal", "uphill", "down")
WALL_ORIGINS = ("hull", "marker")
GOALS = ("extract", "place", "pick", "push")
ACTIONS = ("insert", "scoop", "place", "push")
AXES = ("uphill", "x", "y")
COST_MODES = ("hard", "soft", "reward")
LAYERS = ("wall", "liner", "object_other", "clearance_uphill", "target_valley", "pallet_edge", "height_mismatch", "pattern_slot",
          "conveyor_edge", "unobserved", "free", "object")
ORDERING_KEYS = ("tier_top_first", "valley_count_desc", "uphill_pos_desc", "id_asc", "pattern_order", "x_asc", "y_asc")


@dataclass
class Card:
    class_name: str
    rigidity: str
    geometry: dict
    find: dict
    pose: dict
    scene: dict
    gate: dict
    extras: dict
    lift: dict
    path: Path          # 카드 폴더
    raw: dict

    @property
    def reduced(self) -> bool:
        """축소 모드: 변형체이거나 형상 없음 → 포즈 생략, bbox3d 만."""
        return self.rigidity == "deformable" or self.geometry["type"] == "none" or self.pose.get("backend") == "none"

    @property
    def mesh_path(self) -> Optional[Path]:
        m = self.pose.get("mesh")
        return (self.path / m) if m else None

    @property
    def refs_dir(self) -> Path:
        return self.path / self.find.get("refs_dir", "refs")

    def nominal_radius_m(self) -> float:
        """이웃 판정용 대표 반경(m): cylinder = radius, box = size 최대/2, 그 외 = bbox 로 대체(호출자)."""
        gp = self.geometry.get("params") or {}
        if self.geometry["type"] == "cylinder":
            return float(gp["radius_mm"]) / 1000.0
        if self.geometry["type"] == "box":
            return max(gp["size_mm"]) / 2000.0
        return float("nan")


# ---------- 공통 검사 도우미 ----------
def _req(d, key: str, where: str, types=None):
    if not isinstance(d, dict) or key not in d:
        raise CardError(f"{where}: 필수 필드 '{key}' 없음")
    v = d[key]
    if types is not None and not isinstance(v, types):
        raise CardError(f"{where}.{key}: 타입 {type(v).__name__} (기대 {types})")
    return v


def _num(d, key, where, positive=True, types=(int, float)):
    v = _req(d, key, where, types)
    if isinstance(v, bool) or (positive and v <= 0):
        raise CardError(f"{where}.{key}: 양수여야 함, 지금 {v!r}")
    return v


def _enum(d, key, where, allowed):
    v = _req(d, key, where)
    if v not in allowed:
        raise CardError(f"{where}.{key}: {allowed} 중 하나, 지금 {v!r}")
    return v


def _range(v, where: str, positive=True):
    """[lo, hi] 또는 숫자 → [lo, hi] (float)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v]
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)) or v[0] > v[1]:
        raise CardError(f"{where}: [lo, hi] (lo <= hi) 이어야 함, 지금 {v!r}")
    if positive and v[0] <= 0:
        raise CardError(f"{where}: 양수여야 함, 지금 {v!r}")
    return [float(v[0]), float(v[1])]


def _load_yaml(path, kind: str, default_name: str):
    p = Path(path)
    file = p / default_name if p.is_dir() else p
    if not file.exists():
        raise CardError(f"{file}: {kind} 파일 없음")
    with open(file, encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise CardError(f"{file}: 최상위가 매핑이 아님")
    return raw, file


# ---------- 물체 카드 ----------
def _hsv_ranges(v, where):
    if not isinstance(v, list) or not v:
        raise CardError(f"{where}: [[[h,s,v],[h,s,v]], ...] 목록이어야 함")
    for rng in v:
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, list) and len(x) == 3 for x in rng)):
            raise CardError(f"{where}: 각 항목은 [[h,s,v],[h,s,v]], 지금 {rng!r}")
        for x in rng:
            if not all(isinstance(c, int) and 0 <= c <= m for c, m in zip(x, (179, 255, 255))):
                raise CardError(f"{where}: HSV 는 정수 H 0~179, S/V 0~255, 지금 {x!r}")


def _color_hint(ch: dict, where: str, gtype: str):
    for k in ("loose", "strict"):
        _hsv_ranges(_req(ch, k, where), f"{where}.{k}")
    _num(ch, "min_area_frac", where)
    if ch.get("morph") is not None:
        for k in ("close", "open"):
            if ch["morph"].get(k) is not None and not (isinstance(ch["morph"][k], list) and len(ch["morph"][k]) == 2):
                raise CardError(f"{where}.morph.{k}: [w, h]")
    if ch.get("trim") is not None:
        for k in ("min_row_px", "min_rows"):
            _num(ch["trim"], k, f"{where}.trim", types=int)
    if ch.get("split") is not None:
        if gtype != "cylinder":
            raise CardError(f"{where}.split: geometry.type=cylinder 에서만 (지름·길이비 필요), 지금 {gtype}")
        for k in SPLIT_KEYS:
            _req(ch["split"], k, f"{where}.split")
        for k in ("min_sel", "min_valid_frac", "blur_ksize"):
            _req(ch["split"]["profile"], k, f"{where}.split.profile")
    sp = ch.get("sam_prompt") or {}
    if sp.get("point", "box_center") not in ("box_center", "none"):
        raise CardError(f"{where}.sam_prompt.point: box_center | none")
    if not sp.get("box", True) and sp.get("point", "box_center") == "none":
        raise CardError(f"{where}.sam_prompt: 점과 상자를 둘 다 끌 수 없음")
    clip = _req(ch, "clip", where, dict)
    for k in ("y_frac_box", "x_frac_img"):
        _num(clip, k, f"{where}.clip", positive=False)
    vd = _req(ch, "verdict", where, dict)
    for k in ("area_min_frac", "area_max_frac", "min_elongation", "min_strict_frac"):
        _num(vd, k, f"{where}.verdict", positive=False)


def _confidence_rule(rule, where):
    if not isinstance(rule, dict):
        raise CardError(f"{where}: 매핑이어야 함")
    if rule.get("kind") == "identity":
        return
    if "rms_mm_good" in rule:
        keys = ("rms_mm_good", "rms_mm_bad")
    elif "score_good" in rule:
        keys = ("score_good", "score_bad")
    else:
        raise CardError(f"{where}: {{rms_mm_good, rms_mm_bad}} 또는 {{score_good, score_bad}} 또는 {{kind: identity}}")
    for k in keys:
        _num(rule, k, where, positive=False)
    if rule[keys[0]] == rule[keys[1]]:
        raise CardError(f"{where}: good 과 bad 가 같음")


def validate_card(raw: dict, where: str) -> dict:
    """스키마 검증 + 정규화(length_mm/observe_distance/fit_valid → [lo, hi]). 반환: 정규화된 raw."""
    cls = _req(raw, "class", where, str)
    if not cls.isidentifier():
        raise CardError(f"{where}.class: 식별자 형식이어야 함, 지금 {cls!r}")
    _enum(raw, "rigidity", where, RIGIDITY)
    g = _req(raw, "geometry", where, dict)
    gt = _enum(g, "type", f"{where}.geometry", GEOMETRY_TYPES)
    gp = g.get("params") or {}
    if gt == "cylinder":
        _num(gp, "radius_mm", f"{where}.geometry.params")
        gp["length_mm"] = _range(_req(gp, "length_mm", f"{where}.geometry.params"), f"{where}.geometry.params.length_mm")
    elif gt == "box":
        s = _req(gp, "size_mm", f"{where}.geometry.params", list)
        if len(s) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in s):
            raise CardError(f"{where}.geometry.params.size_mm: 양수 3개, 지금 {s!r}")
    f = _req(raw, "find", where, dict)
    methods = _req(f, "methods", f"{where}.find", list)
    if not methods or any(m not in FIND_METHODS for m in methods):
        raise CardError(f"{where}.find.methods: {FIND_METHODS} 의 비지 않은 부분집합, 지금 {methods!r}")
    if "color_rule" in methods:
        _color_hint(_req(f, "color_hint", f"{where}.find", dict), f"{where}.find.color_hint", gt)
    if "sam3_text" in methods and not f.get("text_prompt"):
        raise CardError(f"{where}.find.text_prompt: sam3_text 를 쓰면 필수")
    p = _req(raw, "pose", where, dict)
    be = _enum(p, "backend", f"{where}.pose", BACKENDS)
    if p.get("mesh"):
        _enum(p, "mesh_units", f"{where}.pose", tuple(MESH_UNITS))
    if gt == "mesh" and not p.get("mesh"):
        raise CardError(f"{where}.pose.mesh: geometry.type=mesh 면 필수")
    if p.get("refine") not in (None, "geom_fit"):
        raise CardError(f"{where}.pose.refine: null 또는 geom_fit")
    reduced = raw["rigidity"] == "deformable" or gt == "none" or be == "none"
    if not reduced and (be == "geom_fit" or p.get("refine") == "geom_fit"):
        gf = p.get("geom_fit")
        if not isinstance(gf, dict):
            raise CardError(f"{where}.pose.geom_fit: geom_fit 을 쓰면 맞춤 상수 매핑 필수")
        if gt == "cylinder":
            for k in CYL_FIT_KEYS:
                _req(gf, k, f"{where}.pose.geom_fit")
    if raw.get("lift") is not None:
        if (raw["lift"].get("outlier") or {}).get("method", "none") not in OUTLIER_METHODS:
            raise CardError(f"{where}.lift.outlier.method: {OUTLIER_METHODS}")
    sc = raw.get("scene")
    if sc is not None:
        rp = _req(sc, "reference_plane", f"{where}.scene", dict)
        _enum(rp, "type", f"{where}.scene.reference_plane", PLANE_TYPES)
        if rp["type"] != "none":
            _enum(rp, "plane_from", f"{where}.scene.reference_plane", PLANE_FROM)
        if not isinstance(sc.get("neighbors", False), bool):
            raise CardError(f"{where}.scene.neighbors: bool")
        if sc.get("walls") is not None:                                   # Q24: 벽은 카드 치수로 그린다
            wl, ww = sc["walls"], f"{where}.scene.walls"
            inner = _req(wl, "inner_mm", ww, list)
            if len(inner) != 3 or not all(isinstance(v, (int, float)) and v > 0 for v in inner):
                raise CardError(f"{ww}.inner_mm: [축 방향 길이, 경사 방향 폭, 깊이] mm 3개")
            for k in ("lip_mm", "thickness_mm"):
                if k in wl:
                    _num(wl, k, ww, positive=False)
            _enum(wl, "origin", ww, WALL_ORIGINS)
            if wl["origin"] == "marker" or wl.get("marker") is not None:
                mk = _req(wl, "marker", ww, dict)
                _num(mk, "id", f"{ww}.marker", types=int, positive=False)
                _num(mk, "size_mm", f"{ww}.marker")
                if mk.get("dict", "DICT_4X4_50") not in ("DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250"):
                    raise CardError(f"{ww}.marker.dict: 지원 사전이 아님")
                tbm = mk.get("T_box_marker")
                if tbm is not None and not (isinstance(tbm, list) and (len(tbm) == 6 or (len(tbm) == 4 and all(isinstance(r, list) and len(r) == 4 for r in tbm)))):
                    raise CardError(f"{ww}.marker.T_box_marker: null | [x y z mm, rx ry rz deg] | 4x4")
        if sc.get("tier_cluster_mm") is not None:
            _num(sc, "tier_cluster_mm", f"{where}.scene")
    gate = _req(raw, "gate", where, dict)
    cm = _req(gate, "confidence_map", f"{where}.gate", dict)
    if not reduced and be not in cm:
        raise CardError(f"{where}.gate.confidence_map: backend '{be}' 의 규칙 없음")
    for name, rule in cm.items():
        _confidence_rule(rule, f"{where}.gate.confidence_map.{name}")
    if not isinstance(gate.get("stable_frames"), int) or gate["stable_frames"] < 1:
        raise CardError(f"{where}.gate.stable_frames: 1 이상 정수")
    _num(gate, "stable_tol_mm", f"{where}.gate")
    gtol = _req(gate, "grasp_tolerance", f"{where}.gate", dict)
    for k in ("pos_mm", "axis_deg"):
        _num(gtol, k, f"{where}.gate.grasp_tolerance")
    if "observe_distance_m" in gate:
        raise CardError(f"{where}.gate.observe_distance_m: 카드가 아니라 tasks/<name>/rules.yaml 로 이동 (Q14, 9/18)")
    if not reduced and (not isinstance(gate.get("min_points"), int) or gate["min_points"] < 3):
        raise CardError(f"{where}.gate.min_points: 3 이상 정수")
    for k, v in (gate.get("fit_valid") or {}).items():
        gate["fit_valid"][k] = _range(v, f"{where}.gate.fit_valid.{k}")
    if gate.get("length_penalty") is not None:
        _num(gate["length_penalty"], "per_mm", f"{where}.gate.length_penalty", positive=False)
    ex = _req(raw, "extras", where, dict)
    comp = _req(ex, "compute", f"{where}.extras", list)
    if any(c not in EXTRAS for c in comp):
        raise CardError(f"{where}.extras.compute: {EXTRAS} 의 부분집합, 지금 {comp!r}")
    if any(c in comp for c in ("grooves", "row_level")):
        vp = _req(ex, "valley", f"{where}.extras", dict)
        for k in ("dist_min_mm", "dist_max_mm", "max_abs_dot_u_axis", "max_abs_dot_u_n"):
            _num(vp, k, f"{where}.extras.valley", positive=False)
        npar = _req(ex, "normal", f"{where}.extras", dict)
        _num(npar, "min_bundles", f"{where}.extras.normal", types=int)
        _num(npar, "max_abs_dot_s_axis", f"{where}.extras.normal", positive=False)
        _num(ex, "tie_mm", f"{where}.extras")
    return raw


def check_mesh_scale(card: "Card", tol_frac: float = 0.03, auto_scale: bool = True) -> None:
    """mesh extents(mesh_units → m) 와 카드 치수 비교. 3% 초과면 경고 + card.pose['scale_correction'] (auto_scale 이면 보정값, 아니면 1.0)."""
    mp = card.mesh_path
    card.pose["scale_correction"] = 1.0
    if not mp or not mp.exists():
        return
    import trimesh
    m = trimesh.load(str(mp), force="mesh")
    ext = np.sort(np.asarray(m.extents, dtype=float) * MESH_UNITS[card.pose["mesh_units"]])[::-1]     # m, 큰 순
    card.pose["mesh_extents_m"] = [float(v) for v in ext]
    gp, gt = card.geometry.get("params") or {}, card.geometry["type"]
    if gt == "cylinder":
        lo, hi = (v / 1000.0 for v in gp["length_mm"])
        d = 2.0 * gp["radius_mm"] / 1000.0
        ok = lo * (1 - tol_frac) <= ext[0] <= hi * (1 + tol_frac) and all(abs(e / d - 1) <= tol_frac for e in ext[1:])
        expected = np.array([(lo + hi) / 2, d, d])
    elif gt == "box":
        expected = np.sort(np.asarray(gp["size_mm"], float) / 1000.0)[::-1]
        ok = all(abs(e / x - 1) <= tol_frac for e, x in zip(ext, expected))
    else:
        return
    if not ok:
        corr = float(expected[0] / ext[0])
        log.warning("%s: mesh extents %s m 이 카드 치수 %s m 와 %.0f%% 넘게 다름 → scale_correction %.4f%s", mp, np.round(ext, 4), np.round(expected, 4),
                    tol_frac * 100, corr, " (자동 보정)" if auto_scale else " (기록만)")
        card.pose["scale_correction"] = corr if auto_scale else 1.0


def load_card(path, cfg: Optional[dict] = None) -> Card:
    """objects/<class> 폴더 또는 card.yaml → Card. 스키마 오류는 CardError. mesh 가 있으면 스케일 검증."""
    raw, file = _load_yaml(path, "카드", "card.yaml")
    raw = validate_card(raw, str(file))
    card = Card(class_name=raw["class"], rigidity=raw["rigidity"], geometry=raw["geometry"], find=raw["find"], pose=raw["pose"],
                scene=raw.get("scene") or {}, gate=raw["gate"], extras=raw["extras"], lift=raw.get("lift") or {}, path=file.parent, raw=raw)
    mcfg = (cfg or {}).get("mesh", {})
    check_mesh_scale(card, float(mcfg.get("tol_frac", 0.03)), bool(mcfg.get("auto_scale", True)))
    return card


# ---------- 그리퍼 커널 ----------
def validate_gripper(raw: dict, where: str) -> dict:
    _req(raw, "name", where, str)
    _num(raw, "px_mm", where)
    for k in ("insert_profile", "support_profile"):
        _req(raw, k, where, str)
    ap = _req(raw, "approach", where, dict)
    _enum(ap, "direction", f"{where}.approach", APPROACH_DIRS)
    _num(ap, "offset_mm", f"{where}.approach", positive=False)
    _num(raw, "insert_depth_mm", where, positive=False)
    if raw.get("blade_profile") is not None:                                 # Q23: 2.5D 날 높이 프로파일(선택)
        if not isinstance(raw["blade_profile"], str):
            raise CardError(f"{where}.blade_profile: png 파일명")
        _num(raw, "blade_zero_mm", where, positive=False) if "blade_zero_mm" in raw else raw.setdefault("blade_zero_mm", 0.0)
    cn = _req(raw, "clearance_need_mm", where, dict)
    for k in ("below", "side", "uphill"):
        _num(cn, k, f"{where}.clearance_need_mm", positive=False)
    if not isinstance(raw.get("two_point", False), bool):
        raise CardError(f"{where}.two_point: bool")
    if raw.get("two_point") and not raw.get("spacing_rule"):
        raise CardError(f"{where}.spacing_rule: two_point 면 필수 (예: 'L - 2*end_inset')")
    _num(raw, "end_inset_mm", where, positive=False)
    ms = _req(raw, "measures", where, dict)
    for k in ("width_encoder", "force_at_root"):
        if not isinstance(ms.get(k), bool):
            raise CardError(f"{where}.measures.{k}: bool")
    return raw


def load_kernel_png(path, px_mm: float, res_mm: float) -> np.ndarray:
    """커널 png(흰색=점유) → bool HxW. px_mm ≠ res_mm 이면 NEAREST 리샘플해 1픽셀 = res_mm."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise CardError(f"{path}: 커널 png 읽기 실패")
    if abs(px_mm - res_mm) > 1e-9:
        s = px_mm / res_mm
        img = cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_NEAREST)
    return img > 127


def load_gripper(path, res_mm: Optional[float] = None) -> dict:
    """grippers/<name> 폴더 또는 kernel.yaml → dict(+ path, kernels{insert, support} bool 배열, res_mm)."""
    raw, file = _load_yaml(path, "그리퍼 커널", "kernel.yaml")
    raw = validate_gripper(raw, str(file))
    raw["path"] = file.parent
    for k in ("insert_profile", "support_profile"):
        if not (file.parent / raw[k]).exists():
            raise CardError(f"{file}: {k} 파일 없음: {raw[k]} (scripts/register.py gripper --gen 으로 생성)")
    res = float(res_mm if res_mm is not None else raw["px_mm"])
    raw["res_mm"] = res
    raw["kernels"] = {"insert": load_kernel_png(file.parent / raw["insert_profile"], float(raw["px_mm"]), res),
                      "support": load_kernel_png(file.parent / raw["support_profile"], float(raw["px_mm"]), res)}
    if raw.get("blade_profile"):                                             # 픽셀값 = 그 칸에서 날이 차지하는 아래 방향 깊이 mm (0 = 날 없음)
        bp = file.parent / raw["blade_profile"]
        if not bp.exists():
            raise CardError(f"{file}: blade_profile 파일 없음: {raw['blade_profile']} (scripts/register.py gripper --gen-blade)")
        img = cv2.imread(str(bp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise CardError(f"{bp}: 읽기 실패")
        if abs(float(raw["px_mm"]) - res) > 1e-9:
            s = float(raw["px_mm"]) / res
            img = cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_NEAREST)
        blade = img.astype(np.float32)
        if blade.shape != raw["kernels"]["insert"].shape:
            raise CardError(f"{file}: blade_profile {blade.shape} 와 insert_profile {raw['kernels']['insert'].shape} 크기가 다름")
        raw["kernels"]["blade"] = blade
    else:
        raw["kernels"]["blade"] = None
    return raw


# ---------- 작업 규칙 ----------
def validate_task(raw: dict, where: str) -> dict:
    _req(raw, "name", where, str)
    _enum(raw, "goal", where, GOALS)
    _enum(raw, "action", where, ACTIONS)
    _enum(raw, "axis", where, AXES)
    terms = _req(raw, "cost_terms", where, list)
    for i, t in enumerate(terms):
        w = f"{where}.cost_terms[{i}]"
        _enum(t, "layer", w, LAYERS)
        mode = _enum(t, "mode", w, COST_MODES)
        if mode in ("soft", "reward"):
            _num(t, "weight", w, positive=False)
        if mode == "soft" and "decay_mm" in t:
            _num(t, "decay_mm", w)
        if "saturate_mm" in t:
            _num(t, "saturate_mm", w)
        if "margin_mm" in t:
            _num(t, "margin_mm", w, positive=False)
    order = _req(raw, "ordering", where, list)
    if any(o not in ORDERING_KEYS for o in order):
        raise CardError(f"{where}.ordering: {ORDERING_KEYS} 의 부분집합, 지금 {order!r}")
    rots = _req(raw, "rotations_deg", where, list)
    if not rots or not all(isinstance(r, (int, float)) for r in rots):
        raise CardError(f"{where}.rotations_deg: 숫자 목록")
    _num(raw, "top_k", where, types=int)
    _num(raw, "nms_mm", where)
    for k in ("observe_distance_m", "preferred_distance_m"):          # 유효성 한계 / 선호 범위 (Q21)
        if raw.get(k) is not None:
            raw[k] = _range(raw[k], f"{where}.{k}")
    if raw.get("preferred_distance_m") and raw.get("observe_distance_m"):
        lo, hi = raw["observe_distance_m"]
        if not (lo <= raw["preferred_distance_m"][0] and raw["preferred_distance_m"][1] <= hi):
            raise CardError(f"{where}.preferred_distance_m: observe_distance_m {[lo, hi]} 안에 있어야 함")
    if raw.get("disturbance_exclude") is not None and not isinstance(raw["disturbance_exclude"], list):
        raise CardError(f"{where}.disturbance_exclude: 목록")
    return raw


def load_task(path) -> dict:
    raw, file = _load_yaml(path, "작업 규칙", "rules.yaml")
    raw = validate_task(raw, str(file))
    raw["path"] = file.parent
    return raw


# ---------- 설정 ----------
def load_config(path=None) -> dict:
    """configs/default.yaml (또는 지정 경로)."""
    file = Path(path) if path else PKG_ROOT / "configs" / "default.yaml"
    with open(file, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    # 전달 폴더를 옮겨도 설정의 상대 경로가 동작하도록 한다.
    paths = cfg.get("paths", {})
    for key in ("calib_npz", "T_base_cam_npz", "mobile_sam_ckpt", "original_scripts"):
        value = paths.get(key)
        if value and not Path(value).is_absolute():
            paths[key] = str((PKG_ROOT / value).resolve())
    for key, value in paths.get("datasets", {}).items():
        if value and not Path(value).is_absolute():
            paths["datasets"][key] = str((PKG_ROOT / value).resolve())
    cfg["_path"] = str(file)
    return cfg
