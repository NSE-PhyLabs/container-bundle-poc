"""합성 카메라 (v3 §7.3): trimesh 로 장면을 만들고 표면 점 샘플링 + z-buffer 로 RGB-D 를 만든다. Isaac 없이 채점하기 위한 것.

정답 `Frame.extra['gt']` = {poses(4x4 base), inst_masks(HxW int, 0=배경), visibility, in_frame_frac, object_distance_m,
dims, class_name, T_base_cam, preset, distance_m, splat_px}. seed 를 주면 같은 장면이 재현된다.
- `visibility` = **화면 안에서** 가려지지 않고 보이는 비율(가림만 본다). 화면 밖으로 잘린 부분은 분모·분자에서 함께 빠져 반영되지 않는다.
- `in_frame_frac` = 카메라 앞(z>0) 표면점 중 **화면 안**에 든 비율(잘림). 채점기는 이 둘을 따로 본다 — 잘린 물체를 "다 보인다"로 채점하지 않기 위해서다.
- `object_distance_m` = 물체별 카메라 거리. `distance_m` 은 프레임 공칭값(주시점까지)이라 둘은 다르다.
- `inst_masks` 는 splat 팽창을 되돌려(침식) 준다 — 렌더 파라미터가 정답 실루엣을 부풀리지 않게.

렌더: 각 메시 표면을 넓이에 비례해 점으로 뽑아(density_pt_per_mm2) 카메라로 투영하고, 먼 점부터 찍어 가까운 점이 덮게 한다(z-buffer).
splat 반경 1 px 로 구멍을 메운다. 노이즈: 깊이 가우시안 σ, 무작위 드롭아웃, 유효 영역 가장자리 침식.

좌표: 카메라는 ROS 광학(x 오른쪽, y 아래, z 앞), base 는 로봇(x 앞, y 왼쪽, z 위). `T_base_cam` 을 알고 있으므로 frame_id='base_link' 로 낸다
— 실물(항등, 'camera')과 달리 base 경로를 실제로 검사하게 된다.
"""
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from ..errors import NotAvailable
from ..pose.base import invert_T, make_T, rot_axis
from ..registry import Card
from ..types import Frame
from .base import CameraAdapter

log = logging.getLogger(__name__)
MM = 1000.0
PRESETS = ("ramen_box_4x3", "pallet_layer", "conveyor_single", "single_random")
DEFAULT_NOISE = dict(depth_sigma_mm=3.0, dropout=0.02, edge_erode_px=1)
BG_GRAY = (60, 110)


# ---------- 기하 도우미 (make_T·invert_T·rot_axis 는 pose/base.py) ----------
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """카메라 위치·주시점 → T_base_cam (ROS 광학: x 오른쪽, y 아래, z 앞)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    nr = np.linalg.norm(r)
    r = r / nr if nr > 1e-9 else np.array([0.0, -1.0, 0.0])
    d = np.cross(f, r)
    return make_T(np.stack([r, d, f], axis=1), eye)


# ---------- 물체 메시 ----------
def card_mesh(card: Card, length_m: Optional[float] = None):
    """카드 → trimesh. cylinder 는 축이 +x, box 는 size 순서대로, mesh 는 bbox 중심으로 옮긴다."""
    import trimesh
    gt, gp = card.geometry["type"], card.geometry.get("params") or {}
    if gt == "cylinder":
        r = gp["radius_mm"] / MM
        L = length_m if length_m is not None else float(np.mean(gp["length_mm"])) / MM
        m = trimesh.creation.cylinder(radius=r, height=L, sections=48)
        m.apply_transform(make_T(rot_axis([0, 1, 0], 90), [0, 0, 0]))       # z축 → x축
        return m
    if gt == "box":
        return trimesh.creation.box(extents=np.asarray(gp["size_mm"], float) / MM)
    if gt == "mesh":
        mp = card.mesh_path
        if mp is None or not mp.exists():
            raise NotAvailable(f"{card.class_name}: mesh 없음 ({mp})")
        m = trimesh.load(str(mp), force="mesh")
        scale = {"mm": 0.001, "m": 1.0}[card.pose["mesh_units"]] * float(card.pose.get("scale_correction", 1.0))
        m.apply_scale(scale)
        m.apply_translation(-m.bounds.mean(axis=0))                         # bbox 중심을 원점으로 (ASSUMPTIONS 5)
        return m
    raise NotAvailable(f"{card.class_name}: geometry.type={gt!r} 는 합성 대상이 아님")


def object_color(card: Card) -> Tuple[int, int, int]:
    c = (card.raw.get("synthetic") or {}).get("color_bgr")
    if c:
        return tuple(int(v) for v in c)
    strict = ((card.find.get("color_hint") or {}).get("strict") or [[[0, 200, 200], [10, 255, 255]]])[0]
    import cv2
    hsv = np.uint8([[[int((strict[0][0] + strict[1][0]) / 2), 220, 210]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


# ---------- 프리셋 ----------
def preset_ramen_box(card, rng, p):
    """기울어진 박스 안 4줄×3단. 박스 좌표 B: x 묶음 축, y 경사 아래(+), z 개구부(+). 경사 tilt_deg 만큼 기울어 개구부가 로봇 쪽(-x)·위를 향한다."""
    rows, tiers = int(p.get("rows", 4)), int(p.get("tiers", 3))
    tilt = float(p.get("tilt_deg", 45.0))
    r_m = card.geometry["params"]["radius_mm"] / MM
    pitch = 2 * r_m + float(rng.uniform(*p.get("gap_mm", (0.0, 10.0)))) / MM
    L0 = float(np.mean(card.geometry["params"]["length_mm"])) / MM
    # 박스 L → base: 묶음 축(박스 x)을 로봇 좌우(base y)로 돌린 뒤 tilt 만큼 눕힌다 (sim/box_scene.py ROT_BOX 와 같은 규약).
    # 결과: 축 = base +y, 개구부 법선 = 로봇 쪽(-x)·위(+z), 경사 아래 = 로봇 쪽·아래.
    R_bb = rot_axis([0, 1, 0], -tilt) @ rot_axis([0, 0, 1], 90.0)
    origin = np.array(p.get("box_origin_m", [0.75, 0.0, 0.45]), float)
    objs, dims = [], []
    for t in range(tiers):
        for r in range(rows):
            L = L0 + float(rng.normal(0, p.get("length_jitter_mm", 10.0) / 3)) / MM
            c_B = np.array([0.0, (r - (rows - 1) / 2) * pitch, (t + 0.5) * pitch])
            R = R_bb @ rot_axis([0, 0, 1], float(rng.normal(0, p.get("yaw_jitter_deg", 3.0) / 3)))
            objs.append((make_T(R, origin + R_bb @ c_B), dict(length_m=L, row=r, tier=t)))
            dims.append(L)
    top = origin + R_bb @ np.array([0.0, 0.0, (tiers - 0.5) * pitch])        # 맨 위 단 중심 = 주시점
    # 박스 벽·바닥 [추측: config/params_grasp.yaml 내부 730(축)×590(경사)×440(깊이), 판 10 mm — 실측은 QUESTIONS 25]. 줄은 경사 아래 벽에 붙어 있다.
    Lb, Wb, Db = (float(v) / MM for v in p.get("box_inner_mm", [730, 590, 440]))   # [축 방향 길이, 경사 방향 폭, 깊이]
    tb = float(p.get("wall_mm", 10.0)) / MM
    y_down = (rows - 1) / 2 * pitch + r_m                                      # 경사 아래 벽 안쪽 면 (박스 y)
    y_up = y_down - Wb                                                         # 경사 위 벽 안쪽 면
    yc = (y_down + y_up) / 2
    import trimesh
    walls = [("floor", [0.0, yc, -tb / 2], [Lb + 2 * tb, Wb + 2 * tb, tb]),
             ("wall_down", [0.0, y_down + tb / 2, Db / 2], [Lb + 2 * tb, tb, Db]),
             ("wall_up", [0.0, y_up - tb / 2, Db / 2], [Lb + 2 * tb, tb, Db]),
             ("wall_xneg", [-(Lb / 2 + tb / 2), yc, Db / 2], [tb, Wb + 2 * tb, Db]),
             ("wall_xpos", [(Lb / 2 + tb / 2), yc, Db / 2], [tb, Wb + 2 * tb, Db])]
    bg = [dict(name=f"ramen_box/{n}", T=make_T(R_bb, origin + R_bb @ np.asarray(c, float)), mesh=trimesh.creation.box(extents=e),
               color_bgr=(120, 150, 170)) for n, c, e in walls]                # 골판지: 카드 strict 빨강(H≤10,S≥150) 밖 (S≈75)
    T_base_box = make_T(R_bb, origin + R_bb @ np.array([-Lb / 2, y_up, 0.0]))   # 박스 안쪽 좌표계: 원점 = (축−끝, 경사 위 벽, 바닥) 구석
    return objs, top, dict(tilt_deg=tilt, pitch_m=pitch, rows=rows, tiers=tiers, box_origin_m=origin.tolist(), bg=bg,
                           box=dict(inner_mm=[Lb * MM, Wb * MM, Db * MM], wall_mm=tb * MM, T_base_box=T_base_box.tolist(),
                                    gap_uphill_mm=float((Wb - rows * pitch + (pitch - 2 * r_m)) * MM)))


def preset_pallet_layer(card, rng, p):
    """팔레트 위 한 층. 상자 n개를 격자로, 요 ±5°·위치 지터 ±10 mm."""
    size = np.asarray(card.geometry["params"]["size_mm"], float) / MM
    nx, ny = int(p.get("nx", 2)), int(p.get("ny", 2))
    gap = float(p.get("gap_mm", 20.0)) / MM
    z0 = float(p.get("pallet_z_m", 0.15))
    origin = np.array(p.get("origin_m", [0.6, 0.0, 0.0]), float)
    objs = []
    for iy in range(ny):
        for ix in range(nx):
            c = origin + np.array([(ix - (nx - 1) / 2) * (size[0] + gap), (iy - (ny - 1) / 2) * (size[1] + gap), z0 + size[2] / 2])
            c[:2] += rng.normal(0, p.get("pos_jitter_mm", 10.0) / 3 / MM, 2)
            objs.append((make_T(rot_axis([0, 0, 1], float(rng.normal(0, p.get("yaw_jitter_deg", 5.0) / 3))), c), dict(row=ix, tier=iy)))
    import trimesh
    pw, pd, ph = (float(v) / MM for v in p.get("pallet_mm", [1200, 1000, 144]))   # 유로 팔레트 크기 [추측]
    bg = [dict(name="pallet/top", T=make_T(np.eye(3), origin + np.array([0.0, 0.0, z0 - ph / 2])),
               mesh=trimesh.creation.box(extents=[pw, pd, ph]), color_bgr=(90, 130, 160))]
    return objs, origin + np.array([0, 0, z0 + size[2]]), dict(nx=nx, ny=ny, bg=bg, pallet=dict(size_mm=[pw * MM, pd * MM, ph * MM], top_z_m=z0))


def preset_conveyor_single(card, rng, p):
    """컨베이어 위 1개. x 를 따라 흘러간다."""
    z0 = float(p.get("belt_z_m", 0.30))
    x = float(p.get("x_m", 0.6)) + float(rng.uniform(-0.15, 0.15))
    h = (card.geometry["params"].get("radius_mm", 70) / MM) if card.geometry["type"] == "cylinder" else np.asarray(card.geometry["params"]["size_mm"], float)[2] / 2 / MM
    c = np.array([x, float(rng.normal(0, 0.05)), z0 + h])
    info = dict(length_m=float(np.mean(card.geometry["params"]["length_mm"])) / MM) if card.geometry["type"] == "cylinder" else {}
    import trimesh
    bg = [dict(name="belt", T=make_T(np.eye(3), [0.6, 0.0, z0 - 0.01]), mesh=trimesh.creation.box(extents=[2.0, 0.5, 0.02]), color_bgr=(70, 70, 70))]
    return [(make_T(rot_axis([0, 0, 1], float(rng.uniform(-180, 180))), c), info)], np.array([x, 0.0, z0]), dict(belt_z_m=z0, bg=bg)


def preset_single_random(card, rng, p):
    """임의 자세 1개 (부품 단위 검사용)."""
    c = np.array([float(rng.uniform(0.4, 0.8)), float(rng.normal(0, 0.1)), float(rng.uniform(0.4, 0.9))])
    R = rot_axis(rng.normal(size=3), float(rng.uniform(0, 180)))
    info = dict(length_m=float(np.mean(card.geometry["params"]["length_mm"])) / MM) if card.geometry["type"] == "cylinder" else {}
    return [(make_T(R, c), info)], c, {}


PRESET_FN = dict(ramen_box_4x3=preset_ramen_box, pallet_layer=preset_pallet_layer,
                 conveyor_single=preset_conveyor_single, single_random=preset_single_random)
PRESET_CAM = dict(ramen_box_4x3=dict(distance_m=0.85, pitch_deg=19.0), pallet_layer=dict(distance_m=1.2, pitch_deg=35.0),
                  conveyor_single=dict(distance_m=0.9, pitch_deg=45.0), single_random=dict(distance_m=0.8, pitch_deg=20.0))


# ---------- 렌더 ----------
def _sample_points(mesh, density_pt_per_mm2: float, rng):
    import trimesh
    n = max(2000, int(mesh.area * MM * MM * density_pt_per_mm2))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(0, 2 ** 31 - 1)))
    return np.asarray(pts, float)


def _project(P_cam, K, W, H):
    z = P_cam[:, 2]
    ok = z > 1e-3
    u = np.full(len(P_cam), -1.0)
    v = np.full(len(P_cam), -1.0)
    u[ok] = K[0, 0] * P_cam[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * P_cam[ok, 1] / z[ok] + K[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return ui, vi, z, ok


def _splat(depth, inst, ui, vi, z, ids, W, H, radius: int):
    """먼 점부터 찍어 가까운 점이 덮게 한다(z-buffer)."""
    order = np.argsort(-z)
    u, v, zz, ii = ui[order], vi[order], z[order], ids[order]
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = np.clip(u + du, 0, W - 1), np.clip(v + dv, 0, H - 1)
            closer = zz <= depth[vv, uu]
            depth[vv[closer], uu[closer]] = zz[closer]
            inst[vv[closer], uu[closer]] = ii[closer]


class SyntheticAdapter(CameraAdapter):
    name = "synthetic"

    def __init__(self, cfg: dict, card: Card, preset: str = "ramen_box_4x3", n: int = 30, seed: int = 0,
                 distance_m: Optional[float] = None, noise: Optional[dict] = None, preset_params: Optional[dict] = None,
                 image_size: Optional[Tuple[int, int]] = None, K: Optional[np.ndarray] = None, density_pt_per_mm2: float = 0.25,
                 splat_px: int = 1, cam_jitter: bool = True):
        if preset not in PRESET_FN:
            raise NotAvailable(f"preset={preset!r} — {PRESETS} 중 하나")
        self.cfg, self.card, self.preset = cfg, card, preset
        self.n, self.seed, self.i = n, seed, 0
        self.params = dict(preset_params or {})
        self.noise = dict(DEFAULT_NOISE, **(noise or {}))
        cam = dict(PRESET_CAM[preset])
        self.distance_m = float(distance_m if distance_m is not None else cam["distance_m"])
        self.pitch_deg = float(self.params.get("pitch_deg", cam["pitch_deg"]))
        self.cam_jitter = cam_jitter
        self.density, self.splat = float(density_pt_per_mm2), int(splat_px)
        if K is None:                                                        # 실물 머리 카메라와 같은 내부 파라미터 (거리·크기가 실물과 비교 가능하게)
            from .stereo_head import load_camera_config
            c = load_camera_config()
            W, H = image_size or tuple(c["image_size"])
            r = c["rectified_left"]
            K = np.array([[r["fx"], 0, r["cx"]], [0, r["fy"], r["cy"]], [0, 0, 1]], float)
        else:
            W, H = image_size or (640, 480)
        self.K, self.W, self.H = np.asarray(K, float), int(W), int(H)
        self.color = object_color(card)
        self._mesh_cache, self._bg_cache = {}, {}

    def __len__(self):
        return self.n

    def _bg_points(self, name: str, mesh):
        """배경 메시 표면 표본(밀도는 물체의 0.8배, 이름으로 캐시)."""
        if name not in self._bg_cache:
            self._bg_cache[name] = _sample_points(mesh, self.density * 0.8, np.random.default_rng(1))
        return self._bg_cache[name]

    def _mesh_points(self, length_m):
        """길이별 표면 표본점. 난수는 default_rng(0) 고정이라 프레임 난수 흐름과 무관(재현성). 캐시는 16개까지(누수 방지)."""
        key = round(float(length_m or 0.0), 4)
        if key not in self._mesh_cache:
            if len(self._mesh_cache) >= 16:
                self._mesh_cache.pop(next(iter(self._mesh_cache)))
            self._mesh_cache[key] = _sample_points(card_mesh(self.card, length_m), self.density, np.random.default_rng(0))
        return self._mesh_cache[key]

    def grab(self) -> Frame:
        if self.i >= self.n:
            raise StopIteration
        rng = np.random.default_rng([self.seed, self.i])
        objs, target, meta = PRESET_FN[self.preset](self.card, rng, self.params)
        # 카메라: 주시점에서 distance 만큼, 수평에서 pitch_deg 아래로 내려다본다 (로봇 쪽 -x 에서)
        pitch = np.radians(self.pitch_deg + (float(rng.normal(0, 1.5)) if self.cam_jitter else 0.0))
        d = self.distance_m * (1.0 + (float(rng.normal(0, 0.02)) if self.cam_jitter else 0.0))
        eye = target + np.array([-d * np.cos(pitch), float(rng.normal(0, 0.02)) if self.cam_jitter else 0.0, d * np.sin(pitch)])
        T_base_cam = look_at(eye, target)
        T_cam_base = invert_T(T_base_cam)

        depth = np.full((self.H, self.W), np.inf, np.float64)
        inst = np.zeros((self.H, self.W), np.int32)
        bg = meta.pop("bg", [])
        for j, b in enumerate(bg):                                             # 배경 기하(팔레트·박스 벽·벨트): depth 는 주되 인스턴스는 아님 (id 음수 → 뒤에서 0)
            P = self._bg_points(b["name"], b["mesh"])
            P_cam = (P @ b["T"][:3, :3].T + b["T"][:3, 3]) @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
            ui, vi, z, ok = _project(P_cam, self.K, self.W, self.H)
            _splat(depth, inst, ui[ok], vi[ok], z[ok], np.full(ok.sum(), -(j + 1), np.int32), self.W, self.H, self.splat)
        alone, in_frame, obj_dist = [], [], []
        for k, (T_obj, info) in enumerate(objs):
            P = self._mesh_points(info.get("length_m"))
            P_cam = (P @ T_obj[:3, :3].T + T_obj[:3, 3]) @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
            ui, vi, z, ok = _project(P_cam, self.K, self.W, self.H)
            in_frame.append(float(ok.sum() / max(int((P_cam[:, 2] > 1e-3).sum()), 1)))   # 잘림: 카메라 앞 점 중 화면 안 비율
            obj_dist.append(float(np.linalg.norm(T_cam_base[:3, :3] @ T_obj[:3, 3] + T_cam_base[:3, 3])))
            d1 = np.full((self.H, self.W), np.inf)
            i1 = np.zeros((self.H, self.W), np.int32)
            _splat(d1, i1, ui[ok], vi[ok], z[ok], np.full(ok.sum(), k + 1, np.int32), self.W, self.H, self.splat)
            alone.append(int((i1 > 0).sum()))
            _splat(depth, inst, ui[ok], vi[ok], z[ok], np.full(ok.sum(), k + 1, np.int32), self.W, self.H, self.splat)

        valid = np.isfinite(depth)
        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        rgb[...] = rng.integers(*BG_GRAY)                                    # 배경 단색 회색
        obj_px = inst > 0
        if obj_px.any():
            shade = np.clip(0.75 + 0.5 * (depth[obj_px].max() - depth[obj_px]) / max(float(np.ptp(depth[obj_px])), 1e-6), 0.7, 1.25)
            rgb[obj_px] = np.clip(np.asarray(self.color, float) * np.asarray(shade)[:, None], 0, 255).astype(np.uint8)
        for j, b in enumerate(bg):
            px = inst == -(j + 1)
            if px.any():
                sh = np.clip(0.8 + 0.4 * (depth[px].max() - depth[px]) / max(float(np.ptp(depth[px])), 1e-6), 0.7, 1.2)
                rgb[px] = np.clip(np.asarray(b["color_bgr"], float) * np.asarray(sh)[:, None], 0, 255).astype(np.uint8)
        inst[inst < 0] = 0

        import cv2
        depth_m = np.where(valid, depth, np.nan).astype(np.float32)
        ns = self.noise
        if ns.get("depth_sigma_mm"):
            depth_m = (depth_m + rng.normal(0, ns["depth_sigma_mm"] / MM, depth_m.shape)).astype(np.float32)
        if ns.get("edge_erode_px"):                                          # **드롭아웃보다 먼저**: 실루엣 가장자리만 깎는다.
            k = int(ns["edge_erode_px"])                                      # 뒤에 두면 드롭아웃 구멍마다 (2k+1)² 로 번져 실효 손실이 2% → 17% 가 된다
            keep = cv2.erode(np.isfinite(depth_m).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
            depth_m[~keep] = np.nan
        if ns.get("dropout"):
            depth_m[rng.random(depth_m.shape) < ns["dropout"]] = np.nan

        vis = [float(int((inst == k + 1).sum()) / a) if a else 0.0 for k, a in enumerate(alone)]
        if self.splat > 0:                                                    # 정답 마스크는 splat 팽창을 되돌린다(렌더 파라미터가 정답 실루엣을 부풀리지 않게)
            ker = np.ones((2 * self.splat + 1,) * 2, np.uint8)
            eroded = np.zeros_like(inst)
            for k in range(1, len(objs) + 1):
                eroded[cv2.erode((inst == k).astype(np.uint8), ker) > 0] = k
            inst = eroded
        gp = self.card.geometry.get("params") or {}
        L0 = float(np.mean(gp["length_mm"])) / MM if self.card.geometry["type"] == "cylinder" else None
        gt = dict(poses=[T for T, _ in objs], inst_masks=inst, visibility=vis, in_frame_frac=in_frame,
                  object_distance_m=obj_dist, class_name=self.card.class_name, splat_px=int(self.splat),
                  T_base_cam=T_base_cam, preset=self.preset, distance_m=float(d), pitch_deg=float(np.degrees(pitch)),
                  info=[i for _, i in objs], meta=meta, box=meta.get("box"), pallet=meta.get("pallet"),
                  dims=dict(radius_mm=gp.get("radius_mm"), size_mm=gp.get("size_mm"),
                            length_mm=([float(i.get("length_m", L0) * MM) for _, i in objs] if L0 is not None else None)))
        self.i += 1
        return Frame(rgb=rgb, depth=depth_m, K=self.K.copy(), T_base_cam=T_base_cam, frame_id=self.cfg["frames"]["base"],
                     stamp=time.time(), source=self.name,
                     extra=dict(gt=gt, scene_id=f"{self.preset}/{self.seed}/{self.i - 1:03d}",
                                image_key=f"synthetic:{self.preset}:{self.seed}:{self.i - 1}", n_objects=len(objs)))


def gt_finder_candidates(frame: Frame) -> List:
    """정답 마스크를 그대로 후보로 주는 Finder 대용 (색 규칙이 없는 카드·부품 단위 채점에 쓴다)."""
    from ..types import Candidate
    gt = frame.extra["gt"]
    out = []
    for k in range(1, int(gt["inst_masks"].max()) + 1):
        m = gt["inst_masks"] == k
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        out.append(Candidate(mask=m, bbox=(int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)),
                             score=float(gt["visibility"][k - 1]), finder="gt", instance_id=k - 1))
    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return out


class GtFinder:
    """정답 마스크 Finder (percept.find.base.Finder 와 같은 인터페이스). 합성 채점에서 찾기 오류를 빼고 맞춤만 볼 때."""
    name = "gt"

    def __init__(self, cfg=None):
        self.cfg = cfg

    def find(self, frame: Frame, card: Card):
        return gt_finder_candidates(frame)
