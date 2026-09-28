"""⑤-2 코스트맵 (v3 §7.8). `build(scene, rules, target, hyps, card)` → (cost float32 HxW, hard bool HxW, meta).

cost = Σ 항. 격자 규약 = 정사영 층과 같다(행 = v 축 방향, 열 = u 경사 방향, 셀 = res_mm).
- hard: 층 합집합(margin_mm 있으면 원형 팽창). 커널이 조금이라도 겹치면 불가(generate 에서 판정).
- soft: 층 마스크까지의 거리변환 d(mm) → cost += weight·exp(−d/decay_mm) (기본 decay 15).
- reward: clearance_uphill → 각 free 셀에서 +u 방향 장애물까지 거리(mm)/saturate_mm 를 0~1 로 → cost −= weight·val.
          target_valley → 타깃과 **경사 아래 접촉 이웃**의 접촉선(두 중심의 중점을 지나는 축 방향 선분) 근방 가우시안(σ = rules.valley_sigma_mm, 기본 8) reward.
          pattern_slot → tasks/<name>/pattern.yaml 의 slots [{u_mm, v_mm}] 근방 가우시안(σ 30). 파일 없으면 건너뜀(로그 1회).
- 층 이름: wall · liner(비면 건너뜀) · object_other(타깃 제외 물체) · unobserved(항상 soft weight 1.0) · pallet_edge/conveyor_edge(지지면 밖 = hard) ·
  height_mismatch(지지면 높이 편차 soft).
"""
import logging
from typing import Optional

import numpy as np

from ..scene.plane import to_plane

log = logging.getLogger(__name__)
_warned = set()


def _once(key, msg):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _soft(mask: np.ndarray, res: float, weight: float, decay_mm: float) -> np.ndarray:
    import cv2
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32) * res   # 마스크 셀까지의 거리 mm (마스크 안 = 0)
    return (weight * np.exp(-d / max(decay_mm, 1e-6))).astype(np.float32)


def _close(mask: np.ndarray, size_mm: float, res: float) -> np.ndarray:
    import cv2
    r = int(round(size_mm / res))
    if r <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def _dilate(mask: np.ndarray, margin_mm: float, res: float) -> np.ndarray:
    import cv2
    r = int(round(margin_mm / res))
    if r <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def run_length_uphill(obstacle: np.ndarray) -> np.ndarray:
    """각 셀에서 +u(열 증가) 방향으로 장애물 전까지 이어지는 셀 수(자기 포함). 장애물 셀 = 0."""
    H, W = obstacle.shape
    d = np.zeros((H, W), np.int32)
    nxt = np.zeros(H, np.int32)
    for c in range(W - 1, -1, -1):
        nxt = np.where(obstacle[:, c], 0, nxt + 1)
        d[:, c] = nxt
    return d


def target_index(hyps, target) -> Optional[int]:
    if target is None:
        return None
    for i, o in enumerate(hyps.items):
        if o.id == target.id:
            return i
    return None


def valley_partner(hyps, target, plane):
    """타깃의 골 상대 = 경사 아래(u 작은) 쪽 이웃. 카드 valley 규칙으로 gate 가 만든 extras.grooves 를 먼저, 없으면 접촉 이웃(neighbors.contact).
    반환 (이웃 hyp, u_t, v_t, u_n, v_n) m 또는 None."""
    if target is None:
        return None
    pt = to_plane(plane, np.asarray(target.position, float)[None])[0]
    best = None
    by_id = {o.id: o for o in hyps.items}
    grooves = (target.extras or {}).get("grooves") or []
    cands = [g["with_id"] for g in grooves] if grooves else [nb["id"] for nb in (target.neighbors or []) if nb.get("contact")]
    for nid in cands:
        o = by_id.get(nid)
        if o is None or not o.pose_valid:
            continue
        pn = to_plane(plane, np.asarray(o.position, float)[None])[0]
        if pn[0] < pt[0] and (best is None or pn[0] > best[3]):              # 경사 아래이면서 가장 가까운(u 큰) 이웃
            best = (o, pt[0], pt[1], pn[0], pn[1])
    return best


def build(scene, rules: dict, target=None, hyps=None, card=None, support_tol_mm: float = 20.0) -> tuple:
    L = scene.layers
    res = float(scene.plane["res_mm"])
    H, W = L["object"].shape
    cost = np.zeros((H, W), np.float32)
    hard = np.zeros((H, W), bool)
    meta = dict(used=[], skipped=[], valley=None, support=None)
    tidx = target_index(hyps, target) if hyps is not None else None
    other = L["object"] & (L["instance_id"] != (tidx + 1 if tidx is not None else -1))
    h = L["height_mm"]
    observed = _close(~L["unobserved"], float(rules.get("observed_close_mm", 6.0)), res)   # 1.2 m 에서 픽셀 간격(≈3 mm)이 2 mm 셀보다 넓어 생기는 결손 반점을 메움 [추측 6 mm]
    meta["observed"] = observed
    support = observed & ~L["object"] & (np.abs(np.nan_to_num(h, nan=1e9)) < support_tol_mm)   # 지지면(팔레트 상면·벨트): 기준면 근처의 빈 셀
    support = _close(support, float(rules.get("support_close_mm", 10.0)), res) & ~L["object"]   # 깊이 구멍(1 px 결손)을 메움 [추측 10 mm] — 발자국 전체가 관측돼야 하므로 구멍 하나가 후보를 다 죽인다
    meta["support"] = support
    obstacle = ~(L["free"] | (L["object"] & ~other))                          # free 도 타깃도 아닌 셀 = 장애물 (벽면·이웃·미관측)
    terms = list(rules.get("cost_terms", []))
    if not any(t.get("layer") == "unobserved" for t in terms):
        terms.append(dict(layer="unobserved", mode="soft", weight=1.0, decay_mm=15.0))   # v3 §7.8: unobserved 는 soft 1.0
    for t in terms:
        layer, mode, w = t.get("layer"), t.get("mode"), float(t.get("weight", 1.0))
        mask = None
        if layer == "wall":
            mask = L["wall"]
        elif layer == "liner":
            mask = L["liner"]
            if not mask.any():
                _once("liner", "liner 층이 비어 있어 cost_term liner 를 무시한다(liner_source='none')")
                meta["skipped"].append("liner"); continue
        elif layer == "object_other":
            mask = other
        elif layer == "unobserved":
            mask = ~observed
        elif layer in ("pallet_edge", "conveyor_edge"):
            mask = ~support                                                   # 지지면 밖(미관측·물체·높이 어긋남)은 놓을 수 없다
        elif layer == "height_mismatch":
            dev = np.where(support, np.abs(np.nan_to_num(h, nan=0.0)), 0.0)
            cost += (w * np.minimum(1.0, dev / max(float(t.get("decay_mm", 20.0)), 1e-6))).astype(np.float32)
            meta["used"].append(layer); continue
        elif layer == "clearance_uphill":
            run = run_length_uphill(obstacle) * res
            val = np.minimum(1.0, run / max(float(t.get("saturate_mm", 40.0)), 1e-6)).astype(np.float32)
            cost -= w * val
            meta["used"].append(layer); continue
        elif layer == "target_valley":
            vp = valley_partner(hyps, target, scene.plane) if hyps is not None else None
            if vp is None:
                meta["skipped"].append("target_valley"); continue
            o, ut, vt, un, vn = vp
            u_val = (ut + un) / 2 * 1000.0
            Lm = None                                                         # 골 띠의 v 범위 = 타깃 길이(맞춘 값 우선, 없으면 카드 평균) — generate·compute 와 같은 출처
            if card is not None and card.geometry["type"] == "cylinder":
                Lm = float((target.geometry.get("params_fit") or {}).get("length_mm") or np.mean(card.geometry["params"]["length_mm"]))
            sig = float(rules.get("valley_sigma_mm", 8.0))
            uu = (np.arange(W) + 0.5) * res
            g = np.exp(-((uu - u_val) ** 2) / (2 * sig ** 2)).astype(np.float32)
            band = np.ones(H, np.float32)
            if Lm is not None:
                vv = (np.arange(H) + 0.5) * res
                band = ((vv >= vt * 1000.0 - Lm / 2) & (vv <= vt * 1000.0 + Lm / 2)).astype(np.float32)
            cost -= w * band[:, None] * g[None, :]
            meta["valley"] = dict(partner=o.id, u_mm=u_val, v_mm=vt * 1000.0)
            meta["used"].append(layer); continue
        elif layer == "pattern_slot":
            slots = _pattern_slots(rules)
            if not slots:
                meta["skipped"].append("pattern_slot"); continue
            uu = (np.arange(W) + 0.5) * res
            vv = (np.arange(H) + 0.5) * res
            for k, s_ in enumerate(slots):
                g = np.exp(-(((uu - s_["u_mm"]) ** 2)[None, :] + ((vv - s_["v_mm"]) ** 2)[:, None]) / (2 * 30.0 ** 2)).astype(np.float32)
                cost -= w * g
            meta["slots"] = slots
            meta["used"].append(layer); continue
        else:
            meta["skipped"].append(str(layer)); continue
        if mode == "hard":
            hard |= _dilate(mask, float(t.get("margin_mm", 0.0)), res)
        elif mode == "soft":
            cost += _soft(mask, res, w, float(t.get("decay_mm", 15.0)))
        else:
            meta["skipped"].append(f"{layer}:{mode}"); continue
        meta["used"].append(layer)
    return cost, hard, meta


def _pattern_slots(rules: dict):
    """tasks/<name>/pattern.yaml → [{u_mm, v_mm}] (없으면 [])."""
    path = rules.get("path")
    if not path:
        return []
    from pathlib import Path
    import yaml
    f = Path(path) / "pattern.yaml"
    if not f.exists():
        _once("pattern", f"{f} 없음 → pattern_slot 무시")
        return []
    doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return [dict(u_mm=float(s["u_mm"]), v_mm=float(s["v_mm"])) for s in doc.get("slots", [])]
