"""색 규칙 Finder — detect_container.py v5 이식 (:58-246). 문턱은 전부 카드 find.color_hint 에서 읽는다 (코드 상수 없음).

흐름: strict 색 마스크 → loose 색 후보 상자 → [trim: strict 색이 있는 행으로 다듬기] → [split: 시차 기반 세로 분할] → MobileSAM(점+상자)
→ 상시 절단 → strict 비율 → verdict → 큰 마스크 우선 IoU 중복 제거 → (y0, x0) 정렬. 값·순서를 원본과 같게 유지해야 회귀가 맞는다.
"""
import logging
from typing import List, Optional

import cv2
import numpy as np

from ..lift import m2mm
from ..registry import Card
from ..types import Candidate, Frame
from . import sam
from .base import Finder

log = logging.getLogger(__name__)


def _t(v):
    return tuple(int(x) for x in v)


def color_mask(bgr: np.ndarray, ranges) -> np.ndarray:
    """HSV 범위 목록의 합집합 (uint8 0/255)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = None
    for lo, hi in ranges:
        r = cv2.inRange(hsv, _t(lo), _t(hi))
        m = r if m is None else (m | r)
    return m


def loose_candidates(bgr, ranges, morph: Optional[dict], min_area_frac: float):
    """red_candidates(:69-86): 느슨한 색 → 모폴로지 → 연결성분 상자 (y0, x0) 순."""
    mask = color_mask(bgr, ranges)
    if morph:
        if morph.get("close"):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, _t(morph["close"])))
        if morph.get("open"):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, _t(morph["open"])))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area_frac * H * W:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, mask


def trim_by_strict(sred: np.ndarray, box, min_row_px: int, min_rows: int):
    """trim_by_strict_red(:89-96)."""
    x0, y0, x1, y1 = box
    rows = sred[y0:y1, x0:x1].sum(axis=1)
    good = np.where(rows > min_row_px)[0]
    if good.size < min_rows:
        return None
    return (x0, y0 + int(good[0]), x1, y0 + int(good[-1]) + 1)


def local_diam_px(disp, depth, fx, x0, x1, ya, yb, width_px, baseline_mm, diam_mm, len_diam_ratio, min_samples):
    """local_diam_px(:100-106): 구간의 시차 중앙값으로 그 높이의 물체 두께(px). 시차가 없으면(비스테레오) 깊이로 같은 식.
    유효 표본 ≤ min_samples 면 폭/길이비 폴백."""
    if disp is not None and baseline_mm is not None:
        band = disp[ya:yb, x0:x1]
        v = band[band > 0]
        if v.size > min_samples:
            return diam_mm * float(np.median(v)) / baseline_mm, "depth"
    elif depth is not None:
        band = depth[ya:yb, x0:x1]
        v = band[np.isfinite(band) & (band > 0)]
        if v.size > min_samples:
            return diam_mm * fx / float(m2mm(np.median(v))), "depth"         # D_px = D_mm·fx/Z_mm (짝수 표본에선 disp 중앙값과 ≤0.04 px 차)
    return width_px / len_diam_ratio, "폭"


def brightness_profile(bgr, cmask, box, pcfg: dict):
    """brightness_profile(:109-123)."""
    x0, y0, x1, y1 = box
    H = y1 - y0
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    prof = np.full(H, np.nan, np.float32)
    for i, y in enumerate(range(y0, y1)):
        sel = cmask[y, x0:x1] > 0
        if sel.sum() > pcfg["min_sel"]:
            prof[i] = v[y, x0:x1][sel].mean()
    good = ~np.isnan(prof)
    if good.sum() < H * pcfg["min_valid_frac"]:
        return None
    idx = np.arange(H, dtype=np.float32)
    prof = np.interp(idx, idx[good], prof[good]).astype(np.float32)
    return cv2.GaussianBlur(prof.reshape(-1, 1), (1, int(pcfg["blur_ksize"])), 0).ravel()


def split_adaptive(bgr, cmask, disp, depth, fx, box, baseline_mm, diam_mm, len_diam_ratio, s: dict):
    """split_adaptive(:126-153): 위에서 아래로 그 높이의 두께만큼 내려가며 밝기 골짜기에서 절단."""
    x0, y0, x1, y1 = box
    H, W = y1 - y0, x1 - x0
    prof = brightness_profile(bgr, cmask, box, s["profile"])
    if prof is None or H < s["min_height_px"]:
        return [box]
    cuts, y, guard = [0], 0, 0
    while guard < s["max_cuts"]:
        guard += 1
        probe_end = min(y + max(s["probe_min_px"], int(H * s["probe_frac"])), H)
        ldiam, _src = local_diam_px(disp, depth, fx, x0, x1, y0 + y, y0 + probe_end, W, baseline_mm, diam_mm, len_diam_ratio, s["depth_min_samples"])
        if ldiam < s["min_diam_px"] or H - y < s["remain_factor"] * ldiam:
            break
        c = y + ldiam
        w = s["window_frac"] * ldiam
        lo = int(max(y + s["lo_frac"] * ldiam, c - w))
        hi = int(min(H - s["hi_margin_px"], c + w))
        if lo >= hi:
            break
        cut = lo + int(np.argmin(prof[lo:hi]))
        cuts.append(cut)
        y = cut
    cuts.append(H)
    out = [(x0, y0 + a, x1, y0 + b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= s["min_band_px"]]
    return out if out else [box]


def mask_verdict(m, img_area, strict_frac, v: dict):
    """mask_verdict(:169-184). 통과면 None, 아니면 탈락 사유."""
    a = int(m.sum())
    if a < v["area_min_frac"] * img_area:
        return f"너무 작음({a / img_area * 100:.1f}%)"
    if a > v["area_max_frac"] * img_area:
        return f"너무 큼({a / img_area * 100:.1f}%)"
    ys, xs = np.nonzero(m)
    (w, h) = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[1]
    if min(w, h) < 1:
        return "형태 없음"
    r = max(w, h) / max(min(w, h), 1.0)
    if r < v["min_elongation"]:
        return f"길쭉하지 않음({r:.2f})"
    if strict_frac < v["min_strict_frac"]:
        return f"색 비율 낮음({strict_frac * 100:.0f}%)"
    return None


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return 0.0 if inter == 0 else inter / np.logical_or(a, b).sum()


class ColorRuleFinder(Finder):
    name = "color_rule"

    def __init__(self, cfg: dict, predictor=None):
        self.cfg = cfg
        self.iou_thr = float(cfg["fuse"]["iou_within_finder"])
        self._predictor = predictor
        self.last = {}          # 디버그: prompts, rejects, counts

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = sam.get_predictor(self.cfg["paths"]["mobile_sam_ckpt"])
        return self._predictor

    def find(self, frame: Frame, card: Card) -> List[Candidate]:
        ch = card.find["color_hint"]
        L = frame.rgb
        disp = frame.extra.get("disparity")
        baseline_mm = frame.extra.get("baseline_mm")
        fx = float(frame.K[0, 0])
        g = card.geometry
        diam_mm = 2.0 * float(g["params"]["radius_mm"]) if g["type"] == "cylinder" else None
        len_ratio = (float(ch["split"]["fallback_length_mm"]) / diam_mm) if (diam_mm and ch.get("split")) else None   # LEN_DIAM_RATIO = 650/140

        sred = color_mask(L, ch["strict"]) > 0
        cand, cmask = loose_candidates(L, ch["loose"], ch.get("morph"), float(ch["min_area_frac"]))
        prompts = []
        for b in cand:
            tb = b
            if ch.get("trim"):
                tb = trim_by_strict(sred, b, int(ch["trim"]["min_row_px"]), int(ch["trim"]["min_rows"]))
                if tb is None:
                    continue
            if ch.get("split"):                                    # registry 가 cylinder 카드에서만 허용
                prompts += split_adaptive(L, cmask, disp, frame.depth, fx, tb, baseline_mm, diam_mm, len_ratio, ch["split"])
            else:
                prompts.append(tb)

        dets, rejects = [], []
        if prompts:
            pred = self.predictor
            sam.set_image(pred, cv2.cvtColor(L, cv2.COLOR_BGR2RGB), frame.extra.get("image_key"))
            img_area = L.shape[0] * L.shape[1]
            Himg, Wimg = L.shape[:2]
            clip_cfg, vcfg, pcfg = ch["clip"], ch["verdict"], ch.get("sam_prompt", {})
            for b in prompts:
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                kw = dict(multimask_output=bool(pcfg.get("multimask", False)))
                if pcfg.get("point", "box_center") == "box_center":
                    kw.update(point_coords=np.array([[cx, cy]]), point_labels=np.array([1]))
                if pcfg.get("box", True):
                    kw["box"] = np.array(b)
                masks, scores, _ = pred.predict(**kw)
                best = int(np.argmax(scores)) if kw["multimask_output"] else 0
                m = masks[best].astype(bool)
                mh = b[3] - b[1]                                           # 상시 절단 (:224-229)
                ey0, ey1 = max(0, int(b[1] - clip_cfg["y_frac_box"] * mh)), min(Himg, int(b[3] + clip_cfg["y_frac_box"] * mh))
                ex0, ex1 = max(0, int(b[0] - clip_cfg["x_frac_img"] * Wimg)), min(Wimg, int(b[2] + clip_cfg["x_frac_img"] * Wimg))
                clip = np.zeros_like(m)
                clip[ey0:ey1, ex0:ex1] = True
                m = m & clip
                strict_frac = float(np.logical_and(m, sred).sum()) / max(int(m.sum()), 1)
                verdict = mask_verdict(m, img_area, strict_frac, vcfg)
                if verdict is None:
                    dets.append((m, float(scores[best]), b))
                else:
                    rejects.append(verdict.split("(")[0])

        dets.sort(key=lambda t: -int(t[0].sum()))
        keep = []
        for m, s, b in dets:
            if all(iou(m, k[0]) < self.iou_thr for k in keep):
                keep.append((m, s, b))
        keep.sort(key=lambda t: (t[2][1], t[2][0]))
        self.last = dict(n_cand=len(cand), n_prompts=len(prompts), rejects=rejects, prompts=prompts)
        return [Candidate(mask=m, bbox=(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])), score=s, finder=self.name, instance_id=k)
                for k, (m, s, b) in enumerate(keep)]                       # bbox 는 계약대로 (x, y, w, h)
