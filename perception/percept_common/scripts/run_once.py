#!/usr/bin/env python
"""카드 + 프레임 1장 → Hypotheses JSON (+ Affordances JSON, gripper·task 지정 시) + 시각화 png (마스크·축·후보).
  python scripts/run_once.py --card objects/ramen_bundle --adapter replay --index 31             # 데이터셋 목록(configs paths.datasets 순) 의 N번째
  python scripts/run_once.py --card objects/ramen_bundle --frame /home/nse/container_6d/data/obj_box/031.png
  옵션: --gripper grippers/dummy_hand --task tasks/extract_from_tilted_box --save-scene --out reports/run_once
종료 코드: 0 = 결과 있음, 2 = 빈 결과/전부 무효 (header.reason 에 사유 코드)."""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from percept.camera.replay import ReplayAdapter                       # noqa: E402
from percept.lift import m2mm                                          # noqa: E402
from percept.pipeline import build_default                            # noqa: E402
from percept.pose.base import invert_T                                 # noqa: E402
from percept.registry import load_card, load_config, load_gripper, load_task   # noqa: E402

COLORS = [(0, 255, 0), (0, 200, 255), (255, 140, 0), (255, 0, 200), (60, 120, 255), (200, 255, 0)]


def visualize(pipe, hyps, affs) -> np.ndarray:
    """정렬 왼쪽 영상 위에 마스크(반투명)·축 화살표·id·유효성, 후보 목록."""
    frame, cands, poses = pipe.last["frame"], pipe.last["cands"], pipe.last["poses"]
    img = frame.rgb.copy()
    K, T_cb = frame.K, invert_T(frame.T_base_cam)

    def proj(p_base):
        p = T_cb[:3, :3] @ np.asarray(p_base) + T_cb[:3, 3]
        return int(round(K[0, 0] * p[0] / p[2] + K[0, 2])), int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))

    for i, (c, o) in enumerate(zip(cands, hyps.items)):
        col = COLORS[i % len(COLORS)] if o.pose_valid else (120, 120, 120)
        img[c.mask] = (0.55 * img[c.mask] + 0.45 * np.array(col)).astype(np.uint8)
        x, y, w, h = c.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
        label = f"{o.id} {'ok' if o.pose_valid else o.invalid_reason} c={o.confidence:.2f} nb={len(o.neighbors)}"
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        p = poses[i]
        if p is not None and o.position[0] is not None:
            L = float(p.geometry_fit.get("length_m", 0.3))
            a = p.T_base_obj[:3, 0]
            c0 = np.asarray(o.position)
            cv2.arrowedLine(img, proj(c0 - a * L / 2), proj(c0 + a * L / 2), col, 2, tipLength=0.08)
            cv2.drawMarker(img, proj(c0), col, cv2.MARKER_CROSS, 12, 1)
    if affs is not None:
        for j, cand in enumerate(affs.candidates[:5]):
            cv2.putText(img, f"#{cand.rank} {cand.action} {cand.score:.2f}", (8, 20 + 14 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if affs.invalid_reason:
            cv2.putText(img, f"affordances: {affs.invalid_reason}", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True)
    ap.add_argument("--adapter", default="replay", choices=["replay", "synthetic"])
    ap.add_argument("--index", type=int, default=None, help="replay: 데이터셋 목록의 N번째 (0부터)")
    ap.add_argument("--frame", default=None, help="replay: 좌우 붙임 PNG 경로")
    ap.add_argument("--gripper", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(HERE / "reports" / "run_once"))
    ap.add_argument("--save-scene", action="store_true", help="SceneMap npz 저장 (Step 4)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    card = load_card(args.card, cfg)
    gripper = load_gripper(args.gripper, cfg["scene"]["res_mm"]) if args.gripper else None
    task = load_task(args.task) if args.task else None
    if args.adapter == "synthetic":
        raise SystemExit("synthetic 어댑터는 Step 4")
    if args.frame:
        adapter = ReplayAdapter(cfg, [args.frame])
    else:
        adapter = ReplayAdapter(cfg)
        if args.index is not None:
            adapter.i = args.index
    pipe = build_default(cfg, card, adapter, gripper, task, record=False)
    hyps, affs = pipe.step()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{card.class_name}_{hyps.header['scene_id'].replace('/', '_')}"
    (out / f"{stem}_hypotheses.json").write_text(hyps.to_json(indent=1), encoding="utf-8")
    if affs is not None:
        (out / f"{stem}_affordances.json").write_text(affs.to_json(indent=1), encoding="utf-8")
    if pipe.last.get("frame") is not None:
        cv2.imwrite(str(out / f"{stem}.png"), visualize(pipe, hyps, affs))
    if pipe.last.get("scene") is not None and args.save_scene:
        pipe.last["scene"].save(out / f"{stem}_scene.npz")
    print(json.dumps(dict(scene_id=hyps.header["scene_id"], reason=hyps.header.get("reason"), n=len(hyps.items), n_valid=len(hyps.valid),
                          items=[dict(id=o.id, valid=o.pose_valid, conf=round(o.confidence, 3), reason=o.invalid_reason,
                                      pos_mm=[round(float(v), 1) for v in m2mm(o.position)] if o.position[0] is not None else None,
                                      neighbors=[(n["id"], n["contact"]) for n in o.neighbors]) for o in hyps.items],
                          affordances=None if affs is None else dict(n=len(affs.candidates), invalid_reason=affs.invalid_reason),
                          timing_ms=hyps.header["timing_ms"], out=str(out / stem)), ensure_ascii=False, indent=1))
    return 0 if not hyps.header.get("reason") else 2


if __name__ == "__main__":
    sys.exit(main())
