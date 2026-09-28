#!/usr/bin/env python
"""연속 실행 (Step 2): grab → pipeline.step → JSON 스트림(JSONL, stdout 또는 --out). stable_frames 게이트(연속 N 프레임 중심 이동 < tol → 'locked').
  python scripts/run_live.py --card objects/ramen_bundle --source zmq [--gripper .. --task ..] [--rate 5] [--n 0] [--out live.jsonl] [--max-retries 5]
  python scripts/run_live.py --card objects/ramen_bundle --source dir --dir /home/nse/container_6d/data/obj_box --n 10
camera_unavailable(타임아웃·디코딩 실패)이면 backoff 로 재시도, --max-retries 넘으면 종료 코드 3. ROS 발행은 ros/publish_hypothesis.py(Step 3)."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from percept.camera.stereo_head import StereoHeadAdapter              # noqa: E402
from percept.errors import NotAvailable                               # noqa: E402
from percept.pipeline import build_default                           # noqa: E402
from percept.registry import load_card, load_config, load_gripper, load_task   # noqa: E402

log = logging.getLogger("run_live")


def make_adapter(cfg, source, dirs, repeat):
    if source == "zmq":
        return StereoHeadAdapter(cfg, source="zmq")
    if source == "realsense":                                   # D435i (pyrealsense2 없으면 NotAvailable)
        from percept.camera.realsense import RealSenseAdapter
        return RealSenseAdapter(cfg)
    items = [d for d in dirs for _ in range(repeat)]
    return StereoHeadAdapter(cfg, source="dir", items=items)


def run(cfg, card, adapter, gripper=None, task=None, n=0, rate_hz=5.0, out=None, max_retries=5, record=True):
    """n=0 이면 무한(소스가 끝나면 종료). 반환: 처리한 프레임 수. 종료 코드는 main."""
    pipe = build_default(cfg, card, adapter, gripper, task, continuous=True, record=record)
    fp = open(out, "a", encoding="utf-8") if out else sys.stdout
    period, retries, done = (1.0 / rate_hz if rate_hz > 0 else 0.0), 0, 0
    try:
        while n <= 0 or done < n:
            t0 = time.time()
            try:
                hyps, affs = pipe.step()
            except StopIteration:
                break
            if hyps.header.get("reason") == "camera_unavailable":
                retries += 1
                if retries > max_retries:
                    log.error("camera_unavailable %d회 → 종료", retries)
                    return done, 3
                wait = min(2.0 ** retries * 0.5, 10.0)
                log.warning("camera_unavailable (%d/%d) — %.1f s 후 재시도", retries, max_retries, wait)
                time.sleep(wait)
                continue
            retries = 0
            locked = [o.id for o in hyps.items if o.extras.get("stability", {}).get("locked")]
            line = dict(hypotheses=hyps.to_dict(), affordances=None if affs is None else json.loads(affs.to_json()), locked=locked)
            fp.write(json.dumps(line, ensure_ascii=False) + "\n"); fp.flush()
            done += 1
            dt = time.time() - t0
            if period > dt:
                time.sleep(period - dt)
    finally:
        if out:
            fp.close()
        adapter.close()
    return done, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True); ap.add_argument("--gripper"); ap.add_argument("--task")
    ap.add_argument("--source", default="zmq", choices=["zmq", "dir", "realsense"]); ap.add_argument("--dir", action="append", default=[], help="source=dir 일 때 PNG 폴더/파일 (반복 가능)")
    ap.add_argument("--repeat", type=int, default=1, help="source=dir: 같은 목록을 반복(안정 프레임 검증용)")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--out", default=None); ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--config", default=None); ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    card = load_card(args.card, cfg)
    gripper = load_gripper(args.gripper, cfg["scene"]["res_mm"]) if args.gripper else None
    task = load_task(args.task) if args.task else None
    try:
        adapter = make_adapter(cfg, args.source, args.dir, args.repeat)
    except NotAvailable as e:
        log.error("camera_unavailable: %s", e); return 3
    done, code = run(cfg, card, adapter, gripper, task, args.n, args.rate, args.out, args.max_retries, record=not args.no_record)
    log.info("%d 프레임 처리, 종료 코드 %d", done, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
