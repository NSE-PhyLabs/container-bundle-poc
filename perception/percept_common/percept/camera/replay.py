"""데이터셋 재생 어댑터: 좌우 붙임 PNG 목록을 순서대로 Frame 으로 (source='replay')."""
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2

from ..types import Frame
from .base import CameraAdapter
from .stereo_head import StereoProcessor


class ReplayAdapter(CameraAdapter):
    name = "replay"

    def __init__(self, cfg: dict, items: Optional[Iterable] = None, cam_cfg: Optional[dict] = None):
        """items: 파일/폴더 경로들. None 이면 configs/default.yaml paths.datasets 전부(순서 유지). 폴더는 *.png 정렬."""
        self.proc = StereoProcessor(cfg, cam_cfg)
        self.items: List[Tuple[str, Path]] = []
        if items is None:
            items = list(cfg["paths"]["datasets"].values())
        for it in items:
            p = Path(it)
            if p.is_dir():
                self.items += [(p.name, f) for f in sorted(p.glob("*.png"))]
            else:
                self.items.append((p.parent.name, p))
        self.i = 0

    def __len__(self):
        return len(self.items)

    def grab(self) -> Frame:
        if self.i >= len(self.items):
            raise StopIteration
        set_name, path = self.items[self.i]
        self.i += 1
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise IOError(f"읽기 실패: {path}")
        return self.proc.build_frame(bgr, self.name, extra=dict(set=set_name, stem=path.stem, path=str(path), image_key=str(path.resolve()),
                                                                scene_id=f"{set_name}/{path.stem}"))
