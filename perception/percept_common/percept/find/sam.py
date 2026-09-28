"""MobileSAM 공용 로더 — lazy 싱글턴 (프로세스당 1회 로드; color_rule 과 ref_match 가 같은 인스턴스를 씀, 8 GB GPU)."""
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
_PREDICTOR = None
_KEY = None
_IMAGE_KEY = None


def get_predictor(checkpoint, model_type: str = "vit_t", device: Optional[str] = None):
    """SamPredictor 싱글턴. 같은 (checkpoint, model_type, device) 면 재사용, 다르면 새로 로드."""
    global _PREDICTOR, _KEY, _IMAGE_KEY
    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(Path(checkpoint).resolve()), model_type, dev)
    if _PREDICTOR is None or _KEY != key:
        if not Path(checkpoint).exists():
            from ..errors import NotAvailable
            raise NotAvailable(f"MobileSAM 체크포인트 없음: {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(dev)
        sam.eval()
        _PREDICTOR, _KEY, _IMAGE_KEY = SamPredictor(sam), key, None
        log.info("MobileSAM %s 로드 (장치 %s)", model_type, dev)
    return _PREDICTOR


def set_image(predictor, rgb, image_key=None) -> None:
    """set_image 는 프레임당 1회면 충분. (predictor, image_key) 가 직전과 같으면 건너뜀 (여러 Finder 가 같은 프레임을 볼 때).
    image_key 는 호출자가 프로세스 안에서 유일하게 만들어야 한다(replay: 파일 절대 경로, pipeline: id(pipeline)+스텝 번호+stamp). None 이면 항상 호출."""
    global _IMAGE_KEY
    key = None if image_key is None else (id(predictor), image_key)
    if key is not None and key == _IMAGE_KEY:
        return
    predictor.set_image(rgb)
    _IMAGE_KEY = key
