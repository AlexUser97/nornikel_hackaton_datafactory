"""Инференс обученной сегментации талька (ТЗ task-3).

U-Net (segmentation-models-pytorch, ResNet-18, открытая лицензия), обученный
слабо-supervised на метках папок (оталькованные = тальк, рядовые/труднообогатимые
= силикаты; пиксельной разметки талька в датасете нет). Веса —
``weights/talc_unet.pt``. Инференс на CPU; если весов/smp нет — возвращает None,
и пайплайн использует классический текстурный детектор талька.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "weights" / "talc_unet.pt"
_INPUT = 384
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL = None
_THR = 0.10
_LOAD_FAILED = False


def _lazy_load(path: Path | str = _DEFAULT_PATH):
    global _MODEL, _THR, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    path = Path(path)
    if not path.exists():
        _LOAD_FAILED = True
        return None
    try:
        import segmentation_models_pytorch as smp
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _MODEL = model
        _THR = float(ckpt.get("talc_threshold", 0.10))
    except Exception:  # noqa: BLE001 — нет smp/torch/битые веса -> фолбэк
        _LOAD_FAILED = True
        _MODEL = None
    return _MODEL


def available(path: Path | str = _DEFAULT_PATH) -> bool:
    return _lazy_load(path) is not None


def talc_threshold() -> float:
    _lazy_load()
    return _THR


def predict_talc_mask(image_rgb: np.ndarray, prob_thr: float = 0.5,
                      path: Path | str = _DEFAULT_PATH) -> np.ndarray | None:
    """Маска талька (bool) в разрешении входного снимка, либо None (нет модели)."""
    model = _lazy_load(path)
    if model is None:
        return None
    import cv2
    import torch

    img = np.asarray(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    inp = cv2.resize(img, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    inp = (inp - _MEAN) / _STD
    x = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].numpy()
    mask = (p > prob_thr).astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
