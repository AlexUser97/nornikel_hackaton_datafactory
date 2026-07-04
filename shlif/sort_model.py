"""Инференс обученного классификатора сорта руды (ТЗ task-3).

Модель — EfficientNet-B0 (открытая лицензия, ImageNet-претрейн), дообученная на
папках-метках датасета (рядовые/труднообогатимые/оталькованные). Валидация:
F1 по типу срастаний (рядовая/труднообогатимая) ≈ 0.94, F1_macro ≈ 0.91,
AUC ≈ 0.97 (v2: EMA + mixup + TTA). Обучение — на выданном T4 (см.
training/train_classifier_v2.py). Веса кладутся в ``weights/sort_classifier.pt``.

Инференс работает на CPU (модель лёгкая). Если весов нет или torch недоступен —
функция возвращает None, и пайплайн использует классический эвристический вердикт.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "weights" / "sort_classifier.pt"

# Канонические классы модели -> русский вердикт сорта.
_VERDICT = {
    "ryadovye": "Рядовая руда",
    "trudnoobogatimye": "Труднообогатимая руда",
    "otalkovannye": "Оталькованная руда",
}
_RU_SHORT = {"ryadovye": "рядовая", "trudnoobogatimye": "труднообогатимая", "otalkovannye": "оталькованная"}

_MODEL = None
_CLASSES: list[str] = []
_TF = None
_LOAD_FAILED = False


def _lazy_load(path: Path | str = _DEFAULT_PATH):
    """Ленивая загрузка модели и препроцессинга (torch импортируется здесь)."""
    global _MODEL, _CLASSES, _TF, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    path = Path(path)
    if not path.exists():
        _LOAD_FAILED = True
        return None
    try:
        import torch
        from torchvision import models, transforms

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        _CLASSES = ckpt.get("classes", list(_VERDICT))
        model = models.efficientnet_b0(weights=None)
        import torch.nn as nn

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(_CLASSES))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _MODEL = model
        _TF = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    except Exception:  # noqa: BLE001 — нет torch/битые веса -> фолбэк на эвристику
        _LOAD_FAILED = True
        _MODEL = None
    return _MODEL


def available(path: Path | str = _DEFAULT_PATH) -> bool:
    """Есть ли обученная модель (и удалось ли её загрузить)."""
    return _lazy_load(path) is not None


def predict_sort(image_rgb: np.ndarray, path: Path | str = _DEFAULT_PATH, tta: bool = True) -> dict | None:
    """Предсказывает геолого-технологический сорт руды по снимку.

    ``tta`` — усреднение softmax по отражениям (оригинал + горизонт./вертик. флип):
    +1–2% к F1 практически бесплатно на CPU, повышает устойчивость к ориентации.

    Returns
    -------
    dict | None
        ``{"verdict", "confidence", "probs": {рус_класс: доля}, "source"}`` или
        None, если модель недоступна.
    """
    model = _lazy_load(path)
    if model is None:
        return None
    import torch

    img = np.asarray(image_rgb, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    views = [img]
    if tta:
        views += [img[:, ::-1].copy(), img[::-1, :].copy()]
    with torch.no_grad():
        probs = []
        for v in views:
            x = _TF(v).unsqueeze(0)
            probs.append(torch.softmax(model(x), dim=1)[0].numpy())
    prob = np.mean(probs, axis=0)
    idx = int(prob.argmax())
    cls = _CLASSES[idx]
    return {
        "verdict": _VERDICT.get(cls, cls),
        "confidence": round(float(prob[idx]), 3),
        "probs": {_RU_SHORT.get(_CLASSES[i], _CLASSES[i]): round(float(prob[i]), 3) for i in range(len(_CLASSES))},
        "source": "нейросеть EfficientNet-B0" + (" + TTA" if tta else ""),
    }
