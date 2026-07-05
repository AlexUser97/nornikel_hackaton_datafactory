"""Сегментатор на обученной модели + доменная адаптация (v2).

Сульфиды и магнетит сегментируются моделью, обученной на ВНЕШНЕМ датасете с РЕАЛЬНОЙ
пиксельной разметкой — LumenStone S2 (МГУ: Cu-Ni сульфидные руды пирротин/пентландит/
халькопирит + магнетит, отражённый свет; тот же тип руды, что аншлифы Норникеля). Так мы
компенсируем отсутствие пиксельной разметки в датасете кейса. Доменный разрыв (другой
микроскоп/экспозиция) снимаем **цветовой нормализацией Рейнхарда** к статистике LumenStone.

Тальк — нашим U-Net (обучен на клин-твинах «Области оталькования», без утечки чернил).

4 класса ТЗ:
    0 — тонкие срастания  (красный): сульфиды, сильно замещённые магнетитом/серой фазой;
    1 — обычные срастания (зелёный): крупные слабо замещённые сульфиды;
    2 — тальк             (синий):  зоны оталькования (обученный talc-U-Net);
    3 — вмещающая порода  (серый):  прочая нерудная матрица (без талька).

Если весов LumenStone/talc нет — вызывающий код откатывается на классический ore_petro.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FINE, ORDINARY, TALC, HOST = 0, 1, 2, 3
ORE_LUMEN_CLASS_NAMES = ["тонкие срастания", "обычные срастания", "тальк", "вмещающая порода"]
ORE_LUMEN_COLORS = [(214, 39, 40), (46, 204, 113), (52, 120, 219), (120, 128, 140)]
N_CLASSES = 4

_LUMEN_PATH = Path(__file__).resolve().parent.parent / "weights" / "lumenstone_seg.pt"
_INPUT = 512
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
# Целевая LAB-статистика LumenStone (среднее по train) — для Reinhard color transfer.
_LAB_MEAN = np.array([110.0, 128.1, 130.4], np.float32)
_LAB_STD = np.array([57.5, 3.3, 9.2], np.float32)

_MODEL = None
_LOAD_FAILED = False


def available(path: Path | str = _LUMEN_PATH) -> bool:
    return _lazy_load(path) is not None


def _lazy_load(path: Path | str = _LUMEN_PATH):
    global _MODEL, _LOAD_FAILED
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
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _MODEL = model
    except Exception:  # noqa: BLE001 — нет smp/torch/битые веса -> фолбэк
        _LOAD_FAILED = True
        _MODEL = None
    return _MODEL


def _reinhard(rgb: np.ndarray) -> np.ndarray:
    """Цветовая нормализация к статистике LumenStone (снимает доменный разрыв)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = lab.reshape(-1, 3).mean(0)
    s = lab.reshape(-1, 3).std(0)
    lab = (lab - m) * (_LAB_STD / np.maximum(s, 1e-3)) + _LAB_MEAN
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _classical_phases(rgb: np.ndarray) -> np.ndarray:
    """Классический фолбэк (нет весов LumenStone): 0 матрица / 1 сульфид / 2 магнетит
    по яркости и цвету отражённого света. Сульфиды — светлые/тёплые; магнетит —
    нейтрально-серый; матрица — тёмная."""
    from skimage.filters import threshold_multiotsu
    from .segmentation import to_gray_uint8

    gray = to_gray_uint8(rgb)
    try:
        thr = threshold_multiotsu(gray, classes=3)
        phase = np.digitize(gray, bins=thr)
    except ValueError:
        phase = np.zeros_like(gray, dtype=int)
    R, G, B = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    warm = R - B
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / np.maximum(mx, 1)
    is_neutral = (warm < 10) & (sat < 0.22)
    bright, mid = phase == 2, phase == 1
    out = np.zeros(gray.shape, np.uint8)                    # 0 матрица
    out[(mid & is_neutral) | (bright & is_neutral & (sat < 0.12))] = 2   # магнетит
    out[bright | (mid & ~is_neutral)] = 1                   # сульфид
    return out


def _predict_phases(rgb: np.ndarray) -> np.ndarray:
    """0 матрица / 1 сульфид / 2 магнетит. Обученная модель LumenStone (+ Reinhard
    цвет-нормализация) если веса есть, иначе классический фолбэк."""
    model = _lazy_load()
    if model is None:
        return _classical_phases(rgb)
    import torch

    h, w = rgb.shape[:2]
    norm = _reinhard(rgb)
    x = cv2.resize(norm, (_INPUT, _INPUT)).astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    with torch.no_grad():
        p = model(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()).argmax(1)[0].numpy()
    return cv2.resize(p.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _solidity(blob: np.ndarray, area: int) -> float:
    cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
    return area / hull if hull > 0 else 0.0


def ore_lumen_segment(image: np.ndarray, *, use_blue_annotation: bool = True):
    """Сегментация в 4 класса ТЗ. Returns (mask int32 H×W 0..3, prob H×W×4 float32)."""
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]

    phases = _predict_phases(rgb)                # 0 матрица / 1 сульфид / 2 магнетит
    sulfide = phases == 1
    magnetite = phases == 2

    mask = np.full((h, w), HOST, dtype=np.int32)  # по умолчанию — вмещающая порода

    # --- Тальк (обученный U-Net на клин-твинах), только В НЕРУДНОЙ части ---
    try:
        from .talc_model import predict_talc_mask
        tm = predict_talc_mask(rgb)
    except Exception:  # noqa: BLE001
        tm = None
    if tm is not None:
        talc = np.asarray(tm, dtype=bool) & ~sulfide
        k = max(9, (int(round(min(h, w) * 0.02)) | 1))
        talc = cv2.morphologyEx(talc.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        mask[talc] = TALC

    # --- Сульфидные срастания: обычные vs тонкие по ЗАМЕЩЁННОСТИ магнетитом + размеру ---
    mag_near = cv2.dilate(magnetite.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ore_u8 = cv2.morphologyEx(sulfide.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ore_u8, connectivity=8)
    big = 2e-4 * 12 * h * w
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        blob = labels == lbl
        repl = float((blob & mag_near).sum()) / area           # доля контакта с магнетитом
        ordinary = area >= big and _solidity(blob, area) >= 0.55 and repl < 0.35
        mask[blob] = ORDINARY if ordinary else FINE

    prob = np.zeros((h, w, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        prob[:, :, c] = (mask == c)
    return mask, prob
