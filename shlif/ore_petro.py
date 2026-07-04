"""Классический бейзлайн под задачу ТЗ (task-3) на реальных аншлифах.

Выдаёт маску в цветовой схеме ТЗ:
    0 — тонкие срастания  (красный):   рудные вкрапленники, СИЛЬНО замещённые серой фазой;
    1 — обычные срастания (зелёный):   крупные слабо замещённые рудные вкрапленники;
    2 — тальк             (синий):     тёмные текстурные области зон оталькования;
    3 — оксиды (магнетит) (серый):     нейтрально-серая рудная фаза;
    4 — вмещающая матрица (тёмный):    силикаты/вмещающая порода без талька.

Логика фаз по яркости отражённого света: рудные сульфиды — самые светлые, серая фаза
(магнетит, замещение) — средне-серая, силикаты — тёмные. «Замещённость» вкрапленника
оцениваем по доле серой фазы вокруг/внутри него; тальк — по текстуре в нерудных зонах,
с калибровкой по синей разметке, если она есть (см. :mod:`shlif.annotations`).

Это честный офлайн-бейзлайн (без обучения). Точка ожидания данных — заменить эту
функцию обученной моделью через ``analyze(..., soft_segment_fn=...)``.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_multiotsu

from .annotations import extract_blue_regions
from .segmentation import to_gray_uint8

# Индексы классов и цвета (совпадают со схемой ТЗ).
FINE, ORDINARY, TALC, OXIDE, GANGUE = 0, 1, 2, 3, 4
ORE_PETRO_CLASS_NAMES = ["тонкие срастания", "обычные срастания", "тальк", "оксиды (магнетит)", "вмещающая матрица"]
ORE_PETRO_COLORS = [(214, 39, 40), (46, 204, 113), (52, 120, 219), (150, 160, 175), (40, 40, 55)]
N_CLASSES = 5

# Минимальная площадь вкрапленника (доля кадра) — база для порога «крупного».
_MIN_ORE_FRAC = 2e-4
# Порог доли талька (raw score U-Net) -> оталькованная руда; выше — срастания НЕ ищем.
_TALC_BEARING = 0.15


def _solidity(blob: np.ndarray, area: int) -> float:
    """Плотность формы = площадь / площадь выпуклой оболочки (1 — компактная)."""
    cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
    return area / hull_area if hull_area > 0 else 0.0


def _remove_small(binary: np.ndarray, min_area: float) -> np.ndarray:
    """Убирает связные компоненты площадью меньше ``min_area`` (шумовые крохи)."""
    b = binary.astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = np.zeros(b.shape, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = True
    return out


def _local_std(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Локальное СКО яркости (мера текстуры)."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (ksize, ksize))
    sq = cv2.blur(g * g, (ksize, ksize))
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def detect_talc(gray: np.ndarray, gangue_mask: np.ndarray) -> np.ndarray:
    """Тальк = тёмные текстурные области в нерудных (силикатных) зонах.

    Гладкие силикаты дают низкое локальное СКО; тальк — мелкую «рябь» (повышенное
    СКО). Порог АБСОЛЮТНЫЙ (медиана+разброс по нерудной зоне), поэтому на снимках
    без талька доля выходит низкой, а не фиксированный квантиль. Оставляем только
    связные области (тальк образует зоны, а не одиночные пиксели).
    """
    if gangue_mask.sum() < 200:
        return np.zeros(gray.shape, dtype=bool)
    std = _local_std(gray, 9)
    vals = std[gangue_mask]
    med, sd = float(np.median(vals)), float(np.std(vals))
    thr = max(med + 1.5 * sd, 6.0)  # абсолютный порог текстуры
    cand = (gangue_mask & (std > thr)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Убираем мелкие пятна — тальк образует протяжённые области.
    min_area = int(3e-4 * gray.size)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)


def ore_petro_segment(
    image: np.ndarray, *, use_blue_annotation: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Сегментация аншлифа в классы ТЗ (тонкие/обычные срастания, тальк, фон).

    Returns
    -------
    (mask, prob) : (H×W int32 со значениями 0..3, H×W×4 float32 one-hot)
    """
    rgb = image if np.asarray(image).ndim == 3 else cv2.cvtColor(to_gray_uint8(image), cv2.COLOR_GRAY2RGB)
    rgb = np.asarray(rgb)
    gray = cv2.bilateralFilter(to_gray_uint8(image), d=5, sigmaColor=40, sigmaSpace=5)
    h, w = gray.shape

    # --- Фазы по яркости: матрица(0) / серая(1) / светлая(2) ---
    try:
        thr = threshold_multiotsu(gray, classes=3)
        phase = np.digitize(gray, bins=thr)  # 0 dark .. 2 bright
    except ValueError:
        phase = np.zeros_like(gray, dtype=int)

    # --- Цвет: сульфиды тёплые/яркие, оксиды (магнетит) — нейтрально-серые ---
    # (учёт цвета пикселей, а не только яркости — рекомендация организаторов).
    R, G, B = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    warm = (R - B)  # тёплый оттенок сульфидов > 0
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / np.maximum(mx, 1)  # насыщенность
    is_neutral = (warm < 10) & (sat < 0.22)

    bright, mid, dark = phase == 2, phase == 1, phase == 0
    sulfide = bright | (mid & ~is_neutral)          # тёплое/яркое = сульфид
    oxide = (mid & is_neutral) | (bright & is_neutral & (sat < 0.12))  # нейтрально-серое = оксид
    gangue = dark

    mask = np.full((h, w), GANGUE, dtype=np.int32)
    mask[oxide] = OXIDE

    # --- Сульфидные срастания: классифицируем ПО ВСЕЙ маске (обычные/тонкие) ---
    ore_u8 = cv2.morphologyEx(sulfide.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ore_u8, connectivity=8)
    big_area = _MIN_ORE_FRAC * 12 * h * w   # ~крупный вкрапленник
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        blob = labels == lbl
        ordinary = area >= big_area and _solidity(blob, area) >= 0.55
        mask[blob] = ORDINARY if ordinary else FINE

    # --- Тальк: обученный U-Net -> КОГЕРЕНТНЫЕ ЗОНЫ оталькования (как размечает геолог,
    # см. service_folder/1.jpg): замыкание разрывов синего контура + заполнение. Зона
    # красится ЦЕЛИКОМ поверх матрицы/оксидов, но ЯРКИЕ сульфидные ВКРАПИНКИ внутри
    # сохраняются (важны по ТЗ). Порог/gate по яркости НЕ применяем — авторитет по
    # тальку — обученная модель, иначе зоны средней яркости терялись бы. ---
    from .talc_model import predict_talc_mask
    tm = predict_talc_mask(rgb)
    if tm is not None:
        talc_raw = np.asarray(tm, dtype=bool)
    else:
        talc_raw = np.asarray(detect_talc(gray, gangue), dtype=bool)
        if use_blue_annotation:
            region, frac = extract_blue_regions(rgb)
            if frac > 0.002:
                talc_raw = talc_raw | (region > 0)

    if talc_raw.any():
        k = max(9, (int(round(min(h, w) * 0.02)) | 1))
        tz = cv2.morphologyEx(talc_raw.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        talc_zone = binary_fill_holes(tz > 0)
        talc_zone = _remove_small(talc_zone, 0.002 * h * w)   # убрать шумовые крохи
        bright_ore = bright & (~is_neutral)                   # однозначно рудные вкрапинки
        keep_speck = talc_zone & bright_ore & np.isin(mask, [ORDINARY, FINE])
        prev = mask.copy()
        mask[talc_zone] = TALC
        mask[keep_speck] = prev[keep_speck]                   # вернуть сульфиды внутри зоны

    prob = np.zeros((h, w, N_CLASSES), dtype=np.float32)
    for k in range(N_CLASSES):
        prob[:, :, k] = (mask == k)
    return mask, prob
