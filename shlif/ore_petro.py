"""Классический бейзлайн под задачу ТЗ (task-3) на реальных аншлифах.

Ровно 3 класса по ТЗ (v1):
    0 — тонкие срастания  (красный):   сульфиды, СИЛЬНО замещённые серой/тёмной фазой;
    1 — обычные срастания (зелёный):   крупные слабо замещённые сульфиды;
    2 — тальк             (синий):     ВСЯ нерудная фракция — вмещающая силикатная
                                       матрица (талькосодержащая порода с тёмными
                                       вкраплениями), т.е. всё, что НЕ сульфидное срастание.

Логика: сульфиды — светлые/тёплые в отражённом свете; их классифицируем на обычные/тонкие
по размеру и «замещённости» серой фазой. Всё остальное (силикаты, магнетит, тёмная матрица
с тальком) в v1 относим к ТАЛЬКУ (нерудная фракция) — так размечает геолог зоны оталькования
(см. service_folder/1.jpg: вмещающая порода с чёрными пятнами обведена как тальк).

Это честный офлайн-бейзлайн (без обучения). Модель сегментации можно подключить через
``analyze(..., soft_segment_fn=...)``.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from .segmentation import to_gray_uint8

# Индексы классов и цвета (схема ТЗ: 3 класса, тальк = вся нерудная часть).
FINE, ORDINARY, TALC = 0, 1, 2
ORE_PETRO_CLASS_NAMES = ["тонкие срастания", "обычные срастания", "тальк"]
ORE_PETRO_COLORS = [(214, 39, 40), (46, 204, 113), (52, 120, 219)]
N_CLASSES = 3

# Минимальная площадь вкрапленника (доля кадра) — база для порога «крупного».
_MIN_ORE_FRAC = 2e-4


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
    """Сегментация аншлифа в 3 класса ТЗ: тонкие/обычные срастания + тальк (нерудная фракция).

    Returns
    -------
    (mask, prob) : (H×W int32 со значениями 0..2, H×W×3 float32 one-hot)
    """
    rgb = image if np.asarray(image).ndim == 3 else cv2.cvtColor(to_gray_uint8(image), cv2.COLOR_GRAY2RGB)
    rgb = np.asarray(rgb)
    gray = cv2.bilateralFilter(to_gray_uint8(image), d=5, sigmaColor=40, sigmaSpace=5)
    h, w = gray.shape

    # --- Фазы по яркости: тёмное(0) / среднее(1) / светлое(2) ---
    try:
        thr = threshold_multiotsu(gray, classes=3)
        phase = np.digitize(gray, bins=thr)  # 0 dark .. 2 bright
    except ValueError:
        phase = np.zeros_like(gray, dtype=int)

    # --- Сульфиды: светлые/тёплые в отражённом свете; серая нейтральная фаза (магнетит)
    # не относится к сульфидным срастаниям. Всё, что НЕ сульфид, в v1 -> ТАЛЬК. ---
    R, G, B = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    warm = (R - B)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / np.maximum(mx, 1)
    is_neutral = (warm < 10) & (sat < 0.22)
    bright, mid = phase == 2, phase == 1
    sulfide = bright | (mid & ~is_neutral)          # яркое/тёплое = сульфидное срастание

    # ВСЁ = тальк (нерудная фракция) по умолчанию; сульфиды затем перекрашиваем.
    mask = np.full((h, w), TALC, dtype=np.int32)

    # --- Классификация сульфидных срастаний: обычные (крупные, слабо замещённые) vs
    # тонкие (мелкие / сильно замещённые серой фазой) — по размеру и солидности. ---
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

    prob = np.zeros((h, w, N_CLASSES), dtype=np.float32)
    for k in range(N_CLASSES):
        prob[:, :, k] = (mask == k)
    return mask, prob
