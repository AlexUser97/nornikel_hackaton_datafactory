"""Синтетические данные для демо и тестов, пока не выдан реальный датасет.

Генерируем правдоподобную микроструктуру (зёрна-Вороного + фазы по яркости +
тёмные границы зёрен + шум) с ground truth. Сквозной пайплайн и тесты можно гонять
без реальных данных.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def synthetic_microstructure(
    h: int = 512,
    w: int = 640,
    n_grains: int = 140,
    phase_intensities: tuple[int, ...] = (65, 105, 185),
    phase_probs: tuple[float, ...] = (0.30, 0.25, 0.45),
    seed: int = 0,
    return_labels: bool = False,
):
    """Синтетический шлиф: зёрна Вороного, три фазы по яркости, границы, шум.

    Parameters
    ----------
    return_labels : bool
        Если True — вернуть ещё и истинную маску фаз (ground truth) для оценки
        метрик (IoU/Хаусдорф/F1/AUC) без размеченного датасета.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        RGB uint8 H×W×3, либо ``(rgb, true_mask)`` где true_mask — H×W int32
        индекс истинной фазы (по возрастанию яркости, как у сегментатора).
    """
    rng = np.random.default_rng(seed)

    # Центры зёрен и их разбиение (диаграмма Вороного через ближайший центр).
    pts = rng.uniform([0, 0], [h, w], size=(n_grains, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    _, nearest = cKDTree(pts).query(grid, k=1)
    grain_id = nearest.reshape(h, w)

    # Каждому зерну — фаза (по вероятностям) и базовая яркость + джиттер.
    phase_of_grain = rng.choice(len(phase_intensities), size=n_grains, p=_norm(phase_probs))
    base = np.array(phase_intensities, dtype=np.float32)[phase_of_grain]
    jitter = rng.normal(0, 8, size=n_grains).astype(np.float32)
    grain_val = base + jitter
    img = grain_val[grain_id]

    # Истинная маска фаз (по индексу яркости прототипа) — до наложения границ/шума.
    order = np.argsort(np.asarray(phase_intensities))  # индекс -> ранг яркости
    rank_of_phase = np.argsort(order)
    true_mask = rank_of_phase[phase_of_grain][grain_id].astype(np.int32)

    # Тёмные границы зёрен (там, где меняется id соседа).
    gx = np.abs(np.diff(grain_id, axis=1, prepend=grain_id[:, :1]))
    gy = np.abs(np.diff(grain_id, axis=0, prepend=grain_id[:1, :]))
    boundaries = ((gx + gy) > 0).astype(np.uint8)
    boundaries = cv2.dilate(boundaries, np.ones((2, 2), np.uint8))
    img[boundaries > 0] = 35

    # Текстура + сенсорный шум.
    img = img + rng.normal(0, 5, size=img.shape).astype(np.float32)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), sigmaX=0.6)

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return (rgb, true_mask) if return_labels else rgb


def _norm(p: tuple[float, ...]) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    return a / a.sum()
