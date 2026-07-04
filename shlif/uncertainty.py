"""Модуль неуверенности — главный дифференциатор проекта.

Идея (PROJECT_CONTEXT §2, §10): вместо того чтобы прятать ошибки модели, мы их
показываем. Прогоняем сегментацию по ансамблю лёгких пертурбаций входа и меряем
разброс ответа на каждом пикселе. Там, где решение неустойчиво (границы фаз,
«грязные» зоны), — низкая уверенность; такие зоны идут эксперту на проверку.

Так же честно ведёт себя sim-to-real: чем грязнее снимок, тем выше неуверенность,
а не ложная уверенность.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .simto_real import mild_perturbation


@dataclass
class EnsembleResult:
    """Результат ансамблевой сегментации."""

    mask: np.ndarray            # H×W int32 — итоговая маска (argmax вероятности класса)
    class_prob: np.ndarray      # H×W×K float32 — эмпирическая вероятность класса по ансамблю
    confidence: np.ndarray      # H×W float32 — уверенность = max вероятность класса
    uncertainty: np.ndarray     # H×W float32 0..1 — нормированная энтропия по ансамблю
    per_run_fractions: np.ndarray  # (n_runs, K) — доля каждого класса в каждом прогоне


def ensemble_segment(
    image: np.ndarray,
    soft_segment_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    n_classes: int,
    n_runs: int = 5,
    seed: int = 0,
) -> EnsembleResult:
    """Ансамблевая сегментация с оценкой попиксельной неуверенности.

    Прогон 0 — по чистому снимку, прогоны 1..n-1 — по лёгким пертурбациям.
    Геометрия не меняется, поэтому маски попиксельно сопоставимы. Итоговое
    распределение = среднее мягких вероятностей по прогонам: так учитывается и
    алеаторная неуверенность (размытость границ фаз), и эпистемическая (разброс
    ответа на пертурбациях). Неуверенность = энтропия усреднённого распределения.

    Parameters
    ----------
    image : np.ndarray
        Входной снимок (серый или RGB).
    soft_segment_fn : Callable
        Мягкая сегментация: изображение -> (маска H×W, вероятности H×W×K).
    n_classes : int
        Число классов K (фиксирует размерность распределения).
    n_runs : int
        Размер ансамбля.
    seed : int
        База зерна ГПСЧ для воспроизводимости.
    """
    n_runs = max(1, int(n_runs))
    prob_sum: np.ndarray | None = None
    per_run_fractions_list: list[np.ndarray] = []

    for i in range(n_runs):
        img_i = image if i == 0 else mild_perturbation(image, seed=seed + i)
        m, prob = soft_segment_fn(img_i)
        m = np.asarray(m, dtype=np.int32)
        prob = np.asarray(prob, dtype=np.float32)
        # Приводим ширину распределения к n_classes.
        if prob.shape[2] < n_classes:
            pad = np.zeros((*prob.shape[:2], n_classes - prob.shape[2]), dtype=np.float32)
            prob = np.concatenate([prob, pad], axis=2)
        elif prob.shape[2] > n_classes:
            prob = prob[:, :, :n_classes]

        prob_sum = prob if prob_sum is None else prob_sum + prob

        h, w = m.shape
        total = float(h * w)
        fr = np.array([float((m == k).sum()) / total for k in range(n_classes)], dtype=np.float32)
        per_run_fractions_list.append(fr)

    class_prob = (prob_sum / n_runs).astype(np.float32)
    h, w = class_prob.shape[:2]

    mask = np.argmax(class_prob, axis=2).astype(np.int32)
    confidence = class_prob.max(axis=2).astype(np.float32)

    # Нормированная энтропия усреднённого распределения (0 — уверенно, 1 — хаос).
    eps = 1e-9
    ent = -np.sum(class_prob * np.log(class_prob + eps), axis=2)
    uncertainty = (ent / np.log(n_classes)).astype(np.float32) if n_classes > 1 else np.zeros((h, w), np.float32)

    per_run_fractions = np.stack(per_run_fractions_list, axis=0)

    return EnsembleResult(
        mask=mask,
        class_prob=class_prob,
        confidence=confidence,
        uncertainty=uncertainty,
        per_run_fractions=per_run_fractions,
    )


def find_low_conf_zones(
    result: EnsembleResult,
    class_names: list[str],
    *,
    threshold: float = 0.60,
    min_area: int = 400,
    max_zones: int = 12,
) -> list[dict]:
    """Выделяет связные зоны низкой уверенности для ручной проверки экспертом.

    Для каждой зоны определяются две наиболее «спорящие» фазы — это и есть
    подсказка эксперту вида «бейнит vs перлит».

    Returns
    -------
    list[dict]
        Элементы формата контракта: ``{"bbox": [x,y,w,h], "confidence": float, "note": str}``.
    """
    low = (result.confidence < float(threshold)).astype(np.uint8)
    if low.sum() == 0:
        return []

    # Морфологически чистим одиночные пиксели.
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)

    zones: list[dict] = []
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        region = labels == lbl
        conf = float(result.confidence[region].mean())
        # Две наиболее вероятные (спорящие) фазы усреднённо по зоне.
        mean_prob = result.class_prob[region].mean(axis=0)
        order = np.argsort(mean_prob)[::-1]
        top2 = [class_names[i] for i in order[:2] if i < len(class_names)]
        note = " vs ".join(top2) if len(top2) == 2 else (top2[0] if top2 else "спорная зона")
        zones.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": round(conf, 3),
                "note": note,
                "area_px": int(area),
            }
        )

    # Сначала самые крупные/спорные зоны.
    zones.sort(key=lambda z: (z["confidence"], -z["area_px"]))
    return zones[:max_zones]
