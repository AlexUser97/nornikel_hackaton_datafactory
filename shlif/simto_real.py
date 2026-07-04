"""Синтетическая «загрязнялка» данных для sim-to-real.

Две роли:
1. Демо-фича: показать, что на «грязных» реальных данных модель не врёт уверенно,
   а честно поднимает неуверенность (см. PROJECT_CONTEXT §8, §11.7).
2. Движок ансамбля неуверенности: лёгкие фотометрические пертурбации, по разбросу
   ответов на которых мы и меряем уверенность (см. :mod:`shlif.uncertainty`).

Важно: геометрию НЕ трогаем (только яркость/контраст/шум/блюр/JPEG),
чтобы маски разных прогонов были попиксельно сопоставимы.
"""
from __future__ import annotations

import cv2
import numpy as np


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Приводим вход к одноканальному uint8 (снимок шлифа — по сути серый)."""
    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def corrupt(
    image: np.ndarray,
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gaussian_noise: float = 0.0,
    blur_sigma: float = 0.0,
    jpeg_quality: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Применяет управляемые искажения к серому снимку.

    Parameters
    ----------
    image : np.ndarray
        Входной снимок (H×W или H×W×3), uint8.
    brightness : float
        Аддитивный сдвиг яркости в единицах uint8 (-255..255).
    contrast : float
        Множитель контраста вокруг серого центра 127 (1.0 = без изменений).
    gaussian_noise : float
        СКО аддитивного гауссова шума в единицах uint8.
    blur_sigma : float
        Sigma гауссова размытия (0 — без размытия).
    jpeg_quality : int | None
        Если задан (1..100) — прогоняем через JPEG-компрессию (артефакты сжатия).
    seed : int | None
        Зерно ГПСЧ для воспроизводимости шума.

    Returns
    -------
    np.ndarray
        Искажённый серый uint8-снимок той же формы H×W.
    """
    rng = np.random.default_rng(seed)
    img = _as_uint8_gray(image).astype(np.float32)

    # Контраст вокруг середины + яркость.
    img = (img - 127.0) * float(contrast) + 127.0 + float(brightness)

    if gaussian_noise > 0:
        img = img + rng.normal(0.0, gaussian_noise, size=img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if blur_sigma and blur_sigma > 0:
        # ksize=0 -> вычисляется из sigma автоматически.
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(blur_sigma))

    if jpeg_quality is not None:
        q = int(np.clip(jpeg_quality, 1, 100))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    return img


def mild_perturbation(image: np.ndarray, seed: int) -> np.ndarray:
    """Лёгкая случайная пертурбация для одного прогона ансамбля неуверенности.

    Амплитуды намеренно небольшие: цель — не «сломать» снимок, а прощупать,
    насколько устойчиво решение сегментатора у границ фаз.
    """
    rng = np.random.default_rng(seed)
    return corrupt(
        image,
        brightness=float(rng.uniform(-12, 12)),
        contrast=float(rng.uniform(0.9, 1.1)),
        gaussian_noise=float(rng.uniform(2.0, 8.0)),
        blur_sigma=float(rng.uniform(0.0, 0.8)),
        seed=seed,
    )


# Пресеты для демо-слайдера sim-to-real в UI.
DEMO_PRESETS: dict[str, dict] = {
    "Чисто (как есть)": {},
    "Лёгкое загрязнение": dict(contrast=0.9, gaussian_noise=6.0, blur_sigma=0.6),
    "Среднее загрязнение": dict(
        brightness=-15, contrast=0.8, gaussian_noise=12.0, blur_sigma=1.0, jpeg_quality=60
    ),
    "Сильное загрязнение": dict(
        brightness=-25, contrast=0.7, gaussian_noise=20.0, blur_sigma=1.6, jpeg_quality=35
    ),
}

# Человекочитаемое описание предполагаемого загрязнения (для подсказки в UI).
DEMO_PRESET_DESC: dict[str, str] = {
    "Чисто (как есть)": "Снимок без искусственных искажений.",
    "Лёгкое загрязнение": "Небольшой шум сенсора + лёгкая расфокусировка (σ≈0.6 px).",
    "Среднее загрязнение": "Затемнение, снижен контраст, шум, расфокус (σ≈1.0 px), JPEG-артефакты.",
    "Сильное загрязнение": "Сильное затемнение и потеря контраста, крупный шум, размытие (σ≈1.6 px), "
                           "жёсткое JPEG-сжатие — имитация плохого скана/съёмки.",
}
