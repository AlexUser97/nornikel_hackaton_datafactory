"""Классическая CV-сегментация — честный офлайн-бейзлайн без ML-весов.

Две независимые задачи:
* ``segment_phases`` — деление микроструктуры на K фаз по яркости (multi-Otsu).
  Даёт попиксельную маску индекса фазы — вход для расчёта фазовых долей.
* ``watershed_grains`` — разделение зёрен (distance-transform + watershed) для
  оценки размера зерна и балла ASTM.

Это фолбэк и опорная точка (PROJECT_CONTEXT §9): работает мгновенно на CPU,
не требует данных, интернета и GPU. Сюда же позже втыкается SAM/U-Net — форма
выходной маски совпадает, обвязка не меняется.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Любой вход -> одноканальный uint8 (снимок шлифа физически серый)."""
    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def segment_phases(image: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Сегментация микроструктуры на ``n_classes`` фаз по яркости.

    Используем многоуровневый порог Оцу: интенсивность травленого шлифа
    статистически разделяет составляющие (тёмный перлит/бейнит vs светлый феррит).
    Индекс 0 — самая тёмная фаза, старший индекс — самая светлая (совпадает с
    порядком имён в :data:`shlif.config.DEFAULT_CLASS_NAMES`).

    Returns
    -------
    np.ndarray
        Маска H×W типа int32 со значениями 0..n_classes-1.
    """
    gray = to_gray_uint8(image)
    # Лёгкое сглаживание — убрать зернистость сенсора, сохранив границы фаз.
    gray_s = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=5)

    # Вырожденные случаи: почти однородная картинка -> один класс.
    if gray_s.std() < 3.0 or np.unique(gray_s).size < n_classes:
        return np.zeros(gray.shape, dtype=np.int32)

    try:
        thresholds = threshold_multiotsu(gray_s, classes=n_classes)
    except ValueError:
        # threshold_multiotsu не смог развести классы -> падаем на 2 класса.
        try:
            thresholds = threshold_multiotsu(gray_s, classes=2)
        except ValueError:
            return np.zeros(gray.shape, dtype=np.int32)

    mask = np.digitize(gray_s, bins=thresholds).astype(np.int32)
    return mask


def segment_phases_soft(
    image: np.ndarray, n_classes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Мягкая (вероятностная) версия фазовой сегментации.

    Помимо жёсткой маски возвращает попиксельное распределение вероятностей по
    фазам. Вероятность строится по близости яркости пикселя к «прототипам» фаз
    (средним яркостям классов) с температурой ~ межклассовый зазор. У границ фаз
    и в размытых/зашумлённых зонах распределение размывается — это и есть честная
    алеаторная неуверенность, которую усиливает ансамбль в :mod:`shlif.uncertainty`.

    Returns
    -------
    (mask, prob) : (H×W int32, H×W×K float32)
    """
    gray = to_gray_uint8(image)
    gray_s = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=5).astype(np.float32)

    mask = segment_phases(image, n_classes=n_classes)
    k_eff = int(mask.max()) + 1

    # Прототип класса — средняя яркость его пикселей (робастнее, чем середины порогов).
    centers = np.array(
        [gray_s[mask == k].mean() if np.any(mask == k) else 0.0 for k in range(k_eff)],
        dtype=np.float32,
    )
    if k_eff < 2:
        prob = np.ones((*gray.shape, max(n_classes, 1)), dtype=np.float32)
        prob /= prob.shape[2]
        return mask, prob

    # Температура softmax ~ доля среднего зазора между соседними прототипами.
    # Узкая (0.35·зазор): внутренности фаз уверенны (~0), граничные/размытые
    # пиксели — спорны. Так неуверенность не «заливает» весь кадр.
    gaps = np.diff(np.sort(centers))
    tau = max(float(np.mean(gaps)) * 0.35, 1.0)

    # Softmax по -(I-μ_k)^2 / (2τ^2).
    dist2 = (gray_s[..., None] - centers[None, None, :]) ** 2
    logits = -dist2 / (2.0 * tau * tau)
    logits -= logits.max(axis=2, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=2, keepdims=True)

    # Добиваем до n_classes нулевыми столбцами, если Otsu схлопнул класс.
    if prob.shape[2] < n_classes:
        pad = np.zeros((*gray.shape, n_classes - prob.shape[2]), dtype=np.float32)
        prob = np.concatenate([prob, pad], axis=2)

    return mask, prob.astype(np.float32)


def watershed_grains(
    image: np.ndarray,
    *,
    min_distance: int = 8,
    drop_border: bool = True,
) -> tuple[np.ndarray, int]:
    """Разделение зёрен для оценки их размера (метод границ + watershed).

    Зёрна на травленом шлифе разделены сетью границ. Границу ловим по двум
    признакам сразу: высокий морфологический градиент (перепад яркости) И тёмные
    линии (границы обычно темнее). «Тело» (комплемент границ) размечаем на зёрна,
    а слипшиеся соседи разделяем маркерами из локальных максимумов дистанционного
    преобразования. Зёрна у рамки кадра отбрасываются (обрезаны — их площадь
    занижена, стандартная практика ASTM E112).

    Returns
    -------
    (labels, n_grains)
        ``labels`` — карта меток зёрен (0 = фон/границы), ``n_grains`` — число
        полноценных (не пограничных) зёрен, учтённых в статистике.
    """
    gray = cv2.GaussianBlur(to_gray_uint8(image), (0, 0), sigmaX=0.8)

    # Сеть границ = сильный градиент ∪ тёмные линии.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, ridge_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray < np.percentile(gray, 8)).astype(np.uint8) * 255
    ridge = cv2.max(ridge_grad, dark)

    interior = (ridge == 0).astype(np.uint8)
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if interior.sum() < 50:
        return np.zeros(gray.shape, dtype=np.int32), 0

    distance = ndi.distance_transform_edt(interior)
    coords = peak_local_max(
        distance, min_distance=min_distance, labels=interior.astype(bool), exclude_border=False
    )
    if coords.shape[0] == 0:
        return np.zeros(gray.shape, dtype=np.int32), 0

    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    markers, _ = ndi.label(markers > 0)

    labels = watershed(-distance, markers, mask=interior.astype(bool))

    if drop_border:
        border_ids = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
        border_ids.discard(0)
        for bid in border_ids:
            labels[labels == bid] = 0

    n_grains = int(np.unique(labels[labels > 0]).size)
    return labels.astype(np.int32), n_grains
