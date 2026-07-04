"""Потайловая обработка панорам (ТЗ: до 10000×10000 px, ≤5 мин).

Панорама — мозаика из многих полей зрения; масштабирование целиком теряет
морфологию вкрапленников. Здесь режем изображение на перекрывающиеся плитки,
сегментируем/классифицируем каждую и собираем обратно:

* :func:`segment_tiled` — полноразмерная маска (центр каждой плитки, чтобы швы
  не двоились);
* :func:`classify_tiled` — вердикт сорта по агрегированию предсказаний нейросети
  на плитках (взвешивание по содержанию рудной фазы).
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .segmentation import to_gray_uint8


def iter_tiles(h: int, w: int, tile: int, overlap: int):
    """Итератор плиток: (y0,y1,x0,x1) — область чтения, (cy0,cy1,cx0,cx1) — центр для записи."""
    step = tile - overlap
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    for y0 in ys:
        y1 = min(y0 + tile, h)
        for x0 in xs:
            x1 = min(x0 + tile, w)
            cy0 = y0 + (overlap // 2 if y0 > 0 else 0)
            cx0 = x0 + (overlap // 2 if x0 > 0 else 0)
            cy1 = y1 - (overlap // 2 if y1 < h else 0)
            cx1 = x1 - (overlap // 2 if x1 < w else 0)
            yield (y0, y1, x0, x1, cy0, cy1, cx0, cx1)


def segment_tiled(
    image: np.ndarray,
    soft_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    n_classes: int,
    tile: int = 1024,
    overlap: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Полноразмерная сегментация панорамы плитками.

    Returns
    -------
    (mask, prob) : (H×W int32, H×W×n_classes float32)
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.int32)
    prob = np.zeros((h, w, n_classes), dtype=np.float32)

    for y0, y1, x0, x1, cy0, cy1, cx0, cx1 in iter_tiles(h, w, tile, overlap):
        m, p = soft_fn(img[y0:y1, x0:x1])
        m = np.clip(np.asarray(m, dtype=np.int32), 0, n_classes - 1)
        p = np.asarray(p, dtype=np.float32)
        if p.shape[2] < n_classes:
            pad = np.zeros((*p.shape[:2], n_classes - p.shape[2]), dtype=np.float32)
            p = np.concatenate([p, pad], axis=2)
        # Записываем только центральную часть плитки (без перекрытия) — без швов.
        mask[cy0:cy1, cx0:cx1] = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        prob[cy0:cy1, cx0:cx1] = p[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, :n_classes]
    return mask, prob


def classify_tiled(
    image: np.ndarray,
    *,
    tile: int = 1536,
    stride: int = 1536,
    max_tiles: int = 64,
) -> dict | None:
    """Классификация сорта панорамы по агрегированию нейросети на плитках.

    Плитки взвешиваются по содержанию рудной фазы (яркие пиксели): пустые
    силикатные поля почти не влияют на итоговый вердикт. Возвращает тот же формат,
    что и :func:`shlif.sort_model.predict_sort`, либо None (нет модели).
    """
    from . import sort_model
    from .sort_model import predict_sort

    if sort_model._lazy_load() is None:
        return None
    # рус-метка -> вердикт (после загрузки модели глобали заполнены).
    rus2verdict = {sort_model._RU_SHORT[e]: sort_model._VERDICT[e]
                   for e in sort_model._VERDICT if e in sort_model._RU_SHORT}

    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]

    coords = [(y, x) for y in range(0, max(h - tile, 0) + 1, stride)
              for x in range(0, max(w - tile, 0) + 1, stride)] or [(0, 0)]
    if len(coords) > max_tiles:  # равномерная подвыборка — держим бюджет времени
        idx = np.linspace(0, len(coords) - 1, max_tiles).astype(int)
        coords = [coords[i] for i in idx]

    keys, probs, weights = None, [], []
    for y, x in coords:
        crop = img[y:y + tile, x:x + tile]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        sp = predict_sort(crop)
        if sp is None:
            return None
        if keys is None:
            keys = list(sp["probs"].keys())
        gray = to_gray_uint8(crop)
        ore_frac = float((gray > np.percentile(gray, 92)).mean())  # доля ярких (рудных) пикселей
        probs.append(np.array([sp["probs"][k] for k in keys]))
        weights.append(max(ore_frac, 1e-3))

    if not probs:
        return None
    P = np.average(np.stack(probs), axis=0, weights=np.array(weights))
    i = int(P.argmax())
    return {
        "verdict": rus2verdict.get(keys[i], keys[i]),
        "confidence": round(float(P[i]), 3),
        "probs": {keys[k]: round(float(P[k]), 3) for k in range(len(keys))},
        "source": f"нейросеть EfficientNet-B0, агрегир. по {len(probs)} плиткам (F1≈0.87)",
    }
