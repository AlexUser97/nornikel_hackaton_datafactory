"""Разбор экспертной разметки — синих линий на снимках аншлифов.

По ТЗ (task-3) единственная разметка в датасете — синие линии: в рядовых рудах
ими обведены рудные вкрапленники, в оталькованных/труднообогатимых — области
оталькования (тальк). Функции ниже извлекают синюю разметку в маску, чтобы
использовать её как (слабый) ground truth для талька/ROI: обучение, оценка
метрик и проверка.
"""
from __future__ import annotations

import cv2
import numpy as np


def blue_line_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Бинарная маска пикселей синей разметки (яркая синяя линия/стрелка/текст)."""
    img = np.asarray(image_rgb)
    if img.ndim == 2:
        return np.zeros(img.shape, dtype=np.uint8)
    r, g, b = img[:, :, 0].astype(np.int16), img[:, :, 1].astype(np.int16), img[:, :, 2].astype(np.int16)
    # Насыщенный синий: B заметно выше R и G, и сам по себе яркий.
    mask = (b > 110) & (b - r > 45) & (b - g > 35)
    return mask.astype(np.uint8) * 255


def extract_blue_regions(image_rgb: np.ndarray, min_area_frac: float = 1e-4) -> tuple[np.ndarray, float]:
    """Извлекает области, ОБВЕДЁННЫЕ синим контуром (замкнутые аннотации).

    Замыкаем разрывы линии морфологией, ищем контуры и заполняем те, что образуют
    замкнутую область достаточного размера. Возвращает (маска области, доля площади).

    Note
    ----
    Разметка неоднородна: часть снимков содержит замкнутые контуры (регион),
    часть — стрелки-указатели с подписью «Тальк» (не регион). Для указателей
    замкнутых областей не будет — доля вернётся ~0; такие снимки для площадной
    оценки талька не годятся (честно отражает качество разметки).
    """
    line = blue_line_mask(image_rgb)
    if line.sum() == 0:
        return np.zeros(line.shape, dtype=np.uint8), 0.0

    h, w = line.shape
    # Замыкаем небольшие разрывы контура.
    k = max(3, int(round(min(h, w) * 0.01)) | 1)
    closed = cv2.morphologyEx(line, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros((h, w), dtype=np.uint8)
    min_area = min_area_frac * h * w
    for c in contours:
        area = cv2.contourArea(c)
        # Замкнутость: заполненная площадь заметно больше площади самой линии.
        if area < min_area:
            continue
        fill = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(fill, [c], -1, 255, thickness=cv2.FILLED)
        if fill.sum() > 2.0 * (line[fill > 0] > 0).sum() * 255:  # внутри больше «тела», чем линии
            region = cv2.bitwise_or(region, fill)

    # Убираем сами линии из региона (оставляем внутренность).
    region[line > 0] = 255
    frac = float((region > 0).mean())
    return region, round(frac, 4)


def strip_annotation(image_rgb: np.ndarray) -> np.ndarray:
    """Убирает синюю разметку со снимка (инпейнтинг), чтобы она не влияла на сегментацию."""
    img = np.asarray(image_rgb)
    if img.ndim != 3:
        return img
    line = blue_line_mask(img)
    if line.sum() == 0:
        return img
    dilated = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, dilated, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
