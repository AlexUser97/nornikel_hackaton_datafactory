"""Количественные метрики паспорта: чистая арифметика над масками.

Данные для обучения не нужны (PROJECT_CONTEXT §10):
* доля фазы = площадь пикселей класса / общая площадь (+ ДИ из разброса ансамбля);
* размер зерна = эквивалентный диаметр по калибровке µm/px;
* балл ASTM (E112) = из плотности зёрен на единицу площади;
* дефекты = тёмные округлые включения/поры (простой детектор для стретча).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .segmentation import to_gray_uint8


# Коэффициент 95% ДИ для нормального приближения.
_Z95 = 1.959964


def phase_fractions(
    mask: np.ndarray,
    class_names: list[str],
    per_run_fractions: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Доли фаз в % площади с доверительным интервалом.

    ``value`` берётся из итоговой маски (совпадает с тем, что видно в оверлее),
    ``ci`` — половина 95% ДИ. Источник ДИ:
    * если есть ансамбль (>1 прогона) — из разброса долей между прогонами;
    * иначе — биномиальное приближение по числу «эффективных» пикселей.

    Returns
    -------
    dict
        ``{name: {"value": доля 0..1, "ci": полуширина ДИ 0..1}}``.
    """
    total = float(mask.size)
    out: dict[str, dict[str, float]] = {}
    n_runs = 0 if per_run_fractions is None else per_run_fractions.shape[0]

    for k, name in enumerate(class_names):
        value = float((mask == k).sum()) / total if total else 0.0

        if n_runs > 1:
            ci = float(_Z95 * per_run_fractions[:, k].std(ddof=1))
        else:
            # Биномиальный ДИ; занижаем эффективный N из-за пространственной
            # корреляции пикселей (иначе ДИ нечестно мал). Патч ~ 8×8 px.
            n_eff = max(total / 64.0, 1.0)
            ci = float(_Z95 * math.sqrt(max(value * (1.0 - value), 0.0) / n_eff))

        out[name] = {"value": round(value, 4), "ci": round(min(ci, value, 1.0 - value) if 0 < value < 1 else ci, 4)}

    return out


def grain_metrics(
    grain_labels: np.ndarray,
    n_grains: int,
    scale_um_per_px: float,
) -> dict[str, float | int | None]:
    """Средний размер зерна (µm) и балл ASTM по разделённым зёрнам.

    Плотность зёрен на площадь -> балл G по ASTM E112:
        G = 3.321928 · log10(N_A) − 2.954,
    где N_A — число зёрен на мм² (при 1×).

    Returns
    -------
    dict
        ``grain_size_um`` (эквивалентный диаметр), ``astm_number``,
        ``n_grains``, ``grain_size_std_um``. При отсутствии зёрен — None.
    """
    if n_grains <= 0 or grain_labels is None or grain_labels.max() == 0:
        return {
            "grain_size_um": None,
            "astm_number": None,
            "n_grains": 0,
            "grain_size_std_um": None,
            "reliable": False,
            "note": "зёрна не выделены",
        }

    # Площади зёрен в пикселях (метка 0 = фон/границы отбрасываем).
    counts = np.bincount(grain_labels.ravel())
    areas_px = counts[1:][counts[1:] > 0].astype(np.float64)
    if areas_px.size == 0:
        return {"grain_size_um": None, "astm_number": None, "n_grains": 0, "grain_size_std_um": None}

    # Отбрасываем «осколки» watershed (< 0.15 медианы) — артефакты пересегментации,
    # иначе разброс размера зерна раздувается нефизично.
    if areas_px.size >= 3:
        med = float(np.median(areas_px))
        areas_px = areas_px[areas_px >= 0.15 * med]
    n_grains = int(areas_px.size)

    scale = float(scale_um_per_px) if scale_um_per_px and scale_um_per_px > 0 else 1.0
    areas_um2 = areas_px * (scale ** 2)

    # Эквивалентный круговой диаметр каждого зерна, затем среднее.
    diam_um = 2.0 * np.sqrt(areas_um2 / math.pi)
    grain_size_um = float(diam_um.mean())
    grain_size_std_um = float(diam_um.std(ddof=1)) if diam_um.size > 1 else 0.0

    # N_A: зёрен на мм². Каждое зерно занимает ~mean_area_um2, плотность = 1/area.
    mean_area_um2 = float(areas_um2.mean())
    n_a_per_mm2 = 1.0e6 / mean_area_um2 if mean_area_um2 > 0 else None
    astm = (3.321928 * math.log10(n_a_per_mm2) - 2.954) if n_a_per_mm2 else None

    # Проверка надёжности: балл ASTM E112 физически лежит примерно в 0..14.
    # Значение выше -> метод, скорее всего, «считает» ламели/шум, а не зёрна
    # (мелкодисперсная/пластинчатая структура, зашумлённый скан). Честно помечаем.
    reliable = True
    note = ""
    if astm is not None and astm > 13.5:
        reliable = False
        note = "структура слишком мелкая/пластинчатая — оценка зерна ненадёжна"
    elif n_grains > 2500:
        reliable = False
        note = "очень много объектов в поле — вероятен шум/растр, оценка зерна ненадёжна"

    return {
        "grain_size_um": round(grain_size_um, 2),
        "astm_number": round(astm, 1) if astm is not None else None,
        "n_grains": int(n_grains),
        "grain_size_std_um": round(grain_size_std_um, 2),
        "reliable": reliable,
        "note": note,
    }


def detect_defects(
    image: np.ndarray,
    scale_um_per_px: float,
    *,
    max_defects: int = 30,
) -> list[dict]:
    """Простой детектор пор/включений: тёмные округлые компактные объекты.

    Не претендует на полноту (стретч, PROJECT_CONTEXT §8) — цель показать флаги
    дефектов в паспорте. Округлость отсекает царапины и границы зёрен.
    """
    gray = to_gray_uint8(image)
    # Очень тёмные области относительно общего фона.
    thr = max(int(np.percentile(gray, 2)), 10)
    dark = (gray <= thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = float(scale_um_per_px) if scale_um_per_px and scale_um_per_px > 0 else 1.0

    defects: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:  # шум
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perim * perim)
        if circularity < 0.55:  # не круглое -> вероятно, не пора
            continue
        x, y, w, h = cv2.boundingRect(c)
        d_um = 2.0 * math.sqrt(area / math.pi) * scale
        defects.append(
            {
                "type": "пора/включение",
                "bbox": [int(x), int(y), int(w), int(h)],
                "size_um": round(d_um, 2),
                "circularity": round(float(circularity), 2),
            }
        )

    defects.sort(key=lambda d: -d["bbox"][2] * d["bbox"][3])
    return defects[:max_defects]
