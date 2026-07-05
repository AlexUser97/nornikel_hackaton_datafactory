"""Единый контракт ``analyze()`` — сердце сервиса (PROJECT_CONTEXT §10).

Снимок оптической микроскопии -> количественный «паспорт образца» стабильной формы.
Каждый модуль обёрнут в try/except: если отваливается сегмент/зёрна/дефекты — паспорт
всё равно собирается (принцип модульных фолбэков, §9).
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from .config import (
    DEFAULT_CLASS_NAMES,
    ENSEMBLE_RUNS,
    LOW_CONF_THRESHOLD,
    MIN_LOW_CONF_AREA,
    TALC_THRESHOLD,
)
from .metrics import detect_defects, grain_metrics, phase_fractions
from .ore_classification import classify_ore, conclusion_text
from .segmentation import segment_phases_soft, to_gray_uint8, watershed_grains
from .uncertainty import EnsembleResult, ensemble_segment, find_low_conf_zones


def analyze(
    image: np.ndarray,
    scale_um_per_px: float,
    *,
    class_names: Optional[list[str]] = None,
    n_runs: int = ENSEMBLE_RUNS,
    seed: int = 0,
    with_defects: bool = True,
    max_dim: int = 1400,
    denoise: bool = True,
    astm_applicable: bool = True,
    material_profile: Optional[str] = None,
    soft_segment_fn: Optional[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
    class_colors: Optional[list] = None,
    tile: int = 0,
) -> dict:
    """Анализирует шлиф и возвращает паспорт образца.

    Parameters
    ----------
    image : np.ndarray
        SEM/OM-снимок (серый H×W или RGB H×W×3), uint8.
    scale_um_per_px : float
        Калибровка масштаба (µm на пиксель) — нужна для размеров в микронах.
    class_names : list[str] | None
        Имена фаз по индексу (по возрастанию яркости). По умолчанию 3 фазы.
    n_runs : int
        Размер ансамбля для оценки неуверенности.
    seed : int
        Зерно ГПСЧ (воспроизводимость).
    with_defects : bool
        Включить простой детектор пор/включений.
    max_dim : int
        Рабочее разрешение: снимки крупнее прореживаются по большей стороне (с
        корректировкой µm/px). Ускоряет анализ, уменьшает PDF и подавляет
        пиксельный шум скан/растра реальных снимков.
    denoise : bool
        Лёгкое подавление шума (медиана) — против растра/зерна сенсора на реальных
        снимках, иначе зёрна пересегментируются на шумовые «осколки».

    Returns
    -------
    dict
        Паспорт стабильной формы (см. PROJECT_CONTEXT §10).
    """
    class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
    n_classes = len(class_names)

    # Тайлинг: для панорам держим более высокое рабочее разрешение (морфология
    # вкрапленников важна, ТЗ), обычные снимки прореживаем как раньше.
    work_max = max(max_dim, 4000) if tile else max_dim
    image, scale_um_per_px = _preprocess(image, float(scale_um_per_px), work_max, denoise)
    gray = to_gray_uint8(image)

    # --- Сегментация фаз + ансамблевая неуверенность ------------------------
    # По умолчанию — классический CV-сегментатор (без весов). Точка ожидания
    # данных: сюда можно передать обёртку предобученной модели с открытой лицензией
    # (SAM Apache-2.0 / U-Net MIT), возвращающую (маска H×W, вероятности H×W×K).
    seg_fn = soft_segment_fn or (lambda img: segment_phases_soft(img, n_classes=n_classes))

    if tile:
        # Панорама: полноразмерная маска плитками, без ансамбля (бюджет времени).
        from .tiling import segment_tiled
        tmask, tprob = segment_tiled(image, seg_fn, n_classes=n_classes, tile=int(tile))
        ens = _ensemble_from_prob(tmask, tprob, n_classes)
    else:
        ens = ensemble_segment(image, seg_fn, n_classes=n_classes, n_runs=n_runs, seed=seed)
    mask = ens.mask

    fractions = phase_fractions(mask, class_names, ens.per_run_fractions)

    # --- Зоны низкой уверенности -------------------------------------------
    low_conf_zones = find_low_conf_zones(
        ens, class_names, threshold=LOW_CONF_THRESHOLD, min_area=MIN_LOW_CONF_AREA
    )

    # --- Размер зерна + ASTM (изолированный фолбэк) -------------------------
    grain = {"grain_size_um": None, "astm_number": None, "n_grains": 0, "grain_size_std_um": None}
    try:
        labels, n_grains = watershed_grains(gray)
        grain = grain_metrics(labels, n_grains, scale_um_per_px)
    except Exception:  # noqa: BLE001 — паспорт важнее одной метрики
        pass

    # Балл ASTM E112 применим только к сталям/сплавам (отчёт v2, §6.3). Для руд и
    # прочих материалов оставляем гранулометрию (размер зёрен), но балл не выводим.
    grain["astm_applicable"] = bool(astm_applicable)
    if not astm_applicable:
        grain["astm_number"] = None

    # --- Дефекты (опц., изолированный фолбэк) ------------------------------
    defects: list[dict] = []
    if with_defects:
        try:
            defects = detect_defects(gray, scale_um_per_px)
        except Exception:  # noqa: BLE001
            defects = []

    # --- Классификация сорта по ТЗ (4 класса) --------------------------------
    # Сорт (рядовая/труднообогатимая/оталькованная) даёт обученный КЛАССИФИКАТОР —
    # он выучил экспертную логику. Доли фаз (сульфиды/тальк/срастания) показываем как
    # метрики ТЗ. Тальк — отдельный класс (зоны оталькования), вмещающая порода — прочее.
    obych_idx = next((i for i, n in enumerate(class_names) if "обыч" in n.lower()), None)
    tonk_idx = next((i for i, n in enumerate(class_names) if "тонк" in n.lower()), None)
    talc_idx = next((i for i, n in enumerate(class_names) if "тальк" in n.lower()), None)
    host_idx = next((i for i, n in enumerate(class_names) if "вмещ" in n.lower() or "порода" in n.lower()), None)
    ore_idx = [i for i, n in enumerate(class_names) if "сраст" in n.lower()]
    ore_area = np.isin(mask, ore_idx) if ore_idx else np.zeros_like(mask, bool)
    talc_frac = float((mask == talc_idx).mean()) if talc_idx is not None else 0.0

    ore_class = classify_ore(fractions)
    if ore_class is not None:
        model_img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        ore_class["talc_seg_frac"] = round(talc_frac, 4)
        # классификатор сорта (3 сорта), при панораме — агрегирование по плиткам
        sp, p_ot, p_ry, p_tr = None, 0.0, 0.0, 0.0
        try:
            from .sort_model import predict_sort
            if tile:
                from .tiling import classify_tiled
                sp = classify_tiled(model_img)
            else:
                sp = predict_sort(model_img)
            pr = (sp or {}).get("probs", {})
            p_ot = pr.get("оталькованная", 0.0)
            p_ry, p_tr = pr.get("рядовая", 0.0), pr.get("труднообогатимая", 0.0)
        except Exception:  # noqa: BLE001 — нет весов/torch -> фолбэк на доли фаз
            pass

        if sp is not None and (p_ot + p_ry + p_tr) > 0:
            ore_class["classifier_confidence"] = sp.get("confidence")
            ore_class["needs_review"] = bool(sp.get("needs_review"))
            ore_class["model_probs"] = {"рядовая": round(p_ry, 3), "труднообогатимая": round(p_tr, 3),
                                        "оталькованная": round(p_ot, 3)}
            verdict, pv = max([("Оталькованная руда", p_ot), ("Рядовая руда", p_ry),
                               ("Труднообогатимая руда", p_tr)], key=lambda t: t[1])
            ore_class["verdict"] = verdict
            ore_class["talc_bearing"] = verdict == "Оталькованная руда"
            ore_class["source"] = sp["source"]
            ore_class["model_confidence"] = round(sp.get("confidence", pv), 3)
            ore_class["rule"] = (f"классификатор сорта: {verdict.split()[0].lower()} (p={pv:.2f}); "
                                 f"тальк {talc_frac*100:.0f}%, сульфиды {float(ore_area.mean())*100:.0f}%")
        else:
            # фолбэк без классификатора: по преобладанию срастаний
            ob = float((mask == obych_idx).mean()) if obych_idx is not None else 0.0
            tn = float((mask == tonk_idx).mean()) if tonk_idx is not None else 0.0
            if talc_frac > TALC_THRESHOLD:
                ore_class["verdict"] = "Оталькованная руда"; ore_class["talc_bearing"] = True
                ore_class["rule"] = f"классификатора нет; тальк {talc_frac*100:.0f}% > {TALC_THRESHOLD*100:.0f}%"
            else:
                ore_class["verdict"] = "Рядовая руда" if ob >= tn else "Труднообогатимая руда"
                ore_class["talc_bearing"] = False
                ore_class["rule"] = f"классификатора нет; преобладание {'обычных' if ob >= tn else 'тонких'} срастаний"
            ore_class["source"] = "эвристика по долям фаз"
    conclusion = conclusion_text(fractions, ore_class, ens.uncertainty.mean()) if ore_class else ""

    # --- Доли площадей (ТЗ): общая доля сульфидов, тальк, срастания, вмещающая порода ----
    proportions = None
    if ore_class is not None:
        proportions = {
            "общая доля сульфидов": round(float(ore_area.mean()), 4),
            "доля талька": round(talc_frac, 4),
            "обычные срастания": round(float((mask == obych_idx).mean()), 4) if obych_idx is not None else 0.0,
            "тонкие срастания": round(float((mask == tonk_idx).mean()), 4) if tonk_idx is not None else 0.0,
            "вмещающая порода": round(float((mask == host_idx).mean()), 4) if host_idx is not None else 0.0,
        }

    # --- Карта сортов руды (для панорам): цветовое выделение 3 типов + % ------
    # Прямой ответ на постановку организаторов: «выделить цветами 3 типа руды и их
    # процентное содержание». Строится из уже посчитанной маски (дёшево).
    ore_type_map = None
    if tile and ore_class is not None:
        try:
            # Основной путь — карта тем же классификатором-ансамблем, что и вердикт
            # (согласована с ним). Фолбэк — правило по маске, если классификатора нет.
            from .tiling import ore_type_map_classified, ore_type_map_from_mask
            model_img2 = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            ore_type_map = ore_type_map_classified(model_img2, talc_threshold=TALC_THRESHOLD)
            if ore_type_map is None:
                ore_type_map = ore_type_map_from_mask(mask, class_names)
        except Exception:  # noqa: BLE001 — карта сортов не критична для паспорта
            ore_type_map = None

    work_rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return {
        "mask": mask,
        "ore_type_map": ore_type_map,
        "class_names": class_names,
        "image": work_rgb.astype(np.uint8),  # рабочий снимок (совпадает по размеру с mask)
        "phase_fractions": fractions,
        "grain_size_um": grain["grain_size_um"],
        "astm_number": grain["astm_number"],
        "material_profile": material_profile,
        "astm_applicable": bool(astm_applicable),
        "ore_class": ore_class,
        "conclusion": conclusion,
        "proportions": proportions,
        "colors": list(class_colors) if class_colors else None,
        "uncertainty_map": ens.uncertainty,
        "low_conf_zones": low_conf_zones,
        "defects": defects,
        # --- расширения контракта (не ломают потребителей) ---
        "confidence_map": ens.confidence,
        # На тайлинговом (панорамном) пути не тащим гигантский prob-массив в память.
        "class_prob": None if tile else ens.class_prob,
        "grain_meta": grain,
        "scale_um_per_px": float(scale_um_per_px),
        "n_runs": int(n_runs),
        "mean_uncertainty": float(ens.uncertainty.mean()),
        # Доля площади, где модель не уверена в фазе (confidence ниже порога) —
        # честный индикатор «не определено»: столько площади требует внимания эксперта.
        "undetermined_fraction": float((ens.confidence < LOW_CONF_THRESHOLD).mean()),
        "overall_confidence": float(1.0 - ens.uncertainty.mean()),
    }


def _ensemble_from_prob(mask: np.ndarray, prob: np.ndarray, n_classes: int) -> EnsembleResult:
    """Строит EnsembleResult из готовой маски+вероятностей (для тайлингового пути)."""
    prob = np.asarray(prob, dtype=np.float32)
    confidence = prob.max(axis=2).astype(np.float32)
    eps = 1e-9
    ent = -np.sum(prob * np.log(prob + eps), axis=2)
    uncertainty = (ent / np.log(n_classes)).astype(np.float32) if n_classes > 1 else np.zeros_like(confidence)
    total = float(mask.size)
    fr = np.array([[float((mask == k).sum()) / total for k in range(n_classes)]], dtype=np.float32)
    return EnsembleResult(mask=np.asarray(mask, np.int32), class_prob=prob,
                          confidence=confidence, uncertainty=uncertainty, per_run_fractions=fr)


def _preprocess(
    image: np.ndarray, scale_um_per_px: float, max_dim: int, denoise: bool
) -> tuple[np.ndarray, float]:
    """Прореживание крупных снимков (с корректировкой µm/px) + лёгкий денойз.

    При уменьшении в f раз каждый пиксель покрывает больше микрон, поэтому
    масштаб пересчитывается: new_scale = scale / f.
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    m = max(h, w)
    if max_dim and m > max_dim:
        f = max_dim / float(m)
        new_w, new_h = max(int(round(w * f)), 1), max(int(round(h * f)), 1)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        f_actual = max(new_w, new_h) / float(m)  # фактический коэффициент после округления
        scale_um_per_px = scale_um_per_px / f_actual

    if denoise:
        # Медиана 3×3 гасит растр скана/точечный шум, сохраняя границы фаз и зёрен.
        if img.ndim == 3:
            img = cv2.medianBlur(img, 3)
        else:
            img = cv2.medianBlur(img.astype(np.uint8), 3)

    return img, scale_um_per_px


