"""Ввод/вывод: загрузка снимков, экспорт результатов (JSON/CSV), разбор масштаба.

Никакого облака — только локальные файлы (PROJECT_CONTEXT §5).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def load_image(source) -> np.ndarray:
    """Загружает снимок в RGB uint8 массив.

    ``source`` — путь, bytes или file-like (например, из Streamlit uploader).
    TIFF/PNG/JPG поддерживаются через Pillow.
    """
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def result_to_json_dict(result: dict) -> dict:
    """Числовой паспорт без тяжёлых numpy-полей (для JSON-экспорта первичных данных)."""
    import numpy as _np

    out = {}
    for k, v in result.items():
        if isinstance(v, _np.ndarray) or k in {"xrd", "class_prob", "image"}:
            continue
        out[k] = v
    return out


def result_to_csv(result: dict) -> str:
    """Экспорт первичных данных результата в CSV (раздел; параметр; значение; ед.).

    Воспроизводимость для исследователя (отчёт v2, §7.4): плоская таблица, которую
    легко открыть в Excel/пандасе рядом с PDF-протоколом.
    """
    import csv
    import io as _io

    buf = _io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["Раздел", "Параметр", "Значение", "Единица"])

    for name, v in result.get("phase_fractions", {}).items():
        w.writerow(["Фазовая доля", name, f"{v['value'] * 100:.2f}", "%"])
        w.writerow(["Неопределённость U95", name, f"{v['ci'] * 100:.2f}", "%"])
    w.writerow(["Фазовая доля", "не определено", f"{result.get('undetermined_fraction', 0) * 100:.2f}", "%"])

    gm = result.get("grain_meta", {})
    w.writerow(["Метрика", "Средний размер зёрен", result.get("grain_size_um", ""), "µm"])
    w.writerow(["Метрика", "СКО размера зёрен", gm.get("grain_size_std_um", ""), "µm"])
    w.writerow(["Метрика", "Зёрен учтено", gm.get("n_grains", 0), "шт"])
    w.writerow(["Метрика", "Дефектов (поры/включения)", len(result.get("defects", [])), "шт"])
    w.writerow(["Неопределённость", "Общая уверенность",
                f"{result.get('overall_confidence', 1 - result.get('mean_uncertainty', 0)) * 100:.1f}", "%"])
    w.writerow(["Неопределённость", "Доля «не определено»", f"{result.get('undetermined_fraction', 0) * 100:.1f}", "%"])
    w.writerow(["Неопределённость", "Спорных зон", len(result.get("low_conf_zones", [])), "шт"])

    oc = result.get("ore_class")
    if oc:
        w.writerow(["Классификация", "Вердикт", oc["verdict"], ""])
        w.writerow(["Классификация", "Основание", oc["rule"], ""])
    pr = result.get("proportions")
    if pr:
        def _p(k):
            v = pr.get(k)
            return "не оценивается" if v is None else f"{v*100:.1f}"
        w.writerow(["Доля площади", "Общая доля сульфидов", _p("общая доля сульфидов"), "%"])
        w.writerow(["Доля площади", "Тальк (нерудная фракция)", _p("доля талька (нерудная фракция)"), "%"])
        w.writerow(["Доля площади", "Обычные срастания", _p("обычные срастания"), "%"])
        w.writerow(["Доля площади", "Тонкие срастания", _p("тонкие срастания"), "%"])

    for i, z in enumerate(result.get("low_conf_zones", []), 1):
        w.writerow(["Спорная зона", f"#{i} {z['note']}", f"{int(z['confidence'] * 100)}", "% уверенности"])

    return buf.getvalue()


def load_custom_profiles(path: str = "data/custom_profiles.json") -> dict:
    """Загружает пользовательские профили материала из локального файла (если есть)."""
    import json
    import os

    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_custom_profile(name: str, profile: dict, path: str = "data/custom_profiles.json") -> None:
    """Сохраняет/обновляет пользовательский профиль материала в локальный файл."""
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = load_custom_profiles(path)
    data[name] = profile
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_scale_from_metadata(source) -> float | None:
    """Пытается достать µm/px из EXIF/разрешения TIFF. Возвращает None, если нельзя.

    Многие SEM/OM пишут физическое разрешение в TIFF-теги. Здесь — грубая
    эвристика по DPI; если её нет, пользователь вводит масштаб вручную.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        dpi = img.info.get("dpi")
        if dpi and dpi[0] > 0:
            # 1 дюйм = 25400 µm; µm/px = 25400 / dpi.
            return round(25400.0 / float(dpi[0]), 5)
    except Exception:
        pass
    return None
