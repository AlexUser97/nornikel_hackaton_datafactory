"""Классификация руды по официальному ТЗ (task-3): срастания + тальк.

Логика (из ТЗ):
* тальк > 10% площади  →  «тальксодержащая руда»;
* иначе  →  класс по преобладанию «обычных» vs «тонких» срастаний.

Работает над уже посчитанными долями фаз (чистая доменная логика, данные для
обучения не нужны — это правило геологической классификации, а не модель).
"""
from __future__ import annotations

from .config import TALC_THRESHOLD

# Три геолого-технологических сорта руды (задача ТЗ) — имена, цвета оверлея и
# порядок (индекс = метка на карте сортов панорамы). Цвета строго по палитре ТЗ:
# рядовая — зелёная (как «обычные срастания»), труднообогатимая — красная (как
# «тонкие срастания»), оталькованная — синяя (как «тальк»).
ORE_TYPE_NAMES: list[str] = ["Рядовая руда", "Труднообогатимая руда", "Оталькованная руда"]
ORE_TYPE_COLORS: list[tuple[int, int, int]] = [(46, 204, 113), (214, 39, 40), (52, 120, 219)]


def ore_type_index(verdict: str) -> int:
    """Индекс сорта (0/1/2) по строке вердикта; -1 если не сорт руды."""
    try:
        return ORE_TYPE_NAMES.index(verdict)
    except ValueError:
        return -1


def _find(fractions: dict, *keywords: str) -> float | None:
    """Ищет долю фазы по ключевому слову в её имени (регистронезависимо)."""
    for name, v in fractions.items():
        low = name.lower()
        if any(k in low for k in keywords):
            return float(v["value"]) if isinstance(v, dict) else float(v)
    return None


# Порог преобладания: доля «обычных» срастаний СРЕДИ всех срастаний (обычные+тонкие),
# начиная с которой руда считается рядовой. 0.5 = «преобладает» (>50%), как уточнили
# организаторы. Значение вынесено в параметр — легко поменять, когда порог подтвердят.
DOMINANCE_THRESHOLD: float = 0.50


def classify_ore(fractions: dict, talc_threshold: float = TALC_THRESHOLD,
                 dominance_threshold: float = DOMINANCE_THRESHOLD) -> dict | None:
    """Классифицирует руду по долям {тонкие срастания, обычные срастания, тальк}.

    Правило (ТЗ + уточнения организаторов):
    * тальк > ``talc_threshold`` (10%) → «Оталькованная руда»;
    * иначе по преобладанию срастаний: если доля ОБЫЧНЫХ срастаний среди всех
      срастаний ≥ ``dominance_threshold`` (>50%) → «Рядовая руда» (крупные слабо
      замещённые сульфиды), иначе → «Труднообогатимая руда» (сульфиды существенно
      замещены нерудной/тёмной фазой).

    Returns
    -------
    dict | None
        ``{"verdict", "rule", "talc", "ordinary", "fine", "ordinary_share",
        "talc_bearing"}`` или None, если во фракциях нет талька.
    """
    talc = _find(fractions, "тальк")
    if talc is None:
        return None
    # Классы датасета: обычные срастания ≈ крупные слабо замещённые рудные вкрапленники
    # (рядовая руда); тонкие срастания ≈ существенно замещённые (труднообогатимая).
    ordinary = _find(fractions, "обыч", "крупн", "рядов") or 0.0
    fine = _find(fractions, "тонк", "замещ", "трудн") or 0.0
    total_ig = ordinary + fine
    ordinary_share = (ordinary / total_ig) if total_ig > 1e-9 else 0.0

    if talc > talc_threshold:
        verdict = "Оталькованная руда"
        rule = f"тальк {talc*100:.1f}% > порога {talc_threshold*100:.0f}%"
        talc_bearing = True
    else:
        talc_bearing = False
        if ordinary_share >= dominance_threshold:
            verdict = "Рядовая руда"
        else:
            verdict = "Труднообогатимая руда"
        rule = (f"тальк {talc*100:.1f}% ≤ {talc_threshold*100:.0f}%; доля обычных срастаний "
                f"{ordinary_share*100:.0f}% (обычные {ordinary*100:.1f}% vs тонкие {fine*100:.1f}%), "
                f"порог преобладания {dominance_threshold*100:.0f}%")

    return {
        "verdict": verdict,
        "rule": rule,
        "talc": round(talc, 4),
        "ordinary": round(ordinary, 4),
        "fine": round(fine, 4),
        "ordinary_share": round(ordinary_share, 4),
        "talc_bearing": talc_bearing,
    }


def conclusion_text(fractions: dict, classification: dict, mean_uncertainty: float = 0.0) -> str:
    """Формирует текстовое заключение по образцу (на русском, для отчёта/UI)."""
    if not classification:
        return ""
    talc = classification["talc"] * 100
    ordn = classification["ordinary"] * 100
    fine = classification["fine"] * 100
    lines = [
        f"Классификация образца: {classification['verdict'].upper()}.",
        f"Основание: {classification['rule']}.",
        f"Содержание талька: {talc:.1f}% (порог тальксодержащей руды — "
        f"{TALC_THRESHOLD*100:.0f}%).",
        f"Срастания сульфидов: обычные (крупные, слабо замещённые) — {ordn:.1f}%, "
        f"тонкие (существенно замещённые) — {fine:.1f}%.",
    ]
    if mean_uncertainty:
        lines.append(
            f"Средняя неопределённость сегментации — {mean_uncertainty*100:.0f}%; "
            "зоны низкой уверенности выделены для верификации экспертом.")
    lines.append("Оценка предварительная (неаттестованный метод), требует проверки экспертом-геологом.")
    return " ".join(lines)
