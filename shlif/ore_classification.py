"""Классификация руды по официальному ТЗ (task-3): срастания + тальк.

Логика (из ТЗ):
* тальк > 10% площади  →  «тальксодержащая руда»;
* иначе  →  класс по преобладанию «обычных» vs «тонких» срастаний.

Работает над уже посчитанными долями фаз (чистая доменная логика, данные для
обучения не нужны — это правило геологической классификации, а не модель).
"""
from __future__ import annotations

from .config import TALC_THRESHOLD


def _find(fractions: dict, *keywords: str) -> float | None:
    """Ищет долю фазы по ключевому слову в её имени (регистронезависимо)."""
    for name, v in fractions.items():
        low = name.lower()
        if any(k in low for k in keywords):
            return float(v["value"]) if isinstance(v, dict) else float(v)
    return None


def classify_ore(fractions: dict, talc_threshold: float = TALC_THRESHOLD) -> dict | None:
    """Классифицирует руду по долям {тонкие срастания, обычные срастания, тальк}.

    Returns
    -------
    dict | None
        ``{"verdict", "rule", "talc", "ordinary", "fine", "talc_bearing"}`` или
        None, если во фракциях нет талька (профиль не соответствует задаче ТЗ).
    """
    talc = _find(fractions, "тальк")
    if talc is None:
        return None
    # Классы датасета: обычные срастания ≈ крупные слабо замещённые рудные вкрапленники
    # (рядовая руда); тонкие срастания ≈ существенно замещённые (труднообогатимая).
    ordinary = _find(fractions, "обыч", "крупн", "рядов") or 0.0
    fine = _find(fractions, "тонк", "замещ", "трудн") or 0.0

    if talc > talc_threshold:
        verdict = "Оталькованная руда"
        rule = f"тальк {talc*100:.1f}% > порога {talc_threshold*100:.0f}%"
        talc_bearing = True
    else:
        talc_bearing = False
        if ordinary >= fine:
            verdict = "Рядовая руда"
        else:
            verdict = "Труднообогатимая руда"
        rule = (f"тальк {talc*100:.1f}% ≤ {talc_threshold*100:.0f}%; обычные/крупные срастания "
                f"{ordinary*100:.1f}% vs тонкие/замещённые {fine*100:.1f}%")

    return {
        "verdict": verdict,
        "rule": rule,
        "talc": round(talc, 4),
        "ordinary": round(ordinary, 4),
        "fine": round(fine, 4),
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
