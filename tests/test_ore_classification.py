"""Тесты классификации руды по официальному ТЗ (task-3)."""
from __future__ import annotations

import numpy as np

from shlif import analyze
from shlif.config import MATERIAL_PROFILES
from shlif.demo_data import synthetic_microstructure
from shlif.ore_classification import classify_ore, conclusion_text


def _fr(fine, ordinary, talc):
    return {
        "тонкие срастания": {"value": fine, "ci": 0.0},
        "обычные срастания": {"value": ordinary, "ci": 0.0},
        "тальк": {"value": talc, "ci": 0.0},
    }


def test_talc_bearing_when_talc_over_10pct():
    c = classify_ore(_fr(0.4, 0.45, 0.15))
    assert c["verdict"] == "Оталькованная руда"
    assert c["talc_bearing"] is True


def test_ordinary_dominant_when_talc_low():
    c = classify_ore(_fr(0.30, 0.62, 0.08))
    assert c["verdict"] == "Рядовая руда"
    assert c["talc_bearing"] is False


def test_fine_dominant_when_talc_low():
    c = classify_ore(_fr(0.60, 0.32, 0.08))
    assert c["verdict"] == "Труднообогатимая руда"


def test_threshold_boundary_exactly_10pct_is_not_talc_bearing():
    c = classify_ore(_fr(0.30, 0.60, 0.10))  # ровно 10% — не превышает порог
    assert c["talc_bearing"] is False


def test_classify_returns_none_without_talc():
    assert classify_ore({"феррит": {"value": 1.0, "ci": 0.0}}) is None


def test_conclusion_text_mentions_verdict_and_percentages():
    fr = _fr(0.30, 0.62, 0.08)
    txt = conclusion_text(fr, classify_ore(fr), mean_uncertainty=0.2)
    assert "рядовая руда" in txt.lower()
    assert "8.0%" in txt and "тальк" in txt.lower()


def test_analyze_produces_ore_classification_for_petro_profile():
    """analyze() с профилем ТЗ выдаёт вердикт классификации и цвета."""
    from shlif.ore_petro import ore_petro_segment
    prof = MATERIAL_PROFILES["Аншлиф руды (оптическая микроскопия)"]
    img = synthetic_microstructure(seed=7, phase_probs=(0.40, 0.52, 0.08))
    res = analyze(img, 0.18, class_names=prof["class_names"], n_runs=2, seed=7,
                  class_colors=prof["colors"], material_profile="Аншлиф руды (оптическая микроскопия)",
                  soft_segment_fn=ore_petro_segment)
    assert res["ore_class"] is not None
    assert res["ore_class"]["verdict"]
    assert res["conclusion"]
    assert res["colors"] == prof["colors"]
