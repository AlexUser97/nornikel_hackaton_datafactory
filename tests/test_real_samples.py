"""Тесты на реальных снимках шлифов (public domain, из Wikimedia Commons).

Проверяем, что пайплайн устойчиво отрабатывает на «грязных» реальных данных:
запускается без ошибок, возвращает выровненную маску и честно помечает
ненадёжную оценку зерна на мелкой/зашумлённой структуре. Если файлы-фикстуры
отсутствуют (не выкачаны), тесты пропускаются.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shlif import analyze
from shlif.io_utils import load_image

SAMPLES = Path(__file__).parent / "real_samples"


@pytest.mark.parametrize(
    "fname, scale",
    [("hypereutectoid_steel.png", 0.3), ("pearlite1.jpg", 0.05)],
)
def test_real_sample_runs_and_flags_unreliable_grain(fname, scale):
    path = SAMPLES / fname
    if not path.exists():
        pytest.skip(f"нет фикстуры {fname} (реальный снимок не выкачан)")

    img = load_image(str(path))
    res = analyze(img, scale_um_per_px=scale, n_runs=3, seed=7)

    # Маска выровнена с рабочим (прореженным) снимком.
    assert res["image"].shape[:2] == res["mask"].shape
    # Доли фаз суммируются в ~1.
    assert abs(sum(v["value"] for v in res["phase_fractions"].values()) - 1.0) < 1e-3
    # Мелкая/зашумлённая структура: оценка зерна честно помечена ненадёжной.
    assert res["grain_meta"]["reliable"] is False
    assert res["grain_meta"]["note"]
