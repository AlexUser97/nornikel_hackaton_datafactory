"""Тесты сквозного пайплайна: контракт analyze(), метрики, неуверенность, PDF."""
from __future__ import annotations

import numpy as np
import pytest

from shlif import analyze
from shlif.config import DEFAULT_CLASS_NAMES
from shlif.demo_data import synthetic_microstructure
from shlif.report import build_passport_pdf
from shlif.simto_real import corrupt


@pytest.fixture(scope="module")
def demo_image() -> np.ndarray:
    return synthetic_microstructure(seed=42)


@pytest.fixture(scope="module")
def result(demo_image):
    return analyze(demo_image, scale_um_per_px=0.35, n_runs=5, seed=42)


def test_contract_shape(result, demo_image):
    """Ответ содержит все обязательные поля контракта нужных типов."""
    for key in [
        "mask", "class_names", "phase_fractions", "grain_size_um", "astm_number",
        "uncertainty_map", "low_conf_zones", "defects",
    ]:
        assert key in result, f"нет поля контракта: {key}"

    h, w = demo_image.shape[:2]
    assert result["mask"].shape == (h, w)
    assert result["mask"].dtype.kind in "iu"
    assert result["uncertainty_map"].shape == (h, w)
    assert result["class_names"] == DEFAULT_CLASS_NAMES
    assert isinstance(result["low_conf_zones"], list)
    assert isinstance(result["defects"], list)


def test_phase_fractions_sum_to_one(result):
    """Доли фаз неотрицательны, с ДИ, и суммируются в ~1."""
    total = 0.0
    for name, v in result["phase_fractions"].items():
        assert 0.0 <= v["value"] <= 1.0
        assert v["ci"] >= 0.0
        total += v["value"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_uncertainty_range(result):
    """Карта неуверенности лежит в [0, 1]."""
    u = result["uncertainty_map"]
    assert float(u.min()) >= 0.0
    assert float(u.max()) <= 1.0 + 1e-6


def test_grain_metrics_present(result):
    """На синтетике размер зерна и балл ASTM считаются (не None)."""
    assert result["grain_size_um"] is not None
    assert result["astm_number"] is not None
    assert result["grain_meta"]["n_grains"] > 0


def test_scale_affects_grain_size(demo_image):
    """Больше µm/px -> больше физический размер зерна (линейно)."""
    r1 = analyze(demo_image, scale_um_per_px=0.2, n_runs=1, seed=1)
    r2 = analyze(demo_image, scale_um_per_px=0.4, n_runs=1, seed=1)
    assert r2["grain_size_um"] == pytest.approx(2 * r1["grain_size_um"], rel=1e-3)


def test_uncertainty_rises_with_corruption(demo_image):
    """Sim-to-real: грязный снимок -> выше средняя неуверенность (честность модели)."""
    clean = analyze(demo_image, 0.35, n_runs=5, seed=1)
    dirty_img = corrupt(demo_image, brightness=-25, contrast=0.7, gaussian_noise=20, blur_sigma=1.6, seed=1)
    dirty = analyze(dirty_img, 0.35, n_runs=5, seed=1)
    assert dirty["mean_uncertainty"] > clean["mean_uncertainty"]


def test_low_conf_zone_format(result):
    """Зоны низкой уверенности имеют формат контракта."""
    for z in result["low_conf_zones"]:
        assert set(["bbox", "confidence", "note"]).issubset(z)
        assert len(z["bbox"]) == 4
        assert 0.0 <= z["confidence"] <= 1.0


def test_pdf_builds(result, demo_image):
    """PDF-паспорт генерируется и является валидным PDF-файлом."""
    pdf = build_passport_pdf(demo_image, result, sample_name="test")
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 10_000


def test_analyze_basic_result(demo_image):
    """Базовый анализ собирается и содержит маску и доли фаз."""
    r = analyze(demo_image, 0.35, n_runs=2, seed=1)
    assert r["mask"].shape == demo_image.shape[:2]
    assert r["phase_fractions"]


def test_grayscale_input(demo_image):
    """Принимает и одноканальный вход."""
    gray = demo_image[:, :, 0]
    r = analyze(gray, 0.35, n_runs=1, seed=1)
    assert r["mask"].shape == gray.shape


def test_astm_not_applicable_for_ore(demo_image):
    """Профиль руды: балл ASTM E112 не выводится (отчёт v2, §6.3), размер зёрен остаётся."""
    r = analyze(demo_image, 0.2, n_runs=2, seed=1, astm_applicable=False, material_profile="руда")
    assert r["astm_number"] is None
    assert r["astm_applicable"] is False
    assert r["grain_size_um"] is not None  # гранулометрия остаётся


def test_undetermined_and_overall_confidence(result):
    """В результате есть доля «не определено» и общая уверенность в [0,1]."""
    assert 0.0 <= result["undetermined_fraction"] <= 1.0
    assert 0.0 <= result["overall_confidence"] <= 1.0
    assert result["overall_confidence"] == pytest.approx(1 - result["mean_uncertainty"], abs=1e-6)


def test_banner_status_matches_numbers():
    """§6.1: зелёный статус только при высокой уверенности и малом «не определено»."""
    from shlif import ui
    assert ui.result_status(0.95, 0.02, 0) == "ok"
    assert ui.result_status(0.64, 0.02, 0) == "warn"   # низкая уверенность
    assert ui.result_status(0.95, 0.12, 0) == "warn"   # много «не определено»
    assert ui.result_status(0.95, 0.02, 3) == "warn"   # есть спорные зоны


def test_primary_data_export(result):
    """Экспорт первичных данных: JSON без numpy, CSV с заголовком (§7.4)."""
    from shlif.io_utils import result_to_csv, result_to_json_dict
    j = result_to_json_dict(result)
    assert "mask" not in j and "image" not in j
    assert "phase_fractions" in j
    csv = result_to_csv(result)
    assert csv.startswith("Раздел;Параметр;Значение;Единица")
    assert "Фазовая доля" in csv
