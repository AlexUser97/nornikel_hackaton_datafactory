"""Тесты сегментации талька (обученный U-Net, слабо-supervised)."""
from __future__ import annotations

import numpy as np
import pytest

from shlif import talc_model


def test_talc_model_interface():
    """Интерфейс не падает и корректно деградирует без весов."""
    assert isinstance(talc_model.available(), bool)
    assert talc_model.talc_threshold() > 0


def test_predict_talc_mask_shape_or_none():
    """Если модель есть — маска bool того же размера; иначе None (фолбэк)."""
    img = (np.random.default_rng(0).integers(0, 255, (200, 260, 3))).astype(np.uint8)
    m = talc_model.predict_talc_mask(img)
    if m is None:
        pytest.skip("веса talc_unet.pt не установлены — используется текстурный фолбэк")
    assert m.shape == img.shape[:2]
    assert m.dtype == bool


def test_ore_petro_segment_runs_with_talc():
    """Сегментатор ТЗ отрабатывает и без реального изображения руды (плейсхолдер)."""
    from shlif.ore_petro import ore_petro_segment
    img = (np.random.default_rng(1).integers(0, 255, (256, 320, 3))).astype(np.uint8)
    from shlif.ore_petro import N_CLASSES
    mask, prob = ore_petro_segment(img)
    assert mask.shape == (256, 320)
    assert prob.shape[2] == N_CLASSES
    assert mask.min() >= 0 and mask.max() <= N_CLASSES - 1
