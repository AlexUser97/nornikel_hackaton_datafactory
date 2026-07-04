"""Тесты потайловой обработки панорам."""
from __future__ import annotations

import numpy as np

from shlif import analyze
from shlif.demo_data import synthetic_microstructure
from shlif.ore_petro import ore_petro_segment
from shlif.tiling import iter_tiles, segment_tiled


def test_iter_tiles_covers_image():
    """Плитки покрывают весь кадр (центральные области без пропусков)."""
    h, w, tile, ov = 500, 700, 256, 64
    covered = np.zeros((h, w), dtype=bool)
    for _, _, _, _, cy0, cy1, cx0, cx1 in iter_tiles(h, w, tile, ov):
        covered[cy0:cy1, cx0:cx1] = True
    assert covered.all()


def test_segment_tiled_shape_and_range():
    """Полноразмерная маска совпадает по размеру и лежит в диапазоне классов."""
    from shlif.ore_petro import N_CLASSES
    img = synthetic_microstructure(h=600, w=800, seed=1)
    mask, prob = segment_tiled(img, ore_petro_segment, n_classes=N_CLASSES, tile=256, overlap=64)
    assert mask.shape == (600, 800)
    assert prob.shape == (600, 800, N_CLASSES)
    assert mask.min() >= 0 and mask.max() <= N_CLASSES - 1


def test_analyze_tile_path_matches_shape_and_classifies():
    """analyze(tile=...) даёт маску размера рабочего снимка и выдаёт вердикт/доли."""
    from shlif.ore_petro import ORE_PETRO_CLASS_NAMES, ORE_PETRO_COLORS
    img = synthetic_microstructure(h=700, w=900, seed=2)
    res = analyze(img, 0.5, class_names=ORE_PETRO_CLASS_NAMES,
                  soft_segment_fn=ore_petro_segment, tile=256, material_profile="petro",
                  class_colors=ORE_PETRO_COLORS)
    assert res["mask"].shape == res["image"].shape[:2]
    assert res["ore_class"] is not None
    assert abs(sum(v["value"] for v in res["phase_fractions"].values()) - 1.0) < 1e-3
    assert res["class_prob"] is None  # на тайлинговом пути prob не тащим в память
