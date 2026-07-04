"""Тесты метрик заказчика: IoU, Хаусдорф, F1, AUC."""
from __future__ import annotations

import numpy as np

from shlif import analyze, evaluate_segmentation
from shlif.demo_data import synthetic_microstructure
from shlif.evaluation import hausdorff_per_class, iou_per_class
from shlif.simto_real import corrupt


def test_perfect_prediction_scores_ideal():
    """Идеальное совпадение: IoU=1, Хаусдорф=0."""
    true = np.zeros((64, 64), dtype=np.int32)
    true[:, 32:] = 1
    iou = iou_per_class(true.copy(), true, n_classes=2)
    assert iou["mean"] == 1.0
    hd = hausdorff_per_class(true.copy(), true, n_classes=2)
    assert hd["mean_hd"] == 0.0 and hd["mean_hd95"] == 0.0


def test_iou_range_and_mismatch():
    """Полное несовпадение классов -> IoU=0."""
    a = np.zeros((32, 32), dtype=np.int32)
    b = np.ones((32, 32), dtype=np.int32)
    assert iou_per_class(a, b, n_classes=2)["mean"] == 0.0


def test_evaluate_on_synthetic_ground_truth():
    """На синтетике с ground truth метрики в разумном диапазоне и полной формы."""
    img, true = synthetic_microstructure(seed=3, return_labels=True)
    res = analyze(img, 0.3, n_runs=3, seed=3)
    ev = evaluate_segmentation(res["mask"], true, n_classes=3,
                               prob=res["class_prob"], scale_um_per_px=0.3)
    assert 0.5 <= ev["iou"]["mean"] <= 1.0
    assert 0.5 <= ev["classification"]["f1_macro"] <= 1.0
    assert 0.5 <= ev["classification"]["auc_macro_ovr"] <= 1.0
    assert ev["hausdorff_um"]["mean_hd"] >= ev["hausdorff_um"]["mean_hd95"] >= 0


def test_metrics_degrade_with_corruption():
    """Sim-to-real: на загрязнённом снимке mIoU не выше, чем на чистом."""
    img, true = synthetic_microstructure(seed=5, return_labels=True)
    clean = analyze(img, 0.3, n_runs=3, seed=5)
    dirty = analyze(corrupt(img, brightness=-25, contrast=0.7, gaussian_noise=20, blur_sigma=1.6, seed=5),
                    0.3, n_runs=3, seed=5)
    m_clean = evaluate_segmentation(clean["mask"], true, n_classes=3)["iou"]["mean"]
    m_dirty = evaluate_segmentation(dirty["mask"], true, n_classes=3)["iou"]["mean"]
    assert m_dirty <= m_clean + 1e-6


def test_pretrained_hook_is_used():
    """analyze() принимает внешнюю (предобученную) сегментацию через soft_segment_fn."""
    img = synthetic_microstructure(seed=1)
    called = {"n": 0}

    def fake_soft(im):
        called["n"] += 1
        h, w = im.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        prob = np.zeros((h, w, 3), dtype=np.float32)
        prob[..., 0] = 1.0
        return mask, prob

    res = analyze(img, 0.3, n_runs=2, seed=1, soft_segment_fn=fake_soft)
    assert called["n"] >= 1
    assert res["phase_fractions"][res["class_names"][0]]["value"] == 1.0
