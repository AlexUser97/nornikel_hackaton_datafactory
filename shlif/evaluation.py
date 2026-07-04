"""Метрики качества под критерии заказчика.

Заказчик обозначил метрики (автопроверки нет, но модель обучают/оценивают по ним):
* сегментация — **IoU** и **расстояние Хаусдорфа** (полное HD и робастное HD95);
* классификация — **F1** и **AUC** (попиксельная классификация фаз, one-vs-rest).

Ground truth берём из синтетического генератора (истинная фаза на пиксель известна),
поэтому метрики считаются честно, без размеченного датасета. Когда придут реальные
разметки — те же функции применяются к ним без изменений.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries


def iou_per_class(pred: np.ndarray, true: np.ndarray, n_classes: int) -> dict:
    """IoU (Jaccard) по каждому классу и среднее (mIoU).

    Returns
    -------
    dict
        ``{"per_class": {k: iou|nan}, "mean": mIoU}``. Классы, отсутствующие и в
        pred, и в true, в среднее не входят.
    """
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    per: dict[int, float] = {}
    vals: list[float] = []
    for k in range(n_classes):
        p, t = pred == k, true == k
        union = np.logical_or(p, t).sum()
        if union == 0:
            per[k] = float("nan")  # класса нет ни там, ни там
            continue
        inter = np.logical_and(p, t).sum()
        v = float(inter) / float(union)
        per[k] = round(v, 4)
        vals.append(v)
    return {"per_class": per, "mean": round(float(np.mean(vals)), 4) if vals else float("nan")}


def hausdorff_per_class(
    pred: np.ndarray, true: np.ndarray, n_classes: int, scale_um_per_px: float = 1.0
) -> dict:
    """Расстояние Хаусдорфа между границами классов: полное HD и робастное HD95.

    Для каждого класса берём границы масок pred и true, считаем двунаправленные
    расстояния ближайшего соседа. HD = максимум, HD95 = 95-й перцентиль (устойчив
    к единичным выбросам). Значения переводятся в µm по калибровке.

    Returns
    -------
    dict
        ``{"per_class": {k: {"hd": µm, "hd95": µm}}, "mean_hd": ..., "mean_hd95": ...}``.
    """
    scale = float(scale_um_per_px) if scale_um_per_px and scale_um_per_px > 0 else 1.0
    per: dict[int, dict] = {}
    hds: list[float] = []
    hd95s: list[float] = []
    for k in range(n_classes):
        a = find_boundaries(np.asarray(pred) == k, mode="inner")
        b = find_boundaries(np.asarray(true) == k, mode="inner")
        ca, cb = np.argwhere(a), np.argwhere(b)
        if ca.shape[0] == 0 or cb.shape[0] == 0:
            per[k] = {"hd": float("nan"), "hd95": float("nan")}
            continue
        d_ab = cKDTree(cb).query(ca, k=1)[0]  # для каждой точки A — до ближайшей B
        d_ba = cKDTree(ca).query(cb, k=1)[0]
        both = np.concatenate([d_ab, d_ba])
        hd = float(both.max()) * scale
        hd95 = float(np.percentile(both, 95)) * scale
        per[k] = {"hd": round(hd, 3), "hd95": round(hd95, 3)}
        hds.append(hd)
        hd95s.append(hd95)
    return {
        "per_class": per,
        "mean_hd": round(float(np.mean(hds)), 3) if hds else float("nan"),
        "mean_hd95": round(float(np.mean(hd95s)), 3) if hd95s else float("nan"),
    }


def classification_metrics(
    true: np.ndarray, pred: np.ndarray, prob: np.ndarray | None, n_classes: int
) -> dict:
    """Попиксельная классификация фаз: F1 (macro/micro) и AUC (one-vs-rest, macro).

    AUC считается по мягким вероятностям ``prob`` (H×W×K). Классы без положительных
    примеров в ``true`` в AUC не входят (иначе метрика неопределена).
    """
    from sklearn.metrics import f1_score, roc_auc_score

    y_true = np.asarray(true).ravel()
    y_pred = np.asarray(pred).ravel()
    out = {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
    }

    if prob is not None:
        y_score = np.asarray(prob).reshape(-1, prob.shape[-1])
        aucs: list[float] = []
        for k in range(min(n_classes, y_score.shape[1])):
            pos = (y_true == k)
            if pos.any() and (~pos).any():  # нужны оба класса для ROC
                try:
                    aucs.append(float(roc_auc_score(pos.astype(int), y_score[:, k])))
                except ValueError:
                    pass
        out["auc_macro_ovr"] = round(float(np.mean(aucs)), 4) if aucs else float("nan")
    return out


def evaluate_segmentation(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    n_classes: int,
    prob: np.ndarray | None = None,
    scale_um_per_px: float = 1.0,
) -> dict:
    """Полная оценка: IoU + Хаусдорф (сегментация) и F1 + AUC (классификация).

    Returns
    -------
    dict
        ``{"iou": {...}, "hausdorff_um": {...}, "classification": {...}}``.
    """
    return {
        "iou": iou_per_class(pred, true, n_classes),
        "hausdorff_um": hausdorff_per_class(pred, true, n_classes, scale_um_per_px),
        "classification": classification_metrics(true, pred, prob, n_classes),
    }
