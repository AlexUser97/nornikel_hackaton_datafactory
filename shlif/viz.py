"""Визуализация: оверлеи сегментации/неуверенности и графики для паспорта.

Все изображения — RGB uint8. Графики рендерим через matplotlib (backend Agg,
без экрана) и отдаём как PNG-байты, чтобы их можно было вставить и в Streamlit,
и в PDF.
"""
from __future__ import annotations

import io

import cv2
import matplotlib

matplotlib.use("Agg")  # без GUI, офлайн, безопасно в контейнере
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import PHASE_COLORS  # noqa: E402


def colorize_mask(mask: np.ndarray, colors: list | None = None) -> np.ndarray:
    """Маска индексов фаз -> цветное RGB-изображение по палитре фаз."""
    palette = colors or PHASE_COLORS
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k in range(int(mask.max()) + 1):
        out[mask == k] = palette[k % len(palette)]
    return out


def overlay_segmentation(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                         colors: list | None = None) -> np.ndarray:
    """Полупрозрачный цветной оверлей сегментации поверх исходника."""
    base = _ensure_rgb(image_rgb)
    color = colorize_mask(mask, colors)
    blend = (base.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha)
    return np.clip(blend, 0, 255).astype(np.uint8)


def uncertainty_heatmap(uncertainty: np.ndarray) -> np.ndarray:
    """Карта неуверенности 0..1 -> тепловая карта RGB (inferno)."""
    u8 = (np.clip(uncertainty, 0.0, 1.0) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_uncertainty(image_rgb: np.ndarray, uncertainty: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Тепловая карта неуверенности поверх исходника."""
    base = _ensure_rgb(image_rgb)
    heat = uncertainty_heatmap(uncertainty)
    blend = base.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def draw_low_conf_zones(image_rgb: np.ndarray, zones: list[dict]) -> np.ndarray:
    """Рисует пунктирные рамки зон низкой уверенности с подписью эксперту."""
    out = _ensure_rgb(image_rgb).copy()
    gold = (255, 215, 0)
    for i, z in enumerate(zones, 1):
        x, y, w, h = z["bbox"]
        _dashed_rect(out, x, y, x + w, y + h, gold, thickness=2, dash=10)
        label = f"#{i} {int(z['confidence'] * 100)}%"
        cv2.putText(out, label, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gold, 2, cv2.LINE_AA)
    return out


def _dashed_rect(img, x0, y0, x1, y1, color, thickness=2, dash=10) -> None:
    """Пунктирный прямоугольник (у OpenCV нет штатного)."""
    def dline(a, b, horizontal):
        p = a
        while p < b:
            q = min(p + dash, b)
            if horizontal is not None:
                if horizontal:
                    cv2.line(img, (p, a_fixed), (q, a_fixed), color, thickness, cv2.LINE_AA)
                else:
                    cv2.line(img, (a_fixed, p), (a_fixed, q), color, thickness, cv2.LINE_AA)
            p += 2 * dash

    a_fixed = y0; dline(x0, x1, True)
    a_fixed = y1; dline(x0, x1, True)
    a_fixed = x0; dline(y0, y1, False)
    a_fixed = x1; dline(y0, y1, False)


def render_phase_bar_chart(phase_fractions: dict[str, dict[str, float]],
                           colors: list | None = None) -> bytes:
    """Горизонтальные бары фазовых долей с усами ±ДИ. Возвращает PNG-байты."""
    palette = colors or PHASE_COLORS
    names = list(phase_fractions.keys())
    values = [phase_fractions[n]["value"] * 100 for n in names]
    errs = [phase_fractions[n]["ci"] * 100 for n in names]
    colors = [tuple(c / 255 for c in palette[i % len(palette)]) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(5.2, 0.6 * len(names) + 1.0), dpi=130)
    y = np.arange(len(names))
    ax.barh(y, values, xerr=errs, color=colors, capsize=4, height=0.6, error_kw={"ecolor": "#333", "elinewidth": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Доля площади, %")
    ax.set_xlim(0, max(100, max(values) + max(errs) + 5) if values else 100)
    for yi, v, e in zip(y, values, errs):
        ax.text(v + e + 1, yi, f"{v:.1f}±{e:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
