"""Генерация PDF — протокол испытаний по ГОСТ ISO/IEC 17025-2019 / ГОСТ Р 58973-2020.

Точка интеграции в исследовательский процесс аккредитованной лаборатории — не
товарный «паспорт качества», а протокол испытаний (отчёт аналитического отдела v2,
§3). Соответствие стандарту (шапка лаборатории, уникальный номер, метод, даты,
разделение данных заказчика/лаборатории, блок неопределённости, дисклеймер статуса)
даёт документ, пригодный «сразу в дело» после утверждения лабораторией.

Кириллица: регистрируем шрифт DejaVuSans из поставки matplotlib — работает и на
dev-машине, и в контейнере без установки системных шрифтов.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np

from .config import LAB_DEFAULTS, METHOD_DISCLAIMER, OVERALL_CONF_OK, UNDETERMINED_OK
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import viz

_FONT = "DejaVuSans"
_FONT_BOLD = "DejaVuSans-Bold"
_FONTS_READY = False


def _ensure_fonts() -> None:
    """Регистрирует кириллический шрифт из поставки matplotlib (один раз)."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    import matplotlib

    base = matplotlib.get_data_path()
    try:
        pdfmetrics.registerFont(TTFont(_FONT, f"{base}/fonts/ttf/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont(_FONT_BOLD, f"{base}/fonts/ttf/DejaVuSans-Bold.ttf"))
    except Exception:  # noqa: BLE001 — на крайний случай остаёмся на латинице
        pass
    _FONTS_READY = True


def _np_to_png(image_rgb: np.ndarray, max_px: int = 950) -> io.BytesIO:
    """Изображение-оверлей для вставки в PDF. В паспорте картинка ~85мм, поэтому
    прореживаем до max_px и кладём как JPEG — для фотографичных снимков это в разы
    меньше PNG без заметной потери качества."""
    im = PILImage.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    buf.seek(0)
    return buf


def _fit_image(png: io.BytesIO | bytes, max_w_mm: float, max_h_mm: float) -> RLImage:
    """Вставляет PNG с сохранением пропорций в заданный бокс (мм)."""
    if isinstance(png, (bytes, bytearray)):
        png = io.BytesIO(png)
    png.seek(0)
    w, h = PILImage.open(png).size
    png.seek(0)
    ratio = min((max_w_mm * mm) / w, (max_h_mm * mm) / h)
    return RLImage(png, width=w * ratio, height=h * ratio)


def build_passport_pdf(
    image_rgb: np.ndarray,
    result: dict,
    *,
    sample_name: str = "",
    protocol: dict | None = None,
) -> bytes:
    """Собирает PDF — протокол испытаний из результата :func:`shlif.analyze`.

    Parameters
    ----------
    protocol : dict | None
        Реквизиты протокола/лаборатории (ГОСТ 17025 / 58973). Поля:
        ``number, lab_name, lab_address, accreditation, customer, object_name,
        method, operator, approved_by, issue_date``. Отсутствующие берутся из
        заглушек — заполняются аккредитованной лабораторией.

    Returns
    -------
    bytes
        Содержимое PDF-файла.
    """
    _ensure_fonts()
    p = _protocol_defaults(protocol, sample_name)
    # Оверлеи рисуем на рабочем снимке из результата (совпадает по размеру с mask).
    base = result.get("image")
    if base is not None and np.asarray(base).shape[:2] == result["mask"].shape:
        image_rgb = np.asarray(base, dtype=np.uint8)
    else:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=_FONT_BOLD, fontSize=16, spaceAfter=2, alignment=1)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=_FONT_BOLD, fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#c8102e"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=_FONT, fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555"))
    center = ParagraphStyle("center", parent=small, alignment=1)

    story: list = []

    # --- Шапка лаборатории + заголовок протокола (ГОСТ 17025 п.7.8.2.1) ------
    story.append(Paragraph(f"<b>{p['lab_name']}</b>", center))
    story.append(Paragraph(f"{p['lab_address']} · {p['accreditation']}", center))
    story.append(Spacer(1, 6))
    story.append(Paragraph("ПРОТОКОЛ ИСПЫТАНИЙ", h1))
    story.append(Paragraph(
        f"№ {p['number']}&nbsp;&nbsp;·&nbsp;&nbsp;дата выдачи {p['issue_date']}&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"по ГОСТ ISO/IEC 17025-2019, ГОСТ Р 58973-2020", center))
    story.append(Spacer(1, 6))

    # --- Реквизиты: заказчик / объект / метод / условия ---------------------
    n_low = len(result.get("low_conf_zones", []))
    info_rows = [
        [_cell("Заказчик", bold=True), _cell(p["customer"]), _cell("Оператор", bold=True), _cell(p["operator"])],
        [_cell("Объект испытаний", bold=True), _cell(p["object_name"]), _cell("Утвердил", bold=True), _cell(p["approved_by"])],
        [_cell("Метод", bold=True), _cell(p["method"]), _cell("Дата анализа", bold=True), _cell(p["analysis_date"])],
        [_cell("Масштаб (калибровка)", bold=True), _cell(f"{result.get('scale_um_per_px', 0):.4g} µm/px"),
         _cell("Размер поля", bold=True), _cell(f"{image_rgb.shape[1]}×{image_rgb.shape[0]} px")],
    ]
    info_tbl = Table(info_rows, colWidths=[38 * mm, 55 * mm, 33 * mm, 44 * mm])
    info_tbl.setStyle(_kv4_style())
    story.append(info_tbl)

    # --- Классификация руды по ТЗ (главный результат) ----------------------
    oc = result.get("ore_class")
    if oc:
        story.append(Paragraph("Классификация образца", h2))
        vcol = "#b26a00" if oc.get("talc_bearing") else "#137a4a"
        story.append(Paragraph(f'<b><font color="{vcol}">{oc["verdict"]}</font></b> '
                               f'<font size="8" color="#555">(основание: {oc["rule"]})</font>', body))
        pr = result.get("proportions")
        if pr:
            def _pct(k):
                v = pr.get(k)
                return "не оцен." if v is None else f"{v*100:.1f}"
            prows = [["Доля площади", "%"],
                     ["Общая доля сульфидов", _pct("общая доля сульфидов")],
                     ["Тальк (зона оталькования, целиком)", _pct("тальк (зона оталькования)")],
                     ["Оксиды (магнетит)", _pct("оксиды (магнетит)")],
                     ["Срастания (обычные + тонкие)", _pct("срастания (рудные вкрапленники)")]]
            if pr.get("срастания (рудные вкрапленники)") is not None:
                prows += [["  · обычные срастания", _pct("обычные срастания")],
                          ["  · тонкие срастания", _pct("тонкие срастания")],
                          ["  · срастания внутри талька", _pct("срастания внутри талька")]]
            ptbl = Table(prows, colWidths=[90 * mm, 25 * mm])
            ptbl.setStyle(_grid_style())
            story.append(ptbl)
        if result.get("conclusion"):
            story.append(Paragraph(result["conclusion"], small))

    # --- Сводная оценка неопределённости результата (ГОСТ 17025, §5 отчёта) -
    story.append(Paragraph("Оценка неопределённости результата", h2))
    story.append(_status_table(result, n_low))

    # --- Сегментация + карта неуверенности ---------------------------------
    story.append(Paragraph("Сегментация и карта неопределённости", h2))
    overlay = viz.overlay_segmentation(image_rgb, result["mask"], colors=result.get("colors"))
    overlay = viz.draw_low_conf_zones(overlay, result.get("low_conf_zones", []))
    heat = viz.overlay_uncertainty(image_rgb, result["uncertainty_map"])
    img_tbl = Table(
        [[_fit_image(_np_to_png(overlay), 85, 70), _fit_image(_np_to_png(heat), 85, 70)],
         [Paragraph("Сегментация фаз + зоны на проверку (жёлтым)", small),
          Paragraph("Карта неуверенности (ярче = спорнее)", small)]],
        colWidths=[90 * mm, 90 * mm],
    )
    img_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(img_tbl)

    # --- Фазовые доли -------------------------------------------------------
    story.append(Paragraph("Фазовые доли (доля площади, ± — расширенная неопределённость 95%)", h2))
    story.append(_fit_image(viz.render_phase_bar_chart(result["phase_fractions"], result.get("colors")), 120, 60))
    frac_rows = [["Фаза", "Доля, %", "±U(95%)"]]
    for name, v in result["phase_fractions"].items():
        frac_rows.append([name, f"{v['value'] * 100:.1f}", f"±{v['ci'] * 100:.1f}"])
    und = result.get("undetermined_fraction", 0.0)
    frac_rows.append(["Не определено (низкая уверенность)", f"{und * 100:.1f}", "—"])
    frac_tbl = Table(frac_rows, colWidths=[70 * mm, 25 * mm, 25 * mm])
    frac_tbl.setStyle(_grid_style())
    story.append(frac_tbl)

    # --- Метрики зерна/гранулометрии и дефекты -----------------------------
    story.append(Paragraph("Метрики микроструктуры", h2))
    gm = result.get("grain_meta", {})
    gs = result.get("grain_size_um")
    reliable = gm.get("reliable", True)
    size_label = "Средний размер зёрен (гранулометрия)"
    gs_str = (f"{gs} µm" if gs is not None else "н/д") + ("" if reliable else " ⚠")
    metric_rows = [[size_label, gs_str, f"±{gm.get('grain_size_std_um', '—')} µm"]]
    # Гранулометрия зёрен минералов (µm); балл ASTM E112 к рудам неприменим.
    metric_rows.append(["Зёрен учтено (гранулометрия)", f"{gm.get('n_grains', 0)}", ""])
    metric_rows.append(["Дефектов (поры/включения)", str(len(result.get("defects", []))), ""])
    metric_tbl = Table(metric_rows, colWidths=[55 * mm, 40 * mm, 55 * mm])
    metric_tbl.setStyle(_kv_style())
    story.append(metric_tbl)
    if not reliable and gm.get("note"):
        story.append(Paragraph(f"⚠ {gm['note']}", small))

    # --- Зоны низкой уверенности -------------------------------------------
    if n_low:
        story.append(Paragraph(f"Зоны низкой уверенности — на проверку экспертом ({n_low})", h2))
        zrows = [["#", "bbox [x,y,w,h]", "Уверенность", "Спор фаз"]]
        for i, z in enumerate(result["low_conf_zones"], 1):
            zrows.append([str(i), str(z["bbox"]), f"{int(z['confidence'] * 100)}%", z["note"]])
        ztbl = Table(zrows, colWidths=[10 * mm, 55 * mm, 30 * mm, 55 * mm])
        ztbl.setStyle(_grid_style())
        story.append(ztbl)

    # --- Дисклеймер статуса метода (обязателен, §7.5, §8) ------------------
    story.append(Paragraph("Статус метода и ограничения", h2))
    story.append(Paragraph(METHOD_DISCLAIMER, small))
    story.append(Spacer(1, 6))
    sig = Table([[
        Paragraph(f"Оператор: {p['operator']} _______________", small),
        Paragraph(f"Утвердил: {p['approved_by']} _______________ М.П.", small),
    ]], colWidths=[85 * mm, 85 * mm])
    sig.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), _FONT), ("TOPPADDING", (0, 0), (-1, -1), 8)]))
    story.append(sig)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
        title=f"Протокол испытаний № {p['number']}",
    )
    doc.build(story)
    return buf.getvalue()


def _cell(text, *, bold: bool = False, color: str | None = None) -> Paragraph:
    """Ячейка-Paragraph (переносится по ширине колонки, в отличие от сырой строки)."""
    style = ParagraphStyle(
        f"cell_{bold}_{color}", fontName=_FONT_BOLD if bold else _FONT,
        fontSize=8.2, leading=10, textColor=colors.HexColor(color) if color else colors.black,
    )
    return Paragraph(str(text), style)


def _protocol_defaults(protocol: dict | None, sample_name: str) -> dict:
    """Заполняет реквизиты протокола, подставляя заглушки лаборатории."""
    protocol = dict(protocol or {})
    now = datetime.now()
    auto_number = f"КТШ-{now:%Y%m%d}-{now:%H%M%S}"
    return {
        "number": protocol.get("number") or auto_number,
        "issue_date": protocol.get("issue_date") or f"{now:%Y-%m-%d}",
        "analysis_date": protocol.get("analysis_date") or f"{now:%Y-%m-%d %H:%M}",
        "lab_name": protocol.get("lab_name") or LAB_DEFAULTS["lab_name"],
        "lab_address": protocol.get("lab_address") or LAB_DEFAULTS["lab_address"],
        "accreditation": protocol.get("accreditation") or LAB_DEFAULTS["accreditation"],
        "customer": protocol.get("customer") or "(указать заказчика)",
        "object_name": protocol.get("object_name") or (sample_name or "образец (аншлиф/шлиф)"),
        "method": protocol.get("method") or "Оптическая микроскопия; автоматическая сегментация по яркости + РФА",
        "operator": protocol.get("operator") or "(оператор)",
        "approved_by": protocol.get("approved_by") or "(уполномоченное лицо)",
    }


def _status_table(result: dict, n_low: int) -> Table:
    """Таблица сводной оценки неопределённости (согласована с числами, §6.1)."""
    conf = result.get("overall_confidence", 1 - result.get("mean_uncertainty", 0))
    und = result.get("undetermined_fraction", 0.0)
    ok = conf >= OVERALL_CONF_OK and und <= UNDETERMINED_OK and n_low == 0
    verdict = "Согласован" if ok else "ТРЕБУЕТ ВЕРИФИКАЦИИ ЭКСПЕРТОМ"
    vcolor = "#137a4a" if ok else "#b26a00"
    rows = [
        [_cell("Общая уверенность", bold=True), _cell(f"{conf * 100:.0f}%"),
         _cell("Доля «не определено»", bold=True), _cell(f"{und * 100:.0f}%")],
        [_cell("Спорных зон (на верификацию)", bold=True), _cell(str(n_low)),
         _cell("Прогонов ансамбля", bold=True), _cell(str(result.get("n_runs", "—")))],
        [_cell("Статус результата", bold=True), _cell(verdict, bold=True, color=vcolor),
         _cell("Порог", bold=True), _cell(f"уверенность ≥{OVERALL_CONF_OK*100:.0f}%, «н/о» ≤{UNDETERMINED_OK*100:.0f}%")],
    ]
    tbl = Table(rows, colWidths=[50 * mm, 33 * mm, 43 * mm, 44 * mm])
    tbl.setStyle(_kv4_style())
    return tbl


def _grid_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2d5d9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf5f6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


def _kv_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTNAME", (0, 0), (0, -1), _FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


def _kv4_style() -> TableStyle:
    """Стиль для таблицы 4 колонки: ключ|значение|ключ|значение (реквизиты протокола)."""
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTNAME", (0, 0), (0, -1), _FONT_BOLD),
        ("FONTNAME", (2, 0), (2, -1), _FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
