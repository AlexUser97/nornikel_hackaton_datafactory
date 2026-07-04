"""Оформление дашборда: тёмная тема, карточки, кастомные бары и бейджи.

Чистые функции, возвращающие HTML-строки (без зависимости от Streamlit) + блок
CSS. Дизайн повторяет утверждённый макет «Паспорт шлифа»: тёмные скруглённые
карточки, фазовые бары с ±ДИ, бейдж общей уверенности, карточки метрик.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .config import OVERALL_CONF_OK, PHASE_COLORS, UNDETERMINED_OK

# Акцентные цвета интерфейса.
C_GOOD = "#34d399"
C_WARN = "#f59e0b"
C_BAD = "#ef4444"
C_MUTED = "#8b93a1"


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def format_formula(f: str) -> str:
    """Химформула -> HTML с нижними индексами: 'CuFeS2' -> 'CuFeS<sub>2</sub>'.

    Индексируются числа (и запись типа '1-x'), стоящие после буквы или скобки.
    Так формулы не «съезжают» и не зависят от юникод-подстрочных символов.
    """
    import re

    if not f:
        return ""
    return re.sub(r"(?<=[)\]A-Za-zА-Яа-я])([0-9x][0-9x\-]*)", r"<sub>\1</sub>", f)


def conf_color(conf: float) -> str:
    """Цвет по уровню уверенности: зелёный / янтарный / красный."""
    if conf >= OVERALL_CONF_OK:
        return C_GOOD
    if conf >= 0.6:
        return C_WARN
    return C_BAD


def result_status(overall_conf: float, undetermined: float, n_zones: int) -> str:
    """Сводный статус результата (отчёт v2, §6.1).

    «ok» только если одновременно: уверенность высока, «не определено» мало и нет
    спорных зон. Иначе — «warn»: результат требует верификации экспертом. Так
    убирается противоречие «уверенность 64% + баннер “модель уверена”».
    """
    if overall_conf >= OVERALL_CONF_OK and undetermined <= UNDETERMINED_OK and n_zones == 0:
        return "ok"
    return "warn"


CSS = f"""
<style>
  .block-container {{ padding-top: 3rem; max-width: 1300px; }}
  /* Карточка */
  .kts-card {{
    background: #171b22; border: 1px solid #262b35; border-radius: 14px;
    padding: 16px 18px; margin-bottom: 14px;
  }}
  .kts-card h4 {{ margin: 0 0 12px 0; font-size: 0.95rem; color: #cfd4dc; font-weight: 600; }}
  /* Хедер */
  .kts-header {{
    display: flex; align-items: center; gap: 14px;
    background: #171b22; border: 1px solid #262b35; border-radius: 14px;
    padding: 16px 18px; margin: 4px 0 12px 0;
  }}
  .kts-header > div {{ min-width: 0; }}  /* даёт заголовку сжиматься, не наезжая на бейдж */
  .formula {{ color: {C_MUTED}; font-size: 0.85em; margin-left: 4px; }}
  .formula sub {{ font-size: 0.75em; }}
  .help {{ color: {C_MUTED}; cursor: help; border-bottom: 1px dotted {C_MUTED}; }}
  .kts-logo {{
    width: 42px; height: 42px; border-radius: 10px; flex: 0 0 auto;
    background: linear-gradient(135deg,#7c5cff,#4aa3ff);
    display:flex; align-items:center; justify-content:center; font-size: 22px;
  }}
  .kts-title {{ font-size: 1.15rem; font-weight: 700; color: #f0f2f5; line-height: 1.2; }}
  .kts-sub {{ font-size: 0.82rem; color: {C_MUTED}; margin-top: 2px; }}
  .kts-badge {{
    margin-left: auto; padding: 8px 14px; border-radius: 10px; font-weight: 600;
    font-size: 0.9rem; white-space: nowrap;
  }}
  /* Баннер предупреждения */
  .kts-banner {{
    display:flex; align-items:center; gap:10px;
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4);
    color: #fbbf24; border-radius: 12px; padding: 11px 16px; margin-bottom: 14px;
    font-size: 0.9rem;
  }}
  .kts-banner.ok {{ background: rgba(52,211,153,0.10); border-color: rgba(52,211,153,0.4); color:{C_GOOD}; }}
  /* Фазовые бары */
  .phase-row {{ margin-bottom: 12px; }}
  .phase-top {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 5px; }}
  .phase-name {{ color:#dfe3e8; font-size: 0.9rem; }}
  .phase-val {{ font-weight:700; color:#f0f2f5; font-size:0.92rem; }}
  .phase-ci {{ color:{C_MUTED}; font-weight:500; font-size:0.8rem; }}
  .bar-track {{ background:#0e1117; border-radius: 6px; height: 9px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:6px; }}
  /* XRD-строки */
  .xrd-row {{ display:flex; justify-content:space-between; padding: 7px 0; border-bottom:1px solid #22272f; font-size:0.9rem; }}
  .xrd-row:last-child {{ border-bottom:none; }}
  .xrd-name {{ color:#dfe3e8; }}
  .match-good {{ color:{C_GOOD}; font-weight:600; }}
  .match-mut {{ color:{C_MUTED}; }}
  /* Метрики */
  .metric-wrap {{ display:flex; gap:14px; margin: 2px 0 14px 0; }}
  .metric-card {{
    flex:1; background:#171b22; border:1px solid #262b35; border-radius:14px; padding:16px 18px;
  }}
  .metric-label {{ color:{C_MUTED}; font-size:0.8rem; margin-bottom:6px; }}
  .metric-value {{ font-size:1.7rem; font-weight:700; color:#f0f2f5; line-height:1; }}
  .metric-value .unit {{ font-size:0.95rem; color:{C_MUTED}; font-weight:500; }}
  .metric-sub {{ color:{C_MUTED}; font-size:0.78rem; margin-top:6px; }}
  /* Легенда */
  .legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
  .legend span {{ font-size:0.8rem; color:#cfd4dc; display:inline-flex; align-items:center; gap:5px; }}
  .chip {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
  .chip.dashed {{ background:transparent; border:2px dashed {C_WARN}; }}
  .slide-img {{ width:100%; border-radius:12px; border:1px solid #262b35; display:block; }}
</style>
"""


def img_data_uri(rgb: np.ndarray, max_px: int = 1000, quality: int = 85) -> str:
    """numpy RGB -> data-URI (JPEG) для встраивания в HTML-карточку."""
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def header_html(title: str, subtitle: str, overall_conf: float) -> str:
    col = conf_color(overall_conf)
    return f"""
    <div class="kts-header">
      <div class="kts-logo">🔬</div>
      <div>
        <div class="kts-title">{title}</div>
        <div class="kts-sub">{subtitle}</div>
      </div>
      <div class="kts-badge" style="background:{col}22;color:{col};border:1px solid {col}55;">
        Общая уверенность {overall_conf*100:.0f}%
      </div>
    </div>
    """


def banner_html(overall_conf: float, undetermined: float, n_zones: int) -> str:
    """Баннер сводной оценки неопределённости (согласован с числами, §6.1).

    Зелёный — только при действительно высокой уверенности. Иначе перечисляет
    причины и требует верификации экспертом (язык ГОСТ 17025, §5).
    """
    if result_status(overall_conf, undetermined, n_zones) == "ok":
        return ('<div class="kts-banner ok">✅ Результат согласован: общая уверенность '
                f'{overall_conf*100:.0f}%, доля «не определено» {undetermined*100:.0f}%, '
                'спорных зон нет.</div>')
    reasons = []
    if overall_conf < OVERALL_CONF_OK:
        reasons.append(f"общая уверенность {overall_conf*100:.0f}%")
    if undetermined > UNDETERMINED_OK:
        reasons.append(f"«не определено» {undetermined*100:.0f}%")
    if n_zones:
        reasons.append(f"{n_zones} {_plural(n_zones, 'спорная зона', 'спорные зоны', 'спорных зон')}")
    return (f'<div class="kts-banner">⚠️ Требуется верификация экспертом: '
            f'{"; ".join(reasons)}. Оценка неопределённости результата вне допустимого порога — '
            'ручная проверка металлографом перед утверждением протокола.</div>')


def phase_bars_html(phase_fractions: dict, class_names: list[str], undetermined: float,
                    formulas: list[str] | None = None, und_help: str = "",
                    colors: list | None = None) -> str:
    palette = colors or PHASE_COLORS
    rows = ['<div class="kts-card"><h4>Фазовые доли (% площади)</h4>']
    for i, name in enumerate(class_names):
        v = phase_fractions.get(name, {"value": 0.0, "ci": 0.0})
        color = _hex(palette[i % len(palette)])
        pct = v["value"] * 100
        ci = v["ci"] * 100
        fml = format_formula(formulas[i]) if formulas and i < len(formulas) else ""
        fml_html = f'<span class="formula">{fml}</span>' if fml else ""
        rows.append(f"""
        <div class="phase-row">
          <div class="phase-top">
            <span class="phase-name">{name}{fml_html}</span>
            <span class="phase-val">{pct:.0f}% <span class="phase-ci">±{ci:.0f}</span></span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div></div>
        </div>""")
    # Строка «Не определено» — доля площади под сомнением.
    und = undetermined * 100
    tip = f' title="{und_help}"' if und_help else ""
    rows.append(f"""
        <div class="phase-row">
          <div class="phase-top">
            <span class="phase-name help" style="color:{C_MUTED}"{tip}>Не определено (низкая уверенность)</span>
            <span class="phase-val" style="color:{C_MUTED}">{und:.0f}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:{min(und,100):.1f}%;background:{C_MUTED}"></div></div>
        </div>""")
    rows.append("</div>")
    return "".join(rows)


def verdict_html(ore_class: dict, conclusion: str, proportions: dict | None = None) -> str:
    """Карточка классификации руды по ТЗ (вердикт + доли площадей + заключение)."""
    if not ore_class:
        return ""
    color = C_WARN if ore_class.get("talc_bearing") else C_GOOD
    prop_html = ""
    if proportions:
        sulf = proportions.get("общая доля сульфидов", 0) * 100
        talc = proportions.get("тальк (зона оталькования)", 0) * 100
        oxide = proportions.get("оксиды (магнетит)", 0) * 100
        sras = proportions.get("срастания (рудные вкрапленники)")
        sras_html = ("не оцениваются<br><span class='kts-sub'>(оталькованная руда)</span>"
                     if sras is None else f"{sras*100:.1f}%")
        prop_html = (
            '<div style="display:flex;gap:22px;margin:10px 0 2px;flex-wrap:wrap">'
            f'<div><div class="metric-label">Общая доля сульфидов</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:{_hex(PHASE_COLORS[5])}">{sulf:.1f}%</div></div>'
            f'<div><div class="metric-label">Доля талька (зона оталькования)</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:{_hex(PHASE_COLORS[2])}">{talc:.1f}%</div></div>'
            f'<div><div class="metric-label">Срастания (обычные/тонкие)</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:{_hex(PHASE_COLORS[1])}">{sras_html}</div></div>'
            f'<div><div class="metric-label">Доля оксидов (магнетит)</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:{_hex(PHASE_COLORS[3])}">{oxide:.1f}%</div></div>'
            '</div>')
    src = ore_class.get("source", "эвристика по долям фаз")
    conf = ore_class.get("classifier_confidence", ore_class.get("model_confidence"))
    badge = (f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
             f'padding:2px 8px;border-radius:8px;font-size:0.75rem">{src}'
             + (f' · увер. {conf*100:.0f}%' if conf is not None else "") + '</span>')
    # Human-in-the-loop: низкая откалиброванная уверенность -> явный призыв к эксперту
    if ore_class.get("needs_review"):
        badge += ('<span style="background:#b26a0022;color:#f59e0b;border:1px solid #f59e0b55;'
                  'padding:2px 8px;border-radius:8px;font-size:0.75rem;margin-left:6px">'
                  '⚠ требует проверки эксперта</span>')
    probs = ore_class.get("model_probs")
    probs_html = ""
    if probs:
        probs_html = '<div class="kts-sub" style="margin-top:6px">Вероятности сортов: ' + " · ".join(
            f"{k} {v*100:.0f}%" for k, v in probs.items()) + "</div>"
    return (
        f'<div class="kts-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<h4 style="margin:0">Классификация руды (по ТЗ)</h4>{badge}</div>'
        f'<div style="font-size:1.3rem;font-weight:700;color:{color};margin:8px 0 4px">'
        f'{ore_class["verdict"]}</div>'
        f'<div class="kts-sub">Основание: {ore_class["rule"]}</div>'
        f'{prop_html}'
        f'{probs_html}'
        f'<div style="color:#cfd4dc;font-size:0.9rem;line-height:1.4;margin-top:8px">{conclusion}</div>'
        f'</div>'
    )


def metric_cards_html(items: list[tuple]) -> str:
    """Карточки метрик. Каждый элемент: (label, value, sub[, help])."""
    cards = ['<div class="metric-wrap">']
    for it in items:
        label, value, sub = it[0], it[1], it[2]
        help_txt = it[3] if len(it) > 3 else ""
        tip = f' title="{help_txt}"' if help_txt else ""
        lbl_cls = "metric-label help" if help_txt else "metric-label"
        cards.append(f'<div class="metric-card"><div class="{lbl_cls}"{tip}>{label}</div>'
                     f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>')
    cards.append("</div>")
    return "".join(cards)


def image_card_html(data_uri: str, class_names: list[str], caption: str,
                    formulas: list[str] | None = None, colors: list | None = None) -> str:
    palette = colors or PHASE_COLORS
    chips = ""
    for i, n in enumerate(class_names):
        fml = format_formula(formulas[i]) if formulas and i < len(formulas) else ""
        fml_html = f'<span class="formula">{fml}</span>' if fml else ""
        chips += (f'<span><span class="chip" style="background:{_hex(palette[i % len(palette)])}">'
                  f'</span>{n}{fml_html}</span>')
    chips += '<span><span class="chip dashed"></span>Спорная зона</span>'
    return (f'<div class="kts-card"><img class="slide-img" src="{data_uri}"/>'
            f'<div class="kts-sub" style="margin-top:8px">{caption}</div>'
            f'<div class="legend">{chips}</div></div>')


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many
