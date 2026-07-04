"""«Кто твой шлиф» — Streamlit-дашборд «Паспорт шлифа».

Запуск (офлайн, в контуре НИИ):
    streamlit run app.py
или:
    docker compose up

Пользователь — лаборант/научный сотрудник, не ML-инженер: заходит через браузер,
перетаскивает снимок, подтверждает масштаб, читает паспорт, проверяет спорные
зоны и экспортирует PDF. Никакого облака и интернета в рантайме.
"""
from __future__ import annotations

import json
import os

# Телеметрия наглухо ещё до импорта streamlit (дублирует .streamlit/config.toml).
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import streamlit as st

from shlif import analyze, evaluate_segmentation, viz, ui
from shlif.config import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_PROFILE,
    MATERIAL_PROFILES,
    MAX_CLASSES,
    METHOD_DISCLAIMER,
    METHODS_HELP,
    METRIC_HELP,
)
from shlif.demo_data import synthetic_microstructure
from shlif.io_utils import (
    load_custom_profiles,
    load_image,
    parse_scale_from_metadata,
    result_to_csv,
    result_to_json_dict,
    save_custom_profile,
)
from shlif.ore_petro import ore_petro_segment
from shlif.report import build_passport_pdf
from shlif.simto_real import DEMO_PRESET_DESC, DEMO_PRESETS, corrupt

st.set_page_config(page_title="Кто твой шлиф — Протокол испытаний", page_icon="🔬", layout="wide")

ss = st.session_state
ss.setdefault("source_image", None)   # исходный снимок до загрязнения
ss.setdefault("image", None)          # рабочий снимок (RGB uint8)
ss.setdefault("result", None)         # результат analyze()
ss.setdefault("sample_name", "")
ss.setdefault("profile", DEFAULT_PROFILE)
ss.setdefault("protocol", {})
ss.setdefault("true_mask", None)      # ground truth для демо (оценка метрик)
ss.setdefault("formulas", [])         # формулы фаз (по индексу)
ss.setdefault("labels", [])           # редактируемые метки фаз (по индексу)
ss.setdefault("reviewed", set())


# ---------------------------------------------------------------------------
# Сайдбар — ввод
# ---------------------------------------------------------------------------
def _all_profiles() -> dict:
    """Встроенные + пользовательские (сохранённые) профили материала."""
    profiles = dict(MATERIAL_PROFILES)
    profiles.update(load_custom_profiles())
    return profiles


def sidebar() -> dict:
    st.sidebar.title("🔬 Кто твой шлиф")
    st.sidebar.caption("QC-ассистент металлографа · протокол по ГОСТ ISO/IEC 17025 · офлайн")

    st.sidebar.subheader("1. Снимок шлифа/аншлифа")
    up = st.sidebar.file_uploader("Загрузите SEM/OM-снимок", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    with st.sidebar.expander("Нет своего снимка? Демо-образцы"):
        d1, d2 = st.columns(2)
        demo_ore = d1.button("Руда Ni-Cu", use_container_width=True)
        demo_steel = d2.button("Сталь", use_container_width=True)

    st.sidebar.subheader("2. Профиль материала")
    profiles = _all_profiles()
    profile_names = list(profiles)
    idx = profile_names.index(ss.profile) if ss.profile in profile_names else 0
    profile_name = st.sidebar.selectbox(
        "Материал", profile_names, index=idx,
        help="Определяет имена фаз и применимость балла ASTM E112 (только для сталей). "
             "Свои профили можно сохранять (см. «Фазы и формулы»).")
    profile = dict(profiles[profile_name])

    st.sidebar.subheader("3. Реквизиты протокола")
    sample_id = st.sidebar.text_input("ID образца / объект", value=ss.sample_name or "ORE-0001")
    scale = st.sidebar.number_input("Масштаб, µm/px", min_value=0.001, max_value=100.0,
                                    value=0.18, step=0.01, format="%.4f")
    with st.sidebar.expander("Шапка протокола (ГОСТ 17025)"):
        lab = st.text_input("Лаборатория", value=ss.protocol.get("lab_name", "ИАЦ «Гипроникель»"))
        customer = st.text_input("Заказчик", value=ss.protocol.get("customer", "ГОК «Норильск»"))
        operator = st.text_input("Оператор", value=ss.protocol.get("operator", ""))
        approved_by = st.text_input("Утвердил", value=ss.protocol.get("approved_by", ""))
        number = st.text_input("№ протокола (пусто = авто)", value=ss.protocol.get("number", ""))

    st.sidebar.subheader("4. Фазы и формулы")
    base_names = profile.get("class_names", DEFAULT_CLASS_NAMES)
    base_formulas = profile.get("formulas", [""] * len(base_names))
    n_classes = st.sidebar.number_input(
        "Число фаз", min_value=2, max_value=MAX_CLASSES, value=len(base_names), step=1,
        help="Сколько фаз выделять по яркости (тёмная → светлая). Метки можно править и после анализа.")
    names, formulas = [], []
    with st.sidebar.expander("Имена и формулы фаз (тёмная → светлая)"):
        for i in range(int(n_classes)):
            c1, c2 = st.columns([3, 2])
            nm = c1.text_input(f"Фаза {i+1}", value=base_names[i] if i < len(base_names) else f"фаза {i+1}",
                               key=f"cn_{profile_name}_{i}")
            fm = c2.text_input("формула", value=base_formulas[i] if i < len(base_formulas) else "",
                               key=f"cf_{profile_name}_{i}", placeholder="напр. CuFeS2")
            names.append(nm)
            formulas.append(fm)
        save_name = st.text_input("Сохранить как профиль (имя)", value="", placeholder="Мой материал")
        if st.button("💾 Сохранить профиль") and save_name.strip():
            save_custom_profile(save_name.strip(), {
                "class_names": names, "formulas": formulas,
                "astm_applicable": profile.get("astm_applicable", False),
                "size_label": profile.get("size_label", "Средний размер зёрен"),
                "method": profile.get("method", "Оптическая микроскопия; сегментация по яркости"),
            })
            st.success(f"Профиль «{save_name.strip()}» сохранён.")

    st.sidebar.subheader("5. Sim-to-real (демо)")
    preset = st.sidebar.selectbox("Предполагаемое загрязнение снимка", list(DEMO_PRESETS.keys()), index=0,
                                  help="Честность: грязнее снимок → выше оценка неопределённости.")
    st.sidebar.caption("↳ " + DEMO_PRESET_DESC.get(preset, ""))

    st.sidebar.subheader("6. Параметры")
    n_runs = st.sidebar.slider("Точность оценки неопределённости (повторов)", 1, 9, 5,
                               help=METRIC_HELP["runs"])
    tiling = st.sidebar.checkbox("Крупная панорама (тайлинг)", value=False,
                                 help="Полноразмерная сегментация плитками для панорам (до 10000×10000). "
                                      "Медленнее, но сохраняет морфологию. Для крупных снимков включается авто.")

    run = st.sidebar.button("▶️ Проанализировать", type="primary", use_container_width=True)
    st.sidebar.divider()
    st.sidebar.caption("Стек open-source · веса локально · телеметрия отключена")

    return dict(up=up, demo_ore=demo_ore, demo_steel=demo_steel, profile_name=profile_name, profile=profile,
                sample_id=sample_id, scale=scale, preset=preset,
                n_runs=n_runs, tiling=tiling, names=names, formulas=formulas, n_classes=int(n_classes), run=run,
                protocol=dict(lab_name=lab, customer=customer, operator=operator,
                              approved_by=approved_by, number=number or None, method=profile.get("method", "")))


def load_inputs(inp: dict) -> None:
    ss.sample_name = inp["sample_id"]
    ss.profile = inp["profile_name"]
    ss.protocol = inp["protocol"]

    if inp["demo_ore"] or inp["demo_steel"]:
        # Для задачи «срастания + тальк» подбираем доли так, чтобы тальк < 10%
        # (демонстрирует ветку «преобладание срастаний»).
        probs = (0.40, 0.52, 0.08) if inp["profile"].get("classify") else (0.30, 0.25, 0.45)
        ss.source_image, ss.true_mask = synthetic_microstructure(seed=7, return_labels=True, phase_probs=probs)
        ss.sample_name = inp["sample_id"] or "демо-образец"
    elif inp["up"] is not None:
        ss.source_image = load_image(inp["up"].getvalue())
        ss.true_mask = None  # для реальных снимков ground truth нет
        auto = parse_scale_from_metadata(inp["up"].getvalue())
        if auto:
            st.sidebar.info(f"Масштаб из метаданных: {auto} µm/px")


def run_analysis(inp: dict) -> None:
    if ss.source_image is None:
        st.sidebar.warning("Сначала загрузите снимок или демо-образец.")
        return
    preset_kw = DEMO_PRESETS.get(inp["preset"], {})
    work = corrupt(ss.source_image, **preset_kw) if preset_kw else ss.source_image
    ss.image = viz._ensure_rgb(work)

    profile = inp["profile"]
    ss.formulas = inp["formulas"]
    ss.labels = list(inp["names"])          # редактируемые метки для отображения
    seg_fn = ore_petro_segment if profile.get("segmenter") == "ore_petro" else None
    # Для ore-petro сегментатор детерминирован (правила + U-Net талька) — ансамбль не нужен.
    n_runs = 1 if seg_fn is not None else inp["n_runs"]
    # Тайлинг: по флагу или авто для крупных снимков (>5000 px по стороне).
    tile = 1024 if (inp["tiling"] or max(ss.image.shape[:2]) > 5000) else 0

    spinner = "Анализ панорамы плитками (может занять до нескольких минут)…" if tile else \
        "Анализ: сегментация, классификация, оценка неопределённости, метрики…"
    with st.spinner(spinner):
        ss.result = analyze(ss.image, scale_um_per_px=inp["scale"],
                            class_names=inp["names"], n_runs=n_runs, seed=7,
                            astm_applicable=profile.get("astm_applicable", False),
                            material_profile=inp["profile_name"], class_colors=profile.get("colors"),
                            soft_segment_fn=seg_fn, tile=tile)
    ss.reviewed = set()


# ---------------------------------------------------------------------------
# Основной экран — паспорт
# ---------------------------------------------------------------------------
def _apply_labels(res: dict) -> dict:
    """Применяет отредактированные метки фаз к результату (для отображения и PDF)."""
    labels = ss.labels
    orig = res["class_names"]
    if not labels or len(labels) != len(orig):
        return res
    out = dict(res)
    out["class_names"] = list(labels)
    out["phase_fractions"] = {labels[i]: res["phase_fractions"][orig[i]] for i in range(len(orig))}
    remap = {orig[i]: labels[i] for i in range(len(orig))}
    zones = []
    for z in res.get("low_conf_zones", []):
        note = z.get("note", "")
        for o, n in remap.items():
            note = note.replace(o, n)
        zz = dict(z); zz["note"] = note
        zones.append(zz)
    out["low_conf_zones"] = zones
    return out


def render_passport() -> None:
    res = _apply_labels(ss.result)
    img = res.get("image", ss.image)
    names = res["class_names"]
    colors = res.get("colors")
    formulas = ss.formulas if len(ss.formulas) == len(names) else None
    conf = res.get("overall_confidence", 1 - res["mean_uncertainty"])
    und = res.get("undetermined_fraction", 0.0)
    n_low = len(res["low_conf_zones"])
    st.markdown(ui.CSS, unsafe_allow_html=True)

    # --- Хедер ---
    subtitle = " · ".join(x for x in [ss.profile, ss.sample_name,
                                      f"масштаб {res['scale_um_per_px']:.3g} µm/px"] if x)
    st.markdown(ui.header_html(f"Протокол испытаний · {ss.sample_name or 'образец'}", subtitle, conf),
                unsafe_allow_html=True)
    st.markdown(ui.banner_html(conf, und, n_low), unsafe_allow_html=True)

    # --- Вердикт классификации руды (главный результат по ТЗ) ---
    if res.get("ore_class"):
        st.markdown(ui.verdict_html(res["ore_class"], res.get("conclusion", ""), res.get("proportions")),
                    unsafe_allow_html=True)

    # --- Правка меток фаз после сегментации (авто-обновление) ---
    with st.expander("✏️ Метки фаз (правятся после анализа, всё обновится автоматически)"):
        cols = st.columns(len(names))
        new_labels = []
        for i, c in enumerate(cols):
            new_labels.append(c.text_input(f"Фаза {i+1}", value=names[i], key=f"lbl_{i}"))
        if new_labels != list(ss.labels):
            ss.labels = new_labels
            st.rerun()

    left, right = st.columns([3, 2], gap="medium")

    # --- Левая колонка: изображение со слоями ---
    with left:
        layers = ["Сегментация", "Карта неопределённости", "Зоны на верификацию", "Исходник"]
        layer = _pills("layer", layers)
        if layer == "Исходник":
            im, cap = img, "Рабочий снимок"
        elif layer == "Сегментация":
            im, cap = viz.overlay_segmentation(img, res["mask"], colors=colors), "Фазовая сегментация"
        elif layer == "Карта неопределённости":
            im = viz.overlay_uncertainty(img, res["uncertainty_map"])
            cap = f"Карта неопределённости (средняя {res['mean_uncertainty']*100:.0f}%) — ярче = спорнее"
        else:
            im = viz.draw_low_conf_zones(viz.overlay_segmentation(img, res["mask"], colors=colors),
                                         res["low_conf_zones"])
            cap = "Зоны, требующие верификации экспертом (пунктир)"
        st.markdown(ui.image_card_html(ui.img_data_uri(im), names, cap, formulas, colors),
                    unsafe_allow_html=True)

    # --- Правая колонка: доли фаз ---
    with right:
        st.markdown(ui.phase_bars_html(res["phase_fractions"], names, und, formulas,
                                       und_help=METRIC_HELP["undetermined"], colors=colors),
                    unsafe_allow_html=True)

    # --- Карточки метрик (ASTM — только для сталей, §6.3) ---
    gm = res.get("grain_meta", {})
    gs, astm = res["grain_size_um"], res["astm_number"]
    astm_applicable = res.get("astm_applicable", True)
    reliable = gm.get("reliable", True)
    warn = "" if reliable else " ⚠"
    grain_val = (f'{gs} <span class="unit">µm</span>{warn}' if gs is not None else "н/д")
    grain_sub = gm.get("note") if not reliable else f"±{gm.get('grain_size_std_um','')} µm"
    size_label = "Средний размер зерна" if astm_applicable else "Средний размер зёрен (гранулометрия)"
    if astm_applicable:
        second = ("Балл зерна (ASTM E112)", f"{astm}" if astm is not None else "н/д",
                  f"учтено зёрен: {gm.get('n_grains',0)}", METRIC_HELP["astm"])
    else:
        second = ("Зёрен учтено", f"{gm.get('n_grains',0)}", "балл ASTM неприменим к рудам",
                  METRIC_HELP["grain_count"])
    st.markdown(ui.metric_cards_html([
        (size_label, grain_val, grain_sub or "", METRIC_HELP["grain_size"]),
        second,
        ("Дефекты (поры/вкл.)", f"{len(res['defects'])}", "найдено", METRIC_HELP["defects"]),
    ]), unsafe_allow_html=True)

    # --- Валидация на синтетике (метрики заказчика: IoU/Хаусдорф/F1/AUC) ---
    _validation_panel(res)

    # --- Методы и подсказки ---
    with st.expander("❓ Методы и модели (что под капотом)"):
        st.markdown(METHODS_HELP)

    # --- Дисклеймер статуса метода (§7.5) ---
    st.caption("⚠ " + METHOD_DISCLAIMER)

    # --- Human-in-the-loop + действия ---
    _zones_and_actions(res, img)


def _validation_panel(res: dict) -> None:
    """Метрики качества против ground truth (только для демо-образца)."""
    gt = ss.true_mask
    if gt is None or np.asarray(gt).shape != res["mask"].shape:
        return
    ev = evaluate_segmentation(res["mask"], gt, n_classes=len(res["class_names"]),
                               prob=res.get("class_prob"), scale_um_per_px=res["scale_um_per_px"])
    with st.expander("🎯 Валидация на синтетике (ground truth) — метрики заказчика", expanded=False):
        c = st.columns(4)
        c[0].metric("mIoU", ev["iou"]["mean"])
        c[1].metric("Hausdorff, µm", ev["hausdorff_um"]["mean_hd"], f"HD95 {ev['hausdorff_um']['mean_hd95']}")
        c[2].metric("F1 (macro)", ev["classification"]["f1_macro"])
        c[3].metric("AUC (OvR)", ev["classification"].get("auc_macro_ovr"))
        per = ev["iou"]["per_class"]
        st.caption("IoU по фазам: " + " · ".join(
            f"{n} {per.get(i, per.get(str(i), '—'))}" for i, n in enumerate(res["class_names"])))
        st.caption("Сегментация: IoU + расстояние Хаусдорфа · Классификация: F1 + AUC. "
                   "Ground truth — из синтетического генератора; на реальных снимках метрики "
                   "считаются при наличии разметки.")


def _zones_and_actions(res: dict, img: np.ndarray) -> None:
    zones = res["low_conf_zones"]
    if zones:
        with st.expander(f"⌖ Зоны на верификацию ({len(zones)}) — подтверждение эксперта", expanded=False):
            st.caption("Эксперт подтверждает или отклоняет зоны, требующие верификации (human-in-the-loop).")
            for i, z in enumerate(zones):
                c = st.columns([1, 3, 2, 2])
                c[0].markdown(f"**#{i+1}**")
                c[1].markdown(f"`{z['bbox']}` · {z['note']}")
                c[2].markdown(f"уверенность **{int(z['confidence']*100)}%**")
                if c[3].checkbox("проверено", key=f"rev{i}"):
                    ss.reviewed.add(i)
                else:
                    ss.reviewed.discard(i)
            st.info(f"Проверено экспертом: {len(ss.reviewed)} из {len(zones)}")

    base = (ss.sample_name or "sample").replace(" ", "_")
    pdf = build_passport_pdf(img, res, sample_name=ss.sample_name, protocol=ss.protocol)
    a1, a2, a3, a4 = st.columns(4)
    a1.download_button("📄 Протокол (PDF)", data=pdf, file_name=f"protocol_{base}.pdf",
                       mime="application/pdf", type="primary", use_container_width=True)
    a2.download_button("🧾 Данные (JSON)",
                       data=json.dumps(result_to_json_dict(res), ensure_ascii=False, indent=2),
                       file_name=f"protocol_{base}.json", mime="application/json", use_container_width=True)
    a3.download_button("📊 Данные (CSV)", data=result_to_csv(res).encode("utf-8-sig"),
                       file_name=f"protocol_{base}.csv", mime="text/csv", use_container_width=True)
    if a4.button("✓ Подтвердить протокол", use_container_width=True):
        st.toast("Протокол подтверждён экспертом.", icon="✅")


def _pills(key: str, options: list[str]) -> str:
    """Переключатель слоёв в виде пилюль (segmented_control, с фолбэком на radio)."""
    try:
        val = st.segmented_control("Слой", options, default=options[0], key=key, label_visibility="collapsed")
        return val or options[0]
    except Exception:  # noqa: BLE001 — старые версии Streamlit
        return st.radio("Слой", options, horizontal=True, label_visibility="collapsed", key=key)


def render_preview() -> None:
    """Предпросмотр загруженного снимка до запуска анализа."""
    st.markdown(ui.CSS, unsafe_allow_html=True)
    st.markdown(ui.header_html("Предпросмотр — «Кто твой шлиф»",
                               f"{ss.sample_name or 'образец'} · проверьте снимок, затем «Проанализировать»", 1.0),
                unsafe_allow_html=True)
    st.markdown(f'<div class="kts-card"><img class="slide-img" src="{ui.img_data_uri(viz._ensure_rgb(ss.source_image))}"/>'
                f'<div class="kts-sub" style="margin-top:8px">Исходный снимок</div></div>',
                unsafe_allow_html=True)
    st.info("Нажмите **«▶️ Проанализировать»** в сайдбаре, чтобы получить протокол.")


def render_welcome() -> None:
    st.markdown(ui.CSS, unsafe_allow_html=True)
    st.markdown(ui.header_html("Протокол испытаний — «Кто твой шлиф»",
                               "QC-ассистент металлографа · протокол по ГОСТ ISO/IEC 17025 · офлайн", 1.0),
                unsafe_allow_html=True)
    st.markdown(
        """
        Сервис превращает снимок аншлифа/шлифа (оптическая микроскопия) в **протокол испытаний**:
        классификация сорта руды, сегментация талька и срастаний, доли площадей (тальк, срастания,
        оксиды) — как **предварительная воспроизводимая оценка** для исследователя, снижающая
        субъективность. Не замена аттестованному методу — ассистент эксперта.

        #### С чего начать
        1. Слева **загрузите свой SEM/OM-снимок** (или откройте демо-образец в свёрнутом блоке).
        2. Выберите **профиль материала**, подтвердите **масштаб** (µm/px).
        3. Нажмите **«▶️ Проанализировать»** → проверьте спорные зоны → **экспорт протокола (PDF/JSON/CSV)**.
        """
    )
    with st.expander("❓ Методы и модели (что под капотом)"):
        st.markdown(METHODS_HELP)
    st.info("Совет: измените «Предполагаемое загрязнение» и пересчитайте — увидите, как честно растёт "
            "оценка неопределённости (sim-to-real). Профиль «Сталь» включает балл ASTM E112.")


def main() -> None:
    inp = sidebar()
    load_inputs(inp)
    if inp["run"]:
        run_analysis(inp)
    if ss.result is not None:
        render_passport()
    elif ss.source_image is not None:
        render_preview()
    else:
        render_welcome()


if __name__ == "__main__":
    main()
