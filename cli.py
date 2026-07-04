"""CLI-обёртка над analyze(): снимок -> паспорт (PDF + JSON) без UI.

Страховочный трос на случай, если дашборд не поднимется на сцене — паспорт всё
равно можно получить одной командой. Также удобно для пакетной обработки.

Примеры:
    python cli.py --demo --out data/
    python cli.py -i anshlif.png --scale 0.5 --out data/
    python cli.py -i panorama.jpg --tile --out data/
    python cli.py --batch ./samples --out ./reports
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from shlif import analyze
from shlif.config import DEFAULT_PROFILE, MATERIAL_PROFILES
from shlif.demo_data import synthetic_microstructure
from shlif.io_utils import load_image, result_to_csv, result_to_json_dict
from shlif.ore_petro import ore_petro_segment
from shlif.report import build_passport_pdf
from shlif.simto_real import DEMO_PRESETS, corrupt

_PRESET_ALIASES = {
    "clean": "Чисто (как есть)",
    "mild": "Лёгкое загрязнение",
    "medium": "Среднее загрязнение",
    "heavy": "Сильное загрязнение",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Кто твой шлиф — CLI-протокол испытаний")
    ap.add_argument("-i", "--image", help="Путь к SEM/OM-снимку (jpg/png/tiff)")
    ap.add_argument("--demo", action="store_true", help="Использовать синтетический демо-образец")
    ap.add_argument("--scale", type=float, default=0.35, help="Масштаб µm/px (по умолч. 0.35)")
    ap.add_argument("--profile", choices=list(MATERIAL_PROFILES), default=None,
                    help="Профиль материала (имена фаз, применимость ASTM)")
    ap.add_argument("--corrupt", choices=list(_PRESET_ALIASES), help="Демо sim-to-real загрязнение")
    ap.add_argument("--runs", type=int, default=5, help="Прогонов ансамбля неуверенности")
    ap.add_argument("--customer", default=None, help="Заказчик (в шапку протокола)")
    ap.add_argument("--lab", default=None, help="Наименование лаборатории (в шапку протокола)")
    ap.add_argument("--evaluate", action="store_true",
                    help="Оценить метрики (IoU/Хаусдорф/F1/AUC) на синтетике с ground truth (только --demo)")
    ap.add_argument("--tile", type=int, nargs="?", const=1024, default=0,
                    help="Тайлинг панорам: размер плитки (без значения — 1024). Крупные снимки тайлятся авто.")
    ap.add_argument("--batch", default=None, help="Каталог со снимками — пакетная обработка серии")
    ap.add_argument("--out", default="data", help="Каталог для отчётов (по умолч. data/)")
    args = ap.parse_args(argv)

    # --- профиль материала (по умолчанию — задача ТЗ: аншлиф руды) ---
    profile_name = args.profile or DEFAULT_PROFILE
    profile = MATERIAL_PROFILES[profile_name]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol_meta = {"customer": args.customer, "lab_name": args.lab, "method": profile["method"]}

    def process(image, name: str, true_mask=None) -> None:
        seg_fn = ore_petro_segment if profile.get("segmenter") == "ore_petro" else None
        runs = 1 if seg_fn is not None else args.runs  # ore-petro детерминирован
        # Тайлинг: явный --tile или авто для крупных панорам (>5000 px по стороне).
        tile = args.tile or (1024 if max(image.shape[:2]) > 5000 else 0)
        print(f"Анализ «{name}» · профиль «{profile_name}» (scale={args.scale} µm/px, runs={runs}"
              f"{', тайлинг '+str(tile) if tile else ''})…", file=sys.stderr)
        res = analyze(image, scale_um_per_px=args.scale, class_names=profile["class_names"],
                      n_runs=runs, seed=7,
                      astm_applicable=profile["astm_applicable"], material_profile=profile_name,
                      class_colors=profile.get("colors"), soft_segment_fn=seg_fn, tile=tile)
        (out / f"protocol_{name}.pdf").write_bytes(
            build_passport_pdf(image, res, sample_name=name, protocol=protocol_meta))
        (out / f"protocol_{name}.json").write_text(
            json.dumps(result_to_json_dict(res), ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"protocol_{name}.csv").write_text(result_to_csv(res), encoding="utf-8-sig")

        if res.get("ore_class"):
            print(f"Классификация: {res['ore_class']['verdict']} ({res['ore_class']['rule']})")
        print("Фазы: " + ", ".join(f"{k} {v['value']*100:.1f}±{v['ci']*100:.1f}%"
                                    for k, v in res["phase_fractions"].items()))
        print(f"Уверенность {res['overall_confidence']*100:.0f}% | «не определено» "
              f"{res['undetermined_fraction']*100:.0f}% | зон на верификацию {len(res['low_conf_zones'])}")
        if args.evaluate and true_mask is not None:
            _print_metrics(res, true_mask, out, name)
        print(f"Сохранено: protocol_{name}.pdf/json/csv в {out}")

    # --- пакетный режим: папка со снимками ---
    if args.batch:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        files = sorted(p for p in Path(args.batch).iterdir() if p.suffix.lower() in exts)
        if not files:
            ap.error(f"в каталоге {args.batch} нет изображений")
            return 2
        print(f"Пакетная обработка: {len(files)} снимков из {args.batch}", file=sys.stderr)
        for p in files:
            process(load_image(str(p)), p.stem)
        # Лог параметров запуска (воспроизводимость, требование ТЗ).
        (out / "batch_log.json").write_text(json.dumps({
            "profile": profile_name, "scale_um_per_px": args.scale, "runs": args.runs,
            "n_images": len(files), "files": [p.name for p in files],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    # --- одиночный режим ---
    true_mask = None
    if args.demo:
        probs = (0.40, 0.52, 0.08) if profile.get("classify") else (0.30, 0.25, 0.45)
        image, true_mask = synthetic_microstructure(seed=7, return_labels=True, phase_probs=probs)
        name = "demo"
    elif args.image:
        image = load_image(args.image)
        name = Path(args.image).stem
    else:
        ap.error("нужно указать --image ПУТЬ, --demo или --batch КАТАЛОГ")
        return 2

    if args.corrupt:
        image = corrupt(image, **DEMO_PRESETS[_PRESET_ALIASES[args.corrupt]])

    process(image, name, true_mask)
    return 0


def _print_metrics(res: dict, true_mask, out: Path, name: str) -> None:
    """Печатает и сохраняет метрики против ground truth (IoU/Хаусдорф/F1/AUC)."""
    import cv2 as _cv2

    from shlif.evaluation import evaluate_segmentation
    gt = true_mask
    if gt.shape != res["mask"].shape:
        gt = _cv2.resize(gt, (res["mask"].shape[1], res["mask"].shape[0]), interpolation=_cv2.INTER_NEAREST)
    ev = evaluate_segmentation(res["mask"], gt, n_classes=len(res["class_names"]),
                               prob=res.get("class_prob"), scale_um_per_px=res["scale_um_per_px"])
    print(f"Метрики: mIoU={ev['iou']['mean']} HD95={ev['hausdorff_um']['mean_hd95']}µm "
          f"F1={ev['classification']['f1_macro']} AUC={ev['classification'].get('auc_macro_ovr')}")
    (out / f"metrics_{name}.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
