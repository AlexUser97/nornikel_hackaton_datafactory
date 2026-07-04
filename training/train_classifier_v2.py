"""Классификатор сорта руды v2 (ТЗ task-3) — цель: F1 ≥ 0.90 на типе срастаний
(рядовая ↔ труднообогатимая). Улучшения относительно train_classifier.py:

* стратифицированный train/val сплит (устойчивее оценка на малых данных);
* **EMA весов** (экспоненциальное сглаживание) — убирает «удачную» раннюю эпоху,
  даёт стабильно лучшую генерализацию;
* выбор лучшей модели по метрике ТЗ — **F1 по рядовая/труднообогатимая** (не по
  общей macro, где оталькованная с малым n шумит);
* сильная color-adaptation аугментация (LumenStone) сохранена.

Оталькованная в классификаторе остаётся (3 класса) — её вероятность нужна пайплайну
как второй сигнал ансамбля; но модель выбираем по качеству различения срастаний.
"""
import json
import time
from copy import deepcopy
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

torch.manual_seed(7)
np.random.seed(7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("~/shlif_train/train_ds").expanduser()
CLASSES = ["ryadovye", "trudnoobogatimye", "otalkovannye"]
OUT = Path("~/shlif_train").expanduser()
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
N_EP = 45


class DS(Dataset):
    def __init__(self, items, train):
        self.items = items
        if train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.55, 1.0), ratio=(0.8, 1.25)),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.4, 0.4, p=0.85),
                A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=30, val_shift_limit=22, p=0.75),
                A.RandomGamma(gamma_limit=(70, 140), p=0.5),
                A.CLAHE(clip_limit=3.0, p=0.4),
                A.GaussNoise(p=0.25),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                                hole_width_range=(0.05, 0.15), p=0.25),
                A.Normalize(mean=_MEAN, std=_STD),
            ])
        else:
            self.aug = A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224), A.Normalize(mean=_MEAN, std=_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = np.asarray(Image.open(p).convert("RGB"))
        x = self.aug(image=img)["image"]
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), y


def collect():
    """Стратифицированный сплит: по 15% валидации ИЗ КАЖДОГО класса."""
    rng = np.random.default_rng(7)
    train, val = [], []
    for ci, c in enumerate(CLASSES):
        files = sorted((ROOT / c).glob("*.jpg"))
        rng.shuffle(files)
        nv = max(int(0.15 * len(files)), 3)
        val += [(p, ci) for p in files[:nv]]
        train += [(p, ci) for p in files[nv:]]
    rng.shuffle(train)
    return train, val


class EMA:
    """Экспоненциальное скользящее среднее весов — стабильная генерализация."""
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m, alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


@torch.no_grad()
def evaluate(model, vl):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in vl:
        pr = torch.softmax(model(x.to(DEV)), 1).cpu().numpy()
        probs.append(pr); ps.extend(pr.argmax(1)); ys.extend(y.numpy())
    probs = np.concatenate(probs)
    ys, ps = np.array(ys), np.array(ps)
    # F1 по типу срастаний (метрика ТЗ): классы 0=рядовые, 1=труднообогатимые
    m = np.isin(ys, [0, 1])
    f1_ig = f1_score(ys[m], ps[m], labels=[0, 1], average="macro", zero_division=0)
    f1_all = f1_score(ys, ps, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return f1_ig, f1_all, auc, ys, ps


def run():
    train_items, val_items = collect()
    counts = np.bincount([y for _, y in train_items], minlength=3)
    print("train/val:", len(train_items), len(val_items), "| counts:", counts.tolist(), flush=True)
    weights = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))), dtype=torch.float32, device=DEV)

    tl = DataLoader(DS(train_items, True), batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(DS(val_items, False), batch_size=64, shuffle=False, num_workers=4)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(DEV)
    ema = EMA(model, decay=0.998)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EP)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best_f1ig, best = 0.0, {}
    for ep in range(N_EP):
        model.train(); t = time.time()
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            # MixUp (alpha=0.2) — регуляризация на малых данных
            if np.random.rand() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(x.size(0), device=DEV)
                x = lam * x + (1 - lam) * x[idx]
                loss = lam * crit(model(x), y) + (1 - lam) * crit(model(x), y[idx])
            else:
                loss = crit(model(x), y)
            loss.backward(); opt.step(); ema.update(model)
        sched.step()

        f1_ig, f1_all, auc, _, _ = evaluate(ema.shadow, vl)
        flag = ""
        if f1_ig > best_f1ig:
            best_f1ig = f1_ig
            torch.save({"state_dict": ema.shadow.state_dict(), "classes": CLASSES}, OUT / "sort_classifier.pt")
            _, _, _, ys, ps = evaluate(ema.shadow, vl)
            best = {"epoch": ep, "f1_intergrowth": float(f1_ig), "f1_macro": float(f1_all),
                    "auc_macro_ovr": float(auc),
                    "report": classification_report(ys, ps, target_names=CLASSES, zero_division=0, output_dict=True)}
            (OUT / "metrics.json").write_text(json.dumps(best, ensure_ascii=False, indent=2))
            flag = "  <-- best (saved, EMA)"
        print(f"ep{ep:02d} loss{loss.item():.3f} F1_срастаний={f1_ig:.3f} F1_macro={f1_all:.3f} "
              f"AUC={auc:.3f} ({time.time()-t:.0f}s){flag}", flush=True)
    print("BEST F1_срастаний=", round(best_f1ig, 3), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    run()
