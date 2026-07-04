"""Классификатор геолого-технологического сорта руды (ТЗ task-3) — обучение на T4.

Transfer learning на предобученной модели (открытая лицензия) поверх папок-меток
(рядовые / труднообогатимые / оталькованные). Оптимизируем целевую метрику ТЗ — F1
(и AUC).

Аугментация — по практике LumenStone (MSU, аншлифы руд с 30 месторождений с разными
условиями съёмки): сильная color-adaptation (яркость/контраст/цвет/гамма/CLAHE),
чтобы модель опиралась на структуру (степень замещения), а не на освещение/оттенок
конкретного месторождения → лучше обобщение рядовая↔труднообогатимая.
"""
import json
import time
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


class DS(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train
        if train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), ratio=(0.85, 1.18)),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.4, 0.4, p=0.85),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                A.RandomGamma(gamma_limit=(75, 130), p=0.5),
                A.CLAHE(clip_limit=3.0, p=0.4),          # адаптация освещения (LumenStone V1)
                A.GaussNoise(p=0.25),
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
    items = []
    for ci, c in enumerate(CLASSES):
        for p in (ROOT / c).glob("*.jpg"):
            items.append((p, ci))
    rng = np.random.default_rng(7)
    rng.shuffle(items)
    n_val = int(0.15 * len(items))
    return items[n_val:], items[:n_val]


def run():
    train_items, val_items = collect()
    counts = np.bincount([y for _, y in train_items], minlength=3)
    print("train/val:", len(train_items), len(val_items), "| train counts:", counts.tolist(), flush=True)
    weights = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))), dtype=torch.float32, device=DEV)

    tl = DataLoader(DS(train_items, True), batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(DS(val_items, False), batch_size=64, shuffle=False, num_workers=4)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    N_EP = 32
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EP)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best_f1, best = 0.0, {}
    for ep in range(N_EP):
        model.train()
        t = time.time()
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        ys, ps, probs = [], [], []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(DEV))
                pr = torch.softmax(out, 1).cpu().numpy()
                probs.append(pr)
                ps.extend(pr.argmax(1))
                ys.extend(y.numpy())
        probs = np.concatenate(probs)
        f1m = f1_score(ys, ps, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        print(f"ep{ep:02d} loss{loss.item():.3f} F1_macro={f1m:.3f} AUC={auc:.3f} ({time.time()-t:.0f}s)", flush=True)
        if f1m > best_f1:
            best_f1 = f1m
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, OUT / "sort_classifier.pt")
            best = {"epoch": ep, "f1_macro": float(f1m), "auc_macro_ovr": float(auc),
                    "report": classification_report(ys, ps, target_names=CLASSES, zero_division=0, output_dict=True)}
            (OUT / "metrics.json").write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print("BEST F1_macro=", round(best_f1, 3), flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    run()
