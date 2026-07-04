"""Слабо-supervised сегментация талька (ТЗ task-3) — обучение на T4.

В датасете НЕТ пиксельной разметки талька (синие линии оказались единичными
указателями, а не контурами). Поэтому используем надёжные метки папок:
оталькованные руды (>10% талька) дают положительный сигнал — тёмная нерудная
матрица в них и есть тальк; рядовые/труднообогатимые — отрицательный (там та же
тёмная зона = силикаты). U-Net учится отличать текстуру/тон талька от силикатов.

Модель: U-Net (segmentation-models-pytorch, ResNet-18, ImageNet-претрейн — открытая
лицензия). Валидация — честно на уровне снимка: доля предсказанного талька -> порог
-> детекция «оталькованной руды», F1/precision/recall против меток папок.
"""
import json
import time
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(7)
np.random.seed(7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("~/shlif_train/talc_ds").expanduser()
OUT = Path("~/shlif_train").expanduser()
TALC_THR = 0.10  # порог доли талька -> оталькованная руда


class DS(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train
        # Сильная фото-аугментация (яркость/контраст/цвет/гамма) — рекомендация
        # организаторов: учитывать цвет пикселей и устойчивость к освещению.
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.35, 0.35, p=0.8),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=15, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.GaussNoise(p=0.25),
        ]) if train else None
        self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, y = self.items[i]
        img = np.asarray(Image.open(ROOT / "img" / f"{stem}.png").convert("RGB").resize((384, 384)))
        msk = np.asarray(Image.open(ROOT / "msk" / f"{stem}.png").convert("L").resize((384, 384)))
        if self.aug:
            a = self.aug(image=img, mask=msk); img, msk = a["image"], a["mask"]
        img = self.norm(image=img)["image"]
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        m = torch.from_numpy((msk > 127).astype(np.float32))[None]
        return x, m, y


def collect():
    labels = json.loads((ROOT / "labels.json").read_text())
    items = list(labels.items())
    rng = np.random.default_rng(7)
    rng.shuffle(items)
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    nvp, nvn = max(1, len(pos) // 6), max(1, len(neg) // 6)
    val = pos[:nvp] + neg[:nvn]
    train_pos, train_neg = pos[nvp:], neg[nvn:]
    # Балансируем классы: оверсэмплим позитивы (талька мало) под число негативов.
    if train_pos:
        k = max(1, round(len(train_neg) / len(train_pos)))
        train_pos = train_pos * k
    train = train_pos + train_neg
    rng.shuffle(train)
    return train, val


def run():
    train_items, val_items = collect()
    print(f"train {len(train_items)} / val {len(val_items)} "
          f"(pos train {sum(y for _,y in train_items)}, pos val {sum(y for _,y in val_items)})", flush=True)
    tl = DataLoader(DS(train_items, True), batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(DS(val_items, False), batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=18)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()

    best_f1, best = 0.0, {}
    for ep in range(18):
        model.train(); t = time.time()
        for x, m, _ in tl:
            x, m = x.to(DEV), m.to(DEV)
            opt.zero_grad()
            out = model(x)
            loss = dice(out, m) + bce(out, m)
            loss.backward(); opt.step()
        sched.step()

        model.eval()
        ys, talc_fracs = [], []
        with torch.no_grad():
            for x, m, y in vl:
                p = torch.sigmoid(model(x.to(DEV))).cpu().numpy()
                for k in range(len(y)):
                    talc_fracs.append(float((p[k, 0] > 0.5).mean()))
                    ys.append(int(y[k]))
        ys = np.array(ys); pred = (np.array(talc_fracs) > TALC_THR).astype(int)
        f1 = f1_score(ys, pred, zero_division=0)
        pr = precision_score(ys, pred, zero_division=0)
        rc = recall_score(ys, pred, zero_division=0)
        mean_pos = float(np.mean([f for f, y in zip(talc_fracs, ys) if y == 1]) if (ys == 1).any() else 0)
        mean_neg = float(np.mean([f for f, y in zip(talc_fracs, ys) if y == 0]) if (ys == 0).any() else 0)
        print(f"ep{ep:02d} loss{loss.item():.3f} | оталькован-детекция F1={f1:.3f} P={pr:.3f} R={rc:.3f} "
              f"| talc%% pos={mean_pos*100:.0f} neg={mean_neg*100:.0f} ({time.time()-t:.0f}s)", flush=True)
        if f1 >= best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "arch": "unet_resnet18",
                        "talc_threshold": TALC_THR}, OUT / "talc_unet.pt")
            best = {"epoch": ep, "otalk_f1": float(f1), "precision": float(pr), "recall": float(rc),
                    "mean_talc_pos": mean_pos, "mean_talc_neg": mean_neg}
            (OUT / "talc_metrics.json").write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print("BEST оталькован-детекция F1 =", round(best_f1, 3), flush=True)
    print("TALC_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    run()
