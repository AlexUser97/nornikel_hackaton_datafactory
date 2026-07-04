# -*- coding: utf-8 -*-
"""Обучение сегментации талька с ПИКСЕЛЬНОЙ разметкой эксперта.

В отличие от train_talc.py (слабая супервизия по меткам папок), здесь есть
настоящая GT-маска: заливка замкнутых синих контуров геолога из папки
«Области оталькования» (58 снимков) + сбалансированные негативы (рядовые/тонкие,
талька нет). Проблема дисбаланса классов (позитивов много меньше) решается
oversampling позитивов + pos_weight в BCE (по совету команды).

Даёт веса weights/talc_unet.pt, совместимые с shlif/talc_model.py:
    ckpt = {"state_dict": ..., "talc_threshold": <откалиброванный порог>}

Запуск на T4:  python3 train_talc_supervised.py --data talc_seg_data --epochs 60
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

INPUT = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_manifest(root: Path):
    rows = []
    with open(root / "manifest.csv") as f:
        for r in csv.DictReader(f):
            rows.append((r["stem"], r["kind"], float(r["talc_frac"])))
    return rows


class TalcSet(Dataset):
    def __init__(self, root: Path, items, train: bool):
        self.root = root
        self.items = items
        self.train = train
        if train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(INPUT, INPUT), scale=(0.5, 1.0), ratio=(0.8, 1.25), p=1.0),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                # цвет-адаптация под разные условия съёмки (вывод LumenStone)
                A.RandomBrightnessContrast(0.4, 0.4, p=0.9),
                A.HueSaturationValue(18, 30, 22, p=0.8),
                A.RandomGamma((70, 140), p=0.6),
                A.CLAHE(clip_limit=3.0, p=0.4),
                A.GaussNoise(p=0.3),
            ])
        else:
            self.aug = A.Compose([A.Resize(INPUT, INPUT)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, kind, frac = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / "images" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.root / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        m = (m > 127).astype(np.uint8)
        a = self.aug(image=img, mask=m)
        img, m = a["image"], a["mask"]
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(x).float(), torch.from_numpy(m).float().unsqueeze(0)


def build_loaders(root: Path, val_frac=0.2, seed=0, oversample=4):
    rows = read_manifest(root)
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r[1] == "pos"]
    neg = [r for r in rows if r[1] == "neg"]
    rng.shuffle(pos); rng.shuffle(neg)
    nvp, nvn = max(6, int(len(pos) * val_frac)), max(20, int(len(neg) * val_frac))
    val = pos[:nvp] + neg[:nvn]
    tr_pos, tr_neg = pos[nvp:], neg[nvn:]
    # oversample позитивов, чтобы сбалансировать редкий класс
    train = tr_pos * oversample + tr_neg
    rng.shuffle(train)
    print(f"train: {len(train)} (pos×{oversample}={len(tr_pos)*oversample}, neg={len(tr_neg)}) | "
          f"val: {len(val)} (pos={nvp}, neg={nvn})")
    tl = DataLoader(TalcSet(root, train, True), batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(TalcSet(root, val, False), batch_size=8, shuffle=False, num_workers=2)
    return tl, vl, val


@torch.no_grad()
def evaluate(model, vl, device):
    model.eval()
    inter = union = tp = fp = fn = 0
    probs_by_img, gt_pos = [], []
    for x, y in vl:
        x = x.to(device)
        p = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        y = y[:, 0].numpy()
        for pi, yi in zip(p, y):
            pm = pi > 0.5
            gi = yi > 0.5
            inter += np.logical_and(pm, gi).sum()
            union += np.logical_or(pm, gi).sum()
            tp += np.logical_and(pm, gi).sum(); fp += np.logical_and(pm, ~gi).sum()
            fn += np.logical_and(~pm, gi).sum()
            probs_by_img.append(pi.mean()); gt_pos.append(gi.mean() > 0.012)
    iou = inter / (union + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    # калибровка порога доли талька для детекции оталькованной (image-level)
    probs_by_img = np.array(probs_by_img); gt_pos = np.array(gt_pos)
    best_thr, best_df1 = 0.10, 0
    for t in np.linspace(0.02, 0.4, 39):
        pred = probs_by_img > t
        dtp = (pred & gt_pos).sum(); dfp = (pred & ~gt_pos).sum(); dfn = (~pred & gt_pos).sum()
        df1 = 2 * dtp / (2 * dtp + dfp + dfn + 1e-6)
        if df1 > best_df1:
            best_df1, best_thr = df1, t
    return dict(iou=iou, pix_f1=f1, det_f1=best_df1, thr=float(best_thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="talc_seg_data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="weights/talc_unet.pt")
    ap.add_argument("--pos-weight", type=float, default=3.0)
    args = ap.parse_args()
    root = Path(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tl, vl, _ = build_loaders(root)
    model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_key, best = -1, None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x)
            loss = dice(logit, y) + bce(logit, y)
            loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if ep % 3 == 0 or ep == args.epochs - 1:
            m = evaluate(model, vl, device)
            key = 0.5 * m["iou"] + 0.5 * m["det_f1"]  # баланс пикселей и детекции
            flag = ""
            if key > best_key:
                best_key = key
                best = {"state_dict": model.state_dict(), "talc_threshold": m["thr"]}
                torch.save(best, args.out)
                flag = "  <-- best (saved)"
            print(f"ep {ep:02d} loss {tot/len(tl):.3f} | IoU {m['iou']:.3f} pixF1 {m['pix_f1']:.3f} "
                  f"detF1 {m['det_f1']:.3f} thr {m['thr']:.2f}{flag}")
    print("done. best key:", round(best_key, 4), "| saved:", args.out)


if __name__ == "__main__":
    main()
