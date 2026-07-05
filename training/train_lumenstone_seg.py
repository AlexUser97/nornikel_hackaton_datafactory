# -*- coding: utf-8 -*-
"""Сегментация сульфидов/матрицы/магнетита на внешнем датасете LumenStone S2 (МГУ,
Cu-Ni сульфидные руды, отражённый свет — тот же домен, что аншлифы Норникеля).

Даёт РЕАЛЬНЫЙ пиксельный GT (которого нет в датасете кейса) -> честный IoU/F1 по ТЗ
и обученный сегментатор сульфидов для нашей задачи (обычные/тонкие срастания = сульфиды).

3 класса: 0=матрица(нерудное), 1=сульфид(Po/Pn/Ccp), 2=магнетит.
Запуск на T4:  python3 train_lumenstone_seg.py --data lumen_seg --epochs 80
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

INPUT = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NCLS = 3
NAMES = ["матрица", "сульфид", "магнетит"]


class LS(Dataset):
    def __init__(self, root, split, train):
        self.root, self.split = Path(root), split
        self.items = [r["stem"] for r in csv.DictReader(open(Path(root)/"manifest.csv")) if r["split"] == split]
        if train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(INPUT, INPUT), scale=(0.4, 1.0), ratio=(0.75, 1.33)),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.35, 0.35, p=0.85),
                A.HueSaturationValue(15, 25, 18, p=0.6), A.RandomGamma((75, 135), p=0.5),
                A.CLAHE(3.0, p=0.3), A.GaussNoise(p=0.25),
            ])
        else:
            self.aug = A.Compose([A.Resize(INPUT, INPUT)])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        st = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(self.root/"images"/self.split/f"{st}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.root/"masks"/self.split/f"{st}.png"), cv2.IMREAD_GRAYSCALE)
        a = self.aug(image=img, mask=m); img, m = a["image"], a["mask"]
        x = ((img.astype(np.float32)/255.0 - MEAN)/STD).transpose(2, 0, 1)
        return torch.from_numpy(x).float(), torch.from_numpy(m.astype(np.int64))


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    inter = np.zeros(NCLS); union = np.zeros(NCLS)
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        y = y.numpy()
        for c in range(NCLS):
            pc, yc = (p == c), (y == c)
            inter[c] += np.logical_and(pc, yc).sum()
            union[c] += np.logical_or(pc, yc).sum()
    iou = inter / np.maximum(union, 1)
    return iou, float(iou.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lumen_seg"); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default="weights/lumenstone_seg.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev)
    tl = DataLoader(LS(args.data, "train", True), batch_size=6, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(LS(args.data, "test", False), batch_size=4, shuffle=False, num_workers=2)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=NCLS).to(dev)
    dice = smp.losses.DiceLoss(mode="multiclass")
    ce = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0, 2.0], device=dev))  # магнетит редкий
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); logit = model(x)
            loss = dice(logit, y) + ce(logit, y); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if ep % 4 == 0 or ep == args.epochs-1:
            iou, miou = evaluate(model, vl, dev); flag = ""
            if miou > best:
                best = miou; torch.save({"state_dict": model.state_dict(), "classes": NAMES,
                                         "model_version": "lumenstone-seg-2026-07-05"}, args.out); flag = "  <-- best"
            print(f"ep{ep:02d} loss{tot/len(tl):.3f} mIoU={miou:.3f} | "
                  f"матрица={iou[0]:.3f} сульфид={iou[1]:.3f} магнетит={iou[2]:.3f}{flag}", flush=True)
    print("BEST mIoU=", round(best, 3), "| TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
