"""Train CNN nhận dạng ký tự biển số (36 lớp).

Chạy (CPU, ~10-15 phút):  python tools/train_char_cnn.py      ->  models/char_cnn.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpr.char_cnn import CLASSES, LETTERS_IDX, SIZE, build_model  # noqa: E402


def augment(img, rng=np.random):
    """Tăng cường dữ liệu mô phỏng ảnh thực tế: xoay, co giãn, dịch, độ dày nét, mờ, sáng tối, nhiễu."""
    u8 = (img * 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rng.uniform(-8, 8), rng.uniform(0.85, 1.12))
    M[:, 2] += rng.uniform(-2.5, 2.5, size=2)
    u8 = cv2.warpAffine(u8, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    r = rng.rand()
    if r < 0.2:
        u8 = cv2.erode(u8, np.ones((2, 2), np.uint8))   # nét chữ (tối) dày hơn
    elif r < 0.4:
        u8 = cv2.dilate(u8, np.ones((2, 2), np.uint8))  # nét chữ mảnh hơn
    if rng.rand() < 0.3:
        u8 = cv2.GaussianBlur(u8, (3, 3), rng.uniform(0.3, 1.0))
    out = u8.astype(np.float32) / 255.0
    out = (out - 0.5) * rng.uniform(0.7, 1.2) + 0.5 + rng.uniform(-0.12, 0.12)
    out += rng.normal(0, rng.uniform(0, 0.05), out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


class CharDataset(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = augment(self.X[i]) if self.train else self.X[i]
        return torch.from_numpy(img[None]), int(self.y[i])


def evaluate(model, loader):
    model.eval()
    ok = n = lok = ln = 0
    letters = torch.tensor(LETTERS_IDX)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(1)
            ok += (pred == yb).sum().item()
            n += len(yb)
            m = torch.isin(yb, letters)
            lok += (pred[m] == yb[m]).sum().item()
            ln += m.sum().item()
    return ok / max(n, 1), lok / max(ln, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets/chars.npz"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    d = np.load(args.data)
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]

    # Chữ cái ít mẫu hơn chữ số nhiều lần -> lấy mẫu cân bằng (trọng số ~ 1/sqrt(số mẫu của lớp))
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    sampler = WeightedRandomSampler(torch.tensor(weights[ytr]), num_samples=len(ytr), replacement=True)
    tr = DataLoader(CharDataset(Xtr, ytr, True), batch_size=args.batch, sampler=sampler)
    va = DataLoader(CharDataset(Xva, yva, False), batch_size=512)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    dst = ROOT / "models/char_cnn.pt"
    dst.parent.mkdir(exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
        acc, lacc = evaluate(model, va)
        saved = ""
        if acc > best:
            best = acc
            torch.save(model.state_dict(), dst)
            saved = "  [lưu]"
        print(f"Epoch {ep:2d}/{args.epochs}  loss={tot / len(ytr):.4f}  val_acc={acc:.4f}  "
              f"val_acc(chữ cái)={lacc:.4f}  {time.time() - t0:.0f}s{saved}", flush=True)
    print(f"Hoàn tất. val_acc tốt nhất = {best:.4f} -> {dst}")


if __name__ == "__main__":
    main()
