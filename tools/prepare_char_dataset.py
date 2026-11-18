"""Chuẩn bị dữ liệu ký tự để train CNN.

Nguồn: "Vietnamese License Plate Recognition (YOLO)" - nhãn hộp bao từng ký tự, lớp 0-9 = chữ số, 10-35 = A-Z
    https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md
    Tải file zip, giải nén vào datasets/raw_rec/  (thư mục con yolo_plate_ocr_dataset/)

Biển nào có chuỗi trùng với nhãn tập kiểm tra (tools/test_labels.csv) bị LOẠI để tránh rò rỉ dữ liệu.
Chia train/val theo ẢNH biển số (không theo ký tự) để ký tự của cùng một biển không lọt sang cả hai tập.

Chạy:  python tools/prepare_char_dataset.py      ->  datasets/chars.npz
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpr.char_cnn import CLASSES, char_to_input  # noqa: E402


def read_boxes(txt, w, h):
    boxes = []
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append((int(p[0]), (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
    return boxes


def plate_text(boxes):
    """Khôi phục chuỗi biển số từ nhãn: tách dòng theo y, sắp theo x."""
    if not boxes:
        return ""
    boxes = sorted(boxes, key=lambda b: b[2] + b[4] / 2)
    med_h = np.median([b[4] for b in boxes])
    rows, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if (b[2] + b[4] / 2) - (cur[-1][2] + cur[-1][4] / 2) > 0.5 * med_h:
            rows.append(cur)
            cur = [b]
        else:
            cur.append(b)
    rows.append(cur)
    return "".join(CLASSES[b[0]] for r in rows for b in sorted(r, key=lambda b: b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets/raw_rec/yolo_plate_ocr_dataset"))
    ap.add_argument("--out", default=str(ROOT / "datasets/chars.npz"))
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()

    with open(ROOT / "tools/test_labels.csv", encoding="utf-8") as f:
        test_texts = {r["text"] for r in csv.DictReader(f)}

    files = sorted(p for p in Path(args.src).glob("*.jpg")
                   if p.with_suffix(".txt").exists() and p.name != "0.jpg")  # 0.jpg là bảng mẫu font
    random.Random(0).shuffle(files)
    n_val = int(len(files) * args.val)

    X, y = {"train": [], "val": []}, {"train": [], "val": []}
    skipped = 0
    for i, p in enumerate(files):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        boxes = read_boxes(p.with_suffix(".txt"), w, h)
        if plate_text(boxes) in test_texts:
            skipped += 1
            continue
        split = "val" if i < n_val else "train"
        for c, bx, by, bw, bh in boxes:
            x0, y0, x1, y1 = int(round(bx)), int(round(by)), int(round(bx + bw)), int(round(by + bh))
            crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if crop.shape[0] >= 6 and crop.shape[1] >= 2:
                X[split].append(char_to_input(crop))
                y[split].append(c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_train=np.stack(X["train"]), y_train=np.array(y["train"]),
                        X_val=np.stack(X["val"]), y_val=np.array(y["val"]))
    print(f"Train: {len(y['train'])} ký tự | Val: {len(y['val'])} ký tự | "
          f"Loại {skipped} biển trùng tập kiểm tra -> {out}")


if __name__ == "__main__":
    main()
