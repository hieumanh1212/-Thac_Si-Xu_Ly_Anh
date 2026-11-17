"""Đánh giá độ chính xác trên một thư mục ảnh.

Nhãn đúng lấy từ tên file:  <dòng1>_<dòng2>.<ext>   vd. 59X1_12345.jpg, 29B1_1234.png
(các phần sau dấu '_' thứ hai bị bỏ qua, vd. 59X1_12345_02.jpg vẫn hợp lệ)

Chỉ số:
    - Tỉ lệ phát hiện biển số
    - Độ chính xác toàn biển (đúng hết mọi ký tự)
    - Độ chính xác mức ký tự (dựa trên khoảng cách Levenshtein)

Chạy:  python tools/evaluate.py samples  [--csv ket_qua.csv]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpr import LicensePlateRecognizer  # noqa: E402
from lpr.utils import read_image  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print("Không có ảnh trong thư mục.")
        return

    rec = LicensePlateRecognizer()
    rows, n_found, n_exact, char_total, char_err, t_total = [], 0, 0, 0, 0, 0.0
    for p in files:
        parts = p.stem.split("_")
        gt = norm("".join(parts[:2]))
        img = read_image(p)
        if img is None:
            print(f"[bỏ qua] {p.name}")
            continue
        r = rec.run(img, with_steps=False)
        pred = norm(r["text"])
        dist = levenshtein(pred, gt)
        n_found += r["found"]
        n_exact += pred == gt
        char_total += len(gt)
        char_err += min(dist, len(gt))
        t_total += r["time_ms"]["total"]
        mark = "OK " if pred == gt else "SAI"
        print(f"{mark} {p.name:30s} GT={gt:10s} PRED={pred:10s} ({r['time_ms']['total']:.0f} ms)")
        rows.append([p.name, gt, pred, int(r["found"]), int(pred == gt), dist, r["confidence"],
                     r["time_ms"]["total"]])

    n = len(rows)
    print("\n========== KẾT QUẢ ==========")
    print(f"Số ảnh                : {n}")
    print(f"Phát hiện biển số     : {n_found}/{n} = {n_found / n:.1%}")
    print(f"Đúng toàn biển        : {n_exact}/{n} = {n_exact / n:.1%}")
    print(f"Độ chính xác ký tự    : {1 - char_err / max(char_total, 1):.1%}")
    print(f"Thời gian trung bình  : {t_total / n:.0f} ms/ảnh")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "ground_truth", "prediction", "found", "exact", "edit_distance",
                        "confidence", "time_ms"])
            w.writerows(rows)
        print(f"Đã lưu {args.csv}")


if __name__ == "__main__":
    main()
