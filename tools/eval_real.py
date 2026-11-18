"""Đánh giá độ chính xác nhận dạng trên ẢNH XE MÁY THẬT đã gán nhãn chữ (tools/test_labels.csv).

Ảnh lấy từ bộ "Vietnamese License Plate Detection (YOLO)", giải nén vào datasets/raw_det/.
So sánh 2 bộ nhận dạng ký tự: EasyOCR (đa dụng) và CNN (train riêng cho font biển số VN).

Chạy:  python tools/eval_real.py               (so sánh cả hai)
       python tools/eval_real.py --ocr cnn     (một bộ)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpr import LicensePlateRecognizer  # noqa: E402
from lpr.utils import read_image  # noqa: E402


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


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
    ap.add_argument("--images", default=str(ROOT / "datasets/raw_det/yolo_plate_dataset"))
    ap.add_argument("--labels", default=str(ROOT / "tools/test_labels.csv"))
    ap.add_argument("--ocr", default="both", choices=["both", "easyocr", "cnn"])
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        labels = [(r["file"], r["text"]) for r in csv.DictReader(f)]
    rec = LicensePlateRecognizer()
    engines = ["easyocr", "cnn"] if args.ocr == "both" else [args.ocr]
    if "cnn" in engines and not rec.chars.available:
        print("Chưa có models/char_cnn.pt -> bỏ qua CNN")
        engines.remove("cnn")
    rec.warmup()

    stats = {e: {"exact": 0, "chars": 0, "err": 0, "time": 0.0, "used_cnn": 0} for e in engines}
    errors = {e: [] for e in engines}
    n = 0
    for name, gt in labels:
        img = read_image(Path(args.images) / name)
        if img is None:
            print(f"[thiếu ảnh] {name}")
            continue
        n += 1
        for e in engines:
            r = rec.run(img, with_steps=False, ocr=e)
            pred = norm(r["text"])
            s = stats[e]
            s["exact"] += pred == gt
            s["chars"] += len(gt)
            s["err"] += min(levenshtein(pred, gt), len(gt))
            s["time"] += r["time_ms"]["total"]
            s["used_cnn"] += r["ocr_engine"] == "CNN"
            if pred != gt:
                errors[e].append(f"{name:16s} GT={gt:10s} PRED={pred}")

    print(f"\n========= NHẬN DẠNG TRÊN {n} ẢNH XE MÁY THẬT =========")
    print(f"{'Chỉ số':30s}" + "".join(f"{e:>12s}" for e in engines))
    rows = [
        ("Đúng toàn biển", lambda s: f"{s['exact'] / n:.1%}"),
        ("Độ chính xác ký tự", lambda s: f"{1 - s['err'] / s['chars']:.1%}"),
        ("Thời gian TB (ms/ảnh)", lambda s: f"{s['time'] / n:.0f}"),
    ]
    for title, fn in rows:
        print(f"{title:30s}" + "".join(f"{fn(stats[e]):>12s}" for e in engines))
    if "cnn" in stats:
        print(f"(CNN được dùng ở {stats['cnn']['used_cnn']}/{n} ảnh; còn lại không tách đủ 2 dòng -> EasyOCR)")
    if args.show_errors:
        for e in engines:
            print(f"\n--- Ảnh sai ({e}) ---")
            print("\n".join(errors[e]) or "(không có)")


if __name__ == "__main__":
    main()
