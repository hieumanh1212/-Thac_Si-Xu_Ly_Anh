"""Sinh ảnh biển số xe máy TỔNG HỢP để kiểm thử nhanh pipeline.

Lưu ý: ảnh tổng hợp chỉ để kiểm tra chương trình chạy đúng. Khi làm thực nghiệm cho báo cáo
cần dùng ảnh chụp thật.

Tên file = nhãn đúng, dạng  <dòng1>_<dòng2>.jpg  (vd. 59X1_12345.jpg) -> dùng được cho tools/evaluate.py

Chạy:  python tools/make_samples.py --out samples --n 10
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
LETTERS = "ABCDEFGHKLMNPSTUVXYZ"


def load_font(size):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def random_plate():
    province = random.randint(11, 99)
    series = random.choice(LETTERS) + str(random.randint(1, 9))
    number = "".join(random.choice("0123456789") for _ in range(random.choice([4, 5, 5])))
    return f"{province}{series}", number


def draw_plate(line1, line2, w=380, h=280):
    img = Image.new("RGB", (w, h), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=14, outline=(20, 20, 20), width=7)
    font = load_font(108)
    t1 = f"{line1[:2]}-{line1[2:]}"
    t2 = f"{line2[:3]}.{line2[3:]}" if len(line2) == 5 else line2
    for text, cy in ((t1, h * 0.29), (t2, h * 0.71)):
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((w - tw) / 2 - bbox[0], cy - th / 2 - bbox[1]), text, font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def compose(plate, W=960, H=720):
    # Nền giả lập: gradient + khối màu + nhiễu
    bg = np.zeros((H, W, 3), np.uint8)
    base = np.random.randint(40, 120, 3)
    for y in range(H):
        bg[y] = np.clip(base + (y / H) * 60, 0, 255)
    for _ in range(8):
        x1, y1 = random.randint(0, W), random.randint(0, H)
        cv2.rectangle(bg, (x1, y1), (x1 + random.randint(40, 300), y1 + random.randint(40, 200)),
                      tuple(int(v) for v in np.random.randint(0, 160, 3)), -1)

    ph, pw = plate.shape[:2]
    scale = random.uniform(0.6, 1.0)
    tw, th = int(pw * scale), int(ph * scale)
    cx, cy = random.randint(tw, W - tw), random.randint(th, H - th)
    jitter = lambda: random.uniform(-0.08, 0.08) * tw  # noqa: E731
    dst = np.float32([[cx - tw / 2 + jitter(), cy - th / 2 + jitter()],
                      [cx + tw / 2 + jitter(), cy - th / 2 + jitter()],
                      [cx + tw / 2 + jitter(), cy + th / 2 + jitter()],
                      [cx - tw / 2 + jitter(), cy + th / 2 + jitter()]])
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (W, H))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]

    out = cv2.GaussianBlur(out, (3, 3), 0)
    noise = np.random.normal(0, 6, out.shape)
    out = np.clip(out.astype(np.float32) * random.uniform(0.75, 1.1) + noise, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(args.n):
        l1, l2 = random_plate()
        img = compose(draw_plate(l1, l2))
        path = out / f"{l1}_{l2}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(path))  # hỗ trợ đường dẫn Unicode
        print("->", path)


if __name__ == "__main__":
    main()
