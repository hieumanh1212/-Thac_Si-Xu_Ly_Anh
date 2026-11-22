"""Các hàm tiện ích dùng chung."""

import base64

import cv2
import numpy as np


def read_image(path):
    """Đọc ảnh từ đường dẫn có ký tự Unicode (cv2.imread lỗi với tiếng Việt trên Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_image_bytes(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def to_data_uri(img, ext=".jpg", quality=88):
    """Mã hoá ảnh OpenCV thành data URI để hiển thị trực tiếp trên web."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return ""
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=interp)


def order_points(pts):
    """Sắp xếp 4 đỉnh theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái (chuẩn hình học)."""
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    # Sắp xếp theo trục y tăng dần (2 điểm trên, 2 điểm dưới)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    # Điểm có x nhỏ hơn là left, lớn hơn là right
    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]
    br = bottom_two[np.argmax(bottom_two[:, 0])]
    bl = bottom_two[np.argmin(bottom_two[:, 0])]
    return np.array([tl, tr, br, bl], dtype="float32")


def quad_size(quad):
    """Chiều rộng / cao trung bình của tứ giác đã sắp xếp."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return w, h


def warp_quad(img, quad, size):
    """Biến đổi phối cảnh (perspective transform) tứ giác về hình chữ nhật kích thước size=(W, H)."""
    W, H = size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.asarray(quad, dtype="float32"), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
