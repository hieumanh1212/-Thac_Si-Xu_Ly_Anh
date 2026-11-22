"""Bước 1 - Tiền xử lý ảnh đầu vào.

Chuỗi xử lý:
    ảnh màu -> ảnh xám -> tăng tương phản cục bộ (CLAHE) -> lọc song phương (bilateral)
            -> phát hiện biên Canny -> phép đóng hình thái học (nối các đoạn biên đứt)
            -> (song song) ngưỡng Otsu để tìm vùng sáng (nền biển số màu trắng)
"""

import cv2


def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: cân bằng histogram thích nghi, giới hạn tương phản -> giảm ảnh hưởng của ánh sáng không đều
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Lọc song phương: khử nhiễu nhưng vẫn giữ biên sắc nét (khác Gaussian làm nhoè biên)
    blur = cv2.bilateralFilter(clahe, d=11, sigmaColor=17, sigmaSpace=17)

    # Phát hiện biên Canny
    edges = cv2.Canny(blur, 30, 200)

    # Phép đóng (dilate rồi erode) để nối các đoạn biên bị đứt của khung biển số
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Vùng sáng: biển số xe máy VN thường nền trắng/vàng -> ngưỡng Otsu tách vùng sáng,
    # phép mở để tách vùng biển số khỏi các vùng sáng dính liền xung quanh
    _, bright = cv2.threshold(cv2.GaussianBlur(clahe, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Ngưỡng thích nghi cục bộ: phát hiện biển số trong bóng râm hoặc chênh sáng phức tạp
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    return {
        "gray": gray,
        "clahe": clahe,
        "blur": blur,
        "edges": edges,
        "closed": closed,
        "bright": bright,
        "adapt": adapt,
    }
