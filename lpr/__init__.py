"""Gói nhận dạng biển số xe máy Việt Nam (License Plate Recognition).

Các module:
    utils        - hàm tiện ích (resize, nắn phối cảnh, mã hoá ảnh base64)
    preprocess   - tiền xử lý ảnh (xám hoá, CLAHE, lọc nhiễu, biên, hình thái học)
    detect       - định vị vùng biển số bằng contour
    segment      - nhị phân hoá biển số và phân đoạn ký tự
    recognize    - nhận dạng ký tự (EasyOCR)
    postprocess  - chuẩn hoá chuỗi theo định dạng biển số xe máy VN
    pipeline     - ghép toàn bộ các bước
"""

from .pipeline import LicensePlateRecognizer

__all__ = ["LicensePlateRecognizer"]
