"""Ứng dụng Web nhận dạng biển số xe máy - Flask backend.

Chạy:  python app.py   rồi mở http://127.0.0.1:5000
"""

import base64
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from lpr import LicensePlateRecognizer
from lpr.utils import decode_image_bytes

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Không cache static file để cập nhật UI ngay lập tức

recognizer = LicensePlateRecognizer(gpu=False)
# Nạp mô hình EasyOCR ở luồng nền để không chặn lúc khởi động
threading.Thread(target=recognizer.warmup, daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(recognizer.status())


@app.get("/api/samples")
def list_samples():
    """Danh sách ảnh mẫu trong thư mục samples/ (để demo nhanh)."""
    if not SAMPLES_DIR.is_dir():
        return jsonify([])
    files = sorted(p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if "29D2_39999_double.jpg" in files:
        files.remove("29D2_39999_double.jpg")
        files.insert(0, "29D2_39999_double.jpg")
    return jsonify(files[:24])


@app.get("/samples/<path:name>")
def sample_file(name):
    if Path(name).suffix.lower() not in IMG_EXTS:
        abort(404)
    return send_from_directory(SAMPLES_DIR, name)


@app.post("/api/recognize")
def recognize():
    data = None
    if "image" in request.files:
        data = request.files["image"].read()
    else:
        payload = request.get_json(silent=True) or {}
        uri = payload.get("image", "")
        if "," in uri:  # data:image/jpeg;base64,....
            uri = uri.split(",", 1)[1]
        if uri:
            try:
                data = base64.b64decode(uri)
            except (ValueError, TypeError):
                data = None

    if not data:
        return jsonify({"error": "Không nhận được ảnh."}), 400
    img = decode_image_bytes(data)
    if img is None:
        return jsonify({"error": "Định dạng ảnh không hợp lệ."}), 400

    try:
        result = recognizer.run(img)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Lỗi xử lý ảnh")
        return jsonify({"error": f"Lỗi xử lý: {exc}"}), 500
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Ảnh quá lớn (tối đa 16 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
