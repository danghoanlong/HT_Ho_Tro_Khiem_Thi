"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              BLIND ASSISTANT AI — core_engine.py                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Chứa toàn bộ logic AI, xử lý tín hiệu và tiện ích chia sẻ giữa các file. ║
║                                                                             ║
║  MỤC LỤC                                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  §1  Hằng số cấu hình      (TARGET_CLASSES, CLASS_VI, REAL_HEIGHT_M, ...)  ║
║  §2  CalibrationManager    (học khoảng cách per-class, lưu calib.json)     ║
║  §3  BackgroundSpeaker     (TTS tiếng Việt, gTTS + pygame, non-blocking)   ║
║  §4  AnnouncementManager   (quản lý cooldown thông báo giọng nói)          ║
║  §5  DistanceEstimator     (ước lượng khoảng cách thông minh)              ║
║  §6  FaceRecognizer        (nhận diện khuôn mặt từ encodings.pickle)       ║
║  §7  NavigationGuide       (chỉ dẫn đường đi trong nhà)                   ║
║  §8  MoneyDetector         (nhận diện tiền VN — model custom)              ║
║  §9  OCRReader             (đọc văn bản — EasyOCR background thread)       ║
║  §10 TrafficLightAnalyzer  (nhận diện đèn giao thông — HSV + voting)      ║
║  §11 TimeReader            (đọc giờ tiếng Việt qua TTS)                   ║
║  §12 SessionLogger         (ghi lịch sử phiên — CSV + JSON)               ║
║  §13 Hàm vẽ               (put_vi_text, draw_box_*, draw_nav_overlay)     ║
║  §14 Hàm khởi động        (build_preload_texts)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cài đặt:
    pip install ultralytics opencv-python gtts pygame Pillow easyocr
    pip install face_recognition  (Windows: pip install cmake dlib trước)
"""

# ── Thư viện chuẩn ────────────────────────────────────────────────────────────
import csv
import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from datetime import datetime

# ── Thư viện bên thứ ba ───────────────────────────────────────────────────────
import cv2
import numpy as np
import pygame
from gtts import gTTS
from ultralytics import YOLO

# ── Pillow (render chữ tiếng Việt có dấu) ────────────────────────────────────
try:
    from PIL import Image as _PILI, ImageDraw as _PILD, ImageFont as _PILF
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── EasyOCR (tùy chọn) ───────────────────────────────────────────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[OCR] ⚠  easyocr chưa cài. Chạy: pip install easyocr")


# ══════════════════════════════════════════════════════════════════════════════
#  §1  HẰNG SỐ CẤU HÌNH
#  Tất cả giá trị điều chỉnh hành vi hệ thống tập trung tại đây.
#  Chỉnh sửa phần này khi muốn thay đổi ngưỡng, đường dẫn, v.v.
# ══════════════════════════════════════════════════════════════════════════════

# ── 1a. Nhóm class YOLO ───────────────────────────────────────────────────────
TARGET_CLASSES_HIGH = {          # Ưu tiên cao: phương tiện + người + đèn
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "traffic light", "stop sign",
}
TARGET_CLASSES_HOME = {          # Đồ vật trong nhà
    "chair", "couch", "bed", "dining table", "toilet", "sink",
    "refrigerator", "microwave", "oven", "toaster", "tv",
    "laptop", "cell phone", "book", "clock", "vase",
    "bottle", "cup", "bowl", "knife", "fork", "spoon",
    "potted plant", "backpack", "handbag", "suitcase",
    "umbrella", "bench", "door", "stairs",
    "keyboard", "mouse", "remote",
}
TARGET_CLASSES = TARGET_CLASSES_HIGH | TARGET_CLASSES_HOME

# ── 1b. Tên tiếng Việt ───────────────────────────────────────────────────────
CLASS_VI = {
    "person":        "người lạ",     "car":           "ô tô",
    "motorcycle":    "xe máy",       "bicycle":       "xe đạp",
    "bus":           "xe buýt",      "truck":         "xe tải",
    "traffic light": "đèn giao thông","stop sign":    "biển dừng",
    "chair":         "ghế",          "couch":         "ghế sofa",
    "bed":           "giường",       "dining table":  "bàn ăn",
    "toilet":        "bồn cầu",      "sink":          "bồn rửa tay",
    "door":          "cửa ra vào",   "stairs":        "cầu thang",
    "bench":         "ghế dài",      "refrigerator":  "tủ lạnh",
    "microwave":     "lò vi sóng",   "oven":          "lò nướng",
    "toaster":       "máy nướng bánh","tv":           "ti vi",
    "laptop":        "máy tính xách tay","cell phone":"điện thoại",
    "keyboard":      "bàn phím",     "mouse":         "chuột máy tính",
    "remote":        "điều khiển từ xa","bottle":     "chai",
    "cup":           "cốc",          "bowl":          "bát",
    "knife":         "dao",          "fork":          "nĩa",
    "spoon":         "muỗng",        "book":          "quyển sách",
    "clock":         "đồng hồ",      "vase":          "lọ hoa",
    "potted plant":  "chậu cây",     "backpack":      "ba lô",
    "handbag":       "túi xách",     "suitcase":      "vali",
    "umbrella":      "ô dù",
}

# ── 1c. Màu bounding box (BGR) ───────────────────────────────────────────────
CLASS_COLORS = {
    "person":        (0,   220,   0),   "car":           (0,     0, 220),
    "motorcycle":    (220,   0, 220),   "bicycle":       (255, 140,   0),
    "bus":           (0,     0, 180),   "truck":         (0,    30, 200),
    "stairs":        (0,     0, 255),   "chair":         (0,   165, 255),
    "couch":         (0,   200, 255),   "bed":           (0,   255, 200),
    "dining table":  (50,  200, 200),   "door":          (200, 100,   0),
    "refrigerator":  (180, 180,   0),   "traffic light": (255, 255,   0),
    "default":       (120, 120, 120),
}

# ── 1d. Khoảng cách — chiều cao & chiều rộng thực tế (mét) ──────────────────
REAL_HEIGHT_M = {
    "person": 1.70, "car": 1.50, "motorcycle": 1.10, "bicycle": 1.00,
    "bus": 3.00, "truck": 2.50, "chair": 0.90, "couch": 0.85,
    "bed": 0.55, "dining table": 0.75, "toilet": 0.70, "sink": 0.50,
    "refrigerator": 1.70, "tv": 0.60, "laptop": 0.25, "door": 2.00,
    "stairs": 0.20, "bottle": 0.25, "cup": 0.10, "backpack": 0.45,
    "suitcase": 0.60, "default": 0.80,
}
REAL_WIDTH_M = {
    "car": 1.80, "bus": 2.50, "truck": 2.40, "motorcycle": 0.70,
    "bicycle": 0.55, "person": 0.50, "dining table": 1.20,
    "couch": 1.80, "bed": 1.40, "refrigerator": 0.70, "tv": 1.00,
    "default": 0.60,
}

# Hệ số bù YOLO underdetect: bbox YOLO thường nhỏ hơn vật thực
# (vd: person 0.78 → YOLO detect ~78% chiều cao thực)
BBOX_COVERAGE = {
    "person": 0.78, "cell phone": 0.65, "laptop": 0.72,
    "car": 0.88, "motorcycle": 0.85, "bicycle": 0.82,
    "bus": 0.90, "truck": 0.88, "chair": 0.80, "couch": 0.82,
    "bed": 0.75, "dining table": 0.80, "refrigerator": 0.85,
    "tv": 0.85, "door": 0.88, "bottle": 0.80, "cup": 0.78,
    "backpack": 0.80, "suitcase": 0.82, "default": 0.80,
}

# ── 1e. Khoảng cách — ngưỡng cảnh báo ───────────────────────────────────────
#   FOCAL_LENGTH: tiêu cự webcam (px). Tự động được cập nhật khi calibrate.
#   Nhấn C (terminal) hoặc nút Calibrate (GUI) để hiệu chỉnh.
FOCAL_LENGTH    = 615.0   # px — giá trị mặc định cho webcam 640×480
WARN_DIST_M     = 1.5     # mét → trạng thái "near" (cảnh báo)
CRITICAL_DIST_M = 0.80    # mét → trạng thái "critical" (nguy hiểm)
CALIB_FILE      = "calib.json"

# ── 1f. Frame skipping ───────────────────────────────────────────────────────
YOLO_SKIP  = 3   # Chạy YOLO mỗi N frame
MONEY_SKIP = 5   # Chạy model tiền mỗi N frame
FACE_SKIP  = 6   # Chạy nhận diện khuôn mặt mỗi N frame

# ── 1g. YOLO inference ───────────────────────────────────────────────────────
YOLO_CONF  = 0.40
YOLO_IOU   = 0.50
MONEY_CONF = 0.55

# ── 1h. Đường dẫn file ───────────────────────────────────────────────────────
FACE_ENCODINGS   = "Model/encodings.pickle"
MONEY_MODEL_PATH = "Model/money_v8n.pt"

# ── 1i. Cooldown TTS (giây) — khoảng cách tối thiểu giữa 2 lần thông báo ────
ANNOUNCE_COOLDOWN        = 5.0    # vật cản near
CRIT_COOLDOWN            = 5.0    # vật cản critical
FACE_ANNOUNCE_CD         = 10.0   # nhận ra người quen (lần đầu)
KNOWN_PERSON_ANNOUNCE_CD = 15.0   # người quen đang ở gần
MONEY_COOLDOWN           = 3.0    # tiền
TRAFFIC_COOLDOWN         = 2.5    # đèn giao thông
NAV_COOLDOWN             = 3.0    # điều hướng
TTS_CACHE_DIR            = os.path.join(tempfile.gettempdir(), "blind_tts_cache")

# ── 1j. Tiền Việt Nam — ánh xạ label model → tên đọc tiếng Việt ─────────────
# Hỗ trợ 3 định dạng label model hay trả về: số nguyên, "k", dấu chấm
MONEY_VI = {
    # Số nguyên
    "200": "hai trăm đồng",        "500": "năm trăm đồng",
    "1000": "một nghìn đồng",      "2000": "hai nghìn đồng",
    "5000": "năm nghìn đồng",      "10000": "mười nghìn đồng",
    "20000": "hai mươi nghìn đồng","50000": "năm mươi nghìn đồng",
    "100000": "một trăm nghìn đồng","200000": "hai trăm nghìn đồng",
    "500000": "năm trăm nghìn đồng",
    # Dạng "k"
    "0.2k": "hai trăm đồng",       "0.5k": "năm trăm đồng",
    "1k": "một nghìn đồng",        "2k": "hai nghìn đồng",
    "5k": "năm nghìn đồng",        "10k": "mười nghìn đồng",
    "20k": "hai mươi nghìn đồng",  "50k": "năm mươi nghìn đồng",
    "100k": "một trăm nghìn đồng", "200k": "hai trăm nghìn đồng",
    "500k": "năm trăm nghìn đồng",
    # Dạng dấu chấm
    "1.000": "một nghìn đồng",     "2.000": "hai nghìn đồng",
    "5.000": "năm nghìn đồng",     "10.000": "mười nghìn đồng",
    "20.000": "hai mươi nghìn đồng","50.000": "năm mươi nghìn đồng",
    "100.000": "một trăm nghìn đồng","200.000": "hai trăm nghìn đồng",
    "500.000": "năm trăm nghìn đồng",
}

# ── 1k. Đèn giao thông ───────────────────────────────────────────────────────
# HSV: H 0-179 · S 0-255 · V 0-255
TRAFFIC_HSV = {
    "red":    [(0, 120, 80, 10, 255, 255), (165, 120, 80, 179, 255, 255)],
    "green":  [(50, 80, 70, 95, 255, 255)],
    "yellow": [(18, 100, 80, 42, 255, 255)],
}
TRAFFIC_COLORS_BGR = {
    "red":    (0,   0, 220), "green":   (0, 200,   0),
    "yellow": (0, 200, 220), "unknown": (80,  80,  80),
}
TRAFFIC_VI = {
    "red":    "Đèn đỏ, dừng lại",  "green":   "Đèn xanh, được đi",
    "yellow": "Đèn vàng, chú ý",   "unknown": "Phía trước có đèn giao thông",
}

# ── 1l. Điều hướng trong nhà ─────────────────────────────────────────────────
ROOM_MAP: dict[str, dict] = {
    "phòng bếp":   {"direction_hint": "Đi thẳng rồi rẽ phải"},
    "nhà vệ sinh": {"direction_hint": "Đi thẳng rồi rẽ trái"},
    "phòng ngủ":   {"direction_hint": "Rẽ phải, đi thẳng"},
    "phòng khách": {"direction_hint": "Đi thẳng"},
    "cửa ra":      {"direction_hint": "Quay lại, đi thẳng"},
    "tivi":        {"direction_hint": "Rẽ trái, đi thẳng"},
}
ROOM_ALIASES: dict[str, str] = {
    "bếp": "phòng bếp", "toilet": "nhà vệ sinh", "wc": "nhà vệ sinh",
    "ngủ": "phòng ngủ", "khách": "phòng khách", "cửa": "cửa ra",
    "tv": "tivi",
}



# ══════════════════════════════════════════════════════════════════════════════
#  §2  CALIBRATION MANAGER
#  Lưu/load tiêu cự focal và correction scale per-class vào calib.json.
#  Học từ nhiều điểm đo (multi-point averaging) để ngày càng chính xác.
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationManager:
    """
    Quản lý calibration FOCAL_LENGTH để khoảng cách chính xác theo camera thực.

    Vì sao cần calibrate:
    ─────────────────────────────────────────────────────────────────────────
    FOCAL_LENGTH = 615px là ước lượng lý thuyết cho webcam 640×480 FOV 60°.
    Camera thực tế có thể lệch 20-50% tuỳ loại:
      - Webcam USB thường: 500-700px
      - Laptop built-in: 550-650px
      - Điện thoại: 700-900px (góc rộng hơn)

    Khi FOCAL sai 3x → khoảng cách sai 3x:
      - Bạn 50cm → báo 150cm: FOCAL cần giảm xuống còn 1/3

    Cách calibrate:
    ─────────────────────────────────────────────────────────────────────────
    1. Đứng trước camera, giữ vật tham chiếu (mặc định: 'person' = bản thân)
    2. Nhấn C (Terminal) hoặc nút Calibrate (GUI)
    3. Nhập khoảng cách thực (cm) và class đang dùng
    4. Hệ thống tính FOCAL mới và lưu vào calib.json
    5. Lần sau khởi động tự load — không cần calibrate lại
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, calib_file: str = CALIB_FILE):
        self._file   = calib_file
        self.focal   = FOCAL_LENGTH          # Sẽ được override khi load
        self.scales: dict[str, float] = {}   # Per-class extra correction
        self._load()

    def _load(self):
        """Load focal length và scales đã calibrate từ file."""
        import json
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.focal  = float(data.get("focal_length", FOCAL_LENGTH))
            self.scales = data.get("class_scales", {})
            print(f"[Calib] ✓ Loaded focal={self.focal:.1f}px "
                  f"từ {self._file}")
            if self.scales:
                print(f"[Calib]   Per-class: {self.scales}")
        except FileNotFoundError:
            print(f"[Calib] Chưa có {self._file} — dùng focal mặc định "
                  f"{FOCAL_LENGTH}px")
            print(f"[Calib] → Nhấn C để calibrate cho camera của bạn")
        except Exception as e:
            print(f"[Calib] Lỗi load: {e} — dùng focal mặc định")

    def save(self):
        """Lưu calibration ra file JSON."""
        import json
        data = {
            "focal_length":  round(self.focal, 2),
            "class_scales":  {k: round(v, 4) for k, v in self.scales.items()},
        }
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[Calib] ✓ Đã lưu focal={self.focal:.1f}px → {self._file}")
        except Exception as e:
            print(f"[Calib] Lỗi lưu: {e}")

    def calibrate_from_bbox(self, cls_name: str,
                            x1: int, y1: int, x2: int, y2: int,
                            true_dist_m: float,
                            frame_w: int = 640, frame_h: int = 480) -> float:
        """
        Calibrate per-class scale sao cho estimate() trả về đúng true_dist_m.

        Fix bug quan trọng:
        ─────────────────────────────────────────────────────────────────
        Trước đây tính est bằng công thức nội bộ ≠ công thức của estimate().
        → Scale sai cơ sở → sau calibrate vẫn bị lệch.

        Giờ dùng estimate() VỚI scale=1.0 làm cơ sở tính scale.
        Đảm bảo: estimate() * scale = true_dist_m (chính xác).
        ─────────────────────────────────────────────────────────────────
        """
        # Bước 1: Tạm thời đặt scale = 1.0 để lấy estimate thuần
        old_scale = self.scales.get(cls_name, 1.0)
        self.scales[cls_name] = 1.0

        # Bước 2: Gọi estimate() với scale=1.0 — đây là baseline thực sự
        # Không truyền track_key để tránh EMA làm sai
        est_no_scale = DistanceEstimator.estimate(
            cls_name, x1, y1, x2, y2,
            frame_w=frame_w, frame_h=frame_h,
            track_key=None,
        )

        # Bước 3: Tính scale để estimate() * scale = true_dist_m
        if est_no_scale <= 0:
            print(f"[Calib] ✗ estimate() trả về 0 — hủy calibration")
            self.scales[cls_name] = old_scale
            return old_scale

        new_scale = true_dist_m / est_no_scale

        # Bước 4: Lưu scale mới
        self.scales[cls_name] = round(new_scale, 4)

        # Bước 5: Xóa EMA cache cho class này — tránh buffer cũ kéo sai
        DistanceEstimator.clear_ema_for_class(cls_name)

        print(f"[Calib] ─────────────────────────────────────────────")
        print(f"[Calib] Class  : {cls_name}")
        print(f"[Calib] bbox   : ({x1},{y1}) → ({x2},{y2})")
        print(f"[Calib] est_raw: {est_no_scale*100:.0f} cm  (scale=1.0)")
        print(f"[Calib] thực tế: {true_dist_m*100:.0f} cm")
        print(f"[Calib] scale  : {old_scale:.4f} → {new_scale:.4f}  "
              f"({'giảm' if new_scale < 1 else 'tăng'} {abs(1-new_scale)*100:.0f}%)")
        print(f"[Calib] kiểm tra: {est_no_scale*new_scale*100:.0f} cm "
              f"(phải = {true_dist_m*100:.0f} cm)")
        print(f"[Calib] ─────────────────────────────────────────────")

        self.save()
        return new_scale

    def add_class_measurement(self, cls_name: str,
                              x1: int, y1: int, x2: int, y2: int,
                              true_dist_m: float,
                              frame_w: int = 640, frame_h: int = 480) -> float:
        """
        Thêm 1 điểm đo, tính scale trung bình (multi-point averaging).

        Fix bug: dùng estimate() với scale=1.0 làm baseline,
        đảm bảo scale mới áp vào estimate() cho kết quả đúng true_dist_m.
        """
        import json

        # Baseline thực sự = estimate() không có scale
        old_scale = self.scales.get(cls_name, 1.0)
        self.scales[cls_name] = 1.0
        est_no_scale = DistanceEstimator.estimate(
            cls_name, x1, y1, x2, y2,
            frame_w=frame_w, frame_h=frame_h,
            track_key=None,
        )
        self.scales[cls_name] = old_scale   # Khôi phục để tính history

        if est_no_scale <= 0:
            return old_scale

        new_scale = true_dist_m / est_no_scale

        # Đọc lịch sử
        history: list[float] = []
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("measurement_history", {}).get(cls_name, [])
        except Exception:
            data = {}

        history.append(round(new_scale, 4))
        history = history[-10:]   # giữ 10 điểm gần nhất

        # Tính trung bình, lọc outlier ±30%
        if len(history) >= 3:
            med      = sorted(history)[len(history) // 2]
            filtered = [s for s in history if abs(s - med) / max(med, 0.001) <= 0.30]
            final    = sum(filtered) / max(len(filtered), 1)
        else:
            final = sum(history) / len(history)

        final = round(final, 4)
        self.scales[cls_name] = final

        # Xóa EMA để kết quả mới được phản ánh ngay
        DistanceEstimator.clear_ema_for_class(cls_name)

        print(f"[Calib] '{cls_name}': "
              f"est_raw={est_no_scale*100:.0f}cm → nhập={true_dist_m*100:.0f}cm  "
              f"scale_mới={new_scale:.4f}  avg({len(history)} điểm)={final:.4f}")

        # Lưu vào file
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        if "measurement_history" not in data:
            data["measurement_history"] = {}
        data["measurement_history"][cls_name] = history
        data["focal_length"]  = round(self.focal, 2)
        data["class_scales"]  = {k: round(v, 4) for k, v in self.scales.items()}
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Calib] Lỗi lưu: {e}")

        return final

    def get_measurement_count(self, cls_name: str) -> int:
        import json
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                return len(json.load(f).get("measurement_history", {}).get(cls_name, []))
        except Exception:
            return 0

    def get_all_history(self) -> dict[str, list[float]]:
        import json
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                return json.load(f).get("measurement_history", {})
        except Exception:
            return {}

    def get_scale(self, cls_name: str) -> float:
        """Trả về correction scale cho class (mặc định 1.0)."""
        return self.scales.get(cls_name, 1.0)


# Singleton instance dùng toàn hệ thống
_calibration = CalibrationManager()

# Alias để tương thích với gui_app.py và main.py
DistanceCalibrator = CalibrationManager

# Bổ sung constants OCR và ROOM_MAP (chưa có trong §1)
OCR_LANGUAGES = ["vi", "en"]
OCR_MIN_CONF  = 0.50
OCR_MAX_CHARS = 120

ROOM_MAP = {
    "phòng bếp":   {"landmark": "refrigerator", "direction_hint": "Hãy đi về phía tủ lạnh",    "arrived_when": 1.2},
    "nhà vệ sinh": {"landmark": "toilet",        "direction_hint": "Hãy đi về phía bồn cầu",     "arrived_when": 1.0},
    "phòng ngủ":   {"landmark": "bed",           "direction_hint": "Hãy đi về phía giường",      "arrived_when": 1.5},
    "phòng khách": {"landmark": "couch",         "direction_hint": "Hãy đi về phía ghế sofa",    "arrived_when": 1.5},
    "bàn ăn":      {"landmark": "dining table",  "direction_hint": "Hãy đi về phía bàn ăn",      "arrived_when": 1.0},
    "cửa ra":      {"landmark": "door",          "direction_hint": "Hãy đi về phía cửa ra vào",  "arrived_when": 1.0},
    "tivi":        {"landmark": "tv",            "direction_hint": "Hãy đi về phía ti vi",        "arrived_when": 1.2},
    "bồn rửa tay": {"landmark": "sink",          "direction_hint": "Hãy đi về phía bồn rửa tay", "arrived_when": 0.8},
}
ROOM_ALIASES = {
    "bếp": "phòng bếp", "wc": "nhà vệ sinh", "toilet": "nhà vệ sinh",
    "ngủ": "phòng ngủ", "khách": "phòng khách",
}


def normalize_money_label(raw_label: str) -> str:
    """Chuẩn hóa label tiền từ model về key trong MONEY_VI.
    Hỗ trợ: '10k', '10K', '10.000', '10000', 'VND10000', v.v."""
    import re
    s = raw_label.strip().lower().replace(" ", "")
    s = re.sub(r'^[a-z]+(?=[\d])', '', s)
    if s in MONEY_VI: return s
    s2 = re.sub(r'[a-zđ]+$', '', s)
    if s2 in MONEY_VI: return s2
    s3 = s2.replace(".", "").replace(",", "")
    if s3 in MONEY_VI: return s3
    m = re.match(r'^(\d+(?:\.\d+)?)k$', s)
    if m:
        key = str(int(float(m.group(1)) * 1000))
        if key in MONEY_VI: return key
    return raw_label


# ══════════════════════════════════════════════════════════════════════════════
#  §3  BACKGROUND SPEAKER  —  TTS tiếng Việt, non-blocking
# ══════════════════════════════════════════════════════════════════════════════

class BackgroundSpeaker:
    """
    Phát giọng nói tiếng Việt hoàn toàn không chặn luồng camera.
    Cache MD5 → mỗi câu chỉ gọi Google TTS 1 lần, sau đó offline.
    """

    def __init__(self, lang: str = "vi", cache_dir: str = TTS_CACHE_DIR):
        self._lang      = lang
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._q          = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        self._thread = threading.Thread(target=self._worker, name="gTTS-Thread", daemon=True)
        self._thread.start()
        print(f"[TTS] gTTS speaker khởi động. Cache: {cache_dir}")

    def _get_mp3(self, text: str) -> str | None:
        key      = hashlib.md5(f"{self._lang}:{text}".encode()).hexdigest()
        mp3_path = os.path.join(self._cache_dir, f"{key}.mp3")
        if os.path.exists(mp3_path):
            return mp3_path
        try:
            gTTS(text=text, lang=self._lang, slow=False).save(mp3_path)
            return mp3_path
        except Exception as exc:
            print(f"[TTS] Lỗi gTTS: {exc}")
            return None

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                mp3_path = self._q.get(timeout=0.5)
                if mp3_path and os.path.exists(mp3_path):
                    pygame.mixer.music.load(mp3_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        if self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.05)
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[TTS] Lỗi phát âm: {exc}")

    def say(self, text: str, priority: bool = False):
        """Phát câu nói không chặn. priority=True → xóa hàng đợi cũ, phát ngay."""
        mp3_path = self._get_mp3(text)
        if not mp3_path:
            return
        if priority:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        try:
            self._q.put_nowait(mp3_path)
        except queue.Full:
            pass

    def preload(self, texts: list[str]):
        """Pre-cache nhiều câu cùng lúc để giảm độ trễ lần đầu."""
        print(f"[TTS] Đang pre-cache {len(texts)} câu thông báo...")
        for t in texts:
            self._get_mp3(t)
        print("[TTS] Pre-cache hoàn tất.")

    def stop(self):
        self._stop_event.set()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        self._thread.join(timeout=2)


# ══════════════════════════════════════════════════════════════════════════════
#  §4  ANNOUNCEMENT MANAGER  —  Quản lý cooldown thông báo giọng nói
# ══════════════════════════════════════════════════════════════════════════════

class AnnouncementManager:
    """
    Quản lý cooldown để không thông báo liên tục cùng một vật/sự kiện.
    Ưu tiên: critical > near > far (far không thông báo).
    """

    def __init__(self, speaker: BackgroundSpeaker):
        self._speaker = speaker
        self._last: dict[str, float] = {}

    def _can_announce(self, key: str, cooldown: float) -> bool:
        """Trả về True và cập nhật timestamp nếu đã đủ cooldown."""
        now = time.time()
        if now - self._last.get(key, 0) >= cooldown:
            self._last[key] = now
            return True
        return False

    def process_obstacles(self, detections: list[dict]):
        """
        Thông báo vật cản theo mức độ ưu tiên critical > near.

        Quy tắc:
        - Người quen (is_known=True) bị bỏ qua hoàn toàn ở đây.
          Họ được xử lý bởi process_face_nearby() với cooldown dài hơn.
        - Chỉ đọc 1 cảnh báo critical mỗi cycle (vật gần nhất).
        - Cooldown per-label tránh lặp cùng 1 vật liên tục.
        """
        critical_items = []
        near_items     = []

        for det in detections:
            state    = det.get("state", "far")
            lbl      = det["label"]
            is_known = det.get("is_known", False)
            dist     = det.get("dist_m", 99)

            # Người quen KHÔNG được thông báo là "người lạ"
            if is_known:
                continue

            vi      = CLASS_VI.get(lbl, lbl)
            dist_vi = DistanceEstimator.dist_text_vi(dist)

            if state == "critical":
                critical_items.append((vi, dist_vi, lbl))
            elif state == "near":
                near_items.append((vi, dist_vi, lbl))

        # Chỉ đọc 1 cảnh báo critical mỗi cycle
        for vi, dist_vi, lbl in critical_items:
            if self._can_announce(f"crit_{lbl}", CRIT_COOLDOWN):
                msg = f"Nguy hiểm! Phía trước có {vi} cách {dist_vi}!"
                self._speaker.say(msg, priority=True)
                print(f"[Warn‼] {msg}")
                return   # Không đọc thêm near nếu đã có critical

        # Near: gộp câu, cooldown per-label
        to_say = []
        for vi, dist_vi, lbl in near_items:
            if self._can_announce(lbl, ANNOUNCE_COOLDOWN):
                to_say.append(f"{vi} cách {dist_vi}")
        if to_say:
            msg = f"Cảnh báo, phía trước có {', '.join(to_say)}"
            self._speaker.say(msg)
            print(f"[Warn] {msg}")

    def process_face_nearby(self, name: str, state: str, dist_m: float):
        """
        Thông báo riêng khi người quen ở gần hoặc rất gần.
        Cooldown dài (KNOWN_PERSON_ANNOUNCE_CD=15s) để không lặp.
        Chỉ thông báo khi near/critical — xa thì im.
        """
        if state == "far":
            return
        key = f"known_near_{name}"
        if not self._can_announce(key, KNOWN_PERSON_ANNOUNCE_CD):
            return
        dist_vi = DistanceEstimator.dist_text_vi(dist_m)
        if state == "critical":
            msg = f"{name} đang rất gần, cách {dist_vi}"
        else:
            msg = f"Phía trước có {name}, cách {dist_vi}"
        self._speaker.say(msg, priority=(state == "critical"))
        print(f"[Face‑Near] → {msg}")

    def process_money(self, money_label: str):
        norm    = normalize_money_label(money_label)
        vi_name = MONEY_VI.get(norm, f"tờ {money_label}")
        key     = f"money_{norm}"
        if self._can_announce(key, MONEY_COOLDOWN):
            msg = f"Đây là tờ {vi_name}"
            print(f"[Money] → {msg}  (label gốc: '{money_label}' → chuẩn hóa: '{norm}')")
            self._speaker.say(msg, priority=True)

    def process_traffic(self, color: str):
        if self._can_announce(f"traffic_{color}", TRAFFIC_COOLDOWN):
            msg = TRAFFIC_VI.get(color, TRAFFIC_VI["unknown"])
            print(f"[Traffic] → {msg}")
            self._speaker.say(msg, priority=True)

    def process_ocr(self, text: str, ocr_reader: "OCRReader | None" = None):
        """
        Đọc văn bản OCR bằng TTS.
        Nếu truyền ocr_reader → dùng chunking thông minh để đọc từng đoạn.
        """
        if not self._can_announce("ocr", 8.0) or not text.strip():
            return
        msg_prefix = "Văn bản đọc được: "

        if ocr_reader is not None:
            # Dùng chunking: đọc từng đoạn ngắn tự nhiên
            chunks = ocr_reader.get_tts_chunks(text)
            if not chunks:
                return
            # Đọc tiêu đề trước
            self._speaker.say(msg_prefix + chunks[0], priority=True)
            # Đọc các đoạn tiếp theo (không priority để không cắt nhau)
            for chunk in chunks[1:]:
                self._speaker.say(chunk, priority=False)
            print(f"[OCR] → {text[:100]} ({len(chunks)} đoạn)")
        else:
            # Fallback: cắt ngắn và đọc 1 lần
            short = text[:OCR_MAX_CHARS] + ("..." if len(text) > OCR_MAX_CHARS else "")
            msg = msg_prefix + short
            print(f"[OCR] → {msg}")
            self._speaker.say(msg, priority=True)


# ══════════════════════════════════════════════════════════════════════════════
#  §5  DISTANCE ESTIMATOR  —  Ước lượng khoảng cách (calibration-aware)
# ══════════════════════════════════════════════════════════════════════════════

class DistanceEstimator:
    """
    Ước lượng khoảng cách chính xác — tích hợp CalibrationManager.

    Pipeline tính khoảng cách (theo thứ tự):
    ─────────────────────────────────────────────────────────────────
    1. Lấy FOCAL từ CalibrationManager (đã calibrate hoặc mặc định)
    2. Tính D_h = H_real * focal / h_px
       Tính D_w = W_real * focal / w_px
    3. Chia cho BBOX_COVERAGE[class]: bù trừ YOLO underdetect
    4. Fusion D_h và D_w có trọng số theo class và aspect ratio
    5. Nhân per-class correction scale (nếu có từ calibration)
    6. Hiệu chỉnh perspective (vật ở rìa frame)
    7. Hiệu chỉnh visibility (bbox chạm biên frame)
    8. EMA smoothing qua frame (alpha=0.35, reset nếu > 2s)
    ─────────────────────────────────────────────────────────────────
    """

    _EMA_ALPHA  = 0.35
    # {track_key: (smoothed_dist, timestamp)}
    _ema_cache: dict[str, tuple[float, float]] = {}

    # ── Ước lượng 1 chiều ────────────────────────────────────────────────────

    @staticmethod
    def _axis_dist(real_m: float, px: int, focal: float, coverage: float) -> float:
        """D = (real_m * focal) / (px * coverage)"""
        return (real_m * focal) / max(px * coverage, 0.1)

    # ── Hiệu chỉnh perspective ────────────────────────────────────────────────

    @staticmethod
    def _perspective(cx_norm: float) -> float:
        """Hệ số 0.85–1.0: vật ở rìa frame cần điều chỉnh nhỏ xuống."""
        edge = min(cx_norm, 1.0 - cx_norm) * 2   # 0=rìa → 1=giữa
        return 0.88 + 0.12 * edge

    # ── Hiệu chỉnh visibility (bbox chạm biên) ───────────────────────────────

    @staticmethod
    def _visibility(x1: int, y1: int, x2: int, y2: int,
                    fw: int, fh: int) -> float:
        M = 3
        cut = 0.0
        if x1 <= M:       cut += 0.12
        if x2 >= fw - M:  cut += 0.12
        if y1 <= M:        cut += 0.08
        if y2 >= fh - M:  cut += 0.12
        return max(1.0 - cut, 0.60)

    # ── Fusion height + width ─────────────────────────────────────────────────

    @staticmethod
    def _fuse(dh: float, dw: float, h_px: int, w_px: int,
              cls_name: str) -> float:
        aspect = w_px / max(h_px, 1)
        # Mặc định: tin chiều cao hơn
        wH = 0.65
        WIDE = {"car","bus","truck","couch","bed","dining table","tv"}
        if cls_name in WIDE:
            wH = 0.38
        # Penalty nếu aspect bất thường
        if aspect > 3.0:
            wH = min(wH, 0.25)
        if aspect < 0.2:
            wH = max(wH, 0.82)
        wW = 1.0 - wH
        return wH * dh + wW * dw

    # ── EMA smoothing ─────────────────────────────────────────────────────────

    @classmethod
    def _smooth(cls, key: str, raw: float) -> float:
        now = time.time()
        if key in cls._ema_cache:
            prev, ts = cls._ema_cache[key]
            # Reset nếu vật biến mất lâu
            if now - ts > 2.5:
                smoothed = raw
            else:
                smoothed = cls._EMA_ALPHA * raw + (1 - cls._EMA_ALPHA) * prev
        else:
            smoothed = raw
        cls._ema_cache[key] = (smoothed, now)
        return smoothed

    @classmethod
    def cleanup_stale(cls, max_age: float = 4.0):
        now   = time.time()
        stale = [k for k, (_, ts) in cls._ema_cache.items()
                 if now - ts > max_age]
        for k in stale:
            del cls._ema_cache[k]

    @classmethod
    def clear_ema_for_class(cls, cls_name: str):
        """Xóa EMA cache của tất cả track key thuộc class này.
        Gọi sau khi calibrate để kết quả mới phản ánh ngay, không bị kéo bởi buffer cũ."""
        keys = [k for k in list(cls._ema_cache.keys()) if k.startswith(cls_name)]
        for k in keys:
            del cls._ema_cache[k]
        if keys:
            print(f"[Calib] EMA cache cleared for '{cls_name}' ({len(keys)} entries)")

    # ── API chính ─────────────────────────────────────────────────────────────

    @classmethod
    def estimate(cls, cls_name: str,
                 x1: int, y1: int, x2: int, y2: int,
                 frame_w: int = 640, frame_h: int = 480,
                 track_key: str | None = None) -> float:
        """
        Ước lượng khoảng cách (mét) đã tích hợp calibration.

        Returns: khoảng cách (mét), clamp [0.05, 35.0].
        """
        focal    = _calibration.focal
        coverage = BBOX_COVERAGE.get(cls_name, BBOX_COVERAGE["default"])
        h_real   = REAL_HEIGHT_M.get(cls_name, REAL_HEIGHT_M["default"])
        w_real   = REAL_WIDTH_M.get(cls_name,  REAL_WIDTH_M["default"])

        h_px = max(y2 - y1, 1)
        w_px = max(x2 - x1, 1)
        cx   = (x1 + x2) / 2

        # Bước 1: ước lượng theo 2 trục (có coverage correction)
        d_h = cls._axis_dist(h_real, h_px, focal, coverage)
        d_w = cls._axis_dist(w_real, w_px, focal, coverage)

        # Bước 2: fusion
        raw = cls._fuse(d_h, d_w, h_px, w_px, cls_name)

        # Bước 3: per-class scale từ calibration
        raw *= _calibration.get_scale(cls_name)

        # Bước 4: perspective
        raw *= cls._perspective(cx / max(frame_w, 1))

        # Bước 5: visibility
        raw *= cls._visibility(x1, y1, x2, y2, frame_w, frame_h)

        # Clamp
        raw = min(max(raw, 0.05), 35.0)

        # Bước 6: EMA smooth
        if track_key:
            raw = cls._smooth(track_key, raw)

        return round(raw, 2)

    @classmethod
    def estimate_meters(cls, cls_name: str, y1: int, y2: int) -> float:
        """Backward-compat: ước lượng chỉ từ height."""
        return cls.estimate(cls_name, 0, y1, 0, y2)

    @classmethod
    def classify(cls, cls_name: str, y1: int, y2: int,
                 x1: int = 0, x2: int = 0,
                 frame_w: int = 640, frame_h: int = 480,
                 track_key: str | None = None) -> tuple[str, float]:
        """Trả về (state, distance_m). state: 'critical'|'near'|'far'."""
        d = cls.estimate(cls_name, x1, y1, x2, y2, frame_w, frame_h, track_key)
        if d <= CRITICAL_DIST_M: return "critical", d
        if d <= WARN_DIST_M:     return "near", d
        return "far", d

    @staticmethod
    def classify_state(dist_m: float) -> str:
        if dist_m <= CRITICAL_DIST_M: return "critical"
        if dist_m <= WARN_DIST_M:     return "near"
        return "far"

    @staticmethod
    def box_color(state: str) -> tuple[int, int, int]:
        return {"critical": (0,0,255), "near": (0,100,255)}.get(state, (100,100,100))

    @staticmethod
    def format_dist(dist_m: float) -> str:
        """Hiển thị dạng cm."""
        return f"{int(round(dist_m * 100))} cm"

    @staticmethod
    def state_label(state: str, dist_m: float) -> str:
        d = DistanceEstimator.format_dist(dist_m)
        if state == "critical": return f"!! {d} !!"
        if state == "near":     return f"GẦN {d}"
        return d

    @staticmethod
    def dist_text_vi(dist_m: float) -> str:
        """
        Chuỗi khoảng cách tiếng Việt tự nhiên cho TTS.
        Dưới 2m dùng xăng-ti-mét; từ 2m dùng mét (tự nhiên hơn).
        """
        cm = int(round(dist_m * 100))
        if cm <= 5:
            return "chưa đầy 5 xăng ti mét"
        if cm < 200:
            return f"{cm} xăng ti mét"
        # >= 2m: làm tròn 0.5m
        m_half = round(dist_m * 2) / 2
        if m_half == int(m_half):
            return f"{int(m_half)} mét"
        return f"{int(m_half)} mét rưỡi"

    # ── Tiện ích chẩn đoán ────────────────────────────────────────────────────

    @staticmethod
    def get_calib_info() -> str:
        """Trả về chuỗi thông tin calibration hiện tại."""
        return (f"focal={_calibration.focal:.1f}px | "
                f"scales={_calibration.scales}")

    @staticmethod
    def get_calibration() -> "CalibrationManager":
        """Trả về CalibrationManager singleton để GUI/main dùng."""
        return _calibration



# ══════════════════════════════════════════════════════════════════════════════
#  §7  NAVIGATION GUIDE  —  Chỉ dẫn đường đi trong nhà
# ══════════════════════════════════════════════════════════════════════════════


class NavigationGuide:
    # Ngưỡng giai đoạn khoảng cách (mét)
    _STAGE_FAR    = 8.0   # > 8m  → xa
    _STAGE_MED    = 4.0   # 4–8m  → trung bình
    _STAGE_NEAR   = 2.0   # 2–4m  → gần
    _STAGE_VCLOSE = 1.2   # 1.2–2m → rất gần
    # (< arrived_when) → đến nơi

    # Thời gian chờ trước khi nhắc "bị lạc" (giây)
    LOST_TIMEOUT  = 5.0
    SCAN_TIMEOUT  = 12.0

    # Số frame landmark phải xuất hiện liên tiếp trước khi phát hướng dẫn
    STABLE_FRAMES = 3

    def __init__(self, speaker: "BackgroundSpeaker"):
        self._speaker      = speaker
        self._destination  = None
        self._room_info    = None
        self._arrived      = False
        self._last_msg     = ""

        # Timing
        self._last_guide   = 0.0   # Lần phát hướng dẫn gần nhất
        self._last_seen    = 0.0   # Lần cuối thấy landmark
        self._lost_warned  = False  # Đã nhắc "không thấy" chưa
        self._scan_warned  = False  # Đã nhắc "hãy quay" chưa

        # Theo dõi tiến trình
        self._last_stage   = ""    # Giai đoạn khoảng cách lần trước
        self._last_dir     = ""    # Hướng đi lần trước

        # Ổn định detect
        self._stable_count = 0     # Frame liên tiếp thấy landmark
        self._last_dist    = 99.0  # Khoảng cách gần nhất gần đây

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._destination is not None and not self._arrived

    @property
    def destination(self) -> str | None:
        return self._destination

    def set_destination(self, raw_input: str) -> bool:
        """
        Thiết lập điểm đến. Hỗ trợ alias và tìm kiếm gần đúng (fuzzy match).
        Trả về True nếu tìm thấy, False nếu không nhận ra.
        """
        dest = raw_input.strip().lower()
        dest = ROOM_ALIASES.get(dest, dest)
        if dest not in ROOM_MAP:
            for key in ROOM_MAP:
                if dest in key or key in dest:
                    dest = key
                    break
            else:
                return False

        self._destination  = dest
        self._room_info    = ROOM_MAP[dest]
        self._arrived      = False
        self._last_guide   = 0.0
        self._last_seen    = time.time()   # Bắt đầu countdown ngay
        self._lost_warned  = False
        self._scan_warned  = False
        self._last_stage   = ""
        self._last_dir     = ""
        self._stable_count = 0
        self._last_dist    = 99.0

        hint      = self._room_info["direction_hint"]
        landmark  = CLASS_VI.get(self._room_info["landmark"],
                                 self._room_info["landmark"])
        msg = (f"Bắt đầu dẫn đường đến {dest}. "
               f"{hint}. "
               f"Cột mốc cần tìm là {landmark}.")
        self._last_msg = msg
        print(f"[Nav] ▶ Mục tiêu: {dest} | Cột mốc: {self._room_info['landmark']}")
        self._speaker.say(msg, priority=True)
        return True

    def cancel(self):
        """Hủy điều hướng hiện tại."""
        self._destination = None
        self._room_info   = None
        self._arrived     = False

    def repeat_last(self):
        """Phát lại câu chỉ dẫn gần nhất (phím H)."""
        if self._last_msg:
            self._speaker.say(self._last_msg, priority=True)

    # ── Cập nhật mỗi frame ────────────────────────────────────────────────────

    def update(self, detections: list[dict], frame_w: int):
        """
        Gọi mỗi YOLO frame khi active == True.

        detections: list[dict] từ pipeline YOLO, mỗi phần tử có:
            label, dist_m, state, x1, y1, x2, y2
        frame_w: chiều rộng frame (px).
        """
        if not self.active:
            return

        landmark  = self._room_info["landmark"]
        arrived_d = self._room_info["arrived_when"]
        now       = time.time()

        # ── Tìm landmark trong frame ──────────────────────────────────────────
        candidates = [d for d in detections if d.get("label") == landmark]

        if not candidates:
            self._stable_count = 0
            self._handle_lost(now, landmark)
            return

        # Landmark thấy được → reset trạng thái lạc
        self._last_seen   = now
        self._lost_warned = False
        self._scan_warned = False
        self._stable_count += 1

        # Lấy landmark gần nhất
        target = min(candidates, key=lambda d: d["dist_m"])
        dist   = target["dist_m"]
        self._last_dist = dist

        # ── Kiểm tra đã đến nơi chưa ─────────────────────────────────────────
        if dist <= arrived_d:
            if not self._arrived:
                self._arrived = True
                msg = (f"Bạn đã đến {self._destination}! "
                       f"Điểm đến ngay trước mặt bạn.")
                self._last_msg = msg
                self._speaker.say(msg, priority=True)
                print(f"[Nav] ✓ Đã đến: {self._destination}")
            return

        # ── Chờ ổn định trước khi hướng dẫn ─────────────────────────────────
        if self._stable_count < self.STABLE_FRAMES:
            return

        # ── Xây dựng hướng dẫn thông minh ────────────────────────────────────
        stage     = self._get_stage(dist)
        direction = self._get_direction(target, frame_w)
        cooldown  = self._get_cooldown(stage)

        # Phát hướng dẫn khi: (1) đủ cooldown, hoặc (2) đổi giai đoạn/hướng
        stage_changed = (stage != self._last_stage)
        dir_changed   = (direction != self._last_dir and stage != "very_close")
        time_ok       = (now - self._last_guide >= cooldown)

        if not (time_ok or stage_changed or dir_changed):
            return

        # Phân tích vật cản trên đường
        obstacle_msg = self._analyze_obstacles(detections, landmark,
                                               target, frame_w)

        # Tạo câu chỉ dẫn theo ngữ cảnh
        msg = self._build_message(stage, direction, dist,
                                  landmark, obstacle_msg, stage_changed)

        self._last_msg    = msg
        self._last_guide  = now
        self._last_stage  = stage
        self._last_dir    = direction
        self._speaker.say(msg, priority=(stage == "very_close"))
        print(f"[Nav] [{stage}] {msg}")

    # ── Xử lý trạng thái bị lạc ───────────────────────────────────────────────

    def _handle_lost(self, now: float, landmark: str):
        """Nhắc nhở khi không thấy landmark trong một khoảng thời gian."""
        elapsed = now - self._last_seen
        lm_vi   = CLASS_VI.get(landmark, landmark)

        if elapsed >= self.SCAN_TIMEOUT and not self._scan_warned:
            self._scan_warned = True
            msg = (f"Vẫn chưa tìm thấy {lm_vi}. "
                   f"Hãy xoay người từ từ sang trái hoặc phải để quét xung quanh.")
            self._last_msg = msg
            self._speaker.say(msg, priority=True)
            self._last_guide = now
            print(f"[Nav] [SCAN] {msg}")

        elif elapsed >= self.LOST_TIMEOUT and not self._lost_warned:
            self._lost_warned = True
            msg = (f"Không thấy {lm_vi} trong khung hình. "
                   f"Hãy nhìn xung quanh để tìm {lm_vi}.")
            self._last_msg = msg
            self._speaker.say(msg, priority=False)
            self._last_guide = now
            print(f"[Nav] [LOST] {msg}")

    # ── Phân tích vật cản trên đường ──────────────────────────────────────────

    def _analyze_obstacles(self, detections: list[dict], landmark: str,
                           target: dict, frame_w: int) -> str:
        """
        Phân tích vật cản giữa người dùng và landmark.
        Trả về chuỗi mô tả (rỗng nếu không có vật cản đáng kể).

        Thuật toán:
        1. Lọc vật cản critical/near, gần hơn landmark (không phải landmark).
        2. Kiểm tra vật cản có nằm cùng vùng ngang với landmark không.
        3. Nếu có: gợi ý tránh sang bên nào ít cản trở hơn.
        """
        target_cx = (target["x1"] + target["x2"]) / 2
        target_dist = target["dist_m"]

        # Lọc vật cản nguy hiểm (critical/near) gần hơn landmark
        blockers = [
            d for d in detections
            if d.get("label") != landmark
            and d.get("state") in ("critical", "near")
            and d.get("dist_m", 99) < target_dist
        ]

        if not blockers:
            return ""

        # Tìm vật cản gần nhất
        closest = min(blockers, key=lambda d: d["dist_m"])
        obs_vi  = CLASS_VI.get(closest["label"], closest["label"])
        obs_cx  = (closest["x1"] + closest["x2"]) / 2
        obs_dist = closest["dist_m"]

        # Phân tích lối đi tự do: chia frame thành 5 vùng ngang
        # Tính "mật độ vật cản" mỗi vùng → gợi ý vùng trống nhất
        zones     = [0] * 5
        zone_w    = frame_w / 5
        for b in blockers:
            bcx = (b["x1"] + b["x2"]) / 2
            zi  = min(int(bcx / zone_w), 4)
            # Vật gần hơn có trọng số cao hơn
            weight = 2 if b.get("state") == "critical" else 1
            zones[zi] += weight

        # Vùng landmark
        lm_zone = min(int(target_cx / zone_w), 4)

        # Gợi ý hướng tránh: tìm vùng trống bên cạnh landmark
        if lm_zone > 0 and zones[lm_zone - 1] < zones[lm_zone]:
            avoid = "Hãy đi sang trái một chút để tránh vật cản"
        elif lm_zone < 4 and zones[lm_zone + 1] < zones[lm_zone]:
            avoid = "Hãy đi sang phải một chút để tránh vật cản"
        elif obs_cx < frame_w / 2:
            avoid = "Hãy đi sang phải để tránh vật cản bên trái"
        else:
            avoid = "Hãy đi sang trái để tránh vật cản bên phải"

        dist_vi = DistanceEstimator.dist_text_vi(obs_dist)
        urgency = "Dừng lại!" if closest.get("state") == "critical" else "Chú ý!"
        return f"{urgency} Có {obs_vi} cách {dist_vi}. {avoid}."

    # ── Hàm hỗ trợ nội bộ ────────────────────────────────────────────────────

    def _get_stage(self, dist_m: float) -> str:
        """Phân loại khoảng cách thành giai đoạn."""
        if dist_m > self._STAGE_FAR:
            return "far"
        elif dist_m > self._STAGE_MED:
            return "medium"
        elif dist_m > self._STAGE_NEAR:
            return "near"
        elif dist_m > self._STAGE_VCLOSE:
            return "very_close"
        else:
            return "arrived"

    def _get_direction(self, target: dict, frame_w: int) -> str:
        """Tính hướng đi dựa trên vị trí nằm ngang của landmark."""
        cx    = (target["x1"] + target["x2"]) / 2
        # Chia 5 vùng: << < giữa > >>
        fifth = frame_w / 5
        if cx < fifth:
            return "rẽ trái mạnh"
        elif cx < 2 * fifth:
            return "rẽ trái nhẹ"
        elif cx < 3 * fifth:
            return "đi thẳng"
        elif cx < 4 * fifth:
            return "rẽ phải nhẹ"
        else:
            return "rẽ phải mạnh"

    def _get_cooldown(self, stage: str) -> float:
        """Cooldown giữa 2 lần hướng dẫn, ngắn hơn khi gần hơn."""
        return {
            "far":        6.0,
            "medium":     4.5,
            "near":       3.0,
            "very_close": 2.0,
        }.get(stage, NAV_COOLDOWN)

    def _build_message(self, stage: str, direction: str, dist_m: float,
                       landmark: str, obstacle_msg: str,
                       stage_changed: bool) -> str:
        """Tạo câu chỉ dẫn tự nhiên theo ngữ cảnh."""
        lm_vi   = CLASS_VI.get(landmark, landmark)
        dest    = self._destination or ""
        dist_vi = DistanceEstimator.dist_text_vi(dist_m)

        # Tiền tố khi chuyển giai đoạn
        stage_prefix = ""
        if stage_changed:
            stage_prefix = {
                "far":        "",
                "medium":     "Tốt lắm! ",
                "near":       "Bạn đang tiến gần. ",
                "very_close": "Gần đến rồi! ",
            }.get(stage, "")

        # Câu hướng dẫn chính theo giai đoạn
        if stage == "far":
            base = f"Hãy {direction} về phía {lm_vi}. Còn khoảng {dist_vi}."
        elif stage == "medium":
            base = f"{direction.capitalize()} về phía {lm_vi}. Còn {dist_vi}."
        elif stage == "near":
            base = f"Tiếp tục {direction}. {lm_vi} cách {dist_vi}."
        elif stage == "very_close":
            base = (f"{lm_vi} chỉ còn {dist_vi} phía trước. "
                    f"Thêm vài bước nữa là đến {dest}.")
        else:
            base = f"Bạn đã đến {dest}!"

        # Ghép vật cản (ưu tiên vật cản trước câu chỉ dẫn)
        if obstacle_msg:
            return f"{obstacle_msg} Sau đó {base}"
        return f"{stage_prefix}{base}"


# ══════════════════════════════════════════════════════════════════════════════
#  §8  MONEY DETECTOR  —  Nhận diện tiền VN (model custom)
# ══════════════════════════════════════════════════════════════════════════════


class MoneyDetector:
    """
    Nhận diện mệnh giá tiền VN bằng YOLOv8 custom.
    Resize input về 320×320 để tối ưu CPU.
    """

    def __init__(self, model_path: str = MONEY_MODEL_PATH):
        self.model = None
        self.ready = False
        if not os.path.exists(model_path):
            print(f"[Money] ⚠  Không tìm thấy {model_path}. Module tiền bị tắt.")
            return
        try:
            self.model = YOLO(model_path)
            self.ready = True
            print(f"[Money] ✓  Đã load {model_path}")
            print(f"[Money]    Classes: {list(self.model.names.values())}")
        except Exception as exc:
            print(f"[Money] Lỗi load model: {exc}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Trả về list {label, conf, x1, y1, x2, y2} đã scale về kích thước gốc."""
        if not self.ready:
            return []
        small = cv2.resize(frame, (320, 320))
        results = self.model(small, verbose=False, conf=MONEY_CONF, iou=0.45)
        h0, w0 = frame.shape[:2]
        sx, sy = w0 / 320, h0 / 320
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_name   = self.model.names[int(box.cls[0])]
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": cls_name, "conf": conf_score,
                    "x1": int(x1*sx), "y1": int(y1*sy),
                    "x2": int(x2*sx), "y2": int(y2*sy),
                })
        return detections


# ══════════════════════════════════════════════════════════════════════════════
#  §9  OCR READER  —  Đọc văn bản (EasyOCR, background thread)
# ══════════════════════════════════════════════════════════════════════════════


class OCRReader:
    # Kích thước ảnh tối thiểu để OCR hiệu quả
    MIN_SIDE    = 640
    # Ngưỡng confidence để chấp nhận kết quả
    CONF_THRESH = 0.45
    # Số ký tự tối thiểu của 1 text block để chấp nhận
    MIN_CHARS   = 2
    # Kích thước chunk TTS (ký tự)
    CHUNK_CHARS = 80

    def __init__(self):
        self.reader     = None
        self.ready      = False
        self._in_q      = queue.Queue(maxsize=1)
        self._out_q     = queue.Queue(maxsize=1)
        self._busy      = False
        self.last_text  = ""
        self.last_boxes: list = []
        if not EASYOCR_AVAILABLE:
            print("[OCR] ⚠ EasyOCR chưa cài. Chạy: pip install easyocr")
            return
        threading.Thread(target=self._init_reader, daemon=True).start()

    def _init_reader(self):
        print("[OCR] Đang khởi tạo EasyOCR v2 (lần đầu tải model ~500 MB)...")
        try:
            self.reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=False,
                verbose=False,
                # Tắt các option chậm không cần thiết
            )
            self.ready = True
            print("[OCR] ✓  EasyOCR v2 sẵn sàng.")
            threading.Thread(target=self._worker, daemon=True).start()
        except Exception as exc:
            print(f"[OCR] Lỗi khởi tạo: {exc}")

    # ── Worker background thread ───────────────────────────────────────────────

    def _worker(self):
        while True:
            try:
                frame = self._in_q.get(timeout=1)
                self._busy = True
                try:
                    result = self._ocr_pipeline(frame)
                    self._out_q.put_nowait(result)
                except queue.Full:
                    pass
                except Exception as exc:
                    print(f"[OCR] Lỗi pipeline: {exc}")
                    self._out_q.put_nowait(("", []))
                finally:
                    self._busy = False
                    self._in_q.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[OCR] Lỗi worker: {exc}")
                self._busy = False

    # ── Pipeline xử lý ảnh + OCR ─────────────────────────────────────────────

    def _ocr_pipeline(self, frame: np.ndarray) -> tuple[str, list]:
        """
        Pipeline đầy đủ:
        1. Tiền xử lý ảnh
        2. Multi-pass OCR (preprocessed + grayscale)
        3. Lọc & dedup kết quả
        4. Chuẩn hóa text
        """
        h, w = frame.shape[:2]

        # ── Bước 1: Upscale nếu ảnh quá nhỏ ────────────────────────────────
        if max(h, w) < self.MIN_SIDE:
            scale = self.MIN_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        # Giới hạn kích thước tối đa để không quá chậm
        max_side = 1280
        if max(frame.shape[:2]) > max_side:
            s = max_side / max(frame.shape[:2])
            frame = cv2.resize(frame,
                               (int(frame.shape[1]*s), int(frame.shape[0]*s)))

        # ── Bước 2: Tạo ảnh tiền xử lý ──────────────────────────────────────
        preprocessed = self._preprocess(frame)

        # ── Bước 3: Multi-pass OCR ───────────────────────────────────────────
        all_results = []

        # Lượt 1: ảnh đã xử lý (tốt cho text in, signage, label)
        try:
            r1 = self.reader.readtext(
                preprocessed,
                paragraph=False,    # Không gộp để giữ vị trí chính xác
                detail=1,
                batch_size=4,
            )
            all_results.extend(r1)
        except Exception:
            pass

        # Lượt 2: grayscale gốc (tốt cho text tay, màu sắc)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            r2 = self.reader.readtext(
                gray,
                paragraph=False,
                detail=1,
                batch_size=4,
            )
            all_results.extend(r2)
        except Exception:
            pass

        # ── Bước 4: Lọc, dedup, chuẩn hóa ───────────────────────────────────
        good = self._filter_results(all_results)

        # Gộp text theo thứ tự vị trí (từ trên xuống, trái sang phải)
        good.sort(key=lambda x: (x[0][0][1], x[0][0][0]))   # sort by (y, x)

        text_parts = [t for _, t, _ in good]
        raw_text   = " ".join(text_parts)
        clean_text = self._normalize_text(raw_text)

        return clean_text, good

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        4-bước tiền xử lý ảnh để tăng chất lượng OCR.
        """
        # Chuyển grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE — cải thiện tương phản cục bộ
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray   = clahe.apply(gray)

        # Denoise nhẹ (fast non-local means)
        gray   = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                           searchWindowSize=15)

        # Adaptive threshold — tách text khỏi nền phức tạp
        # (dùng khi ảnh có gradient hoặc bóng đổ)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10
        )

        # Morphological opening nhẹ để loại noise nhỏ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    # ── Lọc và chuẩn hóa ──────────────────────────────────────────────────────

    def _filter_results(self, results: list) -> list:
        """
        Lọc kết quả OCR:
        - Bỏ confidence thấp
        - Bỏ text quá ngắn hoặc toàn ký tự rác
        - Dedup: bỏ text trùng về nội dung (từ 2 lượt scan)
        """
        seen_texts = set()
        good       = []

        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            bbox, txt, conf = item[0], item[1], item[2]

            # Lọc confidence
            if conf < self.CONF_THRESH:
                continue

            # Chuẩn hóa text để kiểm tra trùng
            txt_clean = txt.strip()
            if len(txt_clean) < self.MIN_CHARS:
                continue

            # Bỏ chuỗi toàn ký tự đặc biệt / số lẻ
            alpha_count = sum(1 for c in txt_clean if c.isalnum())
            if alpha_count < max(1, len(txt_clean) // 3):
                continue

            # Dedup: so sánh lowercase stripped
            key = txt_clean.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)

            good.append((bbox, txt_clean, conf))

        return good

    def _normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa text để đọc TTS tự nhiên hơn.
        - Rút gọn khoảng trắng thừa
        - Giữ nguyên cấu trúc câu
        """
        import re
        # Rút gọn khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        # Thêm dấu chấm nếu text không kết thúc bằng dấu câu
        if text and text[-1] not in '.!?,;:':
            text += '.'
        return text

    def get_tts_chunks(self, text: str) -> list[str]:
        """
        Chia text thành các đoạn ngắn cho TTS đọc tự nhiên.
        Ưu tiên cắt tại dấu câu (.!?,) hoặc khoảng trắng.
        """
        if len(text) <= self.CHUNK_CHARS:
            return [text] if text.strip() else []

        chunks = []
        while len(text) > self.CHUNK_CHARS:
            # Tìm điểm cắt tốt nhất trong CHUNK_CHARS ký tự cuối
            cut = self.CHUNK_CHARS
            # Ưu tiên cắt tại dấu câu
            for i in range(cut, max(cut - 30, 0), -1):
                if i < len(text) and text[i] in '.!?,':
                    cut = i + 1
                    break
            else:
                # Fallback: cắt tại khoảng trắng
                for i in range(cut, max(cut - 20, 0), -1):
                    if i < len(text) and text[i] == ' ':
                        cut = i
                        break
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            chunks.append(text)
        return [c for c in chunks if c]

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, frame: np.ndarray) -> bool:
        """Gửi frame vào hàng đợi OCR. Trả về False nếu đang bận."""
        if not self.ready or self._busy:
            return False
        try:
            self._in_q.put_nowait(frame.copy())
            return True
        except queue.Full:
            return False

    def get_result(self) -> tuple[str, list] | None:
        """Lấy kết quả nếu đã xong. Trả về None nếu chưa hoàn thành."""
        try:
            text, boxes = self._out_q.get_nowait()
            self.last_text  = text
            self.last_boxes = boxes
            return text, boxes
        except queue.Empty:
            return None

    def draw_results(self, frame: np.ndarray):
        """Vẽ bounding box + text OCR lên frame với màu theo confidence."""
        for bbox, txt, conf in self.last_boxes:
            pts = np.array(bbox, dtype=np.int32)
            # Màu theo confidence: xanh lá (cao) → vàng → đỏ (thấp)
            if conf >= 0.80:
                color = (0, 255, 100)    # Xanh lá
            elif conf >= 0.60:
                color = (0, 220, 255)    # Vàng
            else:
                color = (80, 80, 255)    # Cam nhạt

            cv2.polylines(frame, [pts], True, color, 2)
            x, y = pts[0]
            label = f"{txt[:25]}  {conf:.0%}"
            # Nền cho text label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, max(y-18, 0)), (x+tw+4, max(y, 14)),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (x+2, max(y-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  §10 TRAFFIC LIGHT  —  Nhận diện đèn giao thông (HSV + temporal voting)
# ══════════════════════════════════════════════════════════════════════════════


class TrafficLightAnalyzer:
    """
    Phân tích màu đèn giao thông — 3 cải tiến chính:

    1. Phân vùng dọc (Vertical Zone Analysis):
       Đèn giao thông có 3 bóng theo chiều dọc: Đỏ (trên) → Vàng (giữa) → Xanh (dưới).
       → Chỉ phân tích vùng có độ sáng V cao nhất thay vì toàn bbox.

    2. Xác nhận chéo vị trí ↔ màu:
       Nếu màu phát hiện mâu thuẫn với vị trí bóng sáng nhất → tin vị trí hơn.

    3. Temporal Voting qua nhiều frame:
       Giữ buffer 5 kết quả gần nhất. Màu nào đa số → output.
       Ngăn đèn "nhấp nháy" do nhiễu ánh sáng.
    """

    # Kích thước resize bbox trước khi phân tích (giữ tỷ lệ đứng để 3 vùng rõ)
    _W = 32
    _H = 96
    # Số frame giữ trong buffer vote
    _VOTE_N = 5
    # Vị trí dải → màu kỳ vọng (đỏ trên, vàng giữa, xanh dưới)
    _BAND_COLOR = {0: "red", 1: "yellow", 2: "green"}

    def __init__(self):
        # Buffer vote riêng cho mỗi đèn (key = track_key từ caller)
        self._buffers: dict[str, list[str]] = {}

    # ── Đếm pixel màu trong vùng HSV ─────────────────────────────────────────

    @staticmethod
    def _count(hsv: np.ndarray, ranges: list[tuple]) -> int:
        total = 0
        for r in ranges:
            lo = np.array([r[0], r[1], r[2]], dtype=np.uint8)
            hi = np.array([r[3], r[4], r[5]], dtype=np.uint8)
            total += int(cv2.countNonZero(cv2.inRange(hsv, lo, hi)))
        return total

    # ── Tìm dải dọc sáng nhất trong 3 vùng ──────────────────────────────────

    @staticmethod
    def _brightest_band(hsv: np.ndarray) -> int:
        """Trả về 0=trên(đỏ), 1=giữa(vàng), 2=dưới(xanh) — dải có V cao nhất."""
        h = hsv.shape[0]
        t = h // 3
        means = [
            float(np.mean(hsv[:t,    :, 2])),   # dải trên
            float(np.mean(hsv[t:2*t, :, 2])),   # dải giữa
            float(np.mean(hsv[2*t:,  :, 2])),   # dải dưới
        ]
        return int(np.argmax(means))

    # ── Phân tích 1 frame (chưa qua vote) ────────────────────────────────────

    def _analyze_raw(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown"
        h, w = crop_bgr.shape[:2]
        if h < 15 or w < 8:
            return "unknown"

        small = cv2.resize(crop_bgr, (self._W, self._H),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Tìm dải sáng nhất và mở rộng ±4px để có đủ mẫu màu
        band  = self._brightest_band(hsv)
        t     = self._H // 3
        y0    = max(0,          band * t - 4)
        y1    = min(self._H,    band * t + t + 4)
        region = hsv[y0:y1, :]

        if region.size == 0:
            region = hsv

        # Đếm pixel mỗi màu trong vùng sáng nhất
        counts = {c: self._count(region, r) for c, r in TRAFFIC_HSV.items()}
        total  = region.shape[0] * region.shape[1]
        best   = max(counts, key=counts.get)
        ratio  = counts[best] / max(total, 1)

        # Ngưỡng chấp nhận: màu chiếm >= 6% pixel vùng phân tích
        if ratio < 0.06:
            return "unknown"

        # Xác nhận chéo: nếu màu mâu thuẫn vị trí và chưa rõ ràng → tin vị trí
        expected = self._BAND_COLOR[band]
        if best != expected and ratio < 0.15:
            return expected

        return best

    # ── API công khai — có temporal voting ────────────────────────────────────

    def analyze(self, crop_bgr: np.ndarray,
                track_key: str = "default") -> str:
        """
        Phân tích màu đèn với temporal voting qua nhiều frame.
        track_key: định danh đèn (vd: "tl_120_80") — tách buffer riêng mỗi đèn.
        """
        raw = self._analyze_raw(crop_bgr)

        buf = self._buffers.setdefault(track_key, [])
        buf.append(raw)
        if len(buf) > self._VOTE_N:
            buf.pop(0)

        # Vote — bỏ "unknown" ra trước khi đếm
        meaningful = [c for c in buf if c != "unknown"]
        if not meaningful:
            return "unknown"

        from collections import Counter
        winner, win_count = Counter(meaningful).most_common(1)[0]

        # Chỉ chấp nhận khi đa số >= 40%
        if win_count / len(meaningful) < 0.40:
            return meaningful[-1]   # chưa ổn định → dùng kết quả mới nhất

        return winner

    def clear_track(self, track_key: str):
        """Xóa buffer khi đèn biến mất khỏi frame."""
        self._buffers.pop(track_key, None)


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 3: TIỆN ÍCH VẼ TIẾNG VIỆT LÊN FRAME OPENCV (dùng chung)
#  Cả main.py (terminal) và gui_app.py đều cần vẽ lên frame camera.
# ══════════════════════════════════════════════════════════════════════════════

_VI_FONT_CACHE: dict = {}


def _get_vi_font(size: int):
    """Tìm và cache font TTF hỗ trợ tiếng Việt (Windows / Linux / macOS)."""
    if size in _VI_FONT_CACHE:
        return _VI_FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/arial.ttf",   "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = None
    for p in candidates:
        if os.path.exists(p):
            try:
                font = _PILF.truetype(p, size)
                break
            except Exception:
                pass
    if font is None:
        font = _PILF.load_default()
    _VI_FONT_CACHE[size] = font
    return font


def put_vi_text(frame: np.ndarray, text: str, xy: tuple,
                font_size: int = 16,
                color: tuple = (255, 255, 255),
                bg: tuple = None) -> None:
    """Vẽ text tiếng Việt có dấu lên frame OpenCV in-place dùng Pillow."""
    if not _PIL_OK or not text:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 28, color, 1)
        return
    font   = _get_vi_font(font_size)
    fh, fw = frame.shape[:2]
    x0, y0 = int(xy[0]), int(xy[1])
    dummy  = _PILI.new("RGB", (1, 1))
    bbox_t = _PILD.Draw(dummy).textbbox((0, 0), text, font=font)
    tw = bbox_t[2] - bbox_t[0] + 6
    th = bbox_t[3] - bbox_t[1] + 6
    x1e = min(x0 + tw, fw)
    y1e = min(y0 + th, fh)
    if x1e <= x0 or y1e <= y0:
        return
    patch_bgr = frame[y0:y1e, x0:x1e].copy()
    patch_rgb = _PILI.fromarray(patch_bgr[:, :, ::-1])
    draw = _PILD.Draw(patch_rgb)
    if bg is not None:
        draw.rectangle([0, 0, patch_rgb.width, patch_rgb.height],
                       fill=(bg[2], bg[1], bg[0]))
    draw.text((3, 3), text, font=font, fill=(color[2], color[1], color[0]))
    frame[y0:y1e, x0:x1e] = np.array(patch_rgb)[:, :, ::-1]


def vi_text_size(text: str, font_size: int = 16) -> tuple:
    """Trả về (width, height) ước tính của text."""
    if not _PIL_OK:
        return (int(len(text) * font_size * 0.6), font_size + 4)
    font   = _get_vi_font(font_size)
    dummy  = _PILI.new("RGB", (1, 1))
    bbox_t = _PILD.Draw(dummy).textbbox((0, 0), text, font=font)
    return (bbox_t[2] - bbox_t[0] + 6, bbox_t[3] - bbox_t[1] + 6)


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 4: HÀM VẼ BOUNDING BOX (dùng chung cho Terminal & GUI)
# ══════════════════════════════════════════════════════════════════════════════

def draw_box_obstacle(frame, x1, y1, x2, y2,
                      label, conf, state, dist_m, is_known=False):
    """
    Vẽ bounding box vật cản lên frame.
    is_known=True → vàng cam (người quen); màu còn lại theo khoảng cách.
    """
    color = (0, 200, 255) if is_known else DistanceEstimator.box_color(state)
    lw    = 3 if state in ("near", "critical") else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw)
    if state == "critical" and not is_known:
        cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (0, 0, 180), 2)
    display_name = label if is_known else CLASS_VI.get(label, label)
    line1 = f"{display_name}  {conf:.0%}"
    line2 = DistanceEstimator.state_label(state, dist_m)
    fs    = 14
    tw1, th1 = vi_text_size(line1, fs)
    tw2, th2 = vi_text_size(line2, fs)
    tw       = max(tw1, tw2)
    th_total = th1 + th2
    lx = x1
    ly = max(y1 - th_total - 6, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + 4, ly + th_total + 6), color, -1)
    txt_c = (0, 0, 80) if is_known else (255, 255, 255)
    put_vi_text(frame, line1, (lx + 2, ly + 2),       font_size=fs, color=txt_c)
    put_vi_text(frame, line2, (lx + 2, ly + th1 + 4), font_size=fs, color=txt_c)


def draw_box_money(frame, x1, y1, x2, y2, label, conf):
    """Vẽ bounding box tiền với tên mệnh giá tiếng Việt (hỗ trợ label dạng '10k', '50.000', v.v.)."""
    norm    = normalize_money_label(label)
    vi_name = MONEY_VI.get(norm, label)   # fallback: hiển thị label gốc
    color   = (0, 215, 255)
    text    = f"{vi_name}  {conf:.0%}"
    fs      = 15
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    tw, th = vi_text_size(text, fs)
    ly = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, ly), (x1 + tw + 4, ly + th + 4), color, -1)
    put_vi_text(frame, text, (x1 + 2, ly + 2), font_size=fs, color=(0, 0, 0))


def draw_box_traffic(frame, x1, y1, x2, y2, color_name: str, conf: float):
    """Vẽ bounding box đèn giao thông với màu tương ứng."""
    bgr = TRAFFIC_COLORS_BGR.get(color_name, TRAFFIC_COLORS_BGR["unknown"])
    label_vi = {"red": "Đèn đỏ", "green": "Đèn xanh",
                "yellow": "Đèn vàng"}.get(color_name, "Đèn GT")
    text = f"{label_vi}  {conf:.0%}"
    fs   = 15
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
    if color_name == "red":
        cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (0, 0, 150), 2)
    tw, th = vi_text_size(text, fs)
    ly = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, ly), (x1 + tw + 4, ly + th + 4), bgr, -1)
    put_vi_text(frame, text, (x1 + 2, ly + 2), font_size=fs, color=(255, 255, 255))


def draw_nav_overlay(frame, nav: "NavigationGuide"):
    """Vẽ overlay điều hướng lên góc dưới trái frame với thông tin stage."""
    if not nav.active:
        return
    dest  = nav._destination or ""
    stage = nav._last_stage or ""
    dist  = nav._last_dist

    # Icon và màu theo giai đoạn
    stage_info = {
        "far":        ("→", (100, 200, 100)),
        "medium":     ("→→", (0, 220, 150)),
        "near":       (">>>", (0, 180, 255)),
        "very_close": ("!!!",  (0, 80, 255)),
    }
    icon, color = stage_info.get(stage, ("→", (0, 255, 100)))

    dist_str = f" {dist:.1f}m" if dist < 30 else ""
    msg  = f"{icon} Đang dẫn → {dest}{dist_str}"
    tw, th = vi_text_size(msg, 15)
    fh     = frame.shape[0]
    y0     = fh - th - 8
    cv2.rectangle(frame, (4, y0 - 2), (tw + 10, fh - 4), (0, 60, 0), -1)
    put_vi_text(frame, msg, (6, y0), font_size=15, color=color)


# ══════════════════════════════════════════════════════════════════════════════
#  §11 TIME READER  —  Đọc giờ hiện tại bằng tiếng Việt
# ══════════════════════════════════════════════════════════════════════════════

class TimeReader:
    """
    Đọc thời gian hiện tại bằng tiếng Việt qua TTS.

    Ví dụ output:
        "Bây giờ là 9 giờ 5 phút sáng"
        "Bây giờ là 12 giờ trưa"
        "Bây giờ là 3 giờ 30 phút chiều"
        "Bây giờ là 7 giờ tối"
        "Bây giờ là 11 giờ 45 phút đêm"

    Cách dùng:
        reader = TimeReader(speaker)
        reader.announce()     # phát TTS ngay lập tức
        text = reader.text()  # lấy chuỗi để hiển thị
    """

    def __init__(self, speaker: "BackgroundSpeaker"):
        self._speaker = speaker

    # ── Chuyển giờ số sang tiếng Việt tự nhiên ───────────────────────────────

    @staticmethod
    def _period(hour: int) -> str:
        """Trả về buổi trong ngày bằng tiếng Việt."""
        if hour < 6:   return "đêm"
        if hour < 11:  return "sáng"
        if hour < 13:  return "trưa"
        if hour < 18:  return "chiều"
        return "tối"

    @staticmethod
    def _hour_12(hour: int) -> int:
        """Chuyển giờ 24h → 12h (0 → 12, 13 → 1, v.v.)."""
        h = hour % 12
        return 12 if h == 0 else h

    @staticmethod
    def now_text() -> str:
        """
        Trả về chuỗi giờ hiện tại tự nhiên để đọc TTS.
        Luôn dùng định dạng 12h + buổi.
        """
        now    = datetime.now()
        hour   = now.hour
        minute = now.minute
        h12    = TimeReader._hour_12(hour)
        period = TimeReader._period(hour)

        if minute == 0:
            return f"Bây giờ là {h12} giờ {period}"
        elif minute < 10:
            return f"Bây giờ là {h12} giờ 0{minute} phút {period}"
        else:
            return f"Bây giờ là {h12} giờ {minute} phút {period}"

    @staticmethod
    def now_display() -> str:
        """Chuỗi ngắn để hiển thị trên màn hình: HH:MM  Thứ X  DD/MM/YYYY"""
        now = datetime.now()
        weekdays = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm",
                    "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        wd = weekdays[now.weekday()]
        return (f"{now.strftime('%H:%M')}  {wd}  "
                f"{now.strftime('%d/%m/%Y')}")

    def announce(self, priority: bool = True):
        """Phát âm thanh thời gian hiện tại."""
        text = self.now_text()
        print(f"[Time] {text}")
        self._speaker.say(text, priority=priority)
        return text


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 5: SESSION LOGGER — Lưu lịch sử nhận diện ra file CSV + JSON
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  §12 SESSION LOGGER  —  Ghi lịch sử phiên (CSV + JSON, 3 mức độ)
# ══════════════════════════════════════════════════════════════════════════════

class SessionLogger:
    """
    Ghi lịch sử nhận diện thông minh — kiểm soát cường độ ghi log.

    ── 3 MỨC ĐỘ GHI (LOG_LEVEL) ────────────────────────────────────────────────
    LOG_LEVEL = "summary"   Chỉ ghi tóm tắt cuối phiên (1 dòng/loại vật).
                            → File nhỏ nhất, không spam.
    LOG_LEVEL = "normal"    Ghi sự kiện quan trọng (near/critical, tiền, mặt, OCR).
                            Có cooldown — cùng loại không ghi quá 1 lần/10s.
                            → Cân bằng giữa chi tiết và dung lượng. (Mặc định)
    LOG_LEVEL = "detail"    Ghi tất cả (như cũ). Dùng khi debug.

    ── COOLDOWN PER-EVENT ───────────────────────────────────────────────────────
    Cùng 1 loại vật cản không được ghi lặp trong LOG_COOLDOWN giây.
    Ví dụ: 'person near' chỉ xuất hiện tối đa 1 lần/10s trong CSV.

    ── GIỚI HẠN BỘ NHỚ ─────────────────────────────────────────────────────────
    Chỉ giữ tối đa MAX_EVENTS sự kiện trong RAM.
    Khi đầy → xóa 20% sự kiện cũ nhất (rolling buffer).
    """

    LOG_DIR      = "logs"
    LOG_COOLDOWN = 10.0    # giây — cùng loại event không ghi lặp trong khoảng này
    MAX_EVENTS   = 500     # tối đa 500 sự kiện trong RAM (≈ 150 KB)

    def __init__(self, enabled: bool = True, level: str = "normal"):
        """
        level: "summary" | "normal" | "detail"
        """
        assert level in ("summary", "normal", "detail"), \
            "level phải là 'summary', 'normal' hoặc 'detail'"
        self.enabled     = enabled
        self.level       = level
        self._events:    list[dict]         = []
        self._last_wrote: dict[str, float]  = {}   # Cooldown per event-key
        self._counts:    dict[str, int]     = {}   # Đếm số lần mỗi loại xảy ra
        self._start_time = datetime.now()
        self._csv_path   = ""
        self._json_path  = ""
        self._csv_file   = None
        self._writer     = None

        if not enabled:
            return

        os.makedirs(self.LOG_DIR, exist_ok=True)
        ts = self._start_time.strftime("%Y%m%d_%H%M%S")
        self._csv_path  = os.path.join(self.LOG_DIR, f"session_{ts}.csv")
        self._json_path = os.path.join(self.LOG_DIR, f"session_{ts}.json")

        # Mở CSV (chỉ khi level != summary)
        if level != "summary":
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8-sig")
            self._writer   = csv.DictWriter(self._csv_file, fieldnames=[
                "timestamp", "elapsed_s", "event_type",
                "label_vi", "state", "dist_display", "conf", "note",
            ])
            self._writer.writeheader()
            self._csv_file.flush()

        print(f"[Logger] Level={level} | log → {self._csv_path or '(summary only)'}")

    # ── Cooldown helper ───────────────────────────────────────────────────────

    def _allowed(self, key: str, cooldown: float | None = None) -> bool:
        """Trả về True nếu event này được phép ghi (cooldown chưa hết)."""
        cd  = cooldown if cooldown is not None else self.LOG_COOLDOWN
        now = time.time()
        if now - self._last_wrote.get(key, 0) >= cd:
            self._last_wrote[key] = now
            return True
        return False

    # ── Ghi nội bộ ───────────────────────────────────────────────────────────

    def _write(self, event_type: str, label_vi: str = "", state: str = "",
               dist_m: float | None = None, conf: float | None = None,
               note: str = ""):
        """Ghi 1 dòng vào CSV + thêm vào RAM buffer."""
        if not self.enabled or self.level == "summary":
            return
        now     = datetime.now()
        elapsed = (now - self._start_time).total_seconds()
        dist_d  = DistanceEstimator.format_dist(dist_m) if dist_m is not None else ""
        row = {
            "timestamp":    now.strftime("%H:%M:%S"),
            "elapsed_s":    f"{elapsed:.0f}",
            "event_type":   event_type,
            "label_vi":     label_vi,
            "state":        state,
            "dist_display": dist_d,
            "conf":         f"{int(conf*100)}%" if conf is not None else "",
            "note":         note,
        }
        # Rolling buffer: giữ tối đa MAX_EVENTS
        self._events.append(row)
        if len(self._events) > self.MAX_EVENTS:
            # Xóa 20% cũ nhất
            cut = self.MAX_EVENTS // 5
            self._events = self._events[cut:]

        if self._writer:
            self._writer.writerow(row)
            self._csv_file.flush()

    def _count(self, key: str):
        """Tăng bộ đếm sự kiện (dùng cho summary mode)."""
        self._counts[key] = self._counts.get(key, 0) + 1

    # ── API công khai ─────────────────────────────────────────────────────────

    def log_obstacle(self, label: str, state: str, dist_m: float, conf: float):
        """Ghi vật cản. Chỉ ghi near/critical; có cooldown 10s per-label."""
        if state == "far":
            return
        vi = CLASS_VI.get(label, label)
        self._count(f"obstacle_{label}_{state}")
        if self.level == "detail":
            self._write("obstacle", vi, state, dist_m, conf)
        elif self.level == "normal":
            key = f"obs_{label}_{state}"
            if self._allowed(key):
                self._write("obstacle", vi, state, dist_m, conf)

    def log_face(self, name: str, dist_m: float, conf: float = 0.0):
        """Ghi nhận ra người quen (cooldown 20s per-name)."""
        self._count(f"face_{name}")
        if self.level != "summary":
            if self._allowed(f"face_{name}", 20.0):
                self._write("face", name, "known", dist_m, conf)

    def log_money(self, label_raw: str, conf: float):
        """Ghi nhận tiền (cooldown 5s per-mệnh giá)."""
        norm = normalize_money_label(label_raw)
        vi   = MONEY_VI.get(norm, label_raw)
        self._count(f"money_{norm}")
        if self.level != "summary":
            if self._allowed(f"money_{norm}", 5.0):
                self._write("money", vi, conf=conf)

    def log_traffic(self, color: str):
        """Ghi đèn giao thông (cooldown 5s per-màu)."""
        vi = TRAFFIC_VI.get(color, color)
        self._count(f"traffic_{color}")
        if self.level != "summary":
            if self._allowed(f"traffic_{color}", 5.0):
                self._write("traffic", vi)

    def log_ocr(self, text: str):
        """Ghi OCR — luôn ghi (sự kiện hiếm, quan trọng)."""
        self._count("ocr")
        if self.level != "summary":
            self._write("ocr", "Văn bản", note=text[:120])

    def log_nav(self, destination: str, stage: str, dist_m: float):
        """Ghi điều hướng (cooldown 8s per-stage)."""
        self._count(f"nav_{destination}")
        if self.level != "summary":
            if self._allowed(f"nav_{stage}", 8.0):
                self._write("nav", destination, stage, dist_m)

    def log_system(self, msg: str):
        """Ghi sự kiện hệ thống — luôn ghi (không cooldown)."""
        self._count("system")
        if self.level != "summary":
            self._write("system", note=msg)

    # ── Tóm tắt & đóng ───────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Trả về tóm tắt phiên hiện tại (có thể gọi bất cứ lúc nào)."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        # Gộp counts thành readable dict
        obstacle_totals: dict[str, int] = {}
        for k, v in self._counts.items():
            if k.startswith("obstacle_"):
                parts = k.split("_")
                label = parts[1]
                vi    = CLASS_VI.get(label, label)
                obstacle_totals[vi] = obstacle_totals.get(vi, 0) + v

        face_totals = {k.split("_", 1)[1]: v
                       for k, v in self._counts.items() if k.startswith("face_")}
        money_totals = {k.split("_", 1)[1]: v
                        for k, v in self._counts.items() if k.startswith("money_")}

        return {
            "level":          self.level,
            "duration_s":     round(elapsed, 0),
            "total_detected": sum(self._counts.values()),
            "obstacles":      dict(sorted(obstacle_totals.items(),
                                          key=lambda x: -x[1])[:8]),
            "faces":          face_totals,
            "money":          money_totals,
            "traffic":        {k.split("_",1)[1]: v
                               for k, v in self._counts.items()
                               if k.startswith("traffic_")},
            "ocr_count":      self._counts.get("ocr", 0),
            "events_in_ram":  len(self._events),
            "csv_path":       self._csv_path,
        }

    def close(self) -> dict:
        """Đóng logger, ghi summary JSON, trả về dict tóm tắt."""
        if not self.enabled:
            return {}

        summary = self.get_summary()
        summary["session_start"] = self._start_time.isoformat()

        # Luôn ghi JSON summary (dù level = summary)
        try:
            data: dict = {"summary": summary}
            if self.level == "detail":
                data["events"] = self._events   # Ghi full events chỉ ở detail
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Logger] ✓ JSON → {self._json_path}")
        except Exception as e:
            print(f"[Logger] Lỗi ghi JSON: {e}")

        if self._csv_file:
            self._csv_file.close()
            print(f"[Logger] ✓ CSV  → {self._csv_path}")

        print(f"[Logger] Tóm tắt phiên: "
              f"thời gian={summary['duration_s']:.0f}s  "
              f"phát hiện={summary['total_detected']} lần  "
              f"level={self.level}")
        return summary

    def set_level(self, level: str):
        """Thay đổi mức độ ghi log ngay trong khi chạy."""
        assert level in ("summary", "normal", "detail")
        old = self.level
        self.level = level
        print(f"[Logger] Level: {old} → {level}")
        # Mở CSV nếu chuyển từ summary sang normal/detail
        if old == "summary" and level != "summary" and not self._csv_file:
            try:
                self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8-sig")
                self._writer   = csv.DictWriter(self._csv_file, fieldnames=[
                    "timestamp", "elapsed_s", "event_type",
                    "label_vi", "state", "dist_display", "conf", "note",
                ])
                self._writer.writeheader()
            except Exception as e:
                print(f"[Logger] Không mở được CSV: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 6: HÀM TIỆN ÍCH KHỞI ĐỘNG
# ══════════════════════════════════════════════════════════════════════════════

def build_preload_texts(extra_names: list[str] | None = None) -> list[str]:
    """
    Tạo danh sách câu TTS cần pre-cache khi khởi động.
    extra_names: tên người quen từ FaceRecognizer để pre-cache luôn.
    """
    texts = [
        "Hệ thống hỗ trợ người khiếm thị đã sẵn sàng",
        "Chế độ nhận diện tiền đã bật",
        "Chế độ nhận diện tiền đã tắt",
        "Đang chụp và đọc văn bản",
        "Không tìm thấy văn bản",
        "Chế độ đèn giao thông đã bật",
        "Vui lòng nhập tên điểm đến vào terminal",
        "Điểm đến không tìm thấy, vui lòng thử lại",
        "Đã tắt chế độ dẫn đường",
        "Đã cập nhật danh sách khuôn mặt",
        "Nhận diện khuôn mặt chưa sẵn sàng",
        "Văn bản đọc được:",
        "Tốt lắm!",
        "Bạn đang tiến gần.",
        "Gần đến rồi!",
        "Thêm vài bước nữa là đến.",
        "Hãy nhìn xung quanh để tìm",
        "Hãy xoay người từ từ sang trái hoặc phải để quét xung quanh.",
        "Dừng lại!",
        "Chú ý!",
    ]
    for vi in CLASS_VI.values():
        texts.append(f"Cảnh báo, phía trước có {vi}")
        texts.append(f"Nguy hiểm! Phía trước có {vi}")
        texts.append(f"Hãy đi về phía {vi}")
        texts.append(f"Tiếp tục đi thẳng. {vi} cách")
    for vi_name in MONEY_VI.values():
        texts.append(f"Đây là tờ {vi_name}")
    texts += list(TRAFFIC_VI.values())
    texts += [
        "Calibration thành công",
        "Nhập khoảng cách thực tế bằng xăng ti mét vào terminal",
        "Calibration hoàn tất, hệ thống đã cập nhật",
    ]
    for room, info in ROOM_MAP.items():
        lm_vi = CLASS_VI.get(info["landmark"], info["landmark"])
        texts.append(info["direction_hint"])
        texts.append(f"Bắt đầu dẫn đường đến {room}. {info['direction_hint']}. Cột mốc cần tìm là {lm_vi}.")
        texts.append(f"Bạn đã đến {room}! Điểm đến ngay trước mặt bạn.")
        texts.append(f"Không thấy {lm_vi} trong khung hình. Hãy nhìn xung quanh để tìm {lm_vi}.")
    if extra_names:
        for name in extra_names:
            texts.append(f"Phát hiện {name}")
            texts.append(f"Xin chào {name}")
    return texts