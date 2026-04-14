import cv2
import threading
import queue
import time
import os
import hashlib
import tempfile
import numpy as np
import pygame
from gtts import gTTS
from ultralytics import YOLO

try:
    from PIL import Image as _PILI, ImageDraw as _PILD, ImageFont as _PILF
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[OCR] ⚠  easyocr chưa cài. Chạy: pip install easyocr")


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 1: HẰNG SỐ CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════

# ── Nhóm class YOLO theo mức độ ưu tiên ──────────────────────────────────────
TARGET_CLASSES_HIGH = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "traffic light", "stop sign",
}
TARGET_CLASSES_HOME = {
    "chair", "couch", "bed", "dining table", "toilet", "sink",
    "refrigerator", "microwave", "oven", "toaster", "tv",
    "laptop", "cell phone", "book", "clock", "vase",
    "bottle", "cup", "bowl", "knife", "fork", "spoon",
    "potted plant", "backpack", "handbag", "suitcase",
    "umbrella", "bench", "door", "stairs",
    "keyboard", "mouse", "remote",
}
TARGET_CLASSES = TARGET_CLASSES_HIGH | TARGET_CLASSES_HOME

# ── Tên tiếng Việt ────────────────────────────────────────────────────────────
CLASS_VI = {
    "person":        "người lạ",
    "car":           "ô tô",
    "motorcycle":    "xe máy",
    "bicycle":       "xe đạp",
    "bus":           "xe buýt",
    "truck":         "xe tải",
    "traffic light": "đèn giao thông",
    "stop sign":     "biển dừng",
    "chair":         "ghế",
    "couch":         "ghế sofa",
    "bed":           "giường",
    "dining table":  "bàn ăn",
    "toilet":        "bồn cầu",
    "sink":          "bồn rửa tay",
    "door":          "cửa ra vào",
    "stairs":        "cầu thang",
    "bench":         "ghế dài",
    "refrigerator":  "tủ lạnh",
    "microwave":     "lò vi sóng",
    "oven":          "lò nướng",
    "toaster":       "máy nướng bánh",
    "tv":            "ti vi",
    "laptop":        "máy tính xách tay",
    "cell phone":    "điện thoại",
    "keyboard":      "bàn phím",
    "mouse":         "chuột máy tính",
    "remote":        "điều khiển từ xa",
    "bottle":        "chai",
    "cup":           "cốc",
    "bowl":          "bát",
    "knife":         "dao",
    "fork":          "nĩa",
    "spoon":         "muỗng",
    "book":          "quyển sách",
    "clock":         "đồng hồ",
    "vase":          "lọ hoa",
    "potted plant":  "chậu cây",
    "backpack":      "ba lô",
    "handbag":       "túi xách",
    "suitcase":      "vali",
    "umbrella":      "ô dù",
}

# ── Màu bounding box theo BGR ─────────────────────────────────────────────────
CLASS_COLORS = {
    "person":        (0,   220,   0),
    "car":           (0,     0, 220),
    "motorcycle":    (220,   0, 220),
    "bicycle":       (255, 140,   0),
    "bus":           (0,     0, 180),
    "truck":         (0,    30, 200),
    "stairs":        (0,     0, 255),
    "chair":         (0,   165, 255),
    "couch":         (0,   200, 255),
    "bed":           (0,   255, 200),
    "dining table":  (50,  200, 200),
    "door":          (200, 100,   0),
    "refrigerator":  (180, 180,   0),
    "default":       (120, 120, 120),
    "traffic light": (255, 255,   0),
}

# ── Chiều cao thực tế (mét) — dùng tính khoảng cách ─────────────────────────
REAL_HEIGHT_M = {
    "person":        1.70,
    "car":           1.50,
    "motorcycle":    1.10,
    "bicycle":       1.00,
    "bus":           3.00,
    "truck":         2.50,
    "chair":         0.90,
    "couch":         0.85,
    "bed":           0.55,
    "dining table":  0.75,
    "toilet":        0.70,
    "sink":          0.50,
    "refrigerator":  1.70,
    "tv":            0.60,
    "laptop":        0.25,
    "door":          2.00,
    "stairs":        0.20,
    "bottle":        0.25,
    "cup":           0.10,
    "backpack":      0.45,
    "suitcase":      0.60,
    "default":       0.80,
}

# ── Module khoảng cách ────────────────────────────────────────────────────────
FOCAL_LENGTH    = 615.0   # px — webcam 640×480 thông thường
WARN_DIST_M     = 1.5     # mét — ngưỡng cảnh báo GẦN
CRITICAL_DIST_M = 0.8     # mét — ngưỡng nguy hiểm khẩn

# ── Frame skipping ────────────────────────────────────────────────────────────
YOLO_SKIP        = 3
MONEY_SKIP       = 5
TRAFFIC_SKIP     = 2
FACE_SKIP        = 6      # Chạy face recognition mỗi N frame
FACE_ANNOUNCE_CD = 8.0    # Giây cooldown giữa 2 lần thông báo cùng 1 người

# ── Paths & model files ───────────────────────────────────────────────────────
FACE_ENCODINGS   = "Model/encodings.pickle"
MONEY_MODEL_PATH = "Model/money_v8n.pt"

# ── YOLO inference ────────────────────────────────────────────────────────────
YOLO_CONF  = 0.40
YOLO_IOU   = 0.50
MONEY_CONF = 0.55

# ── TTS cooldown (giây) ───────────────────────────────────────────────────────
ANNOUNCE_COOLDOWN = 4.0
MONEY_COOLDOWN    = 3.0
TRAFFIC_COOLDOWN  = 2.5
NAV_COOLDOWN      = 3.0
TTS_CACHE_DIR     = os.path.join(tempfile.gettempdir(), "blind_tts_cache")

# ── Tiền Việt Nam ─────────────────────────────────────────────────────────────
MONEY_VI = {
    "500":    "năm trăm đồng",
    "1000":   "một nghìn đồng",
    "2000":   "hai nghìn đồng",
    "5000":   "năm nghìn đồng",
    "10000":  "mười nghìn đồng",
    "20000":  "hai mươi nghìn đồng",
    "50000":  "năm mươi nghìn đồng",
    "100000": "một trăm nghìn đồng",
    "200000": "hai trăm nghìn đồng",
    "500000": "năm trăm nghìn đồng",
}

# ── Đèn giao thông ────────────────────────────────────────────────────────────
TRAFFIC_HSV = {
    "red":    [(  0,  80, 80, 10, 255, 255),
               (160,  80, 80, 180, 255, 255)],
    "green":  [(40,   60, 60,  85, 255, 255)],
    "yellow": [(20,   80, 80,  35, 255, 255)],
}
TRAFFIC_COLORS_BGR = {
    "red":    (0,   0,  220),
    "green":  (0, 200,    0),
    "yellow": (0, 200,  220),
    "unknown":(80,  80,  80),
}
TRAFFIC_VI = {
    "red":    "Đèn đỏ, dừng lại",
    "green":  "Đèn xanh, được đi",
    "yellow": "Đèn vàng, chú ý",
    "unknown":"Phía trước có đèn giao thông",
}

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["vi", "en"]
OCR_MIN_CONF  = 0.50
OCR_MAX_CHARS = 120

# ── Bản đồ phòng (Navigation) ─────────────────────────────────────────────────
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
    "bếp":    "phòng bếp",
    "wc":     "nhà vệ sinh",
    "toilet": "nhà vệ sinh",
    "ngủ":    "phòng ngủ",
    "khách":  "phòng khách",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 2: CORE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

# ── MODULE 2: BackgroundSpeaker ───────────────────────────────────────────────

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


# ── MODULE 2b: AnnouncementManager ───────────────────────────────────────────

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
        """Thông báo vật cản theo mức độ ưu tiên critical > near."""
        critical_items = []
        near_items     = []
        for det in detections:
            state   = det.get("state", "far")
            lbl     = det["label"]
            dist    = det.get("dist_m", 99)
            vi      = CLASS_VI.get(lbl, lbl)
            dist_vi = DistanceEstimator.dist_text_vi(dist)
            if state == "critical":
                critical_items.append((vi, dist_vi, lbl))
            elif state == "near":
                near_items.append((vi, dist_vi, lbl))

        # Chỉ đọc 1 vật nguy hiểm nhất mỗi lần
        for vi, dist_vi, lbl in critical_items:
            if self._can_announce(f"crit_{lbl}", 3.0):
                msg = f"Nguy hiểm! Phía trước có {vi} cách {dist_vi}!"
                self._speaker.say(msg, priority=True)
                print(f"[Warn‼] {msg}")
                return

        to_say = []
        for vi, dist_vi, lbl in near_items:
            if self._can_announce(lbl, ANNOUNCE_COOLDOWN):
                to_say.append(f"{vi} cách {dist_vi}")
        if to_say:
            msg = f"Cảnh báo, phía trước có {', '.join(to_say)}"
            self._speaker.say(msg)
            print(f"[Warn] {msg}")

    def process_money(self, money_label: str):
        if self._can_announce(f"money_{money_label}", MONEY_COOLDOWN):
            vi_name = MONEY_VI.get(money_label, f"tờ {money_label}")
            msg = f"Đây là tờ {vi_name}"
            print(f"[Money] → {msg}")
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


# ── MODULE 3: DistanceEstimator ───────────────────────────────────────────────

class DistanceEstimator:
    """
    Ước lượng khoảng cách thực tế bằng công thức quang học:
        D = (H_real * focal_length) / h_bbox
    H_real: chiều cao thực tế vật (mét), focal_length: tiêu cự camera (px).
    """

    @staticmethod
    def estimate_meters(cls_name: str, y1: int, y2: int) -> float:
        h_bbox = max(y2 - y1, 1)
        h_real = REAL_HEIGHT_M.get(cls_name, REAL_HEIGHT_M["default"])
        dist   = (h_real * FOCAL_LENGTH) / h_bbox
        return round(min(max(dist, 0.1), 30.0), 1)

    @staticmethod
    def classify(cls_name: str, y1: int, y2: int) -> tuple[str, float]:
        """Trả về (state, distance_m). state: 'critical' | 'near' | 'far'."""
        d = DistanceEstimator.estimate_meters(cls_name, y1, y2)
        if d <= CRITICAL_DIST_M:
            return "critical", d
        elif d <= WARN_DIST_M:
            return "near", d
        else:
            return "far", d

    @staticmethod
    def box_color(state: str) -> tuple[int, int, int]:
        if state == "critical":
            return (0, 0, 255)
        if state == "near":
            return (0, 100, 255)
        return (100, 100, 100)

    @staticmethod
    def state_label(state: str, dist_m: float) -> str:
        if state == "critical":
            return f"!! {dist_m:.1f}m !!"
        elif state == "near":
            return f"GẦN {dist_m:.1f}m"
        else:
            return f"{dist_m:.1f}m"

    @staticmethod
    def dist_text_vi(dist_m: float) -> str:
        """Chuỗi khoảng cách tự nhiên bằng tiếng Việt để đọc TTS."""
        if dist_m < 1.0:
            return f"{int(dist_m * 100)} xăng ti mét"
        else:
            return f"{dist_m:.1f} mét".replace(".", " phẩy ")


# ── MODULE 8: NavigationGuide (Nâng cấp — Thông minh hơn) ───────────────────

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


# ── MODULE 4: MoneyDetector ───────────────────────────────────────────────────

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


# ── MODULE 5: OCRReader (Nâng cấp — Tiền xử lý ảnh + Smart TTS) ─────────────

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


# ── MODULE 6: TrafficLightAnalyzer ───────────────────────────────────────────

class TrafficLightAnalyzer:
    """
    Phân tích màu đèn giao thông từ crop BGR bằng phân tích HSV.
    Không cần train thêm — dùng YOLO để detect bbox rồi phân tích màu crop.
    """

    @staticmethod
    def _count_color(hsv_crop: np.ndarray, ranges: list[tuple]) -> int:
        total = 0
        for r in ranges:
            lo = np.array([r[0], r[1], r[2]])
            hi = np.array([r[3], r[4], r[5]])
            total += int(cv2.countNonZero(cv2.inRange(hsv_crop, lo, hi)))
        return total

    @staticmethod
    def analyze(crop_bgr: np.ndarray) -> str:
        """Trả về 'red' | 'green' | 'yellow' | 'unknown'."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown"
        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return "unknown"
        small  = cv2.resize(crop_bgr, (32, 64))
        hsv    = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsv    = cv2.GaussianBlur(hsv, (3, 3), 0)
        counts = {c: TrafficLightAnalyzer._count_color(hsv, r)
                  for c, r in TRAFFIC_HSV.items()}
        best   = max(counts, key=counts.get)
        if sum(counts.values()) == 0 or counts[best] / (32 * 64) < 0.05:
            return "unknown"
        return best


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
    """Vẽ bounding box tiền với tên mệnh giá tiếng Việt."""
    color   = (0, 215, 255)
    vi_name = MONEY_VI.get(label, label)
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
#  PHẦN 5: HÀM TIỆN ÍCH KHỞI ĐỘNG
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