import os
import cv2
import pickle
import threading
import numpy as np
import face_recognition


# ══════════════════════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════

# Ngưỡng so sánh: càng nhỏ → càng khắt khe (ít nhầm hơn, có thể bỏ sót)
# Mặc định của face_recognition là 0.6
# Khuyến nghị: 0.45 ~ 0.55 cho môi trường trong nhà
TOLERANCE = 0.50

# Phương pháp detect mặt trong crop realtime
# "hog" = nhanh, đủ dùng cho CPU | "cnn" = chính xác hơn, cần GPU
DETECT_MODEL = "hog"

# Kích thước tối thiểu của crop để thử nhận diện (tránh crop quá nhỏ, mờ)
MIN_CROP_SIZE = 60   # pixel


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

class FaceRecognizer:   
    def __init__(self, encodings_path: str = "encodings.pickle"):
        """
        Tham số:
            encodings_path — đường dẫn file .pickle tạo bởi encode_faces.py
        """
        self._encodings_path = encodings_path
        self._names:     list[str]        = []
        self._encodings: list[np.ndarray] = []
        self._ready      = False
        self._lock       = threading.Lock()   # Thread-safe nếu dùng background thread

        self._load_encodings()

    # ── Load pickle ────────────────────────────────────────────────────────────

    def _load_encodings(self):
        """Load file encodings.pickle vào bộ nhớ."""
        if not os.path.exists(self._encodings_path):
            print(f"[FaceRec] ⚠  Không tìm thấy: {self._encodings_path}")
            print(f"[FaceRec]    Hãy chạy encode_faces.py trước!")
            return

        try:
            with open(self._encodings_path, "rb") as f:
                data = pickle.load(f)

            self._names     = data["names"]
            self._encodings = data["encodings"]
            self._ready     = True

            unique = list(dict.fromkeys(self._names))   # Giữ thứ tự, bỏ trùng
            print(f"[FaceRec] ✓  Loaded {len(self._encodings)} encodings | "
                  f"{len(unique)} người: {', '.join(unique)}")

        except Exception as e:
            print(f"[FaceRec] ✗  Lỗi load {self._encodings_path}: {e}")

    # ── Property ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True nếu đã load encodings thành công và sẵn sàng nhận diện."""
        return self._ready

    @property
    def known_names(self) -> list[str]:
        """Danh sách tên người đã được encode (không trùng)."""
        return list(dict.fromkeys(self._names))

    # ── Tiền xử lý crop từ webcam ─────────────────────────────────────────────

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Chuẩn bị crop BGR từ webcam để đưa vào face_recognition:
          1. Kiểm tra kích thước tối thiểu
          2. Resize nếu quá nhỏ (dưới 120px) — giúp detect tốt hơn
          3. CLAHE trên kênh L (cải thiện ánh sáng kém, ngược sáng)
          4. Chuyển BGR → RGB uint8 C-contiguous

        Trả về RGB numpy uint8 hoặc None nếu crop không hợp lệ.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return None   # Quá nhỏ, bỏ qua

        # Nếu crop nhỏ hơn 120px → upsample để detect dễ hơn
        if h < 120:
            scale    = 120 / h
            new_w    = max(1, int(w * scale))
            new_h    = 120
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)

        # CLAHE — cải thiện độ tương phản, giúp nhận diện trong điều kiện tối
        try:
            lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            crop_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass   # Nếu CLAHE lỗi, tiếp tục với ảnh gốc

        # BGR → RGB, đảm bảo C-contiguous uint8
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # ── Nhận diện chính ────────────────────────────────────────────────────────

    def identify(self, crop_bgr: np.ndarray) -> str:
        """
        Nhận diện khuôn mặt trong crop BGR (lấy từ YOLO bbox).

        Tham số:
            crop_bgr — ảnh BGR numpy, cắt từ frame webcam tại vị trí YOLO bbox
        """
        if not self._ready:
            return "Unknown"

        # Tiền xử lý
        rgb = self._preprocess(crop_bgr)
        if rgb is None:
            return "Unknown"

        with self._lock:
            # Guard cuối: đảm bảo 100% uint8 RGB C-contiguous trước khi vào dlib
            if (rgb.dtype != np.uint8
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                    or not rgb.flags['C_CONTIGUOUS']):
                rgb = np.array(rgb, dtype=np.uint8, order='C')

            # Bước 1: Detect vị trí khuôn mặt trong crop
            try:
                boxes = face_recognition.face_locations(rgb, model=DETECT_MODEL)
            except RuntimeError:
                return "Unknown"
            if not boxes:
                return "Unknown"

            # Bước 2: Encode khuôn mặt
            encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
            if not encodings:
                return "Unknown"

            # Bước 3: So sánh + Voting (theo thuật toán của huytranvan2010)
            # Mỗi encoding trong crop vote cho 1 tên
            counts: dict[str, int] = {}

            for encoding in encodings:
                # compare_faces trả về list[bool] — True = khớp với encoding đó
                matches = face_recognition.compare_faces(
                    self._encodings, encoding, tolerance=TOLERANCE)

                # Đếm vote từng người
                for match, name in zip(matches, self._names):
                    if match:
                        counts[name] = counts.get(name, 0) + 1

            if not counts:
                return "Unknown"

            # Người có nhiều vote nhất = kết quả
            winner = max(counts, key=counts.get)
            return winner

    # ── Reload encodings (không cần restart) ─────────────────────────────────

    def reload(self):
        """
        Load lại file encodings.pickle mà không cần khởi động lại chương trình.
        Hữu ích khi bạn thêm người mới vào dataset và chạy encode_faces.py lại.
        """
        print("[FaceRec] Đang reload encodings...")
        self._names     = []
        self._encodings = []
        self._ready     = False
        self._load_encodings()

    # ── Thống kê ──────────────────────────────────────────────────────────────

    def stats(self) -> str:
        """Trả về chuỗi thống kê trạng thái hiện tại."""
        if not self._ready:
            return "[FaceRec] Chưa sẵn sàng (thiếu encodings.pickle)"
        unique = self.known_names
        return (f"[FaceRec] {len(self._encodings)} encodings | "
                f"{len(unique)} người: {', '.join(unique)} | "
                f"tolerance={TOLERANCE}")
# ══════════════════════════════════════════════════════════════════════════════
#  CHẠY ĐỘC LẬP ĐỂ TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 62)
    print("  FACE MODULE — Kiểm tra nhận diện từ ảnh")
    print("=" * 62)

    rec = FaceRecognizer("encodings.pickle")

    if not rec.is_ready:
        print("\n⚠  Chạy encode_faces.py trước để tạo encodings.pickle!")
        sys.exit(1)

    print(rec.stats())

    # Test với ảnh từ command line nếu có
    if len(sys.argv) > 1:
        for img_path in sys.argv[1:]:
            if not os.path.exists(img_path):
                print(f"✗ Không tìm thấy: {img_path}")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"✗ Không đọc được: {img_path}")
                continue

            name = rec.identify(img_bgr)
            print(f"  {os.path.basename(img_path)} → {name}")
    else:
        # Test realtime qua webcam
        print("\nMở webcam để test realtime (nhấn Q để thoát)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Không mở được webcam")
            sys.exit(1)

        frame_count = 0
        last_name   = "Unknown"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Nhận diện mỗi 6 frame
            if frame_count % 6 == 0:
                # Dùng toàn bộ frame để test (trong main.py sẽ dùng crop)
                last_name = rec.identify(frame)

            # Hiển thị kết quả
            color = (0, 220, 0) if last_name != "Unknown" else (80, 80, 80)
            cv2.putText(frame, f"Name: {last_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Tolerance: {TOLERANCE}  Model: {DETECT_MODEL}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
            cv2.putText(frame, "Q = Thoat",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

            cv2.imshow("Face Module Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()