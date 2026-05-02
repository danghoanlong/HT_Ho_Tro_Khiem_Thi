"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BLIND ASSISTANT AI — gui_app.py  (Giao diện đồ họa)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Giao diện CustomTkinter — bảng điều khiển đầy đủ cho người dùng.         ║
║  Chạy: python gui_app.py                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHÍM TẮT (khi cửa sổ GUI đang focus)                                      ║
║  Q → Thoát     M → Tiền       O → OCR văn bản    T → Đèn giao thông       ║
║  N → Điều hướng  F → Reload mặt  W → Đọc giờ                              ║
║  C → Calibrate khoảng cách       R → Reset calib                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Thư viện chuẩn ────────────────────────────────────────────────────────────
import cv2
import os
import threading
import time
import tkinter as tk
import tkinter.filedialog as filedialog

# ── Thư viện bên thứ ba ───────────────────────────────────────────────────────
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# ── Import từ core_engine ─────────────────────────────────────────────────────
from core_engine import (
    # §1 Hằng số
    TARGET_CLASSES, CLASS_VI, MONEY_VI, TRAFFIC_VI, TRAFFIC_COLORS_BGR,
    ROOM_MAP, ROOM_ALIASES,
    YOLO_CONF, YOLO_IOU, MONEY_MODEL_PATH, FACE_ENCODINGS,
    WARN_DIST_M, CRITICAL_DIST_M,
    YOLO_SKIP, MONEY_SKIP, FACE_SKIP, FACE_ANNOUNCE_CD,
    ANNOUNCE_COOLDOWN, MONEY_COOLDOWN, TRAFFIC_COOLDOWN, NAV_COOLDOWN,
    OCR_MAX_CHARS,
    # §3–§12 Classes AI
    BackgroundSpeaker, AnnouncementManager,
    DistanceEstimator, NavigationGuide,
    MoneyDetector, OCRReader, TrafficLightAnalyzer,
    SessionLogger, TimeReader,
    # §13 Hàm vẽ
    draw_box_obstacle, draw_box_money, draw_box_traffic, draw_nav_overlay,
    put_vi_text, vi_text_size,
    # §14 Tiện ích
    build_preload_texts, normalize_money_label,
    DistanceCalibrator,
)

# ── FaceRecognizer (tùy chọn) ─────────────────────────────────────────────────
try:
    from face_module import FaceRecognizer
    _FACE_MODULE_OK = True
except ImportError:
    _FACE_MODULE_OK = False
    print("[Face] ⚠  face_module.py không tìm thấy.")


# ══════════════════════════════════════════════════════════════════════════════
#  THEME & MÀU SẮC
# ══════════════════════════════════════════════════════════════════════════════

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

C_GREEN   = "#2ecc71"
C_RED     = "#e74c3c"
C_ORANGE  = "#e67e22"
C_BLUE    = "#3498db"
C_PURPLE  = "#9b59b6"
C_YELLOW  = "#f1c40f"
C_GRAY    = "#636e72"
C_CARD    = "#1e2126"
C_PANEL   = "#16191e"


# ══════════════════════════════════════════════════════════════════════════════
#  DUMMY FACE RECOGNIZER
# ══════════════════════════════════════════════════════════════════════════════

class _DummyFaceRec:
    is_ready    = False
    known_names: list[str] = []
    def identify(self, _): return "Unknown"
    def reload(self): pass
    def stats(self): return "[Face] Chưa sẵn sàng"


# ══════════════════════════════════════════════════════════════════════════════
#  WIDGET TIỆN ÍCH: CARD SECTION
# ══════════════════════════════════════════════════════════════════════════════

class _CardSection(ctk.CTkFrame):
    """Frame card với tiêu đề section cho panel điều khiển."""
    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, fg_color=C_CARD, corner_radius=10, **kw)
        self.pack(padx=12, pady=5, fill="x")
        ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_GRAY
        ).pack(anchor="w", padx=14, pady=(8, 4))


# ══════════════════════════════════════════════════════════════════════════════
#  APP CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

class BlindAssistantGUI(ctk.CTk):
    """
    Giao diện đồ họa đầy đủ cho hệ thống hỗ trợ người khiếm thị.
    Kết nối trực tiếp với các class trong core_engine.py.
    """

    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG HỖ TRỢ NGƯỜI KHIẾM THỊ  v5.2  —  GUI")
        self.geometry("1280x800")
        self.minsize(960, 620)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        # Phím tắt toàn cục: W = đọc giờ
        self.bind("<KeyPress-w>", lambda e: self._announce_time())
        self.bind("<KeyPress-W>", lambda e: self._announce_time())

        # ── Biến trạng thái các công tắc ─────────────────────────────────────
        self.mode_money   = tk.BooleanVar(value=False)
        self.mode_traffic = tk.BooleanVar(value=True)
        self.mode_face    = tk.BooleanVar(value=True)
        self.mode_audio   = tk.BooleanVar(value=True)

        # ── Camera ────────────────────────────────────────────────────────────
        self.camera_idx    = 0
        self.cap           = None
        self._cam_running  = False
        self.current_frame = None   # Frame BGR gốc — dùng cho OCR và test

        # ── Runtime counters ──────────────────────────────────────────────────
        self.frame_count = 0
        self._fps_t0     = time.perf_counter()
        self._fps_n      = 0
        self.fps_disp    = 0.0

        # ── Cache nhận diện mặt ───────────────────────────────────────────────
        # key   = (x1//50, y1//50)       — grid thô, ít nhạy với chuyển động nhỏ
        # value = (name, expire_frame, vote_buf)
        self.face_cache: dict[tuple, tuple] = {}
        self.ocr_scanning = False

        # ── Lưu bbox gần nhất để calibrate khoảng cách [C] ──────────────────
        # Per-class bbox cache cho calibration: {cls_name: {cls,x1,y1,x2,y2,dist_m}}
        self._last_bbox_for_calib: dict[str, dict] = {}

        # ── AI modules (khởi tạo trong background thread) ─────────────────────
        self.speaker        = None
        self.announcer      = None
        self.nav            = None
        self.obstacle_model = None
        self.money_det      = None
        self.ocr            = None
        self.traffic_an     = None
        self.face_rec       = _DummyFaceRec()
        self.logger         = None   # SessionLogger — khởi tạo trong _init_ai
        self.time_reader    = None   # TimeReader — khởi tạo sau khi speaker sẵn sàng
        self._ai_ready      = False

        # ── Xây giao diện ────────────────────────────────────────────────────
        self._build_ui()

        # ── Load AI trong nền ─────────────────────────────────────────────────
        threading.Thread(target=self._load_ai, daemon=True, name="LoadAI").start()

    # ══════════════════════════════════════════════════════════════════════════
    #  XÂY DỰNG GIAO DIỆN
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=360)
        self.grid_rowconfigure(0, weight=1)
        self._build_camera_panel()
        self._build_control_panel()

    # ── Panel trái: camera + status bar ───────────────────────────────────────
    def _build_camera_panel(self):
        left = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=0)
        left.grid_columnconfigure(0, weight=1)

        # Khung hiển thị camera
        self.cam_frame = ctk.CTkFrame(left, fg_color="black", corner_radius=14)
        self.cam_frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="nsew")
        self.cam_label = ctk.CTkLabel(
            self.cam_frame,
            text="📷  Đang kết nối camera...",
            font=ctk.CTkFont(size=16), text_color=C_GRAY
        )
        self.cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # Thanh trạng thái dưới camera
        sbar = ctk.CTkFrame(left, fg_color=C_CARD, corner_radius=10, height=50)
        sbar.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="ew")
        sbar.grid_propagate(False)
        for i in range(5):
            sbar.grid_columnconfigure(i, weight=1)

        self.lbl_fps     = ctk.CTkLabel(sbar, text="FPS: --",
                                         font=ctk.CTkFont(size=12, weight="bold"),
                                         text_color=C_GREEN)
        self.lbl_warn    = ctk.CTkLabel(sbar, text=f"⚠ <{WARN_DIST_M}m",
                                         font=ctk.CTkFont(size=11), text_color=C_ORANGE)
        self.lbl_crit    = ctk.CTkLabel(sbar, text=f"🚨 <{CRITICAL_DIST_M}m",
                                         font=ctk.CTkFont(size=11), text_color=C_RED)
        self.lbl_spk     = ctk.CTkLabel(sbar, text="🔊 Chờ...",
                                         font=ctk.CTkFont(size=11), text_color=C_GRAY)
        self.lbl_nav_bar = ctk.CTkLabel(sbar, text="Nav: OFF",
                                         font=ctk.CTkFont(size=11), text_color=C_GRAY)
        self.lbl_fps.grid    (row=0, column=0, padx=8,  pady=12, sticky="w")
        self.lbl_warn.grid   (row=0, column=1, padx=4,  pady=12)
        self.lbl_crit.grid   (row=0, column=2, padx=4,  pady=12)
        self.lbl_spk.grid    (row=0, column=3, padx=4,  pady=12)
        self.lbl_nav_bar.grid(row=0, column=4, padx=8,  pady=12, sticky="e")

    # ── Panel phải: điều khiển (scrollable) ───────────────────────────────────
    def _build_control_panel(self):
        self.ctrl = ctk.CTkScrollableFrame(
            self, width=340, fg_color=C_PANEL, corner_radius=0,
            label_text="  BẢNG ĐIỀU KHIỂN",
            label_font=ctk.CTkFont(size=14, weight="bold"),
            label_fg_color=C_CARD,
        )
        self.ctrl.grid(row=0, column=1, sticky="nsew")

        self._sec_switches()
        self._sec_ocr()
        self._sec_navigation()
        self._sec_face()
        self._sec_camera()
        self._sec_test()
        self._sec_time_history()   # ← Nút giờ + lịch sử
        self._sec_log()

    # ── Section: Công tắc chế độ ──────────────────────────────────────────────
    def _sec_switches(self):
        sec = _CardSection(self.ctrl, "⚙  CHẾ ĐỘ NHẬN DIỆN")

        ctk.CTkSwitch(sec, text="🔊  Bật Âm Thanh (gTTS + pygame)",
                      variable=self.mode_audio, progress_color=C_GREEN,
                      command=self._on_audio_toggle
                      ).pack(padx=16, pady=(4, 3), anchor="w")

        ctk.CTkSwitch(sec, text="👤  Nhận diện Người quen  [F]",
                      variable=self.mode_face, progress_color=C_YELLOW,
                      command=self._on_face_toggle
                      ).pack(padx=16, pady=3, anchor="w")

        ctk.CTkSwitch(sec, text="💰  Nhận diện Tiền tệ  [M]",
                      variable=self.mode_money, progress_color=C_ORANGE,
                      command=self._on_money_toggle
                      ).pack(padx=16, pady=3, anchor="w")

        ctk.CTkSwitch(sec, text="🚦  Đèn Giao thông  [T]",
                      variable=self.mode_traffic, progress_color=C_RED,
                      command=self._on_traffic_toggle
                      ).pack(padx=16, pady=(3, 10), anchor="w")

    # ── Section: OCR ──────────────────────────────────────────────────────────
    def _sec_ocr(self):
        sec = _CardSection(self.ctrl, "📖  ĐỌC VĂN BẢN (OCR)")

        self.btn_ocr = ctk.CTkButton(
            sec, text="📖  ĐỌC VĂN BẢN  [O]",
            fg_color=C_GREEN, hover_color="#27ae60",
            height=42, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._trigger_ocr
        )
        self.btn_ocr.pack(padx=16, pady=(4, 10), fill="x")

    # ── Section: Điều hướng ───────────────────────────────────────────────────
    def _sec_navigation(self):
        sec = _CardSection(self.ctrl, "🗺  CHỈ DẪN ĐƯỜNG ĐI")

        self.btn_nav = ctk.CTkButton(
            sec, text="📍  NHẬP ĐIỂM ĐẾN  [N]",
            fg_color=C_BLUE, hover_color="#2980b9",
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._ask_navigation
        )
        self.btn_nav.pack(padx=16, pady=(4, 4), fill="x")

        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.pack(padx=16, pady=(0, 4), fill="x")
        row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(row, text="🔁  Đọc lại  [H]",
                      fg_color="transparent", border_width=1, border_color=C_BLUE,
                      height=34, command=self._repeat_nav
                      ).grid(row=0, column=0, padx=(0, 3), sticky="ew")

        ctk.CTkButton(row, text="✖  Hủy điều hướng",
                      fg_color="transparent", border_width=1, border_color=C_RED,
                      text_color=C_RED, height=34, command=self._cancel_nav
                      ).grid(row=0, column=1, padx=(3, 0), sticky="ew")

        self.lbl_dest = ctk.CTkLabel(
            sec, text="Điểm đến: —",
            font=ctk.CTkFont(size=11), text_color=C_GRAY
        )
        self.lbl_dest.pack(pady=(2, 10))

    # ── Section: Khuôn mặt ────────────────────────────────────────────────────
    def _sec_face(self):
        sec = _CardSection(self.ctrl, "👤  NHẬN DIỆN KHUÔN MẶT")

        self.btn_reload_face = ctk.CTkButton(
            sec, text="🔄  Reload Khuôn mặt  [F]",
            fg_color=C_PURPLE, hover_color="#8e44ad", height=36,
            command=self._reload_faces
        )
        self.btn_reload_face.pack(padx=16, pady=(4, 4), fill="x")

        self.lbl_face_info = ctk.CTkLabel(
            sec, text="Trạng thái: đang khởi tạo...",
            font=ctk.CTkFont(size=11), text_color=C_GRAY
        )
        self.lbl_face_info.pack(pady=(2, 10))

    # ── Section: Camera ───────────────────────────────────────────────────────
    def _sec_camera(self):
        sec = _CardSection(self.ctrl, "📷  CAMERA")
        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.pack(padx=16, pady=(4, 12), fill="x")
        row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(row, text="Camera 0", height=34,
                      fg_color=C_BLUE, hover_color="#2980b9",
                      command=lambda: self._switch_camera(0)
                      ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(row, text="Camera 1", height=34,
                      fg_color=C_GRAY, hover_color="#4b5563",
                      command=lambda: self._switch_camera(1)
                      ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    # ── Section: Test ảnh ─────────────────────────────────────────────────────
    def _sec_test(self):
        sec = _CardSection(self.ctrl, "🧪  KIỂM THỬ & HIỆU CHỈNH")

        self.btn_test = ctk.CTkButton(
            sec, text="📂  Tải ảnh & Phân tích tổng hợp",
            fg_color=C_ORANGE, hover_color="#d35400",
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._upload_and_test
        )
        self.btn_test.pack(padx=16, pady=(4, 4), fill="x")

        # ── Calibration khoảng cách ──────────────────────────────────────────
        self.btn_calib = ctk.CTkButton(
            sec, text="📏  CALIBRATE KHOẢNG CÁCH  [C]",
            fg_color="#1a5276", hover_color="#21618c",
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._do_calibrate
        )
        self.btn_calib.pack(padx=16, pady=(0, 4), fill="x")

        self.lbl_calib_info = ctk.CTkLabel(
            sec, text="focal: đang load...",
            font=ctk.CTkFont(size=11), text_color=C_GRAY
        )
        self.lbl_calib_info.pack(pady=(0, 10))

    # ── Section: Giờ hiện tại + Xem lịch sử ─────────────────────────────────
    def _sec_time_history(self):
        sec = _CardSection(self.ctrl, "🕐  THỜI GIAN & LỊCH SỬ")

        # Nhãn đồng hồ
        self.lbl_clock = ctk.CTkLabel(
            sec,
            text=TimeReader.now_display(),
            font=ctk.CTkFont(family="Consolas", size=18, weight="bold"),
            text_color=C_YELLOW,
        )
        self.lbl_clock.pack(pady=(6, 4))

        # Nút đọc giờ
        ctk.CTkButton(
            sec,
            text="🕐  ĐỌC GIỜ HIỆN TẠI  [W]",
            fg_color="#1a5276", hover_color="#21618c",
            height=40, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._announce_time,
        ).pack(padx=16, pady=(2, 4), fill="x")

        # ── Log level selector ──────────────────────────────────────────────
        lv_frame = ctk.CTkFrame(sec, fg_color="transparent")
        lv_frame.pack(fill="x", padx=16, pady=(0, 4))

        ctk.CTkLabel(lv_frame, text="Cường độ ghi log:",
                     font=ctk.CTkFont(size=11), text_color=C_GRAY).pack(side="left")

        self._log_level_var = ctk.StringVar(value="normal")
        for lv, lbl, color in [
            ("summary", "Tóm tắt", "#636e72"),
            ("normal",  "Bình thường", "#27ae60"),
            ("detail",  "Chi tiết", "#e17055"),
        ]:
            ctk.CTkRadioButton(
                lv_frame, text=lbl, value=lv,
                variable=self._log_level_var,
                font=ctk.CTkFont(size=11),
                fg_color=color,
                command=lambda lv=lv: self._set_log_level(lv),
            ).pack(side="left", padx=6)

        # Nhãn trạng thái logger
        self.lbl_log_stat = ctk.CTkLabel(
            sec,
            text="🔴 Chờ khởi động...",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=C_GRAY,
        )
        self.lbl_log_stat.pack(pady=(0, 4))

        # Nút xem lịch sử
        ctk.CTkButton(
            sec,
            text="📋  XEM LỊCH SỬ PHIÊN NÀY",
            fg_color="#145a32", hover_color="#1e8449",
            height=40, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._show_history,
        ).pack(padx=16, pady=(0, 4), fill="x")

        # Nút mở thư mục logs
        ctk.CTkButton(
            sec,
            text="📂  Mở thư mục logs/",
            fg_color=C_CARD, hover_color="#333",
            height=32, font=ctk.CTkFont(size=11),
            command=self._open_log_folder,
        ).pack(padx=16, pady=(0, 10), fill="x")

        self._tick_clock()

    def _set_log_level(self, level: str):
        """Thay đổi mức độ ghi log ngay khi đang chạy."""
        if self.logger:
            self.logger.set_level(level)
            self._update_log_stat()
            labels = {"summary": "Tóm tắt (nhỏ nhất)",
                      "normal": "Bình thường", "detail": "Chi tiết (debug)"}
            self._log(f"[Logger] Level → {labels.get(level, level)}", C_YELLOW)

    def _update_log_stat(self):
        """Cập nhật nhãn trạng thái logger."""
        if not hasattr(self, "lbl_log_stat") or not self.logger:
            return
        try:
            sm   = self.logger.get_summary()
            n    = sm.get("events_in_ram", 0)
            tot  = sm.get("total_detected", 0)
            lv   = sm.get("level", "?")
            t    = int(sm.get("duration_s", 0))
            lv_icon = {"summary": "📊", "normal": "📝", "detail": "🔍"}.get(lv, "?")
            txt = (f"{lv_icon} {lv} | {t}s | "
                   f"phát hiện={tot} | RAM={n}/{self.logger.MAX_EVENTS}")
            color = C_GREEN if n < self.logger.MAX_EVENTS * 0.8 else C_ORANGE
            self.lbl_log_stat.configure(text=txt, text_color=color)
        except Exception:
            pass

    def _open_log_folder(self):
        import subprocess, platform
        log_dir = os.path.abspath(SessionLogger.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(log_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
        except Exception as e:
            self._log(f"Không mở được folder: {e}", C_ORANGE)

    def _tick_clock(self):
        try:
            if hasattr(self, "lbl_clock"):
                self.lbl_clock.configure(text=TimeReader.now_display())
            self._update_log_stat()   # Cập nhật trạng thái logger mỗi giây
            self.after(1000, self._tick_clock)
        except Exception:
            pass

    # ── Section: Log ─────────────────────────────────────────────────────────
    def _sec_log(self):
        _CardSection(self.ctrl, "📋  NHẬT KÝ HOẠT ĐỘNG")
        self.txt_log = ctk.CTkTextbox(
            self.ctrl, height=200, corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=C_CARD, text_color="#bdc3c7", wrap="word"
        )
        self.txt_log.pack(padx=12, pady=(0, 16), fill="x")
        self.txt_log.insert("0.0", "Hệ thống khởi động...\n")
        self.txt_log.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    #  LOAD AI (BACKGROUND THREAD)
    # ══════════════════════════════════════════════════════════════════════════

    def _load_ai(self):
        """Khởi tạo tất cả AI modules trong background thread để không block GUI."""

        self._log("Đang khởi động BackgroundSpeaker (gTTS + pygame)...")
        try:
            self.speaker   = BackgroundSpeaker()
            self.announcer = AnnouncementManager(self.speaker)
            self.nav       = NavigationGuide(self.speaker)
            self.time_reader = TimeReader(self.speaker)
            self._log("✓ TTS + TimeReader sẵn sàng", C_GREEN)
        except Exception as e:
            self._log(f"✗ TTS lỗi: {e}", C_RED)
            return

        self._log("Đang load YOLOv8n (vật cản + 30 loại đồ vật)...")
        try:
            self.obstacle_model = YOLO("Model/yolov8n.pt")
            self._log(f"✓ YOLOv8n — {len(TARGET_CLASSES)} classes", C_GREEN)
        except Exception as e:
            self._log(f"✗ YOLOv8n: {e}", C_RED)
            return

        self._log("Đang load MoneyDetector (money_v8n.pt)...")
        try:
            self.money_det = MoneyDetector(MONEY_MODEL_PATH)
            msg = "✓ sẵn sàng" if self.money_det.ready else "⚠ không tìm thấy file"
            col = C_GREEN if self.money_det.ready else C_ORANGE
            self._log(f"Money: {msg}", col)
        except Exception as e:
            self._log(f"⚠ Money: {e}", C_ORANGE)

        self._log("Đang khởi động EasyOCR (background thread)...")
        try:
            self.ocr = OCRReader()
            self._log("✓ OCR đang tải model nền...", C_GREEN)
        except Exception as e:
            self._log(f"⚠ OCR: {e}", C_ORANGE)

        self.traffic_an = TrafficLightAnalyzer()

        # ── SessionLogger ─────────────────────────────────────────────────────
        self.logger = SessionLogger(enabled=True, level="normal")
        self._log("✓ SessionLogger khởi động — lưu log vào logs/", C_GREEN)

        self._log("Đang load FaceRecognizer...")
        if _FACE_MODULE_OK:
            try:
                fr = FaceRecognizer(FACE_ENCODINGS)
                if fr.is_ready:
                    self.face_rec = fr
                    names = ", ".join(fr.known_names)
                    self._log(f"✓ Khuôn mặt: {names}", C_GREEN)
                    self.after(0, lambda: self.lbl_face_info.configure(
                        text=f"✓ {len(fr.known_names)} người  ({FACE_ENCODINGS})",
                        text_color=C_GREEN))
                    # Pre-cache tên người quen để giảm độ trễ TTS
                    for n in fr.known_names:
                        self.speaker._get_mp3(f"Phát hiện {n}")
                        self.speaker._get_mp3(f"Xin chào {n}")
                else:
                    self._log("⚠ Chưa có encodings.pickle — chạy encode_faces.py", C_ORANGE)
                    self.after(0, lambda: self.lbl_face_info.configure(
                        text="⚠ Chưa có encodings.pickle", text_color=C_ORANGE))
            except Exception as e:
                self._log(f"⚠ FaceRecognizer: {e}", C_ORANGE)
                self.after(0, lambda msg=str(e): self.lbl_face_info.configure(
                    text=f"✗ {msg[:45]}", text_color=C_RED))
        else:
            self._log("⚠ face_module.py không tìm thấy", C_ORANGE)
            self.after(0, lambda: self.lbl_face_info.configure(
                text="⚠ face_module.py chưa có", text_color=C_ORANGE))

        # Pre-cache các câu TTS hay dùng
        self.speaker.preload(
            build_preload_texts(extra_names=self.face_rec.known_names))

        self._ai_ready = True
        # Cập nhật thông tin calibration
        calib = DistanceEstimator.get_calibration()
        calib_txt = (f"focal={calib.focal:.0f}px"
                     + (" (đã calibrate)" if calib.scales else " (mặc định)"))
        calib_col = C_GREEN if calib.scales else C_ORANGE
        self.after(0, lambda t=calib_txt, c=calib_col:
                   self.lbl_calib_info.configure(text=t, text_color=c))

        self._log("✅  Hệ thống AI sẵn sàng!", C_GREEN)
        self._safe_speak("Hệ thống hỗ trợ người khiếm thị đã sẵn sàng")
        self.after(0, self._start_camera)

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def _start_camera(self, idx: int | None = None):
        self._cam_running = False
        time.sleep(0.05)
        if self.cap:
            self.cap.release()
        if idx is not None:
            self.camera_idx = idx
        self.cap = cv2.VideoCapture(self.camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,           30)
        if not self.cap.isOpened():
            self._log(f"✗ Không mở được camera {self.camera_idx}", C_RED)
            return
        self._cam_running = True
        threading.Thread(target=self._camera_loop, daemon=True,
                         name="CameraLoop").start()
        self._log(f"✓ Camera {self.camera_idx} kết nối thành công", C_GREEN)

    def _switch_camera(self, idx: int):
        self._start_camera(idx)

    def _camera_loop(self):
        frame_h, frame_w = 480, 640
        while self._cam_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            self.current_frame = frame.copy()
            self.frame_count  += 1
            self._fps_n       += 1

            # Tính FPS
            now = time.perf_counter()
            if now - self._fps_t0 >= 1.0:
                self.fps_disp  = self._fps_n / (now - self._fps_t0)
                self._fps_n, self._fps_t0 = 0, now
                v = self.fps_disp
                self.after(0, lambda v=v: self.lbl_fps.configure(
                    text=f"FPS: {v:4.1f}"))

            # Xử lý AI
            if self._ai_ready:
                annotated = self._process_frame(frame, frame_h, frame_w)
            else:
                annotated = frame.copy()
                cv2.putText(annotated, "Dang khoi dong AI...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 200), 2)

            # Render lên GUI
            try:
                fw = self.cam_frame.winfo_width()
                fh = self.cam_frame.winfo_height()
                if fw > 20 and fh > 20:
                    out = cv2.resize(annotated, (fw, fh))
                else:
                    out = annotated
                img_tk = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
                self.after(0, self._show_frame, img_tk)
            except Exception:
                pass

    def _show_frame(self, img_tk):
        try:
            self.cam_label.configure(image=img_tk, text="")
            self.cam_label.image = img_tk
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  LÕI XỬ LÝ AI — ĐỒNG BỘ VỚI core_engine.py
    # ══════════════════════════════════════════════════════════════════════════

    def _process_frame(self, frame: np.ndarray,
                       frame_h: int, frame_w: int) -> np.ndarray:
        """
        Xử lý một frame: chạy YOLO, đèn GT, khoảng cách, nhận diện mặt,
        tiền, OCR, navigation — giống hệt logic trong main.py nhưng không
        dùng OpenCV window mà render vào cam_label của GUI.
        """
        annotated = frame.copy()
        fc = self.frame_count

        run_yolo  = (fc % YOLO_SKIP  == 0)
        run_money = self.mode_money.get() and (fc % MONEY_SKIP == 0)
        run_face  = (
            self.mode_face.get()
            and self.face_rec.is_ready
            and fc % FACE_SKIP == 0
        )

        # ────────────────────────────────────────────────────────────────────
        #  CHẾ ĐỘ TIỀN — chạy thay thế YOLO obstacle
        # ────────────────────────────────────────────────────────────────────
        if run_money and self.money_det and self.money_det.ready:
            dets = self.money_det.detect(frame)
            for d in dets:
                draw_box_money(annotated, d["x1"], d["y1"],
                               d["x2"], d["y2"], d["label"], d["conf"])
                self.announcer.process_money(d["label"])
                norm = normalize_money_label(d["label"])
                vi   = MONEY_VI.get(norm, d["label"])
                self._spk_bar(f"💰 {vi}")
                if self.logger:
                    self.logger.log_money(d["label"], d["conf"])
            return annotated

        # ────────────────────────────────────────────────────────────────────
        #  YOLO OBSTACLE + ĐÈN GT + KHOẢNG CÁCH + KHUÔN MẶT
        # ────────────────────────────────────────────────────────────────────
        obstacle_det: list[dict] = []

        if run_yolo and self.obstacle_model:
            results = self.obstacle_model(
                frame, verbose=False, conf=YOLO_CONF, iou=YOLO_IOU)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = self.obstacle_model.names[cls_id]
                    conf_sc  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ── Đèn giao thông ───────────────────────────────────────
                    if cls_name == "traffic light" and self.mode_traffic.get():
                        crop = frame[max(0,y1):min(frame_h,y2),
                                     max(0,x1):min(frame_w,x2)]
                        # track_key riêng mỗi đèn → temporal voting không bị lẫn
                        tl_key     = f"tl_{x1//40}_{y1//40}"
                        color_name = self.traffic_an.analyze(crop, tl_key)
                        draw_box_traffic(annotated, x1, y1, x2, y2,
                                         color_name, conf_sc)
                        self.announcer.process_traffic(color_name)
                        vi_msg = TRAFFIC_VI.get(color_name, "Đèn giao thông")
                        self._spk_bar(f"🚦 {vi_msg}")
                        if color_name != "unknown" and self.logger:
                            self.logger.log_traffic(color_name)
                        continue

                    if cls_name not in TARGET_CLASSES:
                        continue

                    # ── Ước lượng khoảng cách thông minh (calibration-aware) ──
                    dist_track_key = f"{cls_name}_{x1//30}_{y1//30}"
                    state, dist_m  = DistanceEstimator.classify(
                        cls_name, y1, y2,
                        x1=x1, x2=x2,
                        frame_w=frame_w, frame_h=frame_h,
                        track_key=dist_track_key,
                    )

                    # Lưu bbox per-class để calibrate riêng từng loại vật
                    if (cls_name not in self._last_bbox_for_calib or
                            dist_m < self._last_bbox_for_calib[cls_name].get("dist_m", 99)):
                        self._last_bbox_for_calib[cls_name] = {
                            "cls": cls_name,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "dist_m": dist_m,
                        }

                    # ── Nhận diện khuôn mặt (chỉ với "person") ───────────────
                    final_label = cls_name
                    is_known    = False

                    if cls_name == "person" and self.face_rec.is_ready:
                        # Grid thô 50px: di chuyển nhẹ vẫn dùng cache cũ
                        tk_key = (x1 // 50, y1 // 50)

                        if tk_key in self.face_cache:
                            cached_name, expire, _buf = self.face_cache[tk_key]
                            if fc < expire:
                                final_label = cached_name
                                is_known    = (cached_name != "Unknown")
                            else:
                                del self.face_cache[tk_key]

                        if tk_key not in self.face_cache and run_face:
                            crop     = frame[max(0,y1):min(frame_h,y2),
                                             max(0,x1):min(frame_w,x2)]
                            new_name = self.face_rec.identify(crop)

                            # Kế thừa vote_buf từ vị trí lân cận (±1 grid)
                            old_buf: list[str] = []
                            for otk, (on, oexp, ovb) in list(self.face_cache.items()):
                                if (abs(otk[0]-tk_key[0]) <= 1 and
                                        abs(otk[1]-tk_key[1]) <= 1 and
                                        fc < oexp):
                                    old_buf = ovb
                                    break

                            vote_buf = (old_buf + [new_name])[-8:]

                            # Majority vote: ưu tiên tên người quen
                            known_hits = [n for n in vote_buf if n != "Unknown"]
                            if known_hits:
                                from collections import Counter
                                winner = Counter(known_hits).most_common(1)[0][0]
                                # Sticky-known: chỉ revert khi 6 lần cuối đều Unknown
                                unknown_streak = sum(
                                    1 for n in reversed(vote_buf) if n == "Unknown"
                                )
                                final_name = "Unknown" if unknown_streak >= 6 else winner
                            else:
                                final_name = "Unknown"

                            # TTL dài: FACE_SKIP × 20 ≈ 2 giây
                            self.face_cache[tk_key] = (
                                final_name, fc + FACE_SKIP * 20, vote_buf
                            )
                            final_label = final_name
                            is_known    = (final_name != "Unknown")

                            if is_known and new_name != "Unknown":
                                self._log(
                                    f"[Face] ★ {final_name}  "
                                    f"(vote: {vote_buf[-4:]})", C_YELLOW)
                                if self.logger:
                                    self.logger.log_face(final_name, dist_m)
                                if self.announcer._can_announce(
                                        f"face_{final_name}", FACE_ANNOUNCE_CD):
                                    self._safe_speak(f"Phát hiện {final_name}")
                                    self._spk_bar(f"👤 {final_name}")

                        elif tk_key in self.face_cache:
                            cached_name, _, _buf = self.face_cache[tk_key]
                            final_label = cached_name
                            is_known    = (cached_name != "Unknown")

                    # Ghi log vật cản (chỉ near/critical)
                    if self.logger:
                        self.logger.log_obstacle(cls_name, state, dist_m, conf_sc)

                    obstacle_det.append({
                        # BUG FIX: luôn dùng final_label (không ép về cls_name khi is_known)
                        "label":    final_label,
                        "is_known": is_known,
                        "state":    state,
                        "dist_m":   dist_m,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

                    # Người quen ở gần → thông báo riêng, cooldown 15s
                    if is_known:
                        self.announcer.process_face_nearby(final_label, state, dist_m)

                    draw_box_obstacle(annotated, x1, y1, x2, y2,
                                      final_label, conf_sc, state, dist_m,
                                      is_known=is_known)

            # ── Thông báo vật cản + cập nhật status bar ──────────────────────
            if obstacle_det:
                # process_obstacles tự bỏ qua is_known=True → không báo "người lạ"
                self.announcer.process_obstacles(obstacle_det)
                # Status bar: ưu tiên vật LẠ gần nhất (không hiện người quen là vật cản)
                stranger_dets = [d for d in obstacle_det if not d.get("is_known")]
                show_dets     = stranger_dets if stranger_dets else obstacle_det
                nearest = min(show_dets, key=lambda d: d["dist_m"])
                vi  = CLASS_VI.get(nearest["label"], nearest["label"])
                dvi = DistanceEstimator.dist_text_vi(nearest["dist_m"])
                st  = nearest["state"]
                icon = "🚨" if st == "critical" else "⚠" if st == "near" else "•"
                self._spk_bar(f"{icon} {vi} • {dvi}")

            # ── Navigation ───────────────────────────────────────────────────
            if self.nav and self.nav.active:
                self.nav.update(obstacle_det, frame_w)
                draw_nav_overlay(annotated, self.nav)
                dest = self.nav._destination or ""
                self.after(0, lambda d=dest: [
                    self.lbl_nav_bar.configure(text=f"Nav→{d}", text_color=C_GREEN),
                    self.lbl_dest.configure(text=f"Điểm đến: {d}", text_color=C_BLUE)
                ])
            else:
                self.after(0, lambda: self.lbl_nav_bar.configure(
                    text="Nav: OFF", text_color=C_GRAY))

        # ────────────────────────────────────────────────────────────────────
        #  OCR — lấy kết quả từ background thread
        # ────────────────────────────────────────────────────────────────────
        if self.ocr_scanning and self.ocr:
            result = self.ocr.get_result()
            if result is not None:
                text, _ = result
                self.ocr_scanning = False
                if text.strip():
                    self.announcer.process_ocr(text, ocr_reader=self.ocr)
                    short = text[:OCR_MAX_CHARS]
                    self._log(f"[OCR] {short}", C_YELLOW)
                    self._spk_bar(f"📖 {short[:40]}")
                    if self.logger:
                        self.logger.log_ocr(text)
                else:
                    self._safe_speak("Không tìm thấy văn bản")
                    self._log("[OCR] Không tìm thấy văn bản", C_GRAY)
                self.after(0, lambda: self.btn_ocr.configure(
                    text="📖  ĐỌC VĂN BẢN  [O]", state="normal"))
            if self.ocr and self.ocr.last_boxes:
                self.ocr.draw_results(annotated)

        return annotated

    # ══════════════════════════════════════════════════════════════════════════
    #  XỬ LÝ CÁC NÚT BẤM
    # ══════════════════════════════════════════════════════════════════════════

    def _on_audio_toggle(self):
        on = self.mode_audio.get()
        self._log(f"🔊 Âm thanh: {'BẬT' if on else 'TẮT'}")

    def _on_face_toggle(self):
        on = self.mode_face.get()
        self._log(f"👤 Nhận diện mặt: {'BẬT' if on else 'TẮT'}")
        if not on:
            self.face_cache.clear()

    def _on_money_toggle(self):
        on = self.mode_money.get()
        self._log(f"💰 Tiền: {'BẬT' if on else 'TẮT'}",
                  C_ORANGE if on else C_GRAY)
        self._safe_speak(f"Chế độ nhận diện tiền đã {'bật' if on else 'tắt'}")

    def _on_traffic_toggle(self):
        on = self.mode_traffic.get()
        self._log(f"🚦 Đèn GT: {'BẬT' if on else 'TẮT'}",
                  C_RED if on else C_GRAY)
        if on:
            self._safe_speak("Chế độ đèn giao thông đã bật")

    def _trigger_ocr(self):
        """Kích hoạt OCR — tương đương phím O trong main.py."""
        if not self.ocr:
            self._log("⚠ OCR chưa sẵn sàng", C_ORANGE); return
        if not self.ocr.ready:
            self._log("⚠ OCR đang tải model, vui lòng chờ...", C_ORANGE); return
        if self.ocr_scanning:
            self._log("⚠ OCR đang xử lý...", C_ORANGE); return
        if self.current_frame is None:
            self._log("⚠ Chưa có frame camera", C_ORANGE); return
        if self.ocr.scan(self.current_frame):
            self.ocr_scanning = True
            self._safe_speak("Đang chụp và đọc văn bản")
            self._log("[OCR] Đang nhận dạng văn bản...", C_YELLOW)
            self.btn_ocr.configure(text="📖  Đang xử lý OCR...", state="disabled")
        else:
            self._log("⚠ OCR bận, thử lại sau", C_ORANGE)

    def _ask_navigation(self):
        """Hiện hộp thoại nhập điểm đến — thay thế terminal input phím N."""
        if not self.nav:
            self._log("⚠ Nav chưa sẵn sàng", C_ORANGE); return
        rooms_str = ", ".join(ROOM_MAP.keys())
        dialog = ctk.CTkInputDialog(
            text=f"Nhập tên điểm đến:\n({rooms_str})",
            title="Chỉ dẫn đường đi"
        )
        dest = dialog.get_input()
        if not dest:
            return
        if self.nav.set_destination(dest):
            self._log(f"🗺 Điều hướng đến: {dest}", C_BLUE)
            self.lbl_dest.configure(text=f"Điểm đến: {dest}", text_color=C_BLUE)
        else:
            self._safe_speak("Điểm đến không tìm thấy, vui lòng thử lại")
            self._log(f"⚠ Không biết: '{dest}'  —  Thử: {rooms_str}", C_ORANGE)

    def _repeat_nav(self):
        """Đọc lại chỉ dẫn — tương đương phím H."""
        if self.nav:
            self.nav.repeat_last()
            self._log("[Nav] Đọc lại chỉ dẫn hiện tại")

    def _cancel_nav(self):
        """Hủy điều hướng."""
        if self.nav:
            self.nav.cancel()
            self._safe_speak("Đã tắt chế độ dẫn đường")
            self._log("[Nav] Đã hủy điều hướng", C_GRAY)
            self.lbl_dest.configure(text="Điểm đến: —", text_color=C_GRAY)
            self.lbl_nav_bar.configure(text="Nav: OFF", text_color=C_GRAY)

    def _reload_faces(self):
        """Reload encodings.pickle — tương đương phím F."""
        if not self.face_rec or not _FACE_MODULE_OK:
            self._log("⚠ FaceRecognizer chưa sẵn sàng", C_ORANGE); return
        self.face_rec.reload()
        self.face_cache.clear()
        if self.face_rec.is_ready:
            names = ", ".join(self.face_rec.known_names)
            self._safe_speak("Đã cập nhật danh sách khuôn mặt")
            self._log(f"✓ Reload face: {names}", C_GREEN)
            self.lbl_face_info.configure(
                text=f"✓ {len(self.face_rec.known_names)} người",
                text_color=C_GREEN)
        else:
            self._safe_speak("Nhận diện khuôn mặt chưa sẵn sàng")
            self._log("⚠ Chưa có encodings.pickle", C_ORANGE)

    def _upload_and_test(self):
        """Tải ảnh từ disk và chạy toàn bộ pipeline phân tích."""
        path = filedialog.askopenfilename(
            title="Chọn ảnh để phân tích",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self._log(f"[Test] {os.path.basename(path)}")
        img = cv2.imread(path)
        if img is None:
            self._log("✗ Không đọc được ảnh", C_RED); return
        h, w = img.shape[:2]

        def _run():
            result = self._process_frame(img, h, w)
            cv2.imshow(f"Kết quả: {os.path.basename(path)}  [Q=đóng]", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self._log("✓ Test hoàn tất!")

        threading.Thread(target=_run, daemon=True).start()

    def _do_calibrate(self):
        """
        Hộp thoại Calibrate Khoảng Cách per-class.

        - Hiển thị tất cả class đang detect trên màn hình.
        - Người dùng nhập khoảng cách thực (cm) cho từng class muốn hiệu chỉnh.
        - Multi-point averaging: mỗi lần nhập = thêm 1 điểm vào lịch sử.
          Hệ thống tính scale trung bình → càng nhiều điểm đo càng chính xác.
        - Lưu vào calib.json → tự load lần sau.
        """
        if not self._ai_ready:
            self._log("⚠ Hệ thống chưa sẵn sàng", C_ORANGE)
            return

        calib   = DistanceEstimator.get_calibration()
        bbox_map = self._last_bbox_for_calib

        if not bbox_map:
            self._log("⚠ Chưa detect vật thể nào. Đặt người/vật trước camera.", C_ORANGE)
            self._safe_speak("Chưa phát hiện vật thể. Hãy đặt vật trước camera.")
            return

        # ── Cửa sổ calibration ──────────────────────────────────────────────
        win = ctk.CTkToplevel(self)
        win.title("📏 Calibrate Khoảng Cách — Học từng loại vật")
        win.geometry("640x680")
        win.grab_set()
        win.resizable(False, False)

        ctk.CTkLabel(
            win,
            text="📏  Hiệu chỉnh khoảng cách theo từng loại vật thể",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(16, 2))

        ctk.CTkLabel(
            win,
            text=(f"focal={calib.focal:.0f}px  |  "
                  f"Đang detect: {', '.join(bbox_map.keys())}  |  "
                  f"Đã học: {len(calib.scales)} class"),
            font=ctk.CTkFont(size=11),
            text_color=C_GRAY,
        ).pack(pady=(0, 8))

        # ── Input per-class ──────────────────────────────────────────────────
        frame_in = ctk.CTkScrollableFrame(
            win, height=240,
            label_text="① Nhập khoảng cách THỰC TẾ (cm) — bỏ trống để bỏ qua"
        )
        frame_in.pack(fill="x", padx=16, pady=(0, 8))

        entries: dict[str, ctk.CTkEntry] = {}

        for cls_name, bbox in sorted(bbox_map.items()):
            vi       = CLASS_VI.get(cls_name, cls_name)
            est_cm   = int(round(bbox["dist_m"] * 100))
            cur_scale = calib.get_scale(cls_name)
            n_pts    = calib.get_measurement_count(cls_name)
            scale_str = (f"scale={cur_scale:.3f}  ({n_pts} điểm đo)"
                         if cur_scale != 1.0 else "chưa học")

            row = ctk.CTkFrame(frame_in, fg_color="transparent")
            row.pack(fill="x", pady=5)

            # Tên class
            ctk.CTkLabel(
                row,
                text=f"{vi}\n({cls_name})",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=130, anchor="w",
            ).pack(side="left", padx=(0, 6))

            # Ước lượng + trạng thái học
            ctk.CTkLabel(
                row,
                text=f"Hiện ước: {est_cm} cm\n{scale_str}",
                font=ctk.CTkFont(family="Consolas", size=10),
                text_color=C_GREEN if cur_scale != 1.0 else C_GRAY,
                width=180,
            ).pack(side="left", padx=4)

            # Entry nhập khoảng cách thực
            ent = ctk.CTkEntry(
                row, width=88,
                placeholder_text="cm thực",
                font=ctk.CTkFont(family="Consolas", size=13),
            )
            # Pre-fill ước lượng hiện tại
            ent.insert(0, str(est_cm))
            ent.select_range(0, "end")
            ent.pack(side="left", padx=4)

            ctk.CTkLabel(row, text="cm", font=ctk.CTkFont(size=11),
                         text_color=C_GRAY).pack(side="left")
            entries[cls_name] = ent

        # ── Bảng lịch sử học ────────────────────────────────────────────────
        frame_hist = ctk.CTkFrame(win, fg_color="#111827", corner_radius=8)
        frame_hist.pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            frame_hist,
            text="② Lịch sử học (calib.json):",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_YELLOW,
        ).pack(anchor="w", padx=12, pady=(8, 0))

        def _make_hist_text():
            lines = []
            all_hist = calib.get_all_history()
            if not calib.scales and not all_hist:
                return "  (Chưa có dữ liệu — nhập khoảng cách thực để bắt đầu học)"
            header = f"  {'Vật thể':<16} {'Scale':>7}  {'Điểm đo':>7}  Lịch sử (4 gần nhất)"
            lines.append(header)
            lines.append("  " + "─" * 60)
            for cls in sorted(set(list(calib.scales.keys()) + list(all_hist.keys()))):
                vi      = CLASS_VI.get(cls, cls)
                scale   = calib.scales.get(cls, 1.0)
                hist    = all_hist.get(cls, [])
                n       = len(hist)
                recent  = "  ".join(f"{s:.3f}" for s in hist[-4:])
                effect  = f"{'+' if scale >= 1 else ''}{(scale-1)*100:.0f}%"
                lines.append(f"  {vi:<16} {scale:>7.4f}  {n:>7}  [{recent}]  {effect}")
            lines.append(f"\n  focal={calib.focal:.1f}px  |  file: {calib._file}")
            return "\n".join(lines)

        hist_lbl = ctk.CTkLabel(
            frame_hist,
            text=_make_hist_text(),
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#9ca3af",
            justify="left",
        )
        hist_lbl.pack(anchor="w", padx=12, pady=(2, 8))

        result_lbl = ctk.CTkLabel(win, text="", font=ctk.CTkFont(size=11),
                                  text_color=C_GREEN, wraplength=580)
        result_lbl.pack(pady=4)

        # ── Nút bấm ─────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(4, 16))

        def _apply():
            applied, errors = [], []
            for cls_name, ent in entries.items():
                raw = ent.get().strip().replace(",", ".")
                if not raw:
                    continue
                try:
                    dist_cm = float(raw)
                    if dist_cm <= 0 or dist_cm > 3000:
                        raise ValueError("Ngoài phạm vi hợp lệ (0–3000 cm)")
                    bbox = bbox_map[cls_name]
                    new_scale = calib.add_class_measurement(
                        cls_name,
                        bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                        dist_cm / 100.0,
                        frame_w=640, frame_h=480,   # kích thước webcam chuẩn
                    )
                    n = calib.get_measurement_count(cls_name)
                    vi = CLASS_VI.get(cls_name, cls_name)
                    applied.append(
                        f"✓ {vi}: {int(round(bbox['dist_m']*100))}cm→{int(dist_cm)}cm  "
                        f"scale={new_scale:.4f}  ({n} điểm đo)"
                    )
                    self._log(
                        f"[Calib] {vi}: ước={int(round(bbox['dist_m']*100))}cm "
                        f"thực={int(dist_cm)}cm → scale={new_scale:.4f} ({n} điểm)",
                        C_GREEN
                    )
                except ValueError as e:
                    errors.append(f"✗ {cls_name}: {e}")

            if applied:
                result_lbl.configure(
                    text="\n".join(applied + (["Lỗi: " + e for e in errors] if errors else [])),
                    text_color=C_GREEN,
                )
                # Cập nhật bảng lịch sử
                hist_lbl.configure(text=_make_hist_text())
                # Cập nhật nhãn bên ngoài
                n_learned = len(calib.scales)
                self.lbl_calib_info.configure(
                    text=f"focal={calib.focal:.0f}px | {n_learned} class đã học",
                    text_color=C_GREEN,
                )
                self._safe_speak("Đã cập nhật khoảng cách, hệ thống học xong")
                if self.logger:
                    self.logger.log_system(
                        "Calibrate: " + " | ".join(applied)
                    )
            else:
                result_lbl.configure(
                    text="Không có thay đổi (bỏ trống hoặc lỗi).",
                    text_color=C_ORANGE,
                )

        def _reset_class():
            """Xóa scale của class được chọn đầu tiên."""
            if not bbox_map:
                return
            cls_to_reset = next(iter(bbox_map))
            if cls_to_reset in calib.scales:
                del calib.scales[cls_to_reset]
                calib.save()
                hist_lbl.configure(text=_make_hist_text())
                result_lbl.configure(
                    text=f"✓ Đã xóa scale của '{cls_to_reset}'",
                    text_color=C_ORANGE,
                )

        def _reset_all():
            calib.scales.clear()
            calib.save()
            hist_lbl.configure(text=_make_hist_text())
            result_lbl.configure(text="✓ Đã xóa toàn bộ scale correction.",
                                 text_color=C_ORANGE)
            self.lbl_calib_info.configure(
                text=f"focal={calib.focal:.0f}px | Đã reset",
                text_color=C_ORANGE,
            )
            self._log("[Calib] Reset toàn bộ per-class scales", C_ORANGE)

        ctk.CTkButton(
            btn_row, text="✓  Áp dụng & Lưu",
            fg_color="#1e8449", hover_color="#27ae60",
            height=44, font=ctk.CTkFont(size=13, weight="bold"),
            command=_apply,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="↺  Reset tất cả",
            fg_color=C_CARD, hover_color="#555",
            height=44, font=ctk.CTkFont(size=11),
            command=_reset_all,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="✖  Đóng",
            fg_color=C_CARD, hover_color="#555",
            height=44,
            command=win.destroy,
        ).pack(side="left")

    def _announce_time(self):
        """Đọc giờ hiện tại qua TTS và hiển thị lên log."""
        if not self._ai_ready or self.time_reader is None:
            self._log("⚠ Hệ thống chưa sẵn sàng", C_ORANGE)
            return
        text = self.time_reader.announce()
        self._log(f"[Giờ] {text}", C_YELLOW)
        self._spk_bar(f"🕐 {text}")
        if self.logger:
            self.logger.log_system(f"Đọc giờ: {text}")

    def _show_history(self):
        """Cửa sổ xem lịch sử phiên — hiển thị tóm tắt trước, chi tiết sau."""
        import tkinter as tk

        if not self.logger or not self.logger.enabled:
            self._log("⚠ Logger chưa khởi tạo", C_ORANGE)
            return

        sm = self.logger.get_summary()

        win = ctk.CTkToplevel(self)
        win.title("📋 Lịch sử phiên hoạt động")
        win.geometry("700x580")
        win.grab_set()

        # ── PHẦN 1: SUMMARY CARD ─────────────────────────────────────────────
        top = ctk.CTkFrame(win, fg_color="#111827", corner_radius=10)
        top.pack(fill="x", padx=16, pady=(14, 6))

        dur  = int(sm.get("duration_s", 0))
        tot  = sm.get("total_detected", 0)
        lv   = sm.get("level", "normal")
        n_ev = sm.get("events_in_ram", 0)
        lv_icon = {"summary": "📊", "normal": "📝", "detail": "🔍"}.get(lv, "?")

        ctk.CTkLabel(
            top,
            text=f"Thời gian chạy: {dur//60}m {dur%60}s  |  "
                 f"Tổng phát hiện: {tot} lần  |  "
                 f"{lv_icon} Level: {lv}  |  RAM: {n_ev}/{self.logger.MAX_EVENTS}",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=C_YELLOW,
        ).pack(pady=8, padx=12)

        # Thanh tiến trình bộ nhớ RAM
        pct = n_ev / max(self.logger.MAX_EVENTS, 1)
        bar_w = int(600 * pct)
        bar_color = C_GREEN if pct < 0.7 else C_ORANGE if pct < 0.9 else C_RED
        bar_frame = ctk.CTkFrame(top, fg_color="#1a1a2e", height=8, corner_radius=4)
        bar_frame.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkFrame(bar_frame, fg_color=bar_color, height=8,
                     width=bar_w, corner_radius=4).place(x=0, y=0)

        # ── PHẦN 2: TÓM TẮT PHÁT HIỆN ───────────────────────────────────────
        mid = ctk.CTkFrame(win, fg_color="transparent")
        mid.pack(fill="x", padx=16, pady=(0, 6))
        mid.grid_columnconfigure((0, 1, 2, 3), weight=1)

        def _stat_card(parent, col, title, data: dict, color: str):
            f = ctk.CTkFrame(parent, fg_color="#1a1a2e", corner_radius=8)
            f.grid(row=0, column=col, padx=4, sticky="nsew")
            ctk.CTkLabel(f, text=title,
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=color).pack(pady=(6, 2))
            if data:
                for k, v in list(data.items())[:5]:
                    ctk.CTkLabel(f,
                                 text=f"{k[:12]}: {v}x",
                                 font=ctk.CTkFont(family="Consolas", size=10),
                                 text_color="#9ca3af").pack()
            else:
                ctk.CTkLabel(f, text="(trống)",
                             font=ctk.CTkFont(size=10),
                             text_color="#4b5563").pack()
            ctk.CTkLabel(f, text="").pack(pady=2)

        _stat_card(mid, 0, "🚧 Vật cản",  sm.get("obstacles", {}), C_RED)
        _stat_card(mid, 1, "👤 Khuôn mặt", sm.get("faces", {}),     C_YELLOW)
        _stat_card(mid, 2, "💰 Tiền",      sm.get("money", {}),      C_GREEN)
        _stat_card(mid, 3, "🚦 Đèn GT",    sm.get("traffic", {}),    C_ORANGE)

        # ── PHẦN 3: DANH SÁCH SỰ KIỆN (có bộ lọc) ───────────────────────────
        events = self.logger._events
        filter_var = ctk.StringVar(value="all")

        tab_frame = ctk.CTkFrame(win, fg_color="transparent")
        tab_frame.pack(fill="x", padx=16, pady=(2, 0))

        ctk.CTkLabel(tab_frame, text="Lọc:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 4))
        for val, lbl in [("all", "Tất cả"), ("obstacle", "Vật cản"),
                          ("face", "Mặt"), ("money", "Tiền"),
                          ("traffic", "Đèn"), ("ocr", "OCR"), ("system", "Hệ thống")]:
            ctk.CTkRadioButton(
                tab_frame, text=lbl, value=val,
                variable=filter_var, font=ctk.CTkFont(size=10),
                command=lambda: _refresh(),
            ).pack(side="left", padx=4)

        txt = ctk.CTkTextbox(
            win, font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#0d1117", text_color="#c9d1d9",
        )
        txt.pack(padx=16, pady=(4, 4), fill="both", expand=True)

        COLORS = {"obstacle": "#ef4444", "face": "#f1c40f", "traffic": "#e67e22",
                  "money": "#2ecc71", "ocr": "#3498db", "nav": "#9b59b6",
                  "system": "#6b7280"}

        def _refresh():
            fv = filter_var.get()
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            shown = [e for e in events if fv == "all" or e["event_type"] == fv]
            if not shown:
                txt.insert("end", "  (Không có sự kiện nào)\n", "gray")
                txt.tag_config("gray", foreground="#4b5563")
            for ev in shown:
                et = ev["event_type"]
                c  = COLORS.get(et, "#888")
                ts = ev.get("timestamp", "")
                vi = ev.get("label_vi", "") or ev.get("note", "")[:50]
                st = ev.get("state", "")
                dd = ev.get("dist_display", "")
                cf = ev.get("conf", "")
                nt = ev.get("note", "")[:60]
                parts = [f"[{ts}]", f"[{et:8s}]", vi]
                if st:  parts.append(st)
                if dd:  parts.append(dd)
                if cf:  parts.append(cf)
                if nt and et in ("ocr", "system"): parts.append(f"→ {nt}")
                line = "  ".join(p for p in parts if p.strip()) + "\n"
                txt.insert("end", line, et)
                txt.tag_config(et, foreground=c)
            txt.configure(state="disabled")
        _refresh()

        # Nút dưới
        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(pady=(2, 12), padx=16, fill="x")
        ctk.CTkButton(btn_row, text="📂  Mở logs/", fg_color=C_BLUE, width=140,
                      command=self._open_log_folder).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="🗑  Xóa RAM", fg_color=C_CARD, width=120,
                      command=lambda: (
                          self.logger._events.clear(),
                          _refresh(),
                          self._log("[Logger] Đã xóa log khỏi RAM", C_ORANGE),
                      )).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="✖  Đóng", fg_color=C_CARD,
                      command=win.destroy).pack(side="right")

    # ══════════════════════════════════════════════════════════════════════════
    #  HÀM TIỆN ÍCH NỘI BỘ
    # ══════════════════════════════════════════════════════════════════════════

    def _safe_speak(self, text: str, priority: bool = False):
        """Phát TTS chỉ khi âm thanh bật và speaker đã sẵn sàng."""
        if not self.mode_audio.get() or self.speaker is None:
            return
        self.speaker.say(text, priority=priority)

    def _spk_bar(self, text: str):
        """Cập nhật nhãn loa trên status bar và reset về Chờ... sau 4 giây."""
        self.after(0, lambda t=text: self.lbl_spk.configure(
            text=t, text_color=C_GREEN))
        self.after(4000, lambda: self.lbl_spk.configure(
            text="🔊 Chờ...", text_color=C_GRAY))

    def _log(self, text: str, color: str | None = None):
        """Thêm dòng vào log box (thread-safe qua after())."""
        def _insert():
            self.txt_log.configure(state="normal")
            self.txt_log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.txt_log.see("end")
            self.txt_log.configure(state="disabled")
        self.after(0, _insert)

    def _on_closing(self):
        """Dọn dẹp khi đóng cửa sổ."""
        self._log("Đang thoát...")
        self._cam_running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
        if self.logger:
            summary = self.logger.close()
            if summary:
                print(f"[Logger] Log đã lưu: {summary.get('csv_path','')}")
        if self.speaker:
            self.speaker.stop()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = BlindAssistantGUI()
    app.mainloop()