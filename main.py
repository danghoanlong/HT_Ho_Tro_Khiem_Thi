import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO

# ── Import toàn bộ logic AI từ core_engine ────────────────────────────────────
from core_engine import (
    TARGET_CLASSES, CLASS_VI, MONEY_VI, TRAFFIC_VI, ROOM_MAP, ROOM_ALIASES,
    YOLO_CONF, YOLO_IOU, MONEY_MODEL_PATH, FACE_ENCODINGS,
    WARN_DIST_M, CRITICAL_DIST_M,
    YOLO_SKIP, MONEY_SKIP, FACE_SKIP, FACE_ANNOUNCE_CD,
    OCR_MAX_CHARS,
    BackgroundSpeaker, AnnouncementManager,
    DistanceEstimator, NavigationGuide,
    MoneyDetector, OCRReader, TrafficLightAnalyzer,
    draw_box_obstacle, draw_box_money, draw_box_traffic, draw_nav_overlay,
    put_vi_text, vi_text_size,
    build_preload_texts,
)

# ── FaceRecognizer (tuỳ chọn, không crash nếu chưa có) ───────────────────────
try:
    from face_module import FaceRecognizer
    _FACE_MODULE_OK = True
except ImportError:
    _FACE_MODULE_OK = False
    print("[Face] ⚠  face_module.py không tìm thấy.")


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD NHẬN INPUT TỪ TERMINAL (phím N — nhập điểm đến)
# ══════════════════════════════════════════════════════════════════════════════

def _input_thread_fn(input_queue: queue.Queue, stop_event: threading.Event):
    """Thread riêng đọc input từ stdin, không block vòng lặp OpenCV."""
    while not stop_event.is_set():
        try:
            raw = input()
            input_queue.put_nowait(raw.strip())
        except EOFError:
            break
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  DUMMY FACE RECOGNIZER (dùng khi chưa có face_module / encodings)
# ══════════════════════════════════════════════════════════════════════════════

class _DummyFaceRec:
    """Placeholder để tránh kiểm tra None ở khắp nơi trong vòng lặp."""
    is_ready    = False
    known_names: list[str] = []
    def identify(self, _): return "Unknown"
    def reload(self): pass
    def stats(self): return "[Face] Chưa sẵn sàng (face_module.py hoặc encodings.pickle thiếu)"


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("   BLIND ASSISTANT AI v5.2  —  GIAO DIỆN TERMINAL")
    print("   Phím: M=Tien  O=OCR  T=DenGT  N=DiDen  H=LapLai  F=FaceReload  Q=Thoat")
    print("=" * 70)
    print(f"[INFO] Đồ vật nhận diện: {len(TARGET_CLASSES)} loại")
    print(f"[INFO] Khoảng cách: cảnh báo < {WARN_DIST_M}m, nguy hiểm < {CRITICAL_DIST_M}m")
    print(f"[INFO] Điểm đến hỗ trợ: {', '.join(ROOM_MAP.keys())}")

    # ── Khởi tạo các module AI ────────────────────────────────────────────────
    print("[INIT] Đang load YOLOv8n (vật cản + đồ vật)...")
    obstacle_model = YOLO("yolov8n.pt")

    print("[INIT] Đang load MoneyDetector...")
    money_detector = MoneyDetector(MONEY_MODEL_PATH)

    print("[INIT] Đang load FaceRecognizer...")
    if _FACE_MODULE_OK:
        face_rec = FaceRecognizer(FACE_ENCODINGS)
        if face_rec.is_ready:
            print(face_rec.stats())
        else:
            print("[Face] ⚠  Chưa có encodings — chạy encode_faces.py để kích hoạt.")
    else:
        face_rec = _DummyFaceRec()

    print("[INIT] Đang khởi động BackgroundSpeaker (gTTS + pygame)...")
    speaker   = BackgroundSpeaker()
    announcer = AnnouncementManager(speaker)
    nav       = NavigationGuide(speaker)

    # Pre-cache các câu TTS hay dùng để giảm độ trễ lần đầu
    speaker.preload(build_preload_texts(extra_names=face_rec.known_names))

    print("[INIT] Đang khởi động OCR (background thread)...")
    ocr              = OCRReader()
    traffic_analyzer = TrafficLightAnalyzer()

    speaker.say("Hệ thống hỗ trợ người khiếm thị đã sẵn sàng")

    # ── Thread nhận input terminal (phím N) ───────────────────────────────────
    stop_event   = threading.Event()
    input_queue  = queue.Queue()
    awaiting_nav = False
    inp_thread   = threading.Thread(
        target=_input_thread_fn, args=(input_queue, stop_event),
        daemon=True, name="InputThread"
    )
    inp_thread.start()

    # ── Mở webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,           30)

    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        speaker.stop()
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[CAM] {frame_w}×{frame_h} @ 30fps. Sẵn sàng.\n")

    # ── Trạng thái runtime ────────────────────────────────────────────────────
    mode_money     = False
    mode_traffic   = True
    ocr_scanning   = False
    frame_count    = 0
    last_annotated = None
    face_cache: dict[tuple, tuple[str, int]] = {}  # {(x//20,y//20): (name, expire)}
    fps_t0, fps_n, fps_disp = time.perf_counter(), 0, 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  VÒNG LẶP CAMERA CHÍNH
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_n       += 1
        now_perf     = time.perf_counter()
        if now_perf - fps_t0 >= 1.0:
            fps_disp = fps_n / (now_perf - fps_t0)
            fps_n, fps_t0 = 0, now_perf

        # ── Xử lý input terminal (phím N đang chờ nhập điểm đến) ────────────
        while not input_queue.empty():
            raw = input_queue.get_nowait()
            if awaiting_nav:
                awaiting_nav = False
                if raw.lower() in ("", "huy", "cancel", "thoat"):
                    nav.cancel()
                    speaker.say("Đã tắt chế độ dẫn đường")
                else:
                    if not nav.set_destination(raw):
                        speaker.say("Điểm đến không tìm thấy, vui lòng thử lại")
                        print(f"[Nav] ⚠ Không biết: '{raw}'. "
                              f"Thử: {', '.join(ROOM_MAP.keys())}")

        # ── Quyết định frame nào chạy module nào (frame skipping) ────────────
        run_yolo  = (frame_count % YOLO_SKIP  == 0)
        run_money = mode_money and (frame_count % MONEY_SKIP == 0)
        run_face  = face_rec.is_ready and (frame_count % FACE_SKIP == 0)

        annotated = last_annotated if last_annotated is not None else frame.copy()

        # ────────────────────────────────────────────────────────────────────
        #  YOLO: vật cản + đèn GT + khoảng cách + khuôn mặt
        # ────────────────────────────────────────────────────────────────────
        obstacle_det: list[dict] = []

        if run_yolo:
            annotated = frame.copy()
            results   = obstacle_model(frame, verbose=False,
                                       conf=YOLO_CONF, iou=YOLO_IOU)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = obstacle_model.names[cls_id]
                    conf_sc  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ── Đèn giao thông ───────────────────────────────────────
                    if cls_name == "traffic light" and mode_traffic:
                        crop = frame[max(0,y1):min(frame_h,y2),
                                     max(0,x1):min(frame_w,x2)]
                        color_name = traffic_analyzer.analyze(crop)
                        draw_box_traffic(annotated, x1, y1, x2, y2,
                                         color_name, conf_sc)
                        announcer.process_traffic(color_name)
                        continue

                    if cls_name not in TARGET_CLASSES:
                        continue

                    # ── Ước lượng khoảng cách ────────────────────────────────
                    state, dist_m = DistanceEstimator.classify(cls_name, y1, y2)

                    # ── Nhận diện khuôn mặt (chỉ với "person") ───────────────
                    final_label = cls_name
                    is_known    = False

                    if cls_name == "person" and face_rec.is_ready:
                        track_key = (x1 // 20, y1 // 20)
                        if track_key in face_cache:
                            cached_name, expire = face_cache[track_key]
                            if frame_count < expire:
                                final_label = cached_name
                                is_known    = (cached_name != "Unknown")
                            else:
                                del face_cache[track_key]
                        if track_key not in face_cache and run_face:
                            crop = frame[max(0,y1):min(frame_h,y2),
                                         max(0,x1):min(frame_w,x2)]
                            person_name = face_rec.identify(crop)
                            face_cache[track_key] = (
                                person_name, frame_count + FACE_SKIP * 5)
                            final_label = person_name
                            is_known    = (person_name != "Unknown")
                            if is_known:
                                print(f"[FaceRec] ★ Phát hiện: {person_name}")
                                if announcer._can_announce(
                                        f"face_{person_name}", FACE_ANNOUNCE_CD):
                                    speaker.say(f"Phát hiện {person_name}",
                                                priority=False)

                    # Dùng cls_name gốc cho navigation (không phải tên người)
                    obstacle_det.append({
                        "label":  final_label if not is_known else cls_name,
                        "state":  state,
                        "dist_m": dist_m,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

                    draw_box_obstacle(annotated, x1, y1, x2, y2,
                                      final_label, conf_sc, state, dist_m,
                                      is_known=is_known)

            announcer.process_obstacles(obstacle_det)
            if nav.active:
                nav.update(obstacle_det, frame_w)

        # ────────────────────────────────────────────────────────────────────
        #  MODULE TIỀN
        # ────────────────────────────────────────────────────────────────────
        if run_money:
            money_dets = money_detector.detect(frame)
            for det in money_dets:
                draw_box_money(annotated, det["x1"], det["y1"],
                               det["x2"], det["y2"], det["label"], det["conf"])
                announcer.process_money(det["label"])

        # ────────────────────────────────────────────────────────────────────
        #  OCR — lấy kết quả từ background thread
        # ────────────────────────────────────────────────────────────────────
        if ocr_scanning:
            result = ocr.get_result()
            if result is not None:
                text, _ = result
                ocr_scanning = False
                if text.strip():
                    announcer.process_ocr(text, ocr_reader=ocr)
                    print(f"[OCR] Kết quả: {text[:100]}")
                else:
                    speaker.say("Không tìm thấy văn bản")
            if ocr.last_boxes:
                ocr.draw_results(annotated)

        last_annotated = annotated

        # ────────────────────────────────────────────────────────────────────
        #  HUD + HIỂN THỊ LÊN CỬA SỔ OPENCV
        # ────────────────────────────────────────────────────────────────────
        display = last_annotated.copy()

        # Nền mờ cho HUD
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (450, 135), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.50, display, 0.50, 0, display)

        cv2.putText(display, f"FPS: {fps_disp:5.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
        cv2.putText(display,
                    f"Warn<{WARN_DIST_M}m | Crit<{CRITICAL_DIST_M}m | Obj:{len(TARGET_CLASSES)}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

        mode_parts = []
        if face_rec.is_ready:
            mode_parts.append(f"[F] {len(face_rec.known_names)} mặt")
        if mode_money:    mode_parts.append("[M] Tiền")
        if mode_traffic:  mode_parts.append("[T] Đèn GT")
        if ocr_scanning:  mode_parts.append("[O] OCR...")
        elif ocr.ready:   mode_parts.append("[O] OCR sẵn")
        if nav.active:    mode_parts.append(f"[N]→{nav._destination}")
        mode_str = "  ".join(mode_parts) if mode_parts else "Chờ..."
        put_vi_text(display, mode_str, (8, 62), font_size=14,
                    color=(0, 255, 100) if nav.active else (255, 215, 0))

        cv2.putText(display,
                    "Q=Quit M=Money O=OCR T=Traffic N=Nav H=Repeat F=FaceReload",
                    (8, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

        if awaiting_nav:
            put_vi_text(display, ">>> Nhap diem den vao terminal <<<",
                        (8, 107), font_size=14, color=(0, 200, 255))

        cv2.rectangle(display, (454, 446), (638, 478), (0, 0, 0), -1)
        cv2.putText(display, f">{WARN_DIST_M}m (gray)",
                    (458, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)
        cv2.putText(display, f"<{WARN_DIST_M}m (orange)",
                    (458, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 255), 1)

        draw_nav_overlay(display, nav)
        cv2.imshow("Blind Assistant AI v5.2  [Q|M|O|T|N|H|F]", display)

        # ────────────────────────────────────────────────────────────────────
        #  XỬ LÝ PHÍM TẮT
        # ────────────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[INFO] Người dùng yêu cầu thoát.")
            break

        elif key == ord("m"):
            mode_money = not mode_money
            speaker.say(f"Chế độ nhận diện tiền đã {'bật' if mode_money else 'tắt'}")
            print(f"[Mode] Tiền: {'BẬT' if mode_money else 'TẮT'}")

        elif key == ord("t"):
            mode_traffic = not mode_traffic
            if mode_traffic:
                speaker.say("Chế độ đèn giao thông đã bật")
            print(f"[Mode] Đèn GT: {'BẬT' if mode_traffic else 'TẮT'}")

        elif key == ord("o"):
            if not ocr.ready:
                print("[OCR] Module chưa sẵn sàng.")
            elif ocr_scanning:
                print("[OCR] Đang xử lý, vui lòng chờ...")
            else:
                if ocr.scan(frame):
                    ocr_scanning = True
                    speaker.say("Đang chụp và đọc văn bản")

        elif key == ord("n"):
            print("\n[NAV] Nhập tên điểm đến (hoặc 'huy' để hủy):")
            print(f"      Các điểm đến: {', '.join(ROOM_MAP.keys())}")
            speaker.say("Vui lòng nhập tên điểm đến vào terminal")
            awaiting_nav = True

        elif key == ord("h"):
            nav.repeat_last()

        elif key == ord("f"):
            print("[FaceRec] Reloading encodings...")
            face_rec.reload()
            face_cache.clear()
            if face_rec.is_ready:
                speaker.say("Đã cập nhật danh sách khuôn mặt")
                print(face_rec.stats())
            else:
                speaker.say("Nhận diện khuôn mặt chưa sẵn sàng")

    # ── Dọn dẹp ──────────────────────────────────────────────────────────────
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    print("[INFO] Thoát hệ thống. Tạm biệt!")


if __name__ == "__main__":
    main()