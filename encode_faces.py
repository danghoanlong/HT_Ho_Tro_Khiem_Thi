import os
import cv2
import pickle
import argparse
import numpy as np
import face_recognition
from pathlib import Path
from PIL import Image, ImageOps

# ══════════════════════════════════════════════════════════════════════════════
#  THAM SỐ DÒNG LỆNH
# ══════════════════════════════════════════════════════════════════════════════

ap = argparse.ArgumentParser(description="Encode faces từ dataset → encodings.pickle")
ap.add_argument("-d", "--dataset",          default="dataset",
                help="Đường dẫn thư mục dataset (mặc định: dataset/)")
ap.add_argument("-o", "--output",           default="encodings.pickle",
                help="File output encodings (mặc định: encodings.pickle)")
ap.add_argument("-m", "--detection-method", default="hog",
                choices=["hog", "cnn"],
                help="Phương pháp detect mặt: hog (CPU nhanh) | cnn (GPU chính xác)")
ap.add_argument("-j", "--jitters",          default=2, type=int,
                help="Số lần jitter khi encode (càng nhiều càng chính xác, càng chậm)")
args = vars(ap.parse_args())


# ══════════════════════════════════════════════════════════════════════════════
#  CÁC ĐỊNH DẠNG ẢNH HỖ TRỢ
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM ĐỌC ẢNH AN TOÀN — luôn trả về uint8 RGB
# ══════════════════════════════════════════════════════════════════════════════

def load_image_safe(path: str) -> np.ndarray | None:
    """
    Đọc ảnh bằng PIL → luôn trả về numpy uint8 RGB.
    Xử lý: EXIF rotation, RGBA, grayscale, 16-bit PNG, tiếng Việt trong path.
    """
    try:
        with open(path, "rb") as f:
            pil = Image.open(f)
            pil.load()                          # Đọc toàn bộ dữ liệu
        pil = ImageOps.exif_transpose(pil)      # Sửa xoay EXIF (ảnh điện thoại)
        pil = pil.convert("RGB")                # Ép về RGB uint8
        return np.ascontiguousarray(np.array(pil), dtype=np.uint8)
    except Exception as e:
        print(f"  ✗ Lỗi đọc {path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION — tạo thêm biến thể để tăng độ chính xác
# ══════════════════════════════════════════════════════════════════════════════

def _to_valid_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Đảm bảo array là uint8 RGB C-contiguous — bắt buộc cho dlib.
    cv2.flip() và np.clip() đôi khi tạo ra array không contiguous hoặc
    sai dtype trên một số phiên bản numpy/dlib cũ.
    """
    arr = np.array(arr, dtype=np.uint8, order='C')   # copy + C-contiguous + uint8
    assert arr.ndim == 3 and arr.shape[2] == 3, f"Shape lỗi: {arr.shape}"
    return arr


def augment(img_rgb: np.ndarray) -> list[np.ndarray]:
    """
    Tạo 4 biến thể từ 1 ảnh: gốc + flip + sáng + tối.
    Dùng cv2.flip() thay vì np.fliplr() — tránh negative stride gây crash dlib.
    """
    # Ảnh RGB gốc đã load bằng PIL → an toàn
    base = _to_valid_rgb(img_rgb)

    # Flip ngang: cv2.flip(src, 1) — luôn trả về array mới, contiguous
    # QUAN TRỌNG: cv2 dùng BGR nhưng flip theo chiều ngang không phụ thuộc
    # thứ tự kênh màu → an toàn khi dùng với RGB
    flipped = _to_valid_rgb(cv2.flip(base, 1))

    # Tăng/giảm độ sáng
    bright = _to_valid_rgb(np.clip(base.astype(np.int32) + 40, 0, 255))
    dark   = _to_valid_rgb(np.clip(base.astype(np.int32) - 40, 0, 255))

    return [base, flipped, bright, dark]


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODE FACES CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def encode_dataset(dataset_dir: str, detection_method: str, jitters: int):
    """
    Duyệt qua toàn bộ dataset, encode từng khuôn mặt.
    Trả về dict {"names": [...], "encodings": [...]}.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        print(f"\n✗ Không tìm thấy thư mục dataset: {dataset_dir}")
        print(  "  Hãy tạo cấu trúc: dataset/<TênNgười>/<ảnh1.jpg> ...")
        return None

    known_names     = []
    known_encodings = []

    # Lấy danh sách thư mục con (mỗi thư mục = 1 người)
    person_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not person_dirs:
        print(f"✗ Thư mục {dataset_dir}/ trống. Thêm thư mục con cho từng người.")
        return None

    print(f"\n[Encode] Phát hiện {len(person_dirs)} người: "
          f"{', '.join(d.name for d in person_dirs)}")
    print(f"[Encode] Detection method: {detection_method} | Jitters: {jitters}\n")

    total_enc = 0

    for person_dir in person_dirs:
        name = person_dir.name
        img_paths = sorted([f for f in person_dir.iterdir()
                            if f.suffix.lower() in IMG_EXTS])

        if not img_paths:
            print(f"  ⚠  {name}: không có ảnh nào trong {person_dir}/")
            continue

        print(f"  ► {name} ({len(img_paths)} ảnh):")
        enc_count = 0

        for img_path in img_paths:
            img_rgb = load_image_safe(str(img_path))
            if img_rgb is None:
                continue

            # Encode ảnh gốc + augmentation
            for variant in augment(img_rgb):
                # Guard cuối cùng — đảm bảo 100% trước khi gọi dlib
                if (variant is None
                        or variant.dtype != np.uint8
                        or variant.ndim != 3
                        or variant.shape[2] != 3
                        or not variant.flags['C_CONTIGUOUS']):
                    variant = np.array(variant, dtype=np.uint8, order='C')

                # Bước 1: Detect vị trí khuôn mặt
                try:
                    boxes = face_recognition.face_locations(
                        variant, model=detection_method)
                except RuntimeError as e:
                    print(f"     ⚠ Bỏ qua variant ({e})")
                    continue

                if not boxes:
                    continue

                # Bước 2: Tạo encoding 128-d cho từng khuôn mặt phát hiện được
                encodings = face_recognition.face_encodings(
                    variant, boxes, num_jitters=jitters)

                for enc in encodings:
                    known_encodings.append(enc)
                    known_names.append(name)
                    enc_count += 1

        if enc_count > 0:
            print(f"     ✓ {enc_count} encodings (từ {len(img_paths)} ảnh × 4 augment)")
            total_enc += enc_count
        else:
            print(f"     ✗ KHÔNG TÌM THẤY KHUÔN MẶT!")
            print(f"       Kiểm tra: ảnh đủ sáng? Mặt chiếm ≥ 30% ảnh? Nhìn rõ mặt?")

    print(f"\n[Encode] Tổng: {total_enc} encodings | {len(set(known_names))} người")

    if total_enc == 0:
        return None

    return {"names": known_names, "encodings": known_encodings}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("   ENCODE FACES — Tạo file encodings.pickle")
    print("=" * 62)

    data = encode_dataset(
        dataset_dir      = args["dataset"],
        detection_method = args["detection_method"],
        jitters          = args["jitters"],
    )

    if data is None:
        print("\n✗ Không tạo được encodings. Kiểm tra lại dataset.")
        return

    # Lưu ra file pickle
    output_path = args["output"]
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✓ Đã lưu encodings → {output_path}")
    print(f"  ({len(data['encodings'])} encodings, {len(set(data['names']))} người)")
    print("\n  Bước tiếp theo: chạy main.py để nhận diện realtime")
    print("=" * 62)


if __name__ == "__main__":
    main()