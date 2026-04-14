# 👁️ Blind Assistant AI v5.2 — Hệ Thống Hỗ Trợ Người Khiếm Thị

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

**Blind Assistant AI** là một hệ thống Trí tuệ Nhân tạo toàn diện, tích hợp Computer Vision (Thị giác máy tính) và NLP (Xử lý ngôn ngữ tự nhiên) nhằm hỗ trợ người khiếm thị trong các sinh hoạt hàng ngày. Hệ thống hoạt động theo thời gian thực (Real-time) qua Camera và giao tiếp với người dùng bằng giọng nói Tiếng Việt tự nhiên.

Dự án cung cấp 2 chế độ: **Giao diện đồ họa (GUI)** thân thiện và **Giao diện dòng lệnh (Terminal)** tối ưu hiệu suất.

---

## ✨ 9 Module Tính Năng Cốt Lõi

| Tiện ích | Công nghệ sử dụng | Mô tả chi tiết |
| :--- | :--- | :--- |
| 🛡️ **Nhận diện vật thể** | `YOLOv8n` + COCO 80 | Nhận diện hơn 80 đồ vật phổ biến trong nhà/ngoài đường và dịch sang Tiếng Việt. |
| 📏 **Đo lường khoảng cách** | Thuật toán tỷ lệ khung hình | Cảnh báo 3 cấp độ (An toàn > Gần > Nguy hiểm) theo đơn vị Mét để tránh va chạm. |
| 🗣️ **Trợ lý Giọng nói** | `gTTS` + `pygame` | Phát âm thanh Tiếng Việt mượt mà với cơ chế Cache `.mp3` chống nói lắp/spam loa. |
| 💵 **Nhận diện Tiền VNĐ** | `YOLOv8n` (Custom Train) | Nhận diện chính xác các mệnh giá tiền Polymer Việt Nam. |
| 🚦 **Đèn giao thông** | `YOLOv8` + `HSV Color` | Xác định vị trí đèn tín hiệu và phân tích màu sắc (Xanh/Đỏ/Vàng) để qua đường an toàn. |
| 👤 **Nhận diện Người quen** | `face_recognition` | Học khuôn mặt từ ảnh và gọi tên người quen khi họ xuất hiện trước camera. |
| 📖 **Đọc văn bản (OCR)** | `EasyOCR` + `Thread` | Chụp ảnh và trích xuất văn bản (tiếng Việt/Anh) chạy ngầm không gây đơ hình. |
| 🗺️ **Chỉ đường trong nhà** | Indoor Navigation Logic | Hướng dẫn di chuyển đến các phòng dựa trên cột mốc (Landmarks) nhận diện được. |
| 🖥️ **Giao diện Điều khiển** | `CustomTkinter` | Bảng điều khiển hiện đại, có công tắc bật/tắt module và theo dõi Log hệ thống. |

---

## 📂 Cấu Trúc Mã Nguồn (Separation of Concerns)

Dự án được cấu trúc chuẩn hóa, tách biệt hoàn toàn phần "Não bộ AI" và "Giao diện", giúp mã nguồn gọn gàng và dễ bảo trì:

```text
Blind-Assistant-AI/
│
├── 🧠 core_engine.py       # "Não bộ": Chứa toàn bộ Class xử lý logic AI, âm thanh, khoảng cách
├── 💻 main.py              # Phiên bản Terminal (gọi core_engine + chạy vòng lặp cv2)
├── 🚀 gui_app.py           # Phiên bản GUI (gọi core_engine + vẽ giao diện CustomTkinter)
├── ⚙️ encode_faces.py      # Script trích xuất dữ liệu khuôn mặt (chạy 1 lần)
├── ⚙️ face_module.py       # Module phụ trợ xử lý nhận diện khuôn mặt
│
├── 📁 models/              # [THƯ MỤC CẦN TẠO] Chứa các model AI
│   ├── yolov8n.pt          # Model YOLO gốc
│   ├── money_v8n.pt        # Model nhận diện tiền tự train
│   └── encodings.pickle    # Sinh ra sau khi chạy encode_faces.py
│
├── 📁 dataset/             # Chứa ảnh khuôn mặt để AI học
│   ├── Long/               # Thư mục chứa các ảnh của Long (Long1.jpg, Long2.jpg)
│ 
│
└── 📄 requirements.txt     # Danh sách thư viện cần cài đặt
**##⚙️ Hướng Dẫn Cài Đặt**
Yêu cầu hệ thống: Python 3.8 - 3.11, Camera (Webcam).

Bước 1: Clone dự án hoặc tải mã nguồn về máy
git clone [https://github.com/Ten-Cua-Ban/Blind-Assistant-AI.git](https://github.com/Ten-Cua-Ban/Blind-Assistant-AI.git)
cd Blind-Assistant-AI
Bước 2: Cài đặt thư viện
Bạn chỉ cần chạy 1 lệnh duy nhất để cài toàn bộ thư viện:
pip install -r requirements.txt
⚠️ Lưu ý cho Windows: Thư viện face_recognition yêu cầu dlib. Nếu bạn gặp lỗi khi cài đặt, hãy cài đặt C++ Build Tools từ Visual Studio trước, hoặc tìm tải file dlib.whl phù hợp với phiên bản Python của bạn để cài thủ công.
**##🚀 Cách Sử Dụng Dự Án**
1. Dạy AI nhận diện người quen (Huấn luyện khuôn mặt)
Trước khi chạy hệ thống, hãy đưa ảnh người quen vào thư mục dataset/Tên_Người/. Sau đó chạy lệnh sau để hệ thống trích xuất đặc trưng:
python encode_faces.py
2. Khởi chạy Giao diện Đồ họa (GUI - Khuyên dùng)
Phiên bản đầy đủ tính năng nhất, dễ thao tác qua các nút bấm:
python gui_app.py
3. Khởi chạy Giao diện Terminal (Dành cho máy yếu)
Nếu bạn cần tối đa hóa tốc độ khung hình (FPS) và điều khiển qua phím tắt:
python main.py
Bảng Phím tắt (Hotkeys) trong chế độ Terminal:
Phím,Chức năng,Thao tác
M,Bật / Tắt nhận diện Tiền,Tức thì
T,Bật / Tắt phân tích Đèn giao thông,Tức thì
O,Quét và Đọc văn bản (OCR),Tức thì
N,Nhập điểm đến để nhận chỉ đường,Chờ nhập ở Terminal
H,Nghe lại chỉ dẫn đường đi gần nhất,Tức thì
F,Cập nhật lại danh sách khuôn mặt,Tức thì
Q,Thoát chương trình,Dọn dẹp & Thoát
**##📸 Ảnh Chụp Màn Hình (Screenshots)**
(Gợi ý: Bạn hãy chụp 1-2 tấm ảnh lúc phần mềm đang chạy nhận diện ra đồ vật, giao diện GUI đang bật và chèn link ảnh vào đây để README thêm sinh động nhé!)
