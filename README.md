# License Plate Recognition System

## Tính năng nổi bật

- **Phát hiện thông minh**: Công nghệ YOLO định vị biển số với độ chính xác tốt.
- **Trích xuất văn bản**: EasyOCR nhận diện ký tự với khả năng xử lý đa ngôn ngữ.
- **Đa phương tiện**: Xử lý linh hoạt cả ảnh tĩnh và video động.
- **Tích hợp dễ dàng**: API RESTful sẵn sàng tích hợp vào hệ thống.

## Cấu trúc dự án

```
license_plate_recognition/
├── app/
│   ├── model.py              # Core ALPR model
│   └── main.py               # REST API implementation
├── demo/
│   ├── main.py               # Desktop GUI application
│   └── output/               # Processing results & logs
├── media/                    # Sample images & videos
├── weights/                  # Pre-trained model weights
├── requirements.txt          # Dependencies
├── Dockerfile               # Container configuration
└── README.md                # Documentation
```

## Khởi tạo dự án

### Phát triển từ mã nguồn

```bash
# Clone repository
git clone https://github.com/marknguyen/license_plate_recognition.git
cd license_plate_recognition

# Thiết lập môi trường ảo
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Cài đặt các phụ thuộc
pip install -r requirements.txt
```

### Xây dựng Docker Image

```bash
docker build -t alpr-api .
```

### Sử dụng Image có sẵn

```bash
docker pull marknguyenn02/alpr-api
```

## Hướng dẫn sử dụng

### Chạy Demo GUI

Khởi động ứng dụng desktop với giao diện đồ họa:

```bash
python -m demo.main
```

### Khởi động API Server

Chạy REST API server:

```bash
python -m app.main
```

API endpoint: `POST /alpr` - Tải lên ảnh để nhận diện biển số

## Triển khai Production

### Google Cloud Platform

* **Lưu ý**: Dịch vụ API trên GCP hiện tạm ngưng hoạt động do giới hạn ngân sách. 
* **URL**: `https://alpr-api-439951883626.asia-southeast1.run.app/` 
* Để sử dụng API, vui lòng liên hệ trực tiếp để được hỗ trợ kích hoạt.

### Self-hosted

```bash
# Sử dụng Docker
docker run -p 8080:8080 alpr-api

# Hoặc chạy trực tiếp
python -m app.main
```

## Yêu cầu hệ thống

- Python 3.8+
- OpenCV
- PyTorch
- EasyOCR
- Flask

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo Pull Request hoặc mở Issue để thảo luận.

## Liên hệ hỗ trợ

Để kích hoạt lại API hoặc được hỗ trợ kỹ thuật, vui lòng liên hệ:
- Email: dungnguyen.workspace@gmail.com
- GitHub: [@marknguyen02](https://github.com/marknguyenn02)