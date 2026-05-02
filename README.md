# DataMining VisionAge Systems - Hướng Dẫn Cài Đặt Và Chạy Dự Án

Tài liệu này hướng dẫn cách chạy dự án sau khi bạn clone/pull mã nguồn từ GitHub về máy tính, cũng như cách khởi chạy các dịch vụ (MinIO, MongoDB) bằng Docker chứa các image (ảnh) có sẵn.

## 1. Yêu cầu hệ thống (Prerequisites)
- **Python** (phiên bản 3.8 trở lên).
- **Docker** và **Docker Compose** đã được cài đặt và đang chạy trên hệ thống.
- **Git** để pull mã nguồn.

---

## 2. Các bước thực hiện

### Bước 1: Clone / Pull dự án từ GitHub
Mở Terminal / Command Prompt và chạy lệnh:
```bash
git pull origin main
```
*(Nếu bạn chưa clone dự án thì dùng lệnh `git clone <link-repo-cua-ban>` và `cd DataMining_VisionAge_Systems`)*

### Bước 2: Cài đặt các thư viện Python cần thiết
Khuyến nghị bạn nên tạo một môi trường ảo (virtual environment) trước khi cài đặt để tránh xung đột thư viện:
```bash
# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

### Bước 3: Khởi chạy MinIO và MongoDB từ Docker
Dự án có sẵn file `docker-compose.yml`. Khi bạn chạy lệnh dưới đây, Docker sẽ sử dụng các image Docker (như `minio/minio` và `mongo:latest`) đã lưu sẵn trên máy bạn (hoặc tự động tải về nếu chưa có), và ánh xạ dữ liệu trực tiếp vào thư mục `./docker_data/` để giữ lại toàn bộ database và file đã được upload trước đó.

Từ thư mục gốc dự án, hãy chạy:
```bash
docker-compose up -d
```

**Cách kiểm tra dịch vụ đã hoạt động hay chưa:**
- **MinIO (Quản lý file ảnh):** Truy cập vào trình duyệt bằng đường dẫn `http://localhost:9001`
  - User: `minioadmin`
  - Pass: `minioadmin`
- **MongoDB (Quản lý Database Metadata):** Sẽ chạy ngầm ở cổng kết nối `localhost:27017` (có thể truy cập bằng MongoDB Compass).

*Lưu ý: Nếu bạn muốn tắt các dịch vụ này, hãy dùng lệnh `docker-compose down`.*

### Bước 4: Chạy các file thực thi (Scripts)
Tùy thuộc vào bước tiến độ của bạn, bạn có thể thực thi các file code của dự án. Ví dụ quan trọng nhất là file đồng bộ ảnh và xử lý dữ liệu:

**1. Thu thập ảnh mới:**
```bash
python data_collector.py
```

**2. Tiền xử lý (Cắt gọn khuôn mặt trong ảnh):**
```bash
python face_cropper.py
```

**3. Đồng bộ hóa ảnh lên MinIO và Metadata lên MongoDB:**
Lệnh dưới đây sẽ đọc toàn bộ thư mục xử lý ảnh, upload ảnh lên dịch vụ MinIO và lưu metadata của ảnh xuống MongoDB để lưu trữ.
```bash
python upload_to_minio.py
```

**4. Phân tích thống kê dữ liệu Dataset:**
```bash
python dataset_stats.py
```

---

## 3. Cấu trúc thư mục chính (Tham khảo)
- `.env` *(Nếu có)*: Chứa các cấu hình environment (Bảo mật).
- `docker-compose.yml`: File chạy Docker cho MinIO và MongoDB.
- `docker_data/`: Thư mục lưu trữ database thực tế tạo bởi Docker.
- `dataset/`: Thư mục chứa dữ liệu ảnh.
- `requirements.txt`: Danh sách các package Python.
