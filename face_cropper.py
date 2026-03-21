import cv2
import os
import glob
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# Khởi tạo mô hình Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_processed_directories():
    """Tạo cấu trúc thư mục chứa dữ liệu đã xử lý"""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    for group in os.listdir(RAW_DIR):
        group_path = os.path.join(RAW_DIR, group)
        if os.path.isdir(group_path):
            processed_group_dir = os.path.join(PROCESSED_DIR, group)
            if not os.path.exists(processed_group_dir):
                os.makedirs(processed_group_dir)

def process_images():
    """Đọc ảnh raw, cắt khuôn mặt và lưu vào thư mục processed"""
    if not os.path.exists(RAW_DIR):
        print(f"Thư mục chứa dữ liệu raw không tồn tại: {RAW_DIR}")
        return
        
    create_processed_directories()
    
    total_processed = 0
    total_faces_found = 0
    
    for group in os.listdir(RAW_DIR):
        group_path = os.path.join(RAW_DIR, group)
        if not os.path.isdir(group_path):
            continue
            
        processed_group_dir = os.path.join(PROCESSED_DIR, group)
        print(f"\n--- Xử lý nhóm tuổi: {group.upper()} ---")
        
        image_files = glob.glob(os.path.join(group_path, "*.jpg"))
        faces_in_group = 0
        
        for img_path in image_files:
            file_name = os.path.basename(img_path)
            
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            total_processed += 1
            
            # Chuyển sang ảnh xám để nhận diện tốt hơn
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Nhận diện khuôn mặt
            faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05, # Giảm xuống để quét kỹ hơn
    minNeighbors=8,    # Tăng lên để tránh bắt nhầm đối tượng rác
    minSize=(60, 60)   # Tăng size tối thiểu để lấy ảnh chất lượng
)
            
            # Xử lý từng khuôn mặt (thường lấy khuôn mặt to nhất)
            if len(faces) > 0:
                # Sắp xếp faces theo diện tích (w*h) giảm dần để lấy khuôn mặt lớn nhất
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                # Mở rộng vùng mặt (Tăng margin lên 30% theo yêu cầu của thầy để không mất chi tiết tóc/cổ)
                margin_x = int(w * 0.3)
                margin_y = int(h * 0.3)
                
                x_start = max(0, x - margin_x)
                y_start = max(0, y - margin_y)
                x_end = min(img.shape[1], x + w + margin_x)
                y_end = min(img.shape[0], y + h + margin_y)
                
                # Cắt ảnh
                face_img = img[y_start:y_end, x_start:x_end]
                
                try:
                    # Resize về chuẩn 224x224 (kích thước MobileNetV2)
                    face_resized = cv2.resize(face_img, (224, 224))
                    
                    # Lưu ảnh
                    save_path = os.path.join(processed_group_dir, file_name)
                    cv2.imwrite(save_path, face_resized)
                    faces_in_group += 1
                    total_faces_found += 1
                    
                except Exception as e:
                    print(f"Lỗi resize ảnh {file_name}: {e}")
                    
        print(f"[{group.upper()}] Tìm thấy và cắt {faces_in_group}/{len(image_files)} khuôn mặt.")
        
    print(f"\n--- TỔNG KẾT ---")
    print(f"Đã xử lý: {total_processed} ảnh gốc.")
    print(f"Số khuôn mặt trích xuất thành công: {total_faces_found}")

if __name__ == "__main__":
    print("Bắt đầu tiền xử lý và cắt khuôn mặt...")
    process_images()
    print("Hoàn tất tiền xử lý dữ liệu!")
