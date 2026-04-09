import os
import shutil
from datetime import datetime
from PIL import Image
from bing_image_downloader import downloader
from minio import Minio
from pymongo import MongoClient
from dotenv import load_dotenv

# Cấu hình dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# Load biến môi trường từ file .env (nếu có)
load_dotenv()

# Cấu hình MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "visionage-dataset")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1", "yes")

# Khởi tạo client MinIO
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    # Kiểm tra bucket và tạo nếu chưa có
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
    USE_MINIO = True
    print(f"Đã cấu hình MinIO. Ảnh sẽ được tự động lưu lên bucket: '{MINIO_BUCKET}'")
except Exception as e:
    USE_MINIO = False
    print(f"Lỗi khởi tạo MinIO, hệ thống sẽ chỉ lưu nội bộ trên máy. Lỗi: {e}")

# Cấu hình MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:adminpassword@localhost:27017/")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping') # Kiểm tra kết nối
    mongo_db = mongo_client["visionage_db"]
    images_collection = mongo_db["images"]
    USE_MONGO = True
    print("Đã cấu hình kết nối MongoDB thành công.")
except Exception as e:
    USE_MONGO = False
    print(f"Lỗi khởi tạo MongoDB, metadata sẽ không được lưu. Lỗi: {e}")

# Định nghĩa các nhóm tuổi và từ khóa tìm kiếm tiếng Anh để có kết quả tốt nhất
AGE_GROUPS = {
    "child": ["toddler face closeup", "baby face portrait"],
    "teen": ["teenager face portrait", "high school student face"],
    "adult": ["middle aged man face", "middle aged woman face"],
    "elderly": ["very old face portrait", "wrinkled face elderly", "senior citizen face"]
}
# Tăng số lượng lên để AI thông minh hơn
IMAGES_PER_GROUP = 50  # Hạ xuống 50 để tải nhanh hơn cho mạng của bạn

def create_directories():
    """Tạo cấu trúc thư mục chứa dữ liệu"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for group in AGE_GROUPS.keys():
        group_dir = os.path.join(DATA_DIR, group)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

def collect_data():
    """Hàm chính để thu thập hình ảnh theo từ khóa dùng Bing"""
    create_directories()
    
    print("Khởi tạo tìm kiếm Bing Image Downloader...")
    for group, keywords in AGE_GROUPS.items():
        print(f"\n[{group.upper()}] Bắt đầu thu thập dữ liệu...")
        group_dir = os.path.join(DATA_DIR, group)
        
        downloaded = len([f for f in os.listdir(group_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(group_dir) else 0
        
        for keyword in keywords:
            if downloaded >= IMAGES_PER_GROUP:
                break
                
            print(f"  Tìm kiếm từ khóa: '{keyword}'")
            try:
                # Tải ảnh qua bing
                downloader.download(
                    keyword, 
                    limit=IMAGES_PER_GROUP - downloaded, 
                    output_dir=group_dir, 
                    adult_filter_off=False, 
                    force_replace=False, 
                    timeout=20, # Giảm timeout xuống để nếu lỗi nhảy luôn sang ảnh khác
                    verbose=True # Hiện chi tiết để bạn thấy nó đang tiếp tục
                )
                
                # Di chuyển và đổi tên các ảnh ra ngoài thư mục group
                kw_dir = os.path.join(group_dir, keyword)
                if os.path.exists(kw_dir):
                    for filename in os.listdir(kw_dir):
                        if downloaded >= IMAGES_PER_GROUP:
                            break
                        
                        src = os.path.join(kw_dir, filename)
                        
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            new_name = f"{group}_{downloaded+1:03d}.jpg"
                            dst = os.path.join(group_dir, new_name)
                            
                            try:
                                img = Image.open(src)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                img.save(dst, "JPEG")
                                
                                # Lấy thông tin dung lượng và độ phân giải
                                try:
                                    file_size_kb = round(os.path.getsize(dst) / 1024, 2)
                                    width, height = img.size
                                    resolution = f"{width}x{height}"
                                except Exception:
                                    file_size_kb = 0
                                    resolution = "unknown"

                                minio_path = ""
                                upload_status = "local_only"

                                # Tải ảnh lên MinIO sau khi lưu xong trên máy
                                if USE_MINIO:
                                    object_name = f"raw/{group}/{new_name}"
                                    try:
                                        minio_client.fput_object(
                                            MINIO_BUCKET, 
                                            object_name, 
                                            dst,
                                            content_type="image/jpeg"
                                        )
                                        minio_path = object_name
                                        upload_status = "success"
                                    except Exception as e:
                                        print(f"    [!] Lỗi upload MinIO với {new_name}: {e}")
                                        upload_status = "minio_error"

                                # Lưu metadata vào MongoDB
                                if USE_MONGO:
                                    metadata = {
                                        "filename": new_name,
                                        "age_group": group,
                                        "search_keyword": keyword,
                                        "source": "bing_image_downloader",
                                        "minio_path": minio_path,
                                        "upload_status": upload_status,
                                        "file_size_kb": file_size_kb,
                                        "resolution": resolution,
                                        "image_format": img.format if img.format else "JPEG",
                                        "created_at": datetime.now()
                                    }
                                    try:
                                        images_collection.insert_one(metadata)
                                    except Exception as e:
                                        print(f"    [!] Lỗi lưu metadata vào MongoDB: {e}")

                                downloaded += 1
                            except Exception as e:
                                pass
                                
                        try:
                            os.remove(src)
                        except:
                            pass
                            
                    try:
                        shutil.rmtree(kw_dir)
                    except:
                        pass
                        
            except Exception as e:
                print(f"    [!] Lỗi truy vấn với {keyword}: {e}")
                
        print(f"[{group.upper()}] Đã tải xong {downloaded}/{IMAGES_PER_GROUP} ảnh.")

if __name__ == "__main__":
    print("Bắt đầu thu thập dữ liệu khuôn mặt...")
    collect_data()
    print("Hoàn tất thu thập dữ liệu!")
