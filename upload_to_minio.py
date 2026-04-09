import os
from datetime import datetime
from minio import Minio
from pymongo import MongoClient
import pymongo
from dotenv import load_dotenv

# Load cấu hình từ file .env
load_dotenv()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "visionage-dataset")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1", "yes")

def sync_dataset_to_minio_and_mongo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Đường dẫn trỏ tới gốc thư mục dataset (chứa cả raw và processed)
    dataset_dir = os.path.join(base_dir, "dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Thư mục {dataset_dir} không tồn tại!")
        return

    # Khởi tạo MinIO client
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        print(f"Lỗi kết nối MinIO: {e}")
        return

    # Khởi tạo MongoDB client
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:adminpassword@localhost:27017/")
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        mongo_db = mongo_client["visionage_db"]
        images_collection = mongo_db["images"]
        # Tạo index để tìm kiếm và cập nhật nhanh hơn (tránh trùng lặp)
        try:
            images_collection.create_index([("minio_path", pymongo.ASCENDING)], unique=True)
        except:
            pass
        USE_MONGO = True
    except Exception as e:
        USE_MONGO = False
        print(f"Lỗi kết nối MongoDB: {e}")

    print("Bắt đầu quét thư mục dataset (raw & processed) và đồng bộ hệ thống...")
    count = 0
    updated_count = 0

    # Quét tất cả các file có trong thư mục dataset/
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Tạo đường dẫn (Object Name) trong MinIO
                # Ví dụ: dataset/raw/child/child_001.jpg -> raw/child/child_001.jpg
                rel_path = os.path.relpath(file_path, dataset_dir)
                object_name = rel_path.replace("\\", "/") # MinIO chuẩn đường dẫn Unix
                
                # Tải file lên MinIO (overwrite nếu đã tồn tại)
                try:
                    client.fput_object(
                        MINIO_BUCKET, 
                        object_name, 
                        file_path,
                        content_type="image/jpeg"
                    )
                    count += 1
                    upload_status = "success"
                except Exception as e:
                    print(f"[!] Lỗi upload MinIO với file {file}: {e}")
                    upload_status = "error"
                
                # Cập nhật thông tin vào MongoDB
                if USE_MONGO and upload_status == "success":
                    try:
                        file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
                        
                        from PIL import Image
                        try:
                            with Image.open(file_path) as img:
                                width, height = img.size
                                resolution = f"{width}x{height}"
                                image_format = img.format if img.format else "JPEG"
                        except Exception:
                            resolution = "unknown"
                            image_format = "JPEG"

                        # Phân loại theo cấu trúc (VD: raw/child/..., processed/child/...)
                        parts = object_name.split("/")
                        source_folder = parts[0] if len(parts) > 0 else "unknown" # 'raw' hoặc 'processed'
                        age_group = parts[1] if len(parts) > 1 else "unknown" # 'child', 'teen', ...

                        metadata = {
                            "filename": file,
                            "age_group": age_group,
                            "source_type": source_folder,  # raw / processed
                            "minio_path": object_name,
                            "upload_status": upload_status,
                            "file_size_kb": file_size_kb,
                            "resolution": resolution,
                            "image_format": image_format,
                            "last_updated": datetime.now()
                        }

                        # Upsert: Nếu chưa có thì thêm mới, đã có thì cập nhật để không bị trùng lặp
                        result = images_collection.update_one(
                            {"minio_path": object_name}, 
                            {"$set": metadata},
                            upsert=True
                        )
                        
                        if result.upserted_id:
                            print(f"[+] [Thêm Mới] {object_name}")
                        elif result.modified_count > 0:
                            print(f"[*] [Cập Nhật] {object_name}")
                            updated_count += 1
                            
                    except Exception as e:
                        print(f"    [!] Lỗi lưu metadata vào MongoDB với {file}: {e}")
                    
    print(f"\n{'='*50}")
    print(f"Hoàn tất! Đã đồng bộ {count} ảnh (Raw & Processed) lên '{MINIO_BUCKET}'.")
    if USE_MONGO:
        print(f"Đã cập nhật (ghi đè data cũ) {updated_count} bản ghi trong Database.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    sync_dataset_to_minio_and_mongo()
