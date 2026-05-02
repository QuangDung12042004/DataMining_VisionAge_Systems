import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import cv2
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset", "unclassified_faces_ready")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "auto_classified")
MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "age_class_indices.json")

# Do bạn dùng cả MobileNetV2 và EfficientNetB0 trong quá trình làm, 
# ta sẽ thử EfficientNetB0 vì file train_model.py đang cấu hình BACKBONE = "efficientnetb0"
BACKBONE = "efficientnetb0" 

def preprocess_image(image_path):
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    if BACKBONE == "efficientnetb0":
        img_array = efficientnet_preprocess_input(img_array)
    else:
        img_array = mobilenet_preprocess_input(img_array)
        
    return np.expand_dims(img_array, axis=0)

def auto_classify_images():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Thư mục {INPUT_DIR} không tồn tại!")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {MODEL_PATH}!")
        return

    print("⏳ Đang load model AI của bạn... Hãy đợi một chút.")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
        
    # Tạo map ngược từ index -> tên class (ví dụ: 0 -> "adult")
    idx_to_class = {v: k for k, v in class_map.items()}
    
    print(f"✅ Đã load model thành công. Các nhóm tuổi nhận diện: {list(idx_to_class.values())}")
    
    # Tạo các thư mục con trong output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for class_name in idx_to_class.values():
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)
        
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if len(image_files) == 0:
        print(f"⚠️ Không có ảnh nào trong {INPUT_DIR} để phân loại!")
        return
        
    print(f"\n🚀 Bắt đầu phân loại tự động {len(image_files)} ảnh...")
    stats = {k: 0 for k in idx_to_class.values()}
    stats["errors"] = 0
    
    for i, file_name in enumerate(image_files):
        img_path = os.path.join(INPUT_DIR, file_name)
        img_array = preprocess_image(img_path)
        
        if img_array is None:
            stats["errors"] += 1
            continue
            
        # Dự đoán
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = idx_to_class[predicted_idx]
        confidence = predictions[0][predicted_idx]
        
        # Di chuyển ảnh vào thư mục tương ứng
        # Có thể thêm tiền tố confidence vào tên file để dễ phân tích
        new_file_name = f"{confidence:.2f}_{file_name}"
        dest_path = os.path.join(OUTPUT_DIR, predicted_class, new_file_name)
        
        shutil.move(img_path, dest_path)
        stats[predicted_class] += 1
        
        if (i + 1) % 50 == 0:
            print(f"Đã xử lý {i + 1}/{len(image_files)} ảnh...")

    print("\n" + "="*50)
    print("📈 KẾT QUẢ PHÂN LOẠI TỰ ĐỘNG BẰNG AI")
    print("="*50)
    for class_name, count in stats.items():
        if class_name != "errors":
            print(f"- {class_name.capitalize()}: {count} ảnh")
    print(f"- Lỗi không đọc được ảnh: {stats['errors']} ảnh")
    print("="*50)
    print(f"🎉 Hoàn tất! Ảnh đã được chia vào các thư mục trong: {OUTPUT_DIR}")
    print(f"👉 Bạn hãy kiểm tra lại bằng mắt thường, sau đó copy ảnh sang dataset/raw để train tiếp nhé!")

if __name__ == "__main__":
    auto_classify_images()
