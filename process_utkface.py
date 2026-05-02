import cv2
import os
import random

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_FOLDER = r"C:\Users\MSI\Desktop\DataMining\DataMining_VisionAge_Systems\UTKFace"
OUTPUT_BASE = r"C:\Users\MSI\Desktop\DataMining\DataMining_VisionAge_Systems\dataset\processed"

# --- THÔNG SỐ LỌC & RESIZE ---
TARGET_SIZE = (200, 200)  
BLUR_THRESHOLD = 60       
DARK_THRESHOLD = 20       
BRIGHT_THRESHOLD = 230    
MAX_IMAGES_PER_GROUP = 3000

def get_age_group(age):
    if age <= 12:
        return "Child"
    elif age <= 26:
        return "Teen"
    elif age <= 59:
        return "Adult"
    else:
        return "Elderly"

def process_utkface():
    print(f"🚀 BẮT ĐẦU LẤY ẢNH TỪ UTKFACE, LỌC CHẤT LƯỢNG VÀ PHÂN LOẠI VÀO: {OUTPUT_BASE}\n")

    # Tạo các thư mục đích nếu chưa có
    categories = ["Child", "Teen", "Adult", "Elderly"]
    for cat in categories:
        cat_path = os.path.join(OUTPUT_BASE, cat)
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)

    if not os.path.exists(INPUT_FOLDER):
        print(f"⚠️ Lỗi: Không tìm thấy thư mục {INPUT_FOLDER}")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files) # Xáo trộn ảnh để không bị dồn cục vào các độ tuổi nhỏ hoặc lớn nhất
    print(f"Tìm thấy tổng cộng {len(files)} ảnh trong thư mục gốc. Đang xử lý...\n")

    total_good = 0
    total_bad = 0
    counts = {"Child": 0, "Teen": 0, "Adult": 0, "Elderly": 0}

    for filename in files:
        # Nếu tất cả các nhóm đều đã đủ 3000 ảnh thì dừng vòng lặp
        if all(count >= MAX_IMAGES_PER_GROUP for count in counts.values()):
            print(f"\n✅ Đã thu thập đủ tối đa {MAX_IMAGES_PER_GROUP} ảnh cho MỖI nhóm tuổi. Kết thúc sớm!")
            break

        # Tách lấy tuổi từ tên file (VD: '24_0_1_2017...jpg' -> tuổi = 24)
        try:
            age = int(filename.split('_')[0])
        except ValueError:
            # Bỏ qua các file không đúng định dạng
            continue
            
        group = get_age_group(age)
        
        # Nếu nhóm này đã đạt số lượng tối đa thì bỏ qua ảnh này
        if counts[group] >= MAX_IMAGES_PER_GROUP:
            continue
        
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            total_bad += 1
            continue

        # Đánh giá chất lượng ảnh (độ mờ, độ sáng)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = gray.mean()

        # Bỏ qua nếu ảnh quá mờ, quá tối, hoặc quá sáng
        if fm < BLUR_THRESHOLD or mean_brightness < DARK_THRESHOLD or mean_brightness > BRIGHT_THRESHOLD:
            total_bad += 1
            continue 

        # Resize ảnh về kích thước chuẩn
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Xử lý tên file, cắt bỏ các đuôi phụ như '.chip.jpg'
        base_name = filename.split('.')[0]
        new_filename = f"{base_name}.png"
        
        # Đường dẫn lưu file mới
        dest_path = os.path.join(OUTPUT_BASE, group, new_filename)
        
        # Ghi ảnh
        cv2.imwrite(dest_path, resized_img)
        total_good += 1
        counts[group] += 1

        if total_good % 1000 == 0:
            print(f"Đã xử lý thành công {total_good} ảnh đạt chuẩn...")

    print("-" * 50)
    print("🎉 HOÀN TẤT LỌC VÀ PHÂN LỚP ẢNH TỪ UTKFACE!")
    print(f"👉 Tổng số ảnh ĐẠT CHUẨN: {total_good} tấm")
    for group in categories:
        print(f"    - {group}: {counts[group]} tấm")
    print(f"👉 Tổng số ảnh BỊ LOẠI (chất lượng kém/lỗi): {total_bad} tấm")

if __name__ == "__main__":
    process_utkface()
