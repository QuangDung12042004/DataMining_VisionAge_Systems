import os
import time
import requests
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS

# Cấu hình dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# Định nghĩa các nhóm tuổi và từ khóa tìm kiếm tiếng Anh để có kết quả tốt nhất
AGE_GROUPS = {
    "child": ["toddler face closeup", "baby face portrait"],
    "teen": ["teenager face portrait", "high school student face"],
    "adult": ["middle aged man face", "middle aged woman face"],
    "elderly": ["very old face portrait", "wrinkled face elderly", "senior citizen face"]
}
# Tăng số lượng lên để AI thông minh hơn
IMAGES_PER_GROUP = 100  # Số lượng ảnh cần tải cho mỗi nhóm

def create_directories():
    """Tạo cấu trúc thư mục chứa dữ liệu"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for group in AGE_GROUPS.keys():
        group_dir = os.path.join(DATA_DIR, group)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

def download_image(url, save_path):
    """Tải một hình ảnh từ URL và lưu vào đĩa"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Tắt verify SSL tạm thời nếu gặp lỗi certificate của một số trang web
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Lỗi khi tải {url[:50]}...: {e}")
        return False

def collect_data():
    """Hàm chính để thu thập hình ảnh theo từ khóa"""
    create_directories()
    
    # Ẩn cảnh báo InsecureRequestWarning khi verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("Khởi tạo tìm kiếm DuckDuckGo...")
    for group, keywords in AGE_GROUPS.items():
        print(f"\n[{group.upper()}] Bắt đầu thu thập dữ liệu...")
        group_dir = os.path.join(DATA_DIR, group)
        downloaded = len([f for f in os.listdir(group_dir) if f.endswith('.jpg')]) if os.path.exists(group_dir) else 0
        
        for keyword in keywords:
            if downloaded >= IMAGES_PER_GROUP:
                break
                
            print(f"  Tìm kiếm từ khóa: '{keyword}'")
            
            try:
                # Khởi tạo DDGS mới mỗi lần tìm kiếm để hạn chế Rate Limit
                with DDGS() as ddgs:
                    results = ddgs.images(
                        keywords=keyword,
                        region="wt-wt",
                        safesearch="moderate",
                        size="Medium",
                        type_image="photo",
                        max_results=IMAGES_PER_GROUP
                    )
                    
                    found_any = False
                    for result in results:
                        found_any = True
                        if downloaded >= IMAGES_PER_GROUP:
                            break
                            
                        image_url = result.get("image")
                        if not image_url:
                            continue
                            
                        file_name = f"{group}_{downloaded+1:03d}.jpg"
                        save_path = os.path.join(group_dir, file_name)
                        
                        if os.path.exists(save_path):
                            continue
                            
                        print(f"    Tải {file_name} từ {image_url[:50]}...")
                        success = download_image(image_url, save_path)
                        
                        if success:
                            downloaded += 1
                        
                        time.sleep(0.5)
                        
                    if not found_any:
                        print(f"    [!] Không tìm thấy ảnh (có thể do Rate Limit ẩn từ DuckDuckGo). Đang chờ 15 giây...")
                        time.sleep(15)
                        
            except Exception as e:
                print(f"Lỗi truy vấn với {keyword}: {e}")
                print("Đang bị Rate Limit, tạm thời chờ 15 giây trước khi tiếp tục...")
                time.sleep(15)
                
        print(f"[{group.upper()}] Đã tải xong {downloaded}/{IMAGES_PER_GROUP} ảnh.")

if __name__ == "__main__":
    print("Bắt đầu thu thập dữ liệu khuôn mặt...")
    collect_data()
    print("Hoàn tất thu thập dữ liệu!")
