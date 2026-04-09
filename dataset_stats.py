import os
import matplotlib.pyplot as plt

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

def count_images(directory):
    counts = {}
    if os.path.exists(directory):
        for group in os.listdir(directory):
            group_path = os.path.join(directory, group)
            if os.path.isdir(group_path):
                img_count = len([f for f in os.listdir(group_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                counts[group] = img_count
    return counts

def plot_stats(raw_counts, processed_counts):
    # Tổng hợp các nhóm tuổi
    groups = list(set(raw_counts.keys()).union(processed_counts.keys()))
    groups.sort() # Sắp xếp alpha-bê
    
    raw_vals = [raw_counts.get(g, 0) for g in groups]
    processed_vals = [processed_counts.get(g, 0) for g in groups]
    
    x = range(len(groups))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([p - width/2 for p in x], raw_vals, width, label='Ảnh Gốc (Raw)', color='#58a6ff')
    rects2 = ax.bar([p + width/2 for p in x], processed_vals, width, label='Đã cắt Face (Processed)', color='#3fb950')
    
    ax.set_ylabel('Số lượng (Tấm)')
    ax.set_title('BIỂU ĐỒ THỐNG KÊ SỐ LƯỢNG ẢNH THEO NHÓM TUỔI')
    ax.set_xticks(x)
    ax.set_xticklabels([g.upper() for g in groups])
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    chart_path = os.path.join(BASE_DIR, "bieu_do_thong_ke.png")
    plt.savefig(chart_path)
    print(f"\n[+] Đã tự động lưu biểu đồ báo cáo tại: {chart_path}")
    plt.show()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"{'BÁO CÁO THỐNG KÊ DATASET MÔN DATA MINING':^60}")
    print(f"{'='*60}\n")
    
    raw_counts = count_images(RAW_DIR)
    processed_counts = count_images(PROCESSED_DIR)
    
    # Hiển thị bảng trên Terminal
    print(f"{'Nhóm Tuổi':<15} | {'Ảnh Gốc (Raw)':<15} | {'Sau khi cắt mặt (Processed)'}")
    print("-" * 60)
    
    total_raw = 0
    total_processed = 0
    all_groups = list(set(raw_counts.keys()).union(processed_counts.keys()))
    all_groups.sort()
    
    for group in all_groups:
        r_count = raw_counts.get(group, 0)
        p_count = processed_counts.get(group, 0)
        total_raw += r_count
        total_processed += p_count
        print(f"{group.upper():<15} | {r_count:<15} | {p_count}")
        
    print("-" * 60)
    print(f"{'TỔNG CỘNG':<15} | {total_raw:<15} | {total_processed}")
    print(f"{'='*60}")
    
    if total_raw > 0:
        print("\nĐang tạo biểu đồ trực quan...")
        plot_stats(raw_counts, processed_counts)
    else:
        print("\n[!] Chưa có dữ liệu ảnh nào trong thư mục dataset/raw/")
