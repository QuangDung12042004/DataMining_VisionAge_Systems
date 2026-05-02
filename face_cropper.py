import os
import cv2
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "raw")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "dataset", "intermediate")

TARGET_SIZE = (200, 200)
VALID_EXT = (".png", )

# =========================================

detector = MTCNN()
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# ================= CLEANING =================
def is_valid_image(img):
    if img is None:
        return False
    h, w = img.shape[:2]
    return h > 80 and w > 80


# 🔥 FIX 1: resize ảnh lớn (tránh OOM)
def resize_if_too_large(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# 🔥 FIX 2: detect an toàn
def detect_single_face(img):
    try:
        results = detector.detect_faces(img)
    except Exception:
        return None

    if len(results) != 1:
        return None

    x, y, w, h = results[0]["box"]

    # lọc mặt quá nhỏ
    if w < 40 or h < 40:
        return None

    return results[0]["box"]


# ================= TRANSFORMATION =================
def crop_head_shoulder(img, box):
    x, y, w, h = box

    # 🔥 FIX 3: tránh bbox âm / lệch
    x = max(0, x)
    y = max(0, y)

    y1 = max(0, y - int(0.2 * h))
    y2 = min(img.shape[0], y + int(1.6 * h))
    x1 = max(0, x - int(0.2 * w))
    x2 = min(img.shape[1], x + int(1.2 * w))

    return img[y1:y2, x1:x2]


# ================= ENCODING =================
def get_embedding(img):
    img = cv2.resize(img, TARGET_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    emb = resnet.predict(arr, verbose=0)
    return emb[0]


# ================= MAIN =================
def process():
    print("RAW_DIR =", RAW_DIR)
    print("Tồn tại:", os.path.exists(RAW_DIR))

    all_images = []

    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                path = os.path.join(root, f)
                source = os.path.basename(root)
                all_images.append((path, source))

    print(f"Tổng ảnh: {len(all_images)}")

    count = 0

    for path, source in tqdm(all_images):
        img = cv2.imread(path)

        # CLEANING
        if not is_valid_image(img):
            continue

        # 🔥 FIX 4: bỏ ảnh quá lớn nguy hiểm
        if img.shape[0] * img.shape[1] > 2000 * 2000:
            continue

        # 🔥 resize trước khi detect
        img = resize_if_too_large(img)

        face_box = detect_single_face(img)
        if face_box is None:
            continue

        # TRANSFORMATION
        face = crop_head_shoulder(img, face_box)

        if face is None or face.size == 0:
            continue

        # NORMALIZATION
        face = cv2.resize(face, TARGET_SIZE)

        # 🔥 OPTIONAL: nếu bị chậm thì comment dòng này
        try:
            _ = get_embedding(face)
        except Exception:
            continue

        # save
        save_dir = os.path.join(INTERMEDIATE_DIR, source)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(save_path, face)

        count += 1

    print(f"Đã xử lý: {count} ảnh")


if __name__ == "__main__":
    process()