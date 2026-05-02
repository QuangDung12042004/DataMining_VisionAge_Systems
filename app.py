import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

st.set_page_config(page_title="Age Classification AI", page_icon="👤", layout="centered")

# --- GIỮ NGUYÊN PHẦN CSS CUSTOM CỦA BẠN ---
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }
    h1 {
        color: #ffffff; font-family: 'Inter', sans-serif; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(#f9ce34, #ee2a7b, #6228d7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 20px;
    }
    p { color: #c9d1d9; font-family: 'Inter', sans-serif; text-align: center; }
    .result-box {
        padding: 20px; border-radius: 15px; background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1); text-align: center; margin-top: 20px;
    }
    .result-label { color: #58a6ff; font-size: 24px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "age_class_indices.json")
BACKBONE = "efficientnetb0"  # Đồng bộ với train_model.py
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

DISPLAY_NAME_MAP = {
    "adult": "Adult (Trưởng thành)",
    "child": "Child (Trẻ em)",
    "elderly": "Elderly (Người già)",
    "teen": "Teen (Thiếu niên)",
    "young_adults": "Young Adults (Thanh niên)"
}

def get_preprocess_fn(backbone_name):
    name = backbone_name.lower()
    if name == "mobilenetv2":
        return mobilenet_preprocess_input
    if name == "efficientnetb0":
        return efficientnet_preprocess_input
    raise ValueError(f"BACKBONE không hợp lệ: {backbone_name}")

def load_class_names():
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        max_idx = max(idx_to_class.keys())
        return [idx_to_class[i] for i in range(max_idx + 1)]
    # Fallback để app vẫn chạy nếu chưa có class map mới
    return ["adult", "child", "elderly", "teen", "young_adults"]

@st.cache_resource
def load_classifier():
    if os.path.exists(MODEL_PATH):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return tf.keras.models.load_model(MODEL_PATH)
    return None

class_names = load_class_names()
display_class_names = [DISPLAY_NAME_MAP.get(name, name) for name in class_names]
preprocess_fn = get_preprocess_fn(BACKBONE)
model = load_classifier()

st.markdown("<h1>Age Classification Vision AI</h1>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Không tìm thấy mô hình! Hãy chạy `train_model.py` trước.")
else:
    # Lựa chọn nguồn ảnh: Tải lên hoặc Webcam
    source = st.radio("Chọn nguồn ảnh:", ("Tải ảnh lên", "Sử dụng Webcam"), horizontal=True)
    
    img_input = None
    if source == "Tải ảnh lên":
        uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_input = Image.open(uploaded_file).convert('RGB')
    else:
        cam_file = st.camera_input("Chụp ảnh từ Webcam")
        if cam_file:
            img_input = Image.open(cam_file).convert('RGB')

    if img_input is not None:
        col1, col2 = st.columns(2)
        img_array = np.array(img_input)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Tăng minNeighbors lên 8 để nhận diện khuôn mặt chuẩn hơn (giảm bắt nhầm)
        faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(50, 50))

        with col1:
            st.image(img_input, caption="🖼️ Ảnh đầu vào", use_container_width=True)

        with col2:
            if len(faces) == 0:
                st.warning("⚠️ Không tìm thấy khuôn mặt rõ ràng!")
            else:
                (x, y, w, h) = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                # Cắt mặt có thêm chút lề (margin)
                face_img = img_array[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
                face_resized = cv2.resize(face_img, (224, 224))
                
                st.image(face_img, caption="👤 Khuôn mặt phân tích", use_container_width=True)
                
                # Dự đoán
                input_arr = np.expand_dims(face_resized.astype(np.float32), axis=0)
                input_arr = preprocess_fn(input_arr)
                predictions = model.predict(input_arr)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100

                if predicted_class >= len(display_class_names):
                    st.error("⚠️ Số lớp của model không khớp class map. Hãy train lại để đồng bộ.")
                    st.stop()

                # Hiển thị kết quả chính
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">{display_class_names[predicted_class]}</div>
                    <div style="color:#8b949e">Độ tin cậy: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Hiển thị biểu đồ xác suất các nhóm khác
                st.write("---")
                st.write("📊 Chi tiết phân tích:")
                chart_data = pd.DataFrame(predictions[:len(display_class_names)], index=display_class_names, columns=["Xác suất"])
                st.bar_chart(chart_data)