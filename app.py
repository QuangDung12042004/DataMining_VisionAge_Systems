import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

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
MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
CLASS_NAMES = {0: 'Adult (Trưởng thành)', 1: 'Child (Trẻ em)', 2: 'Elderly (Người già)', 3: 'Teen (Thiếu niên)'}

@st.cache_resource
def load_classifier():
    if os.path.exists(MODEL_PATH):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return tf.keras.models.load_model(MODEL_PATH)
    return None

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
                input_arr = np.expand_dims(face_resized, axis=0) / 255.0
                predictions = model.predict(input_arr)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100

                # Hiển thị kết quả chính
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">{CLASS_NAMES[predicted_class]}</div>
                    <div style="color:#8b949e">Độ tin cậy: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Hiển thị biểu đồ xác suất các nhóm khác
                st.write("---")
                st.write("📊 Chi tiết phân tích:")
                chart_data = pd.DataFrame(predictions, index=CLASS_NAMES.values(), columns=["Xác suất"])
                st.bar_chart(chart_data)