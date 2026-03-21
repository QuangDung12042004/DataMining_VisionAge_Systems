import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Cấu hình dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.h5")

# Params
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Batch nhỏ để phù hợp bộ nhớ
EPOCHS = 20

def create_model(num_classes):
    """Xây dựng mô hình dựa trên MobileNetV2"""
    # Load mô hình MobileNetV2 pre-trained trên ImageNet, bỏ lớp Fully Connected cuối
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Đóng băng các lớp của base_model
    base_model.trainable = False
    
    # Thêm các lớp phân loại mới
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train():
    """Tiến hành huấn luyện mô hình"""
    if not os.path.exists(PROCESSED_DIR):
        print(f"Không tìm thấy thư mục {PROCESSED_DIR}. Vui lòng chạy face_cropper.py trước.")
        return
        
    print("Khởi tạo Data Generator với Data Augmentation...")
    
    # Sử dụng ImageDataGenerator để tăng cường dữ liệu và chia tập train/val trực tiếp
    datagen = ImageDataGenerator(
        rescale=1./255,          # Chuẩn hóa về [0,1]
        rotation_range=20,       # Xoay
        width_shift_range=0.2,   # Dịch ngang
        height_shift_range=0.2,  # Dịch dọc
        horizontal_flip=True,    # Lật ngang
        validation_split=0.2     # Chia 20% cho tập Validation
    )

    print("Load tập Training...")
    train_generator = datagen.flow_from_directory(
        PROCESSED_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    print("Load tập Validation...")
    validation_generator = datagen.flow_from_directory(
        PROCESSED_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Lấy số lượng class tự động dựa trên thư mục
    num_classes = len(train_generator.class_indices)
    print(f"Phát hiện {num_classes} nhóm tuổi: {train_generator.class_indices}")
    
    if num_classes == 0:
        print("Không có ảnh trong thư mục processed!")
        return

    model = create_model(num_classes)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Bắt đầu huấn luyện...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    print(f"Huấn luyện hoàn tất. Mô hình tốt nhất đã được lưu tại {MODEL_PATH}")

if __name__ == "__main__":
    train()
