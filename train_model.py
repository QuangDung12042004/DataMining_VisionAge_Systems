import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')
# Bật Memory Growth cho GPU để tránh OOM trên card 4GB VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Đã bật cấu hình Memory Growth cho GPU.")
    except RuntimeError as e:
        print(e)
# Cấu hình dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "age_class_indices.json")
TEST_METRICS_PATH = os.path.join(BASE_DIR, "test_metrics.json")
CLASSIFICATION_REPORT_PATH = os.path.join(BASE_DIR, "classification_report.json")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.csv")
SPLIT_INFO_PATH = os.path.join(BASE_DIR, "split_info.json")

# Params
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Ổn định hơn cho GTX 1650 4GB VRAM
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
FINE_TUNE_AT = 50
FINE_TUNE_STAGE2_EPOCHS = 8
FINE_TUNE_STAGE3_FREEZE_AT = 20
BACKBONE = "efficientnetb0"  # "mobilenetv2" hoặc "efficientnetb0"
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_CLASS_NAMES = ("adult", "child", "elderly", "teen")
CLASS_NAME_ALIASES = {
    "young_adults": "adult"
}
USE_BALANCED_TRAIN_SAMPLING = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_preprocess_fn(backbone_name):
    """Trả về hàm preprocess theo backbone."""
    name = backbone_name.lower()
    if name == "mobilenetv2":
        return mobilenet_preprocess_input
    if name == "efficientnetb0":
        return efficientnet_preprocess_input
    raise ValueError(f"BACKBONE không hợp lệ: {backbone_name}")

def get_backbone_model(backbone_name):
    """Trả về backbone model theo cấu hình."""
    name = backbone_name.lower()
    if name == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif name == "efficientnetb0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    else:
        raise ValueError(f"BACKBONE không hợp lệ: {backbone_name}")
    return base_model

def create_model(num_classes, backbone_name):
    """Xây dựng mô hình với backbone có thể cấu hình."""
    base_model = get_backbone_model(backbone_name)
    
    # Đóng băng các lớp của base_model
    base_model.trainable = False
    
    # Thêm các lớp phân loại mới
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=3e-4), 
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
                  metrics=['accuracy'])
    return model, base_model

def _normalize_class_name(class_name):
    """Chuẩn hóa tên lớp theo đề tài (ví dụ gộp young_adults vào adult)."""
    name = class_name.lower()
    return CLASS_NAME_ALIASES.get(name, name)

def collect_labeled_images():
    """Quét ảnh và ánh xạ về đúng 4 lớp mục tiêu của đề tài."""
    raw_class_names = sorted(
        [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    )

    images_by_class = {class_name: [] for class_name in TARGET_CLASS_NAMES}
    ignored_dirs = []

    for raw_class_name in raw_class_names:
        normalized_class_name = _normalize_class_name(raw_class_name)
        if normalized_class_name not in images_by_class:
            ignored_dirs.append(raw_class_name)
            continue

        class_dir = os.path.join(PROCESSED_DIR, raw_class_name)
        for file_name in sorted(os.listdir(class_dir)):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(VALID_EXTENSIONS):
                images_by_class[normalized_class_name].append(file_path)

    if ignored_dirs:
        print(f"Bỏ qua các thư mục không thuộc lớp mục tiêu: {sorted(ignored_dirs)}")

    class_names = [class_name for class_name in TARGET_CLASS_NAMES if len(images_by_class[class_name]) > 0]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    image_paths = []
    labels = []

    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        class_image_paths = images_by_class[class_name]
        image_paths.extend(class_image_paths)
        labels.extend([class_idx] * len(class_image_paths))

    return np.array(image_paths), np.array(labels, dtype=np.int32), class_names

def split_dataset(image_paths, labels):
    """Tách train/val/test theo tỉ lệ 70/15/15, ưu tiên stratified split."""
    holdout_ratio = VAL_RATIO + TEST_RATIO
    test_in_holdout = TEST_RATIO / holdout_ratio

    if len(image_paths) < 3:
        raise ValueError("Dataset quá ít mẫu để tách train/validation/test.")

    try:
        train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
            image_paths,
            labels,
            test_size=holdout_ratio,
            random_state=SEED,
            stratify=labels
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            holdout_paths,
            holdout_labels,
            test_size=test_in_holdout,
            random_state=SEED,
            stratify=holdout_labels
        )
    except ValueError as e:
        print(f"Cảnh báo: Không thể stratify đầy đủ ({e}). Chuyển sang random split.")
        try:
            train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
                image_paths,
                labels,
                test_size=holdout_ratio,
                random_state=SEED,
                stratify=None
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                holdout_paths,
                holdout_labels,
                test_size=test_in_holdout,
                random_state=SEED,
                stratify=None
            )
        except ValueError as random_split_error:
            raise ValueError(
                "Không thể tách dataset thành train/validation/test. "
                "Hãy tăng dữ liệu hoặc điều chỉnh tỉ lệ split."
            ) from random_split_error

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }

def load_and_preprocess_image(file_path, label, num_classes, preprocess_fn, training=False):
    """Đọc ảnh từ đĩa, augment nhẹ cho train và chuẩn hóa theo backbone."""
    image_bytes = tf.io.read_file(file_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    image = tf.cast(image, tf.float32)

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.8, 1.2)

        image = tf.clip_by_value(image, 0.0, 255.0)

    image = preprocess_fn(image)
    one_hot_label = tf.one_hot(label, depth=num_classes)
    return image, one_hot_label

def build_tf_dataset(paths, labels, num_classes, preprocess_fn, training=False, use_cache=False):
    """Tạo tf.data pipeline để huấn luyện/đánh giá."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 5000), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, num_classes, preprocess_fn, training=training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if use_cache:
        dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_balanced_train_dataset(paths, labels, num_classes, preprocess_fn):
    """Tạo tập train cân bằng lớp bằng sample_from_datasets."""
    per_class_datasets = []

    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_paths = paths[class_mask]
        class_labels = labels[class_mask]

        if len(class_paths) == 0:
            raise ValueError(f"Không có dữ liệu cho class_id={class_id}; không thể cân bằng lớp.")

        class_dataset = tf.data.Dataset.from_tensor_slices((class_paths, class_labels))
        class_dataset = class_dataset.shuffle(
            buffer_size=min(len(class_paths), 1000),
            seed=SEED,
            reshuffle_each_iteration=True
        ).repeat()
        per_class_datasets.append(class_dataset)

    balanced_dataset = tf.data.Dataset.sample_from_datasets(
        per_class_datasets,
        weights=[1.0 / num_classes] * num_classes,
        seed=SEED
    )
    balanced_dataset = balanced_dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, num_classes, preprocess_fn, training=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    balanced_dataset = balanced_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Do dataset được repeat vô hạn, cần chỉ định số bước mỗi epoch.
    steps_per_epoch = int(np.ceil(len(paths) / BATCH_SIZE))
    return balanced_dataset, steps_per_epoch

def _count_labels(labels, class_names):
    return {class_names[idx]: int(np.sum(labels == idx)) for idx in range(len(class_names))}

def save_split_info(train_labels, val_labels, test_labels, class_names):
    """Lưu thống kê phân bổ mẫu của từng split để dễ báo cáo."""
    info = {
        "ratios": {
            "train": 1.0 - (VAL_RATIO + TEST_RATIO),
            "validation": VAL_RATIO,
            "test": TEST_RATIO
        },
        "counts": {
            "train": _count_labels(train_labels, class_names),
            "validation": _count_labels(val_labels, class_names),
            "test": _count_labels(test_labels, class_names)
        }
    }

    with open(SPLIT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def evaluate_on_test_set(model, test_dataset, test_labels, class_names):
    """Đánh giá tập test và xuất các metric quan trọng."""
    class_ids = np.arange(len(class_names))

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    y_prob = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels,
        y_pred,
        average='macro',
        zero_division=0
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy": float(accuracy_score(test_labels, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro)
    }

    report = classification_report(
        test_labels,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    conf_matrix = confusion_matrix(test_labels, y_pred, labels=class_ids)

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    np.savetxt(CONFUSION_MATRIX_PATH, conf_matrix, fmt="%d", delimiter=",")

    print(f"Đã lưu test metrics tại {TEST_METRICS_PATH}")
    print(f"Đã lưu classification report tại {CLASSIFICATION_REPORT_PATH}")
    print(f"Đã lưu confusion matrix tại {CONFUSION_MATRIX_PATH}")

def train():
    tf.keras.backend.clear_session() # Dọn dẹp đồ đá cũ trong VRAM
    if not os.path.exists(PROCESSED_DIR):
        print(f"Không tìm thấy thư mục {PROCESSED_DIR}. Vui lòng chạy face_cropper.py trước.")
        return


    print("Quét dữ liệu đã cắt và tạo split train/val/test...")
    image_paths, labels, class_names = collect_labeled_images()

    if len(image_paths) == 0:
        print("Không có ảnh hợp lệ trong dataset/processed!")
        return

    num_classes = len(class_names)
    print(f"Phát hiện {num_classes} nhóm tuổi mục tiêu: {class_names}")
    print(f"Phân bổ dữ liệu ban đầu: {_count_labels(labels, class_names)}")

    if num_classes < 2:
        print("Cần tối thiểu 2 lớp để huấn luyện phân loại.")
        return

    splits = split_dataset(image_paths, labels)
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    if len(train_paths) == 0 or len(val_paths) == 0 or len(test_paths) == 0:
        print("Không đủ dữ liệu để tạo đủ train/validation/test.")
        return

    print(f"Train/Val/Test: {len(train_paths)}/{len(val_paths)}/{len(test_paths)} ảnh")

    save_split_info(train_labels, val_labels, test_labels, class_names)
    print(f"Đã lưu thống kê split tại {SPLIT_INFO_PATH}")

    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu class map tại {CLASS_MAP_PATH}")

    preprocess_fn = get_preprocess_fn(BACKBONE)

    train_steps_per_epoch = None
    if USE_BALANCED_TRAIN_SAMPLING:
        print("Bật balanced train sampling để giảm lệch lớp khi huấn luyện...")
        train_dataset, train_steps_per_epoch = build_balanced_train_dataset(
            train_paths,
            train_labels,
            num_classes,
            preprocess_fn
        )
        class_weights_dict = None
    else:
        train_dataset = build_tf_dataset(train_paths, train_labels, num_classes, preprocess_fn, training=True, use_cache=False)

        # Khắc phục mất cân bằng dữ liệu bằng class_weight
        print("Tính toán trọng số cân bằng lớp (class weights)...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_ids = np.unique(train_labels)
        class_weights_dict = {int(class_id): float(weight) for class_id, weight in zip(class_ids, class_weights)}
        print(f"Trọng số áp dụng: {class_weights_dict}")

    validation_dataset = build_tf_dataset(val_paths, val_labels, num_classes, preprocess_fn, training=False, use_cache=True)
    test_dataset = build_tf_dataset(test_paths, test_labels, num_classes, preprocess_fn, training=False, use_cache=True)

    print(f"Sử dụng backbone: {BACKBONE}")
    model, base_model = create_model(num_classes, BACKBONE)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print("Bắt đầu huấn luyện giai đoạn 1 (feature extraction)...")
    fit_kwargs_stage_1 = {
        "x": train_dataset,
        "validation_data": validation_dataset,
        "epochs": INITIAL_EPOCHS,
        "callbacks": [checkpoint, early_stop, reduce_lr],
        "class_weight": class_weights_dict
    }
    if train_steps_per_epoch is not None:
        fit_kwargs_stage_1["steps_per_epoch"] = train_steps_per_epoch

    model.fit(
        **fit_kwargs_stage_1
    )

    print("Bắt đầu huấn luyện giai đoạn 2 (fine-tuning)...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=3e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    stage2_epochs = min(FINE_TUNE_STAGE2_EPOCHS, FINE_TUNE_EPOCHS)
    stage2_end_epoch = INITIAL_EPOCHS + stage2_epochs

    fit_kwargs_stage_2 = {
        "x": train_dataset,
        "validation_data": validation_dataset,
        "initial_epoch": INITIAL_EPOCHS,
        "epochs": stage2_end_epoch,
        "callbacks": [checkpoint, early_stop, reduce_lr],
        "class_weight": class_weights_dict
    }
    if train_steps_per_epoch is not None:
        fit_kwargs_stage_2["steps_per_epoch"] = train_steps_per_epoch

    model.fit(**fit_kwargs_stage_2)

    if stage2_end_epoch < total_epochs:
        print("Bắt đầu huấn luyện giai đoạn 3 (unfreeze sâu hơn)...")
        base_model.trainable = True
        for layer in base_model.layers[:FINE_TUNE_STAGE3_FREEZE_AT]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        fit_kwargs_stage_3 = {
            "x": train_dataset,
            "validation_data": validation_dataset,
            "initial_epoch": stage2_end_epoch,
            "epochs": total_epochs,
            "callbacks": [checkpoint, early_stop, reduce_lr],
            "class_weight": class_weights_dict
        }
        if train_steps_per_epoch is not None:
            fit_kwargs_stage_3["steps_per_epoch"] = train_steps_per_epoch

        model.fit(**fit_kwargs_stage_3)

    print("Đánh giá trên tập test bằng mô hình tốt nhất...")
    best_model = tf.keras.models.load_model(MODEL_PATH)
    evaluate_on_test_set(best_model, test_dataset, test_labels, class_names)

    print(f"Huấn luyện hoàn tất. Mô hình tốt nhất đã được lưu tại {MODEL_PATH}")

if __name__ == "__main__":
    train()
