import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import csv
import datetime

sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_INFO_PATH = os.path.join(BASE_DIR, "split_info.json")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "age_class_indices.json")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "age_classifier_model.keras")
TEST_METRICS_PATH = os.path.join(BASE_DIR, "test_metrics.json")

def load_split_ratios(path):
    default_ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    if not os.path.exists(path):
        return default_ratios
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ratios = data.get("ratios", {})
    return {
        "train": float(ratios.get("train", 0.70)),
        "validation": float(ratios.get("validation", 0.15)),
        "test": float(ratios.get("test", 0.15)),
    }

def plot_data_split_pie(ratios):
    labels = ["Train", "Validation", "Test"]
    sizes = [ratios["train"] * 100, ratios["validation"] * 100, ratios["test"] * 100]
    colors = ["#2E86AB", "#F6C85F", "#6FB07F"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.0f}%",
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    ax.set_title("Data Split Pie Chart (70/15/15)", fontsize=16, fontweight="bold", pad=16)
    ax.text(
        0,
        -1.25,
        "Su dung Stratified Split de bao toan ty le 4 nhom tuoi tren ca 3 phan",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "data_split_pie_chart.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def draw_stage_block(ax, x, y, w, h, title, subtitle, trainable_backbone_fraction, classifier_color):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#333333", facecolor="#F9F9F9"))
    ax.text(x + w / 2, y + h - 0.10 * h, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.24 * h, subtitle, ha="center", va="center", fontsize=10)

    bar_x = x + 0.08 * w
    bar_y = y + 0.28 * h
    bar_h = 0.30 * h
    backbone_w = 0.68 * w
    classifier_w = 0.16 * w

    ax.add_patch(Rectangle((bar_x, bar_y), backbone_w, bar_h, facecolor="#D6D6D6", edgecolor="#888888"))

    if trainable_backbone_fraction > 0:
        trainable_w = backbone_w * trainable_backbone_fraction
        ax.add_patch(
            Rectangle(
                (bar_x + backbone_w - trainable_w, bar_y),
                trainable_w,
                bar_h,
                facecolor="#2ECC71",
                edgecolor="#1E8F4E",
            )
        )

    ax.add_patch(
        Rectangle(
            (bar_x + backbone_w, bar_y),
            classifier_w,
            bar_h,
            facecolor=classifier_color,
            edgecolor="#8A2D2D" if classifier_color == "#E74C3C" else "#1E8F4E",
        )
    )

    ax.text(bar_x + backbone_w / 2, bar_y + bar_h / 2, "Backbone", ha="center", va="center", fontsize=9)
    ax.text(bar_x + backbone_w + classifier_w / 2, bar_y + bar_h / 2, "Classifier", ha="center", va="center", fontsize=9)

def plot_finetuning_timeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.27, 0.62
    y = 0.18
    x_positions = [0.03, 0.365, 0.70]

    stages = [
        ("Stage 1", "Dong bang Backbone\n(Chi train Classifier)", 0.00, "#E74C3C"),
        ("Stage 2", "Mo bang tu lop 50", 0.35, "#2ECC71"),
        ("Stage 3", "Mo bang tu lop 20", 0.65, "#2ECC71"),
    ]

    for i, (title, subtitle, frac, cls_color) in enumerate(stages):
        draw_stage_block(ax, x_positions[i], y, w, h, title, subtitle, frac, cls_color)

    for i in range(2):
        x1 = x_positions[i] + w + 0.01
        x2 = x_positions[i + 1] - 0.01
        y_mid = y + h * 0.52
        ax.add_patch(FancyArrowPatch((x1, y_mid), (x2, y_mid), arrowstyle="->", mutation_scale=18, linewidth=2))

    ax.set_title("3-Stage Fine-tuning Timeline", fontsize=17, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "three_stage_finetuning_timeline.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def load_confusion_matrix_and_labels(cm_path, class_map_path):
    cm = np.loadtxt(cm_path, delimiter=",", dtype=int)
    if cm.ndim == 1:
        cm = np.expand_dims(cm, axis=0)

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = sorted(class_to_idx.items(), key=lambda x: x[1])
    class_names = [name for name, _ in idx_to_class]
    return cm, class_names

def plot_confusion_matrix_heatmap(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.8,
        linecolor="white",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix Heatmap", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Nhan may doan", fontsize=12)
    ax.set_ylabel("Nhan that", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def load_split_info(path):
    default = {
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "counts": {
            "train": {"adult": 0, "child": 0, "elderly": 0, "teen": 0},
            "validation": {"adult": 0, "child": 0, "elderly": 0, "teen": 0},
            "test": {"adult": 0, "child": 0, "elderly": 0, "teen": 0},
        },
    }
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "ratios" not in data or "counts" not in data:
        return default
    return data


def save_csv_table(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_table_png(path, title, headers, rows):
    fig_w = max(9, len(headers) * 2.0)
    fig_h = max(3.2, 0.65 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.45)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2E86AB")
        else:
            cell.set_facecolor("#F7FBFE" if r % 2 == 0 else "#FFFFFF")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_split_tables(split_info):
    ratios = split_info["ratios"]
    counts = split_info["counts"]

    split_order = ["train", "validation", "test"]
    class_order = ["adult", "child", "elderly", "teen"]

    split_totals = {s: sum(int(counts[s][k]) for k in class_order) for s in split_order}
    grand_total = sum(split_totals.values())

    ratio_rows = [
        ["Train", f"{ratios['train'] * 100:.0f}%", f"{split_totals['train']:,}"],
        ["Validation", f"{ratios['validation'] * 100:.0f}%", f"{split_totals['validation']:,}"],
        ["Test", f"{ratios['test'] * 100:.0f}%", f"{split_totals['test']:,}"],
        ["Total", "100%", f"{grand_total:,}"],
    ]
    ratio_headers = ["Split", "Ratio", "Images"]

    class_rows = []
    for cls in class_order:
        train_n = int(counts["train"][cls])
        val_n = int(counts["validation"][cls])
        test_n = int(counts["test"][cls])
        class_rows.append([cls, f"{train_n:,}", f"{val_n:,}", f"{test_n:,}", f"{train_n + val_n + test_n:,}"])
    class_headers = ["Class", "Train", "Validation", "Test", "Total"]

    save_csv_table(os.path.join(OUTPUT_DIR, "split_ratio_table.csv"), ratio_headers, ratio_rows)
    save_csv_table(os.path.join(OUTPUT_DIR, "class_distribution_table.csv"), class_headers, class_rows)

    save_table_png(
        os.path.join(OUTPUT_DIR, "split_ratio_table.png"),
        "Data Split Summary",
        ratio_headers,
        ratio_rows,
    )
    save_table_png(
        os.path.join(OUTPUT_DIR, "class_distribution_table.png"),
        "Class Distribution by Split",
        class_headers,
        class_rows,
    )

def plot_artifact_overview():
    artifacts = [
        ("Best Model", MODEL_PATH),
        ("Class Index Map", CLASS_MAP_PATH),
        ("Test Metrics", TEST_METRICS_PATH),
        ("Confusion Matrix", CONFUSION_MATRIX_PATH),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Training Artifacts Overview", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_positions = [0.78, 0.58, 0.38, 0.18]
    for (label, path), y in zip(artifacts, y_positions):
        exists = os.path.exists(path)
        bg = "#E9F9EE" if exists else "#FDECEC"
        edge = "#2E9D58" if exists else "#C63D3D"
        status = "FOUND" if exists else "MISSING"

        detail = os.path.basename(path)
        if exists and label == "Best Model":
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            detail = f"{detail} | {size_mb:.2f} MB | updated {mtime}"

        box = FancyBboxPatch(
            (0.05, y - 0.08), 0.90, 0.13,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.8, edgecolor=edge, facecolor=bg
        )
        ax.add_patch(box)
        ax.text(0.08, y, label, ha="left", va="center", fontsize=12, fontweight="bold")
        ax.text(0.40, y, detail, ha="left", va="center", fontsize=10)
        ax.text(0.92, y, status, ha="right", va="center", fontsize=11, fontweight="bold", color=edge)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "artifact_overview.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_class_mapping_table(class_map_path):
    if not os.path.exists(class_map_path):
        return
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    pairs = sorted(class_map.items(), key=lambda x: x[1])
    rows = [[str(idx), name] for name, idx in pairs]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    ax.set_title("Class ID Mapping", fontsize=15, fontweight="bold", pad=10)

    table = ax.table(
        cellText=rows,
        colLabels=["ID", "Age Group"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2E86AB")
        else:
            cell.set_facecolor("#F7FBFE" if r % 2 == 0 else "#FFFFFF")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "class_index_mapping_table.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_test_metrics_dashboard(metrics):
    metric_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1-macro", "Precision-macro", "Recall-macro"]
    values = [metrics[k] for k in metric_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test Set Metrics Dashboard", fontsize=16, fontweight="bold", y=1.02)

    bars = ax1.barh(metric_labels, values, color=["#2E86AB", "#6FB07F", "#F6C85F", "#E27D60"])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Score")
    ax1.grid(axis="x", alpha=0.3)

    for bar, v in zip(bars, values):
        ax1.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=10)

    ax2.axis("off")
    ax2.text(0.0, 0.78, f"Test Loss: {metrics['test_loss']:.4f}", fontsize=14, fontweight="bold")
    ax2.text(0.0, 0.58, f"Test Accuracy: {metrics['test_accuracy']:.4f}", fontsize=14, fontweight="bold")
    ax2.text(0.0, 0.28, "Nguon: test_metrics.json", fontsize=10, color="#666666")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "test_metrics_dashboard.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def load_test_metrics(path):
    default = {
        "test_loss": 0.0,
        "test_accuracy": 0.0,
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
    }
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for k, v in default.items():
        data[k] = float(data.get(k, v))
    return data

if __name__ == "__main__":
    split_info = load_split_info(SPLIT_INFO_PATH)

    ratios = split_info["ratios"]
    plot_data_split_pie(ratios)
    plot_finetuning_timeline()

    cm, class_names = load_confusion_matrix_and_labels(CONFUSION_MATRIX_PATH, CLASS_MAP_PATH)
    plot_confusion_matrix_heatmap(cm, class_names)

    save_split_tables(split_info)

    metrics = load_test_metrics(TEST_METRICS_PATH)
    plot_test_metrics_dashboard(metrics)
    plot_class_mapping_table(CLASS_MAP_PATH)
    plot_artifact_overview()

    print("Da tao xong bieu do va bang trong thu muc:", OUTPUT_DIR)
