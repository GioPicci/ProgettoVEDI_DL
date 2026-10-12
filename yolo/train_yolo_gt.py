import os
import glob
import json
import time

import yaml
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

# Cartella Ground-Truth originale del dataset di Training (sola lettura)
GT_TRAINING_DIR = r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training"
CLASS_MAP_PATH = "./checkpoints_classification/class_map.json"

# Deve combaciare con il val_ratio usato per istanziare VEDIClassificationDataset/VEDIDataset
DINO_VAL_RATIO = 0.2

# Cartella dove Ultralytics salverà i run di training sul Ground Truth (separata da quella distillata)
RUNS_DIR = "./runs_yolo_gt"

# Modelli da allenare (gli stessi usati per la distillation)
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}

# Iperparametri comuni (identici al training distillato per un confronto equo)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 5  # early stopping
SEED = 42


# ==========================================
# 2. COSTRUZIONE data_gt.yaml (split train/val ALLINEATO a DINOv3)
# ==========================================
def compute_dino_split_membership(training_dir, val_ratio=0.2):
    """
    Replica ESATTAMENTE la logica di split di VEDIClassificationDataset/VEDIDataset:
    per ogni cartella-classe, le immagini sono ordinate e 1 immagine ogni int(1/val_ratio)
    va nel validation set.
    """
    membership = {}
    step = int(1 / val_ratio)

    for folder_name in sorted(os.listdir(training_dir)):
        folder_path = os.path.join(training_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        img_dir = os.path.join(folder_path, "images")
        if not os.path.isdir(img_dir):
            continue

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + \
                    sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for idx, img_path in enumerate(img_paths):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            is_val = (idx % step == 0)
            membership[(folder_name, base_name)] = "val" if is_val else "train"

    return membership


def build_gt_data_yaml(training_dir, class_map_path, val_ratio=0.2):
    """
    Scansiona il dataset Ground Truth originale e crea il file .yaml
    con i percorsi train.txt e val.txt per YOLO.
    """
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    names = {i: idx_to_class[i] for i in sorted(idx_to_class.keys())}

    membership = compute_dino_split_membership(training_dir, val_ratio=val_ratio)

    train_images, val_images = [], []

    for folder_name in sorted(os.listdir(training_dir)):
        folder_path = os.path.join(training_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        img_dir = os.path.join(folder_path, "images")
        if not os.path.isdir(img_dir):
            continue

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + \
                    sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for img_path in img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            split = membership.get((folder_name, base_name), "train")

            abs_img_path = os.path.abspath(img_path)
            if split == "val":
                val_images.append(abs_img_path)
            else:
                train_images.append(abs_img_path)

    if not train_images and not val_images:
        raise RuntimeError(f"Nessuna immagine trovata in {training_dir}.")

    train_txt = os.path.join(training_dir, "train_gt.txt")
    val_txt = os.path.join(training_dir, "val_gt.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_images))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_images))

    data_yaml = {
        "path": os.path.abspath(training_dir),
        "train": os.path.abspath(train_txt),
        "val": os.path.abspath(val_txt),
        "names": names,
    }

    data_yaml_path = os.path.join(training_dir, "data_gt.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    return data_yaml_path, len(train_images), len(val_images)


# ==========================================
# 3. TRAINING MULTI-MODELLO (GROUND TRUTH)
# ==========================================
def train_all_models_gt():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--> Device selezionato per il training: {device}")

    data_yaml_path, n_train, n_val = build_gt_data_yaml(
        GT_TRAINING_DIR, CLASS_MAP_PATH, val_ratio=DINO_VAL_RATIO
    )
    print(f"--> data_gt.yaml generato: {data_yaml_path}")
    print(f"--> Immagini train GT: {n_train} | Immagini val GT: {n_val}")

    os.makedirs(RUNS_DIR, exist_ok=True)

    summary = {}

    for tag, weights in MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"TRAINING {tag} SU GROUND TRUTH  ({weights})")
        print(f"{'=' * 80}")

        model = YOLO(weights)

        t0 = time.time()
        model.train(
            data=data_yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            seed=SEED,
            patience=PATIENCE,
            project=RUNS_DIR,
            name=tag,
            exist_ok=True,
            verbose=True,
        )
        train_time_sec = time.time() - t0

        # Validazione esplicita sulle immagini di validation Ground Truth
        metrics = model.val(data=data_yaml_path, imgsz=IMG_SIZE, device=device)

        run_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") else None

        summary[tag] = {
            "weights": weights,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "train_time_min": round(train_time_sec / 60.0, 2),
            "run_dir": run_dir,
        }

    # ==========================================
    # 4. TABELLA COMPARATIVA FINALE (GROUND TRUTH)
    # ==========================================
    print("\n" + "=" * 90)
    print("             CONFRONTO TRA VERSIONI YOLO (nano) SU DATASET GROUND TRUTH")
    print("=" * 90)
    print(
        f"{'MODELLO':<12} | {'mAP@50':<10} | {'mAP@50:95':<10} | {'Precision':<10} | {'Recall':<10} | {'Tempo (min)':<12}")
    print("-" * 90)
    for tag, res in summary.items():
        print(
            f"{tag:<12} | {res['map50'] * 100:>7.2f}%  | {res['map50_95'] * 100:>7.2f}%  | "
            f"{res['precision'] * 100:>7.2f}%  | {res['recall'] * 100:>7.2f}%  | {res['train_time_min']:>10.2f}"
        )
    print("=" * 90)

    summary_path = os.path.join(RUNS_DIR, "comparison_summary_gt.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Riepilogo salvato in: {summary_path}")
    print(f"[+] Pesi e log di ogni run in: {RUNS_DIR}/<nome_modello>/")


if __name__ == "__main__":
    train_all_models_gt()