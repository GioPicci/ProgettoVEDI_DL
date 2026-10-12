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

# Dataset generato dallo script di auto-labeling (teacher DINOv3)
OUTPUT_DATASET_DIR = r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\dataset_yolo_distilled"
CLASS_MAP_PATH = "./checkpoints_classification/class_map.json"

# Cartella ORIGINALE (sola lettura) usata anche da VEDIClassificationDataset per allenare DINO.
# Serve qui SOLO per ricostruire lo stesso split train/val, non viene mai scritta.
DINO_TRAINING_DIR = r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training"

# Deve combaciare con il val_ratio usato per istanziare VEDIClassificationDataset/VEDIDataset
# quando hai allenato DINO. Il default nella classe è 0.2: aggiorna se hai usato un altro valore.
DINO_VAL_RATIO = 0.2

# Cartella dove Ultralytics salverà i run di training (separata dal dataset)
RUNS_DIR = "./runs_yolo_distilled"

# Nano delle ultime 3 versioni "ufficiali" Ultralytics (v8 -> 11 -> 26).
# v9/v10/v12/v13 sono modelli community/sperimentali non raccomandati da Ultralytics
# per produzione (instabilità in training o problemi di riproducibilità).
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}

# Iperparametri comuni (stessi per tutti i modelli, per un confronto equo)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 20          # early stopping
SEED = 42


# ==========================================
# 2. COSTRUZIONE data.yaml (split train/val ALLINEATO a quello di DINOv3)
# ==========================================
def compute_dino_split_membership(training_dir, val_ratio=0.2):
    """
    Replica ESATTAMENTE la logica di split di VEDIClassificationDataset/VEDIDataset:
    per ogni cartella-classe, le immagini sono ordinate (prima tutti i .jpg ordinati,
    poi tutti i .png ordinati) e 1 immagine ogni int(1/val_ratio), per indice, va in val.
    Usare lo stesso criterio qui garantisce che un'immagine "vista" da DINO in training
    non finisca nel val di YOLO (e viceversa), evitando di gonfiare artificialmente le
    metriche di validazione con pseudo-label più "pulite" perché memorizzate dal teacher.

    Ritorna un dict {(nome_cartella, nome_base_senza_estensione): "train" | "val"}.
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


def build_data_yaml(output_dataset_dir, class_map_path, training_dir, val_ratio=0.2):
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    names = {i: idx_to_class[i] for i in sorted(idx_to_class.keys())}

    membership = compute_dino_split_membership(training_dir, val_ratio=val_ratio)

    images_train_dir = os.path.join(output_dataset_dir, "images", "train")
    all_images = sorted(
        os.path.join(images_train_dir, f)
        for f in os.listdir(images_train_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if len(all_images) == 0:
        raise RuntimeError(
            f"Nessuna immagine trovata in {images_train_dir}. "
            "Hai già eseguito lo script di auto-labeling (generate_yolo_labels.py)?"
        )

    train_images, val_images, unmatched = [], [], []

    for img_path in all_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # Il nome file distillato è "<cartella>__<nome_base_originale>" (vedi generate_yolo_labels.py)
        if "__" not in stem:
            unmatched.append(img_path)
            continue
        folder_name, base_name = stem.split("__", 1)
        split = membership.get((folder_name, base_name))
        if split == "val":
            val_images.append(img_path)
        elif split == "train":
            train_images.append(img_path)
        else:
            unmatched.append(img_path)

    if unmatched:
        print(
            f"--> ATTENZIONE: {len(unmatched)} immagini distillate non trovano corrispondenza "
            f"nello split di DINOv3 (nome file inatteso o immagine non più presente in "
            f"{training_dir}). Le metto in train per sicurezza, ma vale la pena controllarle."
        )
        train_images.extend(unmatched)

    train_txt = os.path.join(output_dataset_dir, "train.txt")
    val_txt = os.path.join(output_dataset_dir, "val.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in train_images))
    with open(val_txt, "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in val_images))

    data_yaml = {
        "path": os.path.abspath(output_dataset_dir),
        "train": os.path.abspath(train_txt),
        "val": os.path.abspath(val_txt),
        "names": names,
    }

    data_yaml_path = os.path.join(output_dataset_dir, "data_distilled.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    return data_yaml_path, len(train_images), len(val_images)


# ==========================================
# 3. TRAINING MULTI-MODELLO
# ==========================================
def train_all_models():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--> Device selezionato: {device}")

    data_yaml_path, n_train, n_val = build_data_yaml(
        OUTPUT_DATASET_DIR, CLASS_MAP_PATH, DINO_TRAINING_DIR, val_ratio=DINO_VAL_RATIO
    )
    print(f"--> data.yaml generato: {data_yaml_path}")
    print(f"--> Immagini train: {n_train} | Immagini val: {n_val}")
    print(
        "--> NB: il val set è ALLINEATO allo split di DINOv3 (stesse immagini che erano "
        "'val' per il teacher). Le metriche sotto misurano quanto YOLO replica il teacher "
        "su immagini che DINO stesso non ha mai visto in training, NON l'accuratezza sul "
        "GT umano: per quella serve il test set esterno, valutato separatamente."
    )

    os.makedirs(RUNS_DIR, exist_ok=True)

    summary = {}

    for tag, weights in MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"TRAINING {tag}  ({weights})")
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

        # Validazione esplicita per estrarre le metriche in modo uniforme tra i modelli
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
    # 4. TABELLA COMPARATIVA FINALE
    # ==========================================
    print("\n" + "=" * 90)
    print("             CONFRONTO TRA VERSIONI YOLO (nano) SU DATASET DISTILLATO DA DINOv3")
    print("=" * 90)
    print(f"{'MODELLO':<12} | {'mAP@50':<10} | {'mAP@50:95':<10} | {'Precision':<10} | {'Recall':<10} | {'Tempo (min)':<12}")
    print("-" * 90)
    for tag, res in summary.items():
        print(
            f"{tag:<12} | {res['map50'] * 100:>7.2f}%  | {res['map50_95'] * 100:>7.2f}%  | "
            f"{res['precision'] * 100:>7.2f}%  | {res['recall'] * 100:>7.2f}%  | {res['train_time_min']:>10.2f}"
        )
    print("=" * 90)

    summary_path = os.path.join(RUNS_DIR, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Riepilogo salvato in: {summary_path}")
    print(f"[+] Pesi e log di ogni run in: {RUNS_DIR}/<nome_modello>/")


if __name__ == "__main__":
    train_all_models()