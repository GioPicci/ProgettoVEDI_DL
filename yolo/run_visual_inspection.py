import os
import glob
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CONFIGURAZIONE
# ==========================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "UNICT-VEDI_Dataset_bbox" / "Test"
WEIGHTS_PATH = REPO_ROOT / "runs" / "detect" / "runs_yolo_gt" / "yolov8n" / "weights" / "best.pt"
CLASS_MAP_PATH = REPO_ROOT / "checkpoints_classification" / "class_map.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "visual_inspection_samples"

SAMPLES_PER_CLASS = 3
CONF_THRESHOLD = 0.20


def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def run_visual_inspection():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    idx_to_name = {int(v): k for k, v in class_map.items()}

    model = YOLO(str(WEIGHTS_PATH))

    # Indicizzazione immagini per classe reale (in base al file label corrispondente)
    exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = [p for ext in exts for p in TEST_DIR.rglob(ext)]

    class_samples = {cid: [] for cid in idx_to_name.keys()}

    for img_p in image_paths:
        # Cerca la label corrispondente (cartella labels parallela a images)
        lbl_p = Path(str(img_p).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
        if not lbl_p.exists():
            continue

        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in class_samples and len(class_samples[cls_id]) < SAMPLES_PER_CLASS:
                        if (img_p, lbl_p) not in class_samples[cls_id]:
                            class_samples[cls_id].append((img_p, lbl_p))

    # Generazione dei confronti side-by-side
    for cls_id, samples in class_samples.items():
        cls_name = idx_to_name.get(cls_id, f"class_{cls_id}")
        if not samples:
            continue

        save_cls_dir = OUTPUT_DIR / f"{cls_id:02d}_{cls_name}"
        save_cls_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_p, lbl_p) in enumerate(samples):
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            # 1. Disegna GROUND TRUTH (Verde)
            gt_canvas = img_bgr.copy()
            for c, x1, y1, x2, y2 in load_gt_boxes(lbl_p, w, h):
                cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(gt_canvas, f"GT: {idx_to_name.get(c, c)}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. Disegna PREDIZIONE YOLO (Rosso)
            pred_canvas = img_bgr.copy()
            res = model.predict(source=str(img_p), conf=CONF_THRESHOLD, verbose=False)[0]
            for box in res.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                pcls = int(box.cls[0].item())
                pconf = float(box.conf[0].item())
                cv2.rectangle(pred_canvas, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(pred_canvas, f"Pred: {idx_to_name.get(pcls, pcls)} {pconf:.2f}",
                            (px1, max(py1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 3. Composizione Side-by-Side (Sinistra: GT | Destra: Pred)
            header = np.zeros((40, w * 2, 3), dtype=np.uint8)
            cv2.putText(header, f"GROUND TRUTH: {img_p.name}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(header, f"PREDIZIONE YOLO (conf > {CONF_THRESHOLD})", (w + 20, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            combined = np.vstack([header, np.hstack([gt_canvas, pred_canvas])])
            cv2.imwrite(str(save_cls_dir / f"sample_{i + 1}.jpg"), combined)

    print(f"[+] Campioni salvati ordinati per classe in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_visual_inspection()