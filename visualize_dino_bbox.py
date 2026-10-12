import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
DATASET_DIR = Path(r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training")
CHECKPOINT_PATH = Path("./checkpoints_classification/best_model.pth")
CLASS_MAP_PATH = Path("./checkpoints_classification/class_map.json")
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
OUTPUT_DIR = Path("./visual_sample_dino_vs_gt")

# NUMERO DI IMMAGINI DA CAMPIONARE PER CLASSE
SAMPLES_PER_CLASS = 10
SEED = 42

CAM_STRICTNESS = 0.45
IMG_PROCESS_SIZE = 384

# TRASPARENZA OVERLAY
HEATMAP_ALPHA = 0.40  # Trasparenza della mappa d'attenzione (40%)
IMAGE_BETA = 0.60     # Trasparenza dell'immagine originale (60%)


# ==============================================================================
# ARCHITETTURA MODELLO
# ==============================================================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=False)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, outputs


def generate_linear_cam(outputs, model_classifier, pred_class_id, num_patches, grid_h, grid_w):
    last_hidden_state = outputs.last_hidden_state
    patch_tokens = last_hidden_state[:, -num_patches:, :]
    linear_layer = model_classifier[-1]
    class_weights = linear_layer.weight[pred_class_id]
    cam = torch.matmul(patch_tokens, class_weights.unsqueeze(-1)).squeeze()
    cam = torch.relu(cam)
    return cam.reshape(grid_h, grid_w).cpu().detach().numpy()


# ==============================================================================
# UTILITY
# ==============================================================================
def yolo_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2.0) * w)
    y1 = int((cy - bh / 2.0) * h)
    x2 = int((cx + bw / 2.0) * w)
    y2 = int((cy + bh / 2.0) * h)
    return [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-8)


def parse_gt_labels(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    "class_id": int(parts[0]),
                    "xyxy": yolo_to_xyxy([float(p) for p in parts[1:5]], w, h)
                })
    return boxes


# ==============================================================================
# RUN GENERATORE CON HEATMAP OVERLAY
# ==============================================================================
def run_visual_sampler():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Inizializzazione su {device} | Salvataggio in: {OUTPUT_DIR.resolve()}")

    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}

    model = DINOClassifier(MODEL_NAME, len(class_map))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.size = {"height": IMG_PROCESS_SIZE, "width": IMG_PROCESS_SIZE}
    patch_size = 16
    grid = IMG_PROCESS_SIZE // patch_size
    num_patches = grid * grid

    # Raggruppa le immagini per cartella/classe
    grouped_images = defaultdict(list)
    for img_path in DATASET_DIR.rglob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"] and img_path.parent.name == "images":
            class_folder = img_path.parent.parent.name
            grouped_images[class_folder].append(img_path)

    # Campionamento bilanciato
    selected_images = []
    for folder, paths in sorted(grouped_images.items()):
        sample_count = min(len(paths), SAMPLES_PER_CLASS)
        sampled = random.sample(paths, sample_count)
        selected_images.extend([(folder, p) for p in sampled])

    print(f"--> Selezionate {len(selected_images)} immagini ({SAMPLES_PER_CLASS} per classe).")

    for folder_name, img_path in tqdm(selected_images, desc="Elaborazione con CAM Overlay"):
        label_file = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"

        try:
            raw_image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = raw_image.size
            inputs = processor(images=raw_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        except Exception:
            continue

        gt_boxes = parse_gt_labels(label_file, orig_w, orig_h)

        with torch.no_grad():
            logits, outputs = model(pixel_values)
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            conf = float(probs[0, pred_id].item())

        cam = generate_linear_cam(outputs, model.classifier, pred_id, num_patches, grid, grid)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # 1. Peak Point
        y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        peak_pt = (int(x_max), int(y_max))

        # 2. Maschera binaria ed estrazione bounding box
        h_uint8 = (heatmap * 255).astype(np.uint8)
        _, binary = cv2.threshold(h_uint8, int(CAM_STRICTNESS * h_uint8.max()), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dino_box = None
        if len(contours) > 0:
            best_contour = None
            for c in contours:
                if cv2.pointPolygonTest(c, peak_pt, False) >= 0:
                    best_contour = c
                    break
            if best_contour is None:
                best_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(best_contour)
            dino_box = [bx, by, bx + bw, by + bh]

        # 3. Applicazione Heatmap Overlay in trasparenza
        img_bgr = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(heatmap_color, HEATMAP_ALPHA, img_bgr, IMAGE_BETA, 0)

        # 4. Disegno Ground Truth (Verde)
        best_iou = 0.0
        for gt in gt_boxes:
            gx1, gy1, gx2, gy2 = gt["xyxy"]
            cv2.rectangle(blended, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            gt_name = idx_to_class.get(gt["class_id"], str(gt["class_id"]))
            cv2.putText(blended, f"GT: {gt_name}", (gx1, max(18, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if dino_box is not None:
                iou = compute_iou(dino_box, gt["xyxy"])
                best_iou = max(best_iou, iou)

        # 5. Disegno BBox DINO (Bianco/Rosso visibile)
        if dino_box is not None:
            dx1, dy1, dx2, dy2 = dino_box
            cv2.rectangle(blended, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
            pred_name = idx_to_class.get(pred_id, str(pred_id))
            cv2.putText(blended, f"DINO: {pred_name} ({conf*100:.1f}%) IoU:{best_iou:.2f}",
                        (dx1, min(orig_h - 10, dy2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 6. Disegno Peak Point (Cerchio Giallo + Mirino)
        px, py = peak_pt
        cv2.circle(blended, (px, py), 5, (0, 255, 255), -1)
        cv2.circle(blended, (px, py), 12, (0, 255, 255), 2)
        cv2.line(blended, (px - 16, py), (px + 16, py), (0, 255, 255), 1)
        cv2.line(blended, (px, py - 16), (px, py + 16), (0, 255, 255), 1)

        # Salvataggio
        class_out_dir = OUTPUT_DIR / folder_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = class_out_dir / f"{img_path.stem}_cam_overlay.jpg"
        cv2.imwrite(str(save_path), blended)

    print(f"\n[OK] Elaborazione completata! Output pronti in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_visual_sampler()