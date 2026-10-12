import os
import glob
import json
import random
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torchmetrics.detection import MeanAveragePrecision


# ==========================================
# 1. ARCHITETTURA MODELLO
# ==========================================
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
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, outputs.last_hidden_state


# ==========================================
# 2. FUNZIONI DI UTILITÀ
# ==========================================
def parse_yolo_gt(label_path, w_img, h_img):
    """Converte le etichette YOLO .txt (Ground Truth) in coordinate pixel [xmin, ymin, xmax, ymax]"""
    boxes = []
    labels = []
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, cx, cy, w, h = map(float, parts[:5])
                    xmin = max(0.0, (cx - w / 2.0) * w_img)
                    ymin = max(0.0, (cy - h / 2.0) * h_img)
                    xmax = min(w_img, (cx + w / 2.0) * w_img)
                    ymax = min(h_img, (cy + h / 2.0) * h_img)

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(c))

    if len(boxes) == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64)
        }
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


def predict_wsol_box(model, processor, raw_image, device, negatives_id=57, strictness=0.65):
    """Genera la predizione del Bounding Box WSOL tramite la heatmap di DINOv3"""
    orig_w, orig_h = raw_image.size
    target_size = 384
    processor.size = {"height": target_size, "width": target_size}

    inputs = processor(images=raw_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits, last_hidden_state = model(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        pred_class_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class_id].item()

    if pred_class_id == negatives_id or confidence < 0.1:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64)
        }

    patch_size = 16
    grid_h = target_size // patch_size
    grid_w = target_size // patch_size
    num_patches = grid_h * grid_w

    patch_tokens = last_hidden_state[:, -num_patches:, :]
    linear_layer = model.classifier[-1]
    class_weights = linear_layer.weight[pred_class_id]

    cam = torch.matmul(patch_tokens, class_weights.unsqueeze(-1)).squeeze()
    cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()
    cam = np.maximum(cam, 0)

    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    strict_thresh = int(strictness * heatmap_uint8.max())
    _, binary_mask = cv2.threshold(heatmap_uint8, strict_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64)
        }

    y_max, x_max = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    peak_point = (int(x_max), int(y_max))

    best_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, peak_point, False) >= 0:
            best_contour = contour
            break

    if best_contour is None:
        def contour_energy(c):
            mask = np.zeros_like(heatmap_resized, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            return np.sum(heatmap_resized[mask == 255])

        best_contour = max(contours, key=contour_energy)

    x, y, w, h = cv2.boundingRect(best_contour)

    return {
        "boxes": torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32),
        "scores": torch.tensor([confidence], dtype=torch.float32),
        "labels": torch.tensor([pred_class_id], dtype=torch.int64)
    }


# ==========================================
# 3. MAIN EVALUATION PER CLASSE
# ==========================================
def evaluate():
    DATASET_DIR = "./data/UNICT-VEDI_Dataset_bbox/Training"
    CHECKPOINT_PATH = "./checkpoints_classification/best_model.pth"
    CLASS_MAP_PATH = "./checkpoints_classification/class_map.json"
    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

    SAMPLE_RATIO = 0.25
    STRICTNESS = 0.45  # <-- Soglia selettività BBOX
    SEED = 42

    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device selezionato: {device}")

    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    num_classes = len(class_map)
    negatives_id = class_map.get("Negatives", num_classes - 1)

    model = DINOClassifier(MODEL_NAME, num_classes)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Campionamento Stratificato
    sampled_files = []
    class_gt_counts = defaultdict(int)

    print(f"--> Campionamento del {SAMPLE_RATIO * 100:.0f}% per classe...")
    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        img_dir = os.path.join(folder_path, "images")
        label_dir = os.path.join(folder_path, "labels")

        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not images:
            continue

        k = max(1, int(len(images) * SAMPLE_RATIO))
        selected_imgs = random.sample(images, k)

        for img_path in selected_imgs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            sampled_files.append((img_path, label_path if os.path.exists(label_path) else None))

    # NOTA: Impostiamo class_metrics=True per calcolare le metriche singole per classe!
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    print(f"\n=== VALUTAZIONE SU {len(sampled_files)} IMMAGINI ===")
    for img_path, label_path in tqdm(sampled_files, desc="Processing WSOL"):
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w_img, h_img = raw_image.size

        target_dict = parse_yolo_gt(label_path, w_img, h_img)
        pred_dict = predict_wsol_box(
            model, processor, raw_image, device,
            negatives_id=negatives_id, strictness=STRICTNESS
        )

        # Tracciamo quante etichette Ground Truth reali abbiamo per ciascuna classe
        for lbl in target_dict["labels"]:
            class_gt_counts[lbl.item()] += 1

        metric.update([pred_dict], [target_dict])

    print("\n--> Elaborazione metriche per classe...")
    results = metric.compute()

    global_map_50 = results["map_50"].item()
    global_map_coco = results["map"].item()

    # Estragga i risultati dettagliati per classe
    classes_eval = results.get("classes")
    map_per_class = results.get("map_per_class")
    mar_per_class = results.get("mar_100_per_class")

    class_table = []

    if classes_eval is not None and map_per_class is not None:
        classes_list = classes_eval.tolist()
        map_list = map_per_class.tolist()
        mar_list = mar_per_class.tolist() if mar_per_class is not None else [0.0] * len(classes_list)

        for c_id, c_map, c_mar in zip(classes_list, map_list, mar_list):
            c_name = idx_to_class.get(c_id, f"Class_{c_id}")
            num_gt = class_gt_counts.get(c_id, 0)
            class_table.append({
                "id": c_id,
                "name": c_name,
                "map": c_map,
                "recall": c_mar,
                "gt_count": num_gt
            })

    # Ordiniamo la tabella dal mAP PIÙ BASSO al PIÙ ALTO (per vedere prima le classi rotte!)
    class_table.sort(key=lambda x: x["map"])

    print("\n" + "=" * 78)
    print("                TABELLA PRESTAZIONI DETTAGLIATE PER CLASSE")
    print("=" * 78)
    print(f"{'ID':<4} | {'NOME CLASSE':<32} | {'GT BOXES':<8} | {'mAP@50:95':<12} | {'RECALL':<8}")
    print("-" * 78)

    for item in class_table:
        map_str = f"{item['map'] * 100:.2f}%" if item['map'] >= 0 else "N/A"
        rec_str = f"{item['recall'] * 100:.2f}%" if item['recall'] >= 0 else "N/A"
        print(f"{item['id']:<4} | {item['name'][:32]:<32} | {item['gt_count']:<8} | {map_str:<12} | {rec_str:<8}")

    print("=" * 78)
    print(f" METRICHE GLOBALI -> mAP@50: {global_map_50 * 100:.2f}% | mAP@50:95 (COCO): {global_map_coco * 100:.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    evaluate()