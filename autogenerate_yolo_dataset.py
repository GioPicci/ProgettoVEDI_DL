import os
import glob
import json
import shutil
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# 1. ARCHITETTURA MODELLO (identica allo script di training/benchmark)
# ==========================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes, eager_attn=True):
        super().__init__()
        attn_impl = "eager" if eager_attn else None
        self.backbone = AutoModel.from_pretrained(model_name, attn_implementation=attn_impl)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values, return_attentions=False):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=return_attentions)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, outputs


def generate_cam(method, logits, outputs, model_classifier, pred_class_id, num_patches, grid_h, grid_w):
    last_hidden_state = outputs.last_hidden_state

    if method == "linear_cam":
        patch_tokens = last_hidden_state[:, -num_patches:, :]
        linear_layer = model_classifier[-1]
        class_weights = linear_layer.weight[pred_class_id]
        cam = torch.matmul(patch_tokens, class_weights.unsqueeze(-1)).squeeze()
        cam = torch.relu(cam)
        cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()

    elif method == "dino_attn":
        last_attn = outputs.attentions[-1][0]
        cls_to_patches_attn = last_attn[:, 0, -num_patches:]
        avg_attn = cls_to_patches_attn.mean(dim=0)
        cam = avg_attn.reshape(grid_h, grid_w).cpu().detach().numpy()

    elif method == "eigen_cam":
        patch_tokens = last_hidden_state[0, -num_patches:, :]
        patch_tokens_centered = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
        _, _, V = torch.pca_lowrank(patch_tokens_centered, q=1)
        cam = torch.matmul(patch_tokens_centered, V[:, 0])
        cam = torch.abs(cam)
        cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()

    elif method == "grad_cam":
        last_layer_attn = outputs.attentions[-1]
        score = logits[0, pred_class_id]
        grads = torch.autograd.grad(score, last_layer_attn, retain_graph=True)[0]
        cls_attn = last_layer_attn[0, :, 0, -num_patches:]
        cls_grads = grads[0, :, 0, -num_patches:]
        weighted_attn = torch.relu(cls_grads * cls_attn)
        cam = weighted_attn.mean(dim=0).reshape(grid_h, grid_w).cpu().detach().numpy()

    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    return cam


# ==========================================
# 2. RACCOLTA IMMAGINI DALLE SOTTOCARTELLE (dataset NON piatto)
# ==========================================
def collect_images(dataset_dir, extensions=(".jpg", ".jpeg", ".png")):
    """
    Il dataset ha struttura:
        dataset_dir/
            1.1_Ingresso/images/*.jpg        (+ labels/*.txt che NON tocchiamo mai)
            2.1_RampaS.Nicola/images/*.jpg    (+ labels/*.txt)
            ...
    Ritorna una lista di tuple (nome_cartella_reperto, percorso_immagine).
    """
    samples = []
    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        img_dir = os.path.join(folder_path, "images")
        if not os.path.isdir(img_dir):
            continue

        found = set()
        for ext in extensions:
            found.update(glob.glob(os.path.join(img_dir, f"*{ext}")))
            found.update(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))

        for img_path in sorted(found):
            samples.append((folder, img_path))

    return samples


# ==========================================
# 3. AUTO-LABELER GENERATOR (TEACHER = DINOv3, STUDENT = YOLO)
# ==========================================
def generate_yolo_labels():
    # ---- SETUP PERCORSI ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "data", "UNICT-VEDI_Dataset_bbox", "Training")
    # Cartella di OUTPUT completamente separata: qui scriviamo il nuovo dataset "distillato" per YOLO
    OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset_yolo_distilled")

    CHECKPOINT_PATH = "./checkpoints_classification/best_model.pth"
    CLASS_MAP_PATH = "./checkpoints_classification/class_map.json"
    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

    WSOL_METHOD = "linear_cam"     # metodo usato per generare le CAM/bbox
    STRICTNESS = 0.45
    CONFIDENCE_THRESHOLD = 0.50    # YOLO impara male dal rumore: etichettiamo solo se DINO è sicuro

    # Se True: salta le immagini che hanno GIA' un file .txt manuale in labels/
    # (utile se vuoi che DINO etichetti SOLO le immagini ancora prive di annotazione).
    # Default False: DINO etichetta tutte le immagini, per costruire un training set
    # "distillato" indipendente da confrontare poi con il GT reale.
    SKIP_IF_GT_EXISTS = False

    # ---- SAFETY CHECK: non scrivere MAI dentro il dataset sorgente ----
    dataset_dir_abs = os.path.abspath(DATASET_DIR)
    output_dir_abs = os.path.abspath(OUTPUT_DATASET_DIR)
    if output_dir_abs == dataset_dir_abs or output_dir_abs.startswith(dataset_dir_abs + os.sep):
        raise RuntimeError(
            "OUTPUT_DATASET_DIR non può coincidere con DATASET_DIR (o esserne una sottocartella): "
            "rischieresti di sovrascrivere le label originali. Scegli un percorso di output separato."
        )

    out_img_dir = os.path.join(OUTPUT_DATASET_DIR, "images", "train")
    out_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "labels", "train")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Inizializzazione Auto-Labeling su {device}")
    print(f"--> Sorgente (sola lettura): {DATASET_DIR}")
    print(f"--> Output (nuovo dataset):  {OUTPUT_DATASET_DIR}")

    # ---- Caricamento classi ----
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    num_classes = len(class_map)
    negatives_id = class_map.get("Negatives", num_classes - 1)

    # ---- Caricamento Modello ----
    eager_attn = WSOL_METHOD in ["dino_attn", "grad_cam"]
    needs_grad = WSOL_METHOD == "grad_cam"

    model = DINOClassifier(MODEL_NAME, num_classes, eager_attn=eager_attn)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    target_size = 384
    processor.size = {"height": target_size, "width": target_size}

    patch_size = 16
    grid_h = target_size // patch_size
    grid_w = target_size // patch_size
    num_patches = grid_h * grid_w

    # ---- Raccolta ricorsiva delle immagini in TUTTE le sottocartelle ----
    samples = collect_images(DATASET_DIR)
    n_folders = len(set(folder for folder, _ in samples))
    print(f"--> Trovate {len(samples)} immagini in {n_folders} sottocartelle.")

    # ---- Statistiche per il report finale ----
    stats = {
        "totale_immagini": len(samples),
        "etichettate": 0,
        "saltate_gt_esistente": 0,
        "scartate_negative_o_bassa_confidenza": 0,
        "scartate_nessun_contorno": 0,
        "per_cartella": defaultdict(int),
        "per_classe": defaultdict(int),
    }

    for folder, img_path in tqdm(samples, desc="Generazione BBOX (Teacher DINOv3)"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]

        if SKIP_IF_GT_EXISTS:
            # images/ e labels/ sono cartelle sorelle dentro la stessa cartella-reperto
            labels_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
            gt_label_path = os.path.join(labels_dir, f"{base_name}.txt")
            if os.path.exists(gt_label_path):
                stats["saltate_gt_esistente"] += 1
                continue

        try:
            raw_image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = raw_image.size
            inputs = processor(images=raw_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        except Exception:
            continue

        with torch.set_grad_enabled(needs_grad):
            logits, outputs = model(pixel_values, return_attentions=eager_attn)
            probs = torch.softmax(logits, dim=-1)
            pred_class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class_id].item()

        if pred_class_id == negatives_id or confidence < CONFIDENCE_THRESHOLD:
            stats["scartate_negative_o_bassa_confidenza"] += 1
            continue

        cam = generate_cam(
            WSOL_METHOD, logits, outputs, model.classifier,
            pred_class_id, num_patches, grid_h, grid_w
        )

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        strict_thresh = int(STRICTNESS * heatmap_uint8.max())
        _, binary_mask = cv2.threshold(heatmap_uint8, strict_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            stats["scartate_nessun_contorno"] += 1
            continue

        y_max, x_max = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        peak_point = (int(x_max), int(y_max))
        best_contour = None
        for contour in contours:
            if cv2.pointPolygonTest(contour, peak_point, False) >= 0:
                best_contour = contour
                break
        if best_contour is None:
            best_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(best_contour)

        cx_norm = (x + (w / 2.0)) / orig_w
        cy_norm = (y + (h / 2.0)) / orig_h
        w_norm = w / orig_w
        h_norm = h / orig_h

        cx_norm, cy_norm = max(0.0, min(1.0, cx_norm)), max(0.0, min(1.0, cy_norm))
        w_norm, h_norm = max(0.0, min(1.0, w_norm)), max(0.0, min(1.0, h_norm))

        # ---- Nome file univoco: prefisso con la cartella-reperto di provenienza ----
        # Evita collisioni tra immagini con lo stesso nome in reperti diversi
        # (es. "1.1_Ingresso/images/0001.jpg" vs "2.1_RampaS.Nicola/images/0001.jpg")
        safe_folder = folder.replace(" ", "_")
        unique_name = f"{safe_folder}__{base_name}"

        dest_img = os.path.join(out_img_dir, f"{unique_name}{ext}")
        label_file = os.path.join(out_lbl_dir, f"{unique_name}.txt")

        shutil.copy2(img_path, dest_img)
        with open(label_file, "w") as f:
            f.write(f"{pred_class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        stats["etichettate"] += 1
        stats["per_cartella"][folder] += 1
        stats["per_classe"][idx_to_class.get(pred_class_id, str(pred_class_id))] += 1

    # ---- Report finale ----
    report_path = os.path.join(OUTPUT_DATASET_DIR, "generation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"\n[+] FATTO! {stats['etichettate']} / {stats['totale_immagini']} immagini etichettate con successo!")
    if SKIP_IF_GT_EXISTS:
        print(f"[+] Saltate perché già annotate manualmente: {stats['saltate_gt_esistente']}")
    print(f"[+] Scartate (negative/bassa confidenza): {stats['scartate_negative_o_bassa_confidenza']}")
    print(f"[+] Scartate (nessun contorno valido): {stats['scartate_nessun_contorno']}")
    print(f"[+] Dataset pronto in: {OUTPUT_DATASET_DIR}")
    print(f"[+] Report dettagliato salvato in: {report_path}")
    print(f"[+] Cartelle sorgente NON toccate: {DATASET_DIR}")


if __name__ == "__main__":
    generate_yolo_labels()