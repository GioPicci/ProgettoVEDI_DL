import os
import sys
import csv
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
# 0. SCELTA VERSIONE (parametro)
# ==========================================
# v1 -> modello canonico a 57 classi, NESSUN background (approccio definitivo).
# v2 -> modello sperimentale a 58 classi con background (archiviato in archive/).
VERSION = "v1"  # "v1" (definitivo) oppure "v2" (archiviato); sovrascrivibile da riga di comando

CONFIGS = {
    "v1": {
        "checkpoint": "./checkpoints_classification/best_model.pth",
        "class_map": "./checkpoints_classification/class_map.json",
        "output_dir": "./checkpoints_classification/wsol_benchmark",
        "results_name": "wsol_results_v1.json",
        "csv_name": "wsol_per_class_v1.csv",
        "use_background": False,
    },
    "v2": {
        "checkpoint": "./archive/checkpoints_classification_v2/best_model.pth",
        "class_map": "./archive/checkpoints_classification_v2/class_map.json",
        "output_dir": "./archive/checkpoints_classification_v2/wsol_benchmark",
        "results_name": "wsol_results_v2.json",
        "csv_name": "wsol_per_class_v2.csv",
        "use_background": True,
    },
}

# Tier dimensionali: (nome, limite superiore ESCLUSO) sull'area media delle GT box
# della classe, espressa come frazione dell'area dell'immagine.
SIZE_TIERS = [
    ("Small", 0.30),        # < 0.30
    ("Macro", 0.80),        # 0.30 - 0.79
    ("Full-Frame", float("inf")),  # >= 0.80
]

# Metodi con mappa molto appuntita/diffusa (dino_attn su DINOv3 e grad_cam):
# la soglia relativa 0.45*max li riduce a box minuscole. Per questi usiamo una
# binarizzazione robusta (percentile di attivazione = "top-k% di energia") + un'area
# minima della box. Gli altri (linear_cam, eigen_cam) restano INVARIATI.
ROBUST_METHODS = {"dino_attn", "grad_cam"}
ROBUST_ENERGY_PERCENTILE = 50   # tieni i pixel con valore >= 50esimo percentile
ROBUST_MIN_AREA_FRAC = 0.05     # espandi la box fino ad almeno il 5% dell'immagine


def compute_size_tier(mean_area_ratio):
    for name, upper in SIZE_TIERS:
        if mean_area_ratio < upper:
            return name
    return SIZE_TIERS[-1][0]


# ==========================================
# 1. ARCHITETTURA MODELLO
# ==========================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes, eager_attn=True):
        super().__init__()
        # attn_implementation="eager" serve per accedere alle attenzioni se usiamo dino_attn o grad_cam
        attn_impl = "eager" if eager_attn else None
        self.backbone = AutoModel.from_pretrained(model_name, attn_implementation=attn_impl)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values, return_attentions=False, return_hidden_states=False):
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_attentions=return_attentions,
            output_hidden_states=return_hidden_states,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, outputs


# ==========================================
# 2. FUNZIONI DI UTILITÀ & PARSING YOLO GT
# ==========================================
def parse_yolo_gt(label_path, w_img, h_img):
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


def per_class_value(res, key, c_id):
    """Estrae (in %) il valore per-classe `key` dal dict di MeanAveragePrecision.compute()."""
    classes = res["classes"].reshape(-1).tolist()
    vals = res[key].reshape(-1).tolist()
    if c_id in classes:
        return max(vals[classes.index(c_id)], 0.0) * 100
    return 0.0


# ==========================================
# 3. GENERAZIONE HEATMAP DEI 4 METODI
# ==========================================
def generate_cam(method, logits, outputs, model_classifier, pred_class_id, num_patches, grid_h, grid_w):
    last_hidden_state = outputs.last_hidden_state

    # Presupposti per le dimensioni delle variabili nel contesto del codice:
    # - last_hidden_state: [batch_size, seq_len, hidden_dim]
    # - outputs.attentions: List di tensor [batch_size, num_heads, seq_len, seq_len]
    # - num_patches: Numero di patch nell'immagine (es. 576 per una griglia 24x24)
    # - grid_h, grid_w: Dimensioni della griglia di patch dell'immagine (es. 24, 24)

    if method == "linear_cam":
        # 1. Seleziona i token corrispondenti ai patch dell'immagine, scartando i token speciali
        #    Forma: [batch_size, num_patches, hidden_dim]
        patch_tokens = last_hidden_state[:, -num_patches:, :]

        # 2. Estrae l'ultimo strato del classificatore lineare posizionato sopra il backbone
        linear_layer = model_classifier[-1]

        # 3. Estrae il vettore dei pesi associato alla sola classe predetta (pred_class_id)
        #    Forma: [hidden_dim]
        class_weights = linear_layer.weight[pred_class_id]

        # 4. Esegue il prodotto matrice-vettore (prodotto scalare per ogni patch) tra la feature del patch e i pesi
        #    patch_tokens: [1, num_patches, hidden_dim], class_weights.unsqueeze(-1): [hidden_dim, 1]
        #    Output dopo squeeze(): [num_patches] -> misura l'allineamento tra ogni patch e la classe
        cam = torch.matmul(patch_tokens, class_weights.unsqueeze(-1)).squeeze()

        # 5. Applica la funzione ReLU per azzerare i contributi negativi (mantiene solo le evidenze a favore)
        cam = torch.relu(cam)

        # 6. Riorganizza il vettore 1D dei patch nella griglia 2D dell'immagine (grid_h x grid_w)
        #    Poi sposta il tensor sulla CPU, lo stacca dal grafo computazionale e lo converte in NumPy
        cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()


    elif method == "dino_attn":
        # 1. Tenta di recuperare la lista delle matrici di attenzione prodotte dal Transformer
        attentions = getattr(outputs, "attentions", None)

        # 2. Controllo di sicurezza: se le attenzioni non sono state salvate nel forward pass, restituisce una mappa vuota
        if attentions is None or len(attentions) == 0:
            return np.zeros((grid_h, grid_w))

        # 3. Estrae le attenzioni dell'ultimo layer del Transformer per la prima immagine del batch (indice 0)
        #    Forma: [num_heads, seq_len, seq_len]
        last_attn = attentions[-1][0]

        # 4. Isolamento dell'attenzione del Class Token ([CLS], indice 0) verso i soli token dei patch (-num_patches:)
        #    Forma: [num_heads, num_patches] -> quanta attenzione ogni testa dedica a ciascun patch
        cls_to_patches_attn = last_attn[:, 0, -num_patches:]

        # 5. Calcola la media dei pesi di attenzione lungo l'asse delle teste (dim=0)
        #    Forma risultante: [num_patches]
        avg_attn = cls_to_patches_attn.mean(dim=0)

        # 6. Ridimensiona il vettore 1D nella griglia spaziale 2D dell'immagine e converte in array NumPy
        cam = avg_attn.reshape(grid_h, grid_w).cpu().detach().numpy()


    elif method == "eigen_cam":
        # 1. Estrae le rappresentazioni latenti (token) dei soli patch per la prima immagine del batch
        #    Forma: [num_patches, hidden_dim]
        patch_tokens = last_hidden_state[0, -num_patches:, :]

        # 2. Centra i dati sottraendo la media dei token lungo la dimensione del canale/embedding
        #    Assicura che la distribuzione delle caratteristiche sia centrata in zero per la PCA
        patch_tokens_centered = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)

        # 3. Calcola l'Analisi delle Componenti Principali (PCA) a rango ridotto isolando la 1ª componente (q=1)
        #    V rappresenta gli autovettori / componenti principali nello spazio latente
        _, _, V = torch.pca_lowrank(patch_tokens_centered, q=1)

        # 4. Proietta i patch centrati sul primo autovettore (V[:, 0]) per misurarne la varianza dominante
        #    Forma risultante: [num_patches]
        cam = torch.matmul(patch_tokens_centered, V[:, 0])

        # 5. Prende il valore assoluto per considerare l'ampiezza della deviazione dalla media (sia positiva che negativa)
        cam = torch.abs(cam)

        # 6. Riformatta il risultato 1D nella griglia spaziale 2D e converte in array NumPy per la visualizzazione
        cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()


    elif method == "grad_cam":
        # Grad-CAM standard. NB: la testa legge SOLO il CLS, quindi all'ultimo
        # strato il gradiente rispetto ai patch-token e' nullo. Usiamo l'output
        # del penultimo blocco (hidden_states[-2]): li' i patch influenzano ancora
        # il CLS attraverso l'ultimo blocco di attenzione, quindi il gradiente e' != 0.
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) < 2:
            return np.zeros((grid_h, grid_w))

        feat_full = hidden_states[-2]  # [1, seq_len, hidden]
        score = logits[0, pred_class_id]
        grads = torch.autograd.grad(score, feat_full, retain_graph=True)[0]

        grads = grads[0, -num_patches:, :]                 # [num_patches, hidden]
        feat = feat_full[0, -num_patches:, :].detach()     # [num_patches, hidden]
        alpha = grads.mean(dim=0)                          # [hidden]
        cam = torch.relu((feat * alpha).sum(dim=-1))       # [num_patches]

        cam = cam.reshape(grid_h, grid_w).cpu().numpy()

    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    return cam


# ==========================================
# 4. PREDIZIONE BBOX MULTI-METODO
# ==========================================
def predict_wsol_boxes_multi(
        model, processor, raw_image, device,
        methods=["linear_cam", "dino_attn", "eigen_cam", "grad_cam"],
        negatives_id=57, strictness=0.45
):
    orig_w, orig_h = raw_image.size
    target_size = 384
    processor.size = {"height": target_size, "width": target_size}

    inputs = processor(images=raw_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Calcoliamo le attenzioni solo se un metodo selezionato le richiede
    needs_attentions = "dino_attn" in methods
    # grad_cam richiede gli hidden states intermedi (all'ultimo strato i patch
    # non influenzano il CLS, quindi il loro gradiente sarebbe nullo)
    needs_hidden = "grad_cam" in methods
    needs_grad = "grad_cam" in methods

    with torch.set_grad_enabled(needs_grad):
        logits, outputs = model(
            pixel_values,
            return_attentions=needs_attentions,
            return_hidden_states=needs_hidden,
        )
        probs = torch.softmax(logits, dim=-1)
        pred_class_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class_id].item()

    empty_pred = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64)
    }

    if pred_class_id == negatives_id or confidence < 0.1:
        return {m: empty_pred for m in methods}

    patch_size = 16
    grid_h = target_size // patch_size
    grid_w = target_size // patch_size
    num_patches = grid_h * grid_w

    predictions = {}

    for method in methods:
        cam = generate_cam(
            method, logits, outputs, model.classifier,
            pred_class_id, num_patches, grid_h, grid_w
        )

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        if method in ROBUST_METHODS:
            # Binarizzazione robusta: top-k% di energia (percentile) invece di 0.45*max
            thr = int(np.percentile(heatmap_uint8, ROBUST_ENERGY_PERCENTILE))
            _, binary_mask = cv2.threshold(heatmap_uint8, max(thr, 1), 255, cv2.THRESH_BINARY)
        else:
            strict_thresh = int(strictness * heatmap_uint8.max())
            _, binary_mask = cv2.threshold(heatmap_uint8, strict_thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            predictions[method] = empty_pred
            continue

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

        # Area minima solo per i metodi con mappa appuntita
        if method in ROBUST_METHODS and w * h < ROBUST_MIN_AREA_FRAC * orig_w * orig_h:
            cx, cy = x + w / 2.0, y + h / 2.0
            target = ROBUST_MIN_AREA_FRAC * orig_w * orig_h
            scale = (target / max(1.0, float(w * h))) ** 0.5
            nw, nh = w * scale, h * scale
            x1 = int(max(0, cx - nw / 2.0))
            y1 = int(max(0, cy - nh / 2.0))
            x2 = int(min(orig_w, cx + nw / 2.0))
            y2 = int(min(orig_h, cy + nh / 2.0))
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

        predictions[method] = {
            "boxes": torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32),
            "scores": torch.tensor([confidence], dtype=torch.float32),
            "labels": torch.tensor([pred_class_id], dtype=torch.int64)
        }

    return predictions


# ==========================================
# 5. MAIN EVALUATION BENCHMARK
# ==========================================
def evaluate_benchmark():
    assert VERSION in CONFIGS, f"VERSION deve essere una tra {list(CONFIGS)}"
    cfg = CONFIGS[VERSION]

    DATASET_DIR = "./data/UNICT-VEDI_Dataset_bbox/Training"
    CHECKPOINT_PATH = cfg["checkpoint"]
    CLASS_MAP_PATH = cfg["class_map"]
    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

    # =======================================================
    # SELEZIONA QUI I METODI DA VALUTARE (Default: tutti e 4)
    # Opzioni: "linear_cam", "dino_attn", "eigen_cam", "grad_cam"
    # =======================================================
    SELECTED_METHODS = ["linear_cam", "dino_attn", "eigen_cam", "grad_cam"]

    SAMPLE_RATIO = 1
    STRICTNESS = 0.45
    SEED = 42

    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device selezionato: {device}")
    print(f"--> Metodi selezionati per il Benchmark: {SELECTED_METHODS}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"--> Versione modello: {VERSION}")
    print(f"--> Checkpoint: {CHECKPOINT_PATH}")

    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)
    idx_to_class = {int(v): k for k, v in class_map.items()}
    num_classes = len(class_map)

    if cfg["use_background"]:
        negatives_id = class_map.get("Negatives")
        if negatives_id is None:
            raise ValueError("Config v2 richiede la classe 'Negatives' nella class_map.")
        print(f"--> Background (Negatives) id = {negatives_id}")
    else:
        negatives_id = None
        print("--> Nessuna classe background (v1): nessuna predizione scartata come negativa")

    # Inizializza modello
    eager_attn = any(m in ["dino_attn", "grad_cam"] for m in SELECTED_METHODS)
    model = DINOClassifier(MODEL_NAME, num_classes, eager_attn=eager_attn)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Campionamento Stratificato
    sampled_files = []
    class_gt_counts = defaultdict(int)
    class_area_sum = defaultdict(float)  # somma delle aree relative (box/immagine) per classe

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

    # Per ciascun metodo due valutatori:
    #  - metrics_coco: mAP@[.5:.95] (COCO standard) -> map, mar_100, map_per_class
    #  - metrics_50:   solo IoU=0.5 -> map_per_class qui è l'AP50 per classe
    metrics_coco = {
        method: MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        for method in SELECTED_METHODS
    }
    metrics_50 = {
        method: MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                     iou_thresholds=[0.5])
        for method in SELECTED_METHODS
    }

    print(f"\n=== BENCHMARK SU {len(sampled_files)} IMMAGINI PER {len(SELECTED_METHODS)} METODI ===")
    for img_path, label_path in tqdm(sampled_files, desc="Valutazione Multi-Metodo"):
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w_img, h_img = raw_image.size
        target_dict = parse_yolo_gt(label_path, w_img, h_img)

        # Tracciamo conteggio GT e area relativa delle box per classe
        img_area = float(w_img * h_img)
        for lbl, box in zip(target_dict["labels"], target_dict["boxes"]):
            c = lbl.item()
            class_gt_counts[c] += 1
            class_area_sum[c] += float((box[2] - box[0]) * (box[3] - box[1])) / img_area

        # Genera predizioni per tutti i metodi selezionati
        preds_multi = predict_wsol_boxes_multi(
            model, processor, raw_image, device,
            methods=SELECTED_METHODS,
            negatives_id=negatives_id, strictness=STRICTNESS
        )

        # Aggiorna le metriche di ciascun metodo
        for method in SELECTED_METHODS:
            metrics_coco[method].update([preds_multi[method]], [target_dict])
            metrics_50[method].update([preds_multi[method]], [target_dict])

    # ==========================================
    # CALCOLO METRICHE
    # ==========================================
    res_coco = {m: metrics_coco[m].compute() for m in SELECTED_METHODS}
    res_50 = {m: metrics_50[m].compute() for m in SELECTED_METHODS}

    # ==========================================
    # STAMPA RISULTATI COMPARATIVI GLOBALI
    # ==========================================
    print("\n" + "=" * 80)
    print("             TABELLA COMPARATIVA METRICHE GLOBALI (BENCHMARK)")
    print("=" * 80)
    print(f"{'METODO WSOL':<30} | {'mAP@50':<12} | {'mAP@50:95 (COCO)':<18} | {'RECALL MEDIO':<12}")
    print("-" * 80)

    for method in SELECTED_METHODS:
        map_50 = res_coco[method]["map_50"].item() * 100
        map_coco = res_coco[method]["map"].item() * 100
        mar_100 = res_coco[method]["mar_100"].item() * 100
        print(f"{method:<30} | {map_50:.2f}%{' ' * 6} | {map_coco:.2f}%{' ' * 12} | {mar_100:.2f}%")

    print("=" * 80)

    # ==========================================
    # METRICHE PER CLASSE (con etichetta dimensionale)
    # ==========================================
    all_class_ids = sorted(class_gt_counts.keys())
    per_class_rows = []

    for c_id in all_class_ids:
        n_gt = class_gt_counts[c_id]
        mean_area = class_area_sum[c_id] / n_gt
        row = {
            "class_id": c_id,
            "class_name": idx_to_class.get(c_id, f"Class_{c_id}"),
            "n_gt": n_gt,
            "mean_area_ratio": round(mean_area, 4),
            "size_tier": compute_size_tier(mean_area),
        }
        for method in SELECTED_METHODS:
            row[f"{method}_map50"] = round(per_class_value(res_50[method], "map_per_class", c_id), 4)
            row[f"{method}_map50_95"] = round(per_class_value(res_coco[method], "map_per_class", c_id), 4)
            row[f"{method}_mar100"] = round(per_class_value(res_coco[method], "mar_100_per_class", c_id), 4)
        per_class_rows.append(row)

    # Stampa dettaglio per classe (AP@50)
    print("\n" + "=" * 130)
    print("                 DETTAGLIO AP@50 PER CLASSE E METODO (con tier dimensionale)")
    print("=" * 130)
    header = f"{'ID':<4} | {'NOME CLASSE':<32} | {'GT':<5} | {'AREA':<6} | {'TIER':<10}"
    for method in SELECTED_METHODS:
        header += f" | {method[:10]:<10}"
    print(header)
    print("-" * 130)
    for row in per_class_rows:
        line = (f"{row['class_id']:<4} | {row['class_name'][:32]:<32} | {row['n_gt']:<5} | "
                f"{row['mean_area_ratio']:<6.3f} | {row['size_tier']:<10}")
        for method in SELECTED_METHODS:
            line += f" | {row[f'{method}_map50']:<9.2f}%"
        print(line)
    print("=" * 130)

    # ==========================================
    # RIEPILOGO PER TIER (media non pesata sulle classi del tier)
    # ==========================================
    tier_summary = {}
    for tier_name, _ in SIZE_TIERS:
        rows = [r for r in per_class_rows if r["size_tier"] == tier_name]
        if not rows:
            continue
        tier_summary[tier_name] = {
            "classes_count": len(rows),
            "mean_area_ratio": float(np.mean([r["mean_area_ratio"] for r in rows])),
            "methods": {
                m: {
                    "map_50": float(np.mean([r[f"{m}_map50"] for r in rows])),
                    "map_50_95": float(np.mean([r[f"{m}_map50_95"] for r in rows])),
                    "mar_100": float(np.mean([r[f"{m}_mar100"] for r in rows])),
                }
                for m in SELECTED_METHODS
            },
        }

    print("\n" + "=" * 100)
    print("                 RIEPILOGO PER TIER DIMENSIONALE (mAP@50 | mAP@50:95)")
    print("=" * 100)
    for tier_name, t in tier_summary.items():
        line = f"{tier_name:<11} ({t['classes_count']:>2} classi, area media {t['mean_area_ratio']:.2f})"
        for m in SELECTED_METHODS:
            line += f" | {m[:10]}: {t['methods'][m]['map_50']:.1f}/{t['methods'][m]['map_50_95']:.1f}"
        print(line)
    print("=" * 100)

    # ==========================================
    # SALVATAGGIO RISULTATI SU DISCO (JSON + CSV)
    # ==========================================
    results = {
        "version": VERSION,
        "checkpoint": os.path.abspath(CHECKPOINT_PATH),
        "sample_ratio": SAMPLE_RATIO,
        "strictness": STRICTNESS,
        "seed": SEED,
        "n_images": len(sampled_files),
        "num_classes": num_classes,
        "negatives_id": negatives_id,
        "size_tiers": {name: upper if upper != float("inf") else None for name, upper in SIZE_TIERS},
        "methods": {
            m: {
                "map_50": float(res_coco[m]["map_50"]),
                "map_50_95": float(res_coco[m]["map"]),
                "mar_100": float(res_coco[m]["mar_100"]),
            }
            for m in SELECTED_METHODS
        },
        "tiers": tier_summary,
        "per_class": per_class_rows,
    }
    out_json = os.path.join(cfg["output_dir"], cfg["results_name"])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[+] Risultati salvati in: {out_json}")

    out_csv = os.path.join(cfg["output_dir"], cfg["csv_name"])
    if per_class_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_class_rows)
        print(f"[+] Metriche per classe salvate in: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VERSION = sys.argv[1].lower()
    evaluate_benchmark()