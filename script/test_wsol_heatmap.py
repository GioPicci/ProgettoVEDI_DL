import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ==========================================
# 1. ARCHITETTURA DEL MODELLO
# ==========================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # AGGIUNTA: attn_implementation="eager" forza il calcolo delle mappe di attenzione
        self.backbone = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=True)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, outputs

# ==========================================
# 2. HELPER: TRASFORMA HEATMAP IN BOUNDING BOX
# ==========================================
def heatmap_to_bbox(heatmap, orig_size, thresh_type="otsu", percentile=0.65):
    """
    Riceve una heatmap 2D e restituisce l'immagine con il BBOX disegnato,
    la maschera binaria e le coordinate (x, y, w, h).
    """
    orig_w, orig_h = orig_size
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Normalizzazione tra 0 e 255
    if heatmap_resized.max() > heatmap_resized.min():
        norm_map = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    else:
        norm_map = heatmap_resized

    heatmap_uint8 = (norm_map * 255).astype(np.uint8)

    # Scelta del Thresholding
    if thresh_type == "otsu":
        _, binary_mask = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # Threshold relativo al picco
        strict_thresh = int(percentile * heatmap_uint8.max())
        _, binary_mask = cv2.threshold(heatmap_uint8, strict_thresh, 255, cv2.THRESH_BINARY)

    # Pulizia Morfologica (rimuove rumore)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Estrazione Contorni
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None
    if len(contours) > 0:
        # Prende il contorno con area maggiore
        largest_contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(largest_contour)  # (x, y, w, h)

    return norm_map, binary_mask, bbox


# ==========================================
# 3. IMPLEMENTAZIONE DEI 4 METODI WSOL
# ==========================================

# METODO 1: Linear-CAM (Prodotto scalare Pesi Classifier x Patch Tokens)
def get_linear_cam(last_hidden_state, classifier_weights, pred_class_id, num_patches, grid_h, grid_w):
    patch_tokens = last_hidden_state[0, -num_patches:, :]  # [576, Hidden]
    class_w = classifier_weights[pred_class_id]  # [Hidden]

    cam = torch.matmul(patch_tokens, class_w)  # [576]
    cam = torch.relu(cam)
    cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()
    return cam


# METODO 2: DINO Native Self-Attention (Attenzione del token [CLS])
def get_dino_self_attention(attentions, num_patches, grid_h, grid_w):
    # attentions[-1] ha forma [1, num_heads, seq_len, seq_len]
    last_attn = attentions[-1][0]  # [num_heads, seq_len, seq_len]

    # Prendiamo l'attenzione dal token [CLS] (indice 0) verso i patch token (ultimi N)
    cls_to_patches_attn = last_attn[:, 0, -num_patches:]  # [num_heads, 576]

    # Media tra tutte le teste di attenzione
    avg_attn = cls_to_patches_attn.mean(dim=0)  # [576]
    cam = avg_attn.reshape(grid_h, grid_w).cpu().detach().numpy()
    return cam


# METODO 3: Eigen-CAM (PCA / SVD sui Patch Tokens)
def get_eigen_cam(last_hidden_state, num_patches, grid_h, grid_w):
    patch_tokens = last_hidden_state[0, -num_patches:, :]  # [576, Hidden]

    # Centriamo le feature (SVD per PCA)
    patch_tokens_centered = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(patch_tokens_centered, q=1)  # Primo componente principale

    # Proiezione sul 1° componente principale
    cam = torch.matmul(patch_tokens_centered, V[:, 0])  # [576]
    cam = torch.abs(cam)  # Magnitudo della variazione
    cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()
    return cam


# METODO 4: Grad-CAM per ViT (Gradient-Weighted Patch Activations)
def get_grad_cam(logits, attentions, pred_class_id, num_patches, grid_h, grid_w):
    """
    Grad-CAM corretto per Vision Transformers (ViT / DINO):
    Calcola il gradiente dello score della classe rispetto alla mappa di Self-Attention
    dell'ultimo layer, pesando l'attenzione del token [CLS] per l'importanza della classe.
    """
    if attentions is None or len(attentions) == 0:
        return np.zeros((grid_h, grid_w))

    last_layer_attn = attentions[-1]  # [1, num_heads, seq_len, seq_len]
    score = logits[0, pred_class_id]

    # Calcolo dei gradienti dello score della classe rispetto alle attenzioni
    grads = torch.autograd.grad(score, last_layer_attn, retain_graph=True)[0]

    # Estraiamo l'attenzione e il gradiente dal token [CLS] (indice 0) verso i patch token (ultimi N)
    cls_attn = last_layer_attn[0, :, 0, -num_patches:]  # [num_heads, 576]
    cls_grads = grads[0, :, 0, -num_patches:]           # [num_heads, 576]

    # Pesiamo ogni testa di attenzione con il suo gradiente positivo (ReLU)
    weighted_attn = torch.relu(cls_grads * cls_attn)

    # Media tra le teste di attenzione (num_heads)
    cam = weighted_attn.mean(dim=0)                      # [576]
    cam = cam.reshape(grid_h, grid_w).cpu().detach().numpy()
    return cam


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 4. PIPELINE DI BENCHMARK E VISUALIZZAZIONE
# ==========================================
def run_wsol_benchmark(
        image_path,
        checkpoint_path=None,
        class_map_path=None,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        output_save_path=None
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(REPO_ROOT, "checkpoints_classification", "best_model.pth")
    if class_map_path is None:
        class_map_path = os.path.join(REPO_ROOT, "checkpoints_classification", "class_map.json")
    if output_save_path is None:
        out_dir = os.path.join(REPO_ROOT, "script", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_save_path = os.path.join(out_dir, "wsol_benchmark_comparison.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Esecuzione Benchmark su Device: {device}")

    # 1. Carica Mappatura Classi
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    # 2. Carica Modello e Pesi
    model = DINOClassifier(model_name, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Preprocessing Immagine
    processor = AutoImageProcessor.from_pretrained(model_name)
    target_size = 384
    processor.size = {"height": target_size, "width": target_size}

    raw_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_image.size

    inputs = processor(images=raw_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # 4. Forward Pass con Tracciamento Gradienti
    logits, outputs = model(pixel_values)
    probs = torch.softmax(logits, dim=-1)
    pred_class_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_class_id].item()
    pred_class_name = idx_to_class.get(pred_class_id, f"Class_{pred_class_id}")

    print(f"--> Classe Predetta: '{pred_class_name}' ({confidence * 100:.2f}%)")

    # Parametri della griglia patch
    patch_size = 16
    grid_h = target_size // patch_size
    grid_w = target_size // patch_size
    num_patches = grid_h * grid_w

    # Pesi dell'ultimo layer per Linear-CAM
    classifier_weights = model.classifier[-1].weight

    # 5. Generazione delle Heatmap con i 4 Metodi
    last_hidden_state = outputs.last_hidden_state

    cams = {
        "Linear-CAM (Weights x Tokens)": get_linear_cam(last_hidden_state, classifier_weights, pred_class_id,
                                                        num_patches, grid_h, grid_w),
        "DINO Self-Attention ([CLS])": get_dino_self_attention(outputs.attentions, num_patches, grid_h, grid_w),
        "Eigen-CAM (PCA Feature Map)": get_eigen_cam(last_hidden_state, num_patches, grid_h, grid_w),
        "Grad-CAM (Gradients x Attn)": get_grad_cam(logits, outputs.attentions, pred_class_id, num_patches, grid_h,
                                                    grid_w)
    }

    # 6. Plotting Comparativo (1 Riga, 5 Colonne: Origine + 4 Metodi)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Colonna 0: Immagine Originale
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Input: {pred_class_name}\nConf: {confidence * 100:.1f}%", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Colonne 1-4: I Metodi WSOL
    for idx, (method_name, cam) in enumerate(cams.items(), start=1):
        # Calcolo Bounding Box tramite Otsu Thresholding
        norm_map, _, bbox = heatmap_to_bbox(cam, (orig_w, orig_h), thresh_type="otsu")

        img_with_box = np.array(raw_image).copy()

        if bbox is not None:
            x, y, w, h = bbox
            # Disegna rettangolo verde per il Bounding Box
            cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Rendering
        axes[idx].imshow(img_with_box)
        axes[idx].imshow(norm_map, cmap="jet", alpha=0.5)
        axes[idx].set_title(f"{method_name}", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle(f"WSOL SOTA Benchmark Comparison - DINOv3", fontsize=16, y=1.02)
    plt.tight_layout()

    output_save_path = output_save_path.split(".")[0] + os.path.basename(image_path).split(".")[0] + ".png"
    plt.savefig(output_save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Grafico comparativo salvato con successo in: {output_save_path}")
    plt.show()


if __name__ == "__main__":
    # TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Training/8.11_BustoPietroTacchini/images/frame_000646.jpg"
    # TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Training/4.1_Quadro/images/frame_0005.jpg"
    TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Training/5.4_PortaIngressoRef.Piccolo/images/frame_000149.jpg"
    TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Test/Test3/images/Test3.4 2539.jpg"
    TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Test/Test3/images/Test3.5 3810.jpg"
    #TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Test/Test3/images/Test3.6 1312.jpg"
    #TEST_IMAGE_PATH = "../data/UNICT-VEDI_Dataset_bbox/Training/5.7_S.Agata/images/frame_000149.jpg"

    if os.path.exists(TEST_IMAGE_PATH):
        run_wsol_benchmark(TEST_IMAGE_PATH)
    else:
        print("[-] Seleziona un percorso valido per TEST_IMAGE_PATH per avviare il benchmark!")