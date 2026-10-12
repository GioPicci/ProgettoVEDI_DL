import os
import json
import torch
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision  # <-- Libreria Metriche

from src.dataset import VEDIDataset, collate_fn, natural_sort_key
from src.model import DINOObjectDetector
from src.loss import HungarianMatcher, SetCriterion


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_class_map(training_dir):
    folders = sorted(
        [f for f in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, f))],
        key=natural_sort_key
    )
    class_map = {folder_name: idx for idx, folder_name in enumerate(folders)}
    return class_map


def post_process_detr_outputs(outputs, threshold=0.05):
    """
    Converte i logits e le box di DETR nel formato richiesto da torchmetrics
    """
    pred_logits = outputs["pred_logits"]  # [B, N, num_classes + 1]
    pred_boxes = outputs["pred_boxes"]  # [B, N, 4] in formato [cx, cy, w, h]

    # Probabilità (escludiamo l'ultima classe che è il background)
    probs = pred_logits.softmax(-1)[:, :, :-1]  # [B, N, num_classes]
    scores, labels = probs.max(-1)  # [B, N]

    preds_list = []
    for i in range(len(pred_logits)):
        score_i = scores[i]
        label_i = labels[i]
        box_i = pred_boxes[i]

        # Filtriamo le query a bassissima confidence per velocizzare il calcolo del mAP
        keep = score_i > threshold

        preds_list.append({
            "boxes": box_i[keep],
            "scores": score_i[keep],
            "labels": label_i[keep]
        })

    return preds_list


def train():
    # ==========================================
    # 1. CONFIGURAZIONE & HYPERPARAMETERS
    # ==========================================
    DATASET_DIR = "./data/UNICT-VEDI_Dataset_bbox/Training"
    CHECKPOINT_DIR = "checkpoints_detection"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device selezionato per il training: {DEVICE}")

    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_NORM = 0.1
    USE_V3 = True
    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m" if USE_V3 else "facebook/dinov2-with-registers-small"


    # ==========================================
    # 2. CLASS MAP & MAPPING JSON
    # ==========================================
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Errore: La cartella {DATASET_DIR} non esiste. Verifica il percorso del dataset.")

    class_map = create_class_map(DATASET_DIR)
    num_classes = len(class_map)
    print(f"--> Trovate {num_classes} classi/POI nel dataset.")

    json_path = os.path.join(CHECKPOINT_DIR, "class_map.json")
    with open(json_path, "w") as f:
        json.dump(class_map, f, indent=4)

    # ==========================================
    # 3. PROCESSOR & DATALOADER
    # ==========================================
    print(f"--> Caricamento AutoImageProcessor per {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.size = {"height": 384, "width": 384}

    train_dataset = VEDIDataset(DATASET_DIR, class_map, processor, split="train")
    val_dataset = VEDIDataset(DATASET_DIR, class_map, processor, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 4. MODELLO & LOSS CRITERION
    # ==========================================
    print("--> Inizializzazione DINOObjectDetector...")
    model = DINOObjectDetector(num_classes=num_classes, num_queries=100, use_v3=USE_V3, unfreeze_last_n=0)
    model.to(DEVICE)

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(DEVICE)

    # --- Parametri divisi in due gruppi per LR differenziato ---
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad]

    # Lista FLAT di tutti i parametri allenabili, usata per il grad clipping (serve TUTTI insieme, non per gruppo)
    trainable_params = backbone_params + head_params
    print(
        f"--> Parametri allenabili: backbone={sum(p.numel() for p in backbone_params):,} | head={sum(p.numel() for p in head_params):,}")

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params, "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(1000, total_steps // 20)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)  # la funzione di prima

    val_map_metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    # BUG FIX: Inizializza best_map O best_val_loss FUORI dal ciclo delle epoche!
    best_map_50 = 0.0
    best_val_loss = float("inf")

    print("\n=== INIZIO TRAINING ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoca [{epoch + 1}/{NUM_EPOCHS}] (Train)")

        for pixel_values, targets in progress_bar:
            pixel_values = pixel_values.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            outputs = model(pixel_values)
            total_loss, loss_dict = criterion(outputs, targets)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()

            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.3f}",
                "CE": f"{loss_dict.get('loss_ce', 0):.2f}",
                "L1": f"{loss_dict.get('loss_bbox', 0):.2f}",
                "GIoU": f"{loss_dict.get('loss_giou', 0):.2f}"
            })

        epoch_loss = running_loss / len(train_loader)

        # ==========================================
        # VALIDATION PHASE (Loss + Metriche mAP)
        # ==========================================
        model.eval()
        val_loss = 0.0
        val_map_metric.reset()  # Reset metriche per la nuova epoca

        val_bar = tqdm(val_loader, desc=f"Epoca [{epoch + 1}/{NUM_EPOCHS}] (Val)")

        with torch.no_grad():
            for pixel_values, targets in val_bar:
                pixel_values = pixel_values.to(DEVICE)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                outputs = model(pixel_values)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()

                # Prepara le predizioni e aggiorna le metriche mAP
                preds = post_process_detr_outputs(outputs, threshold=0.05)
                val_map_metric.update(preds, targets)

        avg_val_loss = val_loss / len(val_loader)

        # Calcolo finale delle metriche mAP dell'epoca
        metric_results = val_map_metric.compute()
        map_50 = metric_results["map_50"].item()
        map_50_95 = metric_results["map"].item()

        print(f"\n--> Epoca [{epoch + 1}/{NUM_EPOCHS}] RISULTATI:")
        print(f"    Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(
            f"    [METRICHE] mAP@50: {map_50:.4f} ({map_50 * 100:.2f}%) | mAP@50:95: {map_50_95:.4f} ({map_50_95 * 100:.2f}%)")

        # SALVA IL MODELLO SE MIGLIORA IL mAP@50 (oppure se la val loss scende)
        if map_50 > best_map_50:
            best_map_50 = map_50
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map_50": best_map_50,
                "best_val_loss": best_val_loss,
                "class_map": class_map
            }, checkpoint_path)
            print(f"    [+] Nuovo miglior modello salvato! (mAP@50: {best_map_50 * 100:.2f}%)")
        print("-" * 60)

    print("\n=== TRAINING COMPLETATO ===")


if __name__ == "__main__":
    train()