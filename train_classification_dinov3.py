import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from torchmetrics.classification import Accuracy, MulticlassF1Score

# Importiamo il tuo dataset originale
from src.dataset import VEDIDataset, natural_sort_key


def classification_collate_fn(batch):
    """
    Adatta l'output del VEDIDataset originale per la classificazione:
    Estragga i pixel_values e prende solo il primo label del target (ignorando i BBOX).
    """
    pixel_values = torch.stack([item[0] for item in batch])
    # item[1]["labels"] contiene i class_id per ogni immagine
    labels = torch.tensor([item[1]["labels"][0].item() for item in batch], dtype=torch.long)
    return pixel_values, labels


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


class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super().__init__()
        print(f"--> Caricamento Backbone {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # Linear Classifier Head con Dropout per evitare overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Estragga il CLS token (primo token della sequenza)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


def train():
    # ==========================================
    # 1. CONFIGURAZIONE & HYPERPARAMETERS
    # ==========================================
    DATASET_DIR = "./data/UNICT-VEDI_Dataset_bbox/Training"
    CHECKPOINT_DIR = "./checkpoints_classification"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device selezionato: {DEVICE}")

    # Ora che non usiamo RT-DETR/Queries, possiamo usare un BATCH_SIZE più alto!
    BATCH_SIZE = 16  # Aumenta a 32 se la GPU ha abbastanza VRAM
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

    # ==========================================
    # 2. CLASS MAP & MAPPING JSON
    # ==========================================
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Errore: {DATASET_DIR} non esiste.")

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

    # Usiamo la nostra collate_fn per la classificazione
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=classification_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=classification_collate_fn, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 4. MODELLO, CRITERION & OPTIMIZER
    # ==========================================
    print("--> Inizializzazione DINOClassifier...")
    model = DINOClassifier(model_name=MODEL_NAME, num_classes=num_classes, freeze_backbone=False)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Learning Rate differenziato: backbone più lento (fine-tuning leggero), head più veloce
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.classifier.named_parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params, "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Metriche
    train_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)
    val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)
    val_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(DEVICE)

    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    best_val_acc = 0.0

    print("\n=== INIZIO TRAINING CLASSIFICAZIONE ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_acc_metric.reset()

        progress_bar = tqdm(train_loader, desc=f"Epoca [{epoch + 1}/{NUM_EPOCHS}] (Train)")

        for pixel_values, labels in progress_bar:
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(pixel_values)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_acc_metric.update(logits, labels)

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{train_acc_metric.compute().item() * 100:.1f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = train_acc_metric.compute().item()

        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()
        val_loss = 0.0
        val_acc_metric.reset()
        val_f1_metric.reset()

        val_bar = tqdm(val_loader, desc=f"Epoca [{epoch + 1}/{NUM_EPOCHS}] (Val)")

        with torch.no_grad():
            for pixel_values, labels in val_bar:
                pixel_values = pixel_values.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(pixel_values)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_acc_metric.update(logits, labels)
                val_f1_metric.update(logits, labels)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_acc_metric.compute().item()
        val_f1 = val_f1_metric.compute().item()

        print(f"\n--> RISULTATI Epoca [{epoch + 1}/{NUM_EPOCHS}]:")
        print(f"    Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}% | Val F1-Macro: {val_f1:.4f}")

        # SALVATAGGIO DEL MIGLIOR MODELLO (Basato sull'Accuracy di Validazione)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "val_f1": val_f1,
                "class_map": class_map
            }, checkpoint_path)
            print(f"    [+] Nuovo miglior modello salvato! (Val Acc: {best_val_acc * 100:.2f}%)")
        print("-" * 65)

    print("\n=== TRAINING COMPLETATO ===")


if __name__ == "__main__":
    train()