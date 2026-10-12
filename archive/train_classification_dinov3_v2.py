"""
Training DINOv3 per la classificazione dei POI - versione v2 ("pulita").

Questo script e' una riscrittura ex novo di train_classification_dinov3.py e NON
tocca i file esistenti. Differenze principali:

  1. CLASSE BACKGROUND esplicita. Un frame senza alcun POI (nessun .txt oppure
     .txt vuoto) riceve la classe "Negatives" (indice 57), non piu' la classe
     della cartella. Cosi' il modello impara che "non c'e' nulla" e la pipeline
     WSOL a valle non e' costretta a trovare un POI dove non esiste.
  2. PAIRING IMMAGINE<->LABEL ROBUSTO. I file .txt del dataset usano una padding
     diversa dalle immagini (es. frame_000010.txt vs frame_00010.jpg): qui il
     nome viene normalizzato su (prefisso, numero intero) per non perdere label
     (senza normalizzazione ~810 frame annotati finivano per errore nel background).
  3. SPLIT TRAIN/VAL ROBUSTO. Split per classe con una FINESTRA CONTIGUA di
     validation posizionata a offset casuale (seed per classe) e EMBARGO sui due
     bordi della finestra. Se i frame sono in ordine temporale evita il leakage
     tra frame adiacenti; se sono mescolati la finestra e' comunque un campione
     della sequenza. L'offset casuale per classe evita il bias sistematico di
     prendere sempre la coda.
  4. AUGMENTATION solo in training (RandomResizedCrop, flip, color jitter,
     lieve rotazione, random erasing).
  5. Training piu' solido: AMP, grad clipping, warmup + cosine, label smoothing,
     differential LR backbone/head (attivo quando si sblocca il backbone),
     model selection su F1-macro (non solo acc).

Uso:
    .\\venv\\Scripts\\python.exe train_classification_dinov3_v2.py

NOTA: gli iperparametri sono costanti in cima al file (nessun argomento CLI,
coerentemente col resto del progetto). Lo script NON viene lanciato in automatico.
"""

import os
import re
import json
import math
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from torchmetrics.classification import Accuracy, MulticlassF1Score

from PIL import Image, ImageFile

# Importiamo la class map canonica (ordinamento naturale + "Negatives" = 57)
from src.dataset import create_class_map

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# IPERPARAMETRI
# ==============================================================================
TRAIN_DIR = "./data/UNICT-VEDI_Dataset_bbox/Training"
CHECKPOINT_DIR = "./checkpoints_classification_v2"

MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMG_SIZE = 384

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4          # LR della head
BACKBONE_LR_FACTOR = 0.1      # LR backbone = LEARNING_RATE * questo fattore
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
UNFREEZE_LAST_N = 0           # 0 = backbone congelato (linear probe, ideale per CAM/WSOL).
                              # 1-4 = sblocca gli ultimi N blocchi (attiva il differential LR)

VAL_RATIO = 0.2               # frazione di validation per classe
EMBARGO = 5                   # frame scartati al confine train/val (anti-leakage)
PATIENCE = 5                  # early stopping (epoche senza miglioramento)

LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = False     # pesatura inversa della frequenza delle classi
USE_AMP = True                # mixed precision (solo su CUDA)
SEED = 42
NUM_WORKERS = 4

# ==============================================================================
# SEED / RIPRODUCIBILITA'
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ==============================================================================
# UTILITA' DI PAIRING IMMAGINE <-> LABEL
# ==============================================================================
def stem_key(stem):
    """
    Normalizza un nome file in (prefisso, numero) cosi' che varianti di padding
    diverse (frame_00010 vs frame_000010) producano la stessa chiave.
    """
    m = re.match(r"^(.*?)(\d+)$", stem)
    if not m:
        return (stem, -1)
    return (m.group(1), int(m.group(2)))


def load_label_class(label_path):
    """
    Ritorna la lista degli id-classe presenti nel .txt YOLO.
    Lista vuota se il file manca o e' vuoto (=> frame di background).
    """
    if not label_path or not os.path.exists(label_path):
        return []
    ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                ids.append(int(float(parts[0])))
    return ids


# ==============================================================================
# COSTRUZIONE DEI CAMPIONI E SPLIT ROBUSTO
# ==============================================================================
def build_samples(train_dir, class_map):
    """
    Ritorna (train_samples, val_samples), dove ogni sample e' (img_path, label).
    La label e' la classe della cartella se il frame contiene almeno un POI,
    altrimenti la classe "Negatives" (background).
    """
    background_id = class_map["Negatives"]
    train_samples, val_samples = [], []

    for folder in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, folder)
        if not os.path.isdir(folder_path) or folder not in class_map:
            continue

        class_id = class_map[folder]
        img_dir = os.path.join(folder_path, "images")
        lbl_dir = os.path.join(folder_path, "labels")

        img_paths = []
        for name in os.listdir(img_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(img_dir, name))
        img_paths.sort(key=lambda p: stem_key(os.path.splitext(os.path.basename(p))[0]))

        label_by_key = {}
        key_collisions = 0
        if os.path.isdir(lbl_dir):
            for name in os.listdir(lbl_dir):
                if name.lower().endswith(".txt"):
                    key = stem_key(os.path.splitext(name)[0])
                    if key in label_by_key:
                        key_collisions += 1
                    label_by_key[key] = os.path.join(lbl_dir, name)
        if key_collisions:
            print(f"  [!] {folder}: {key_collisions} collisioni di chiave label (padding) - "
                  f"controlla i nomi file")

        labeled = []
        for img_path in img_paths:
            key = stem_key(os.path.splitext(os.path.basename(img_path))[0])
            label_path = label_by_key.get(key)
            has_poi = len(load_label_class(label_path)) > 0
            label = class_id if has_poi else background_id
            labeled.append((img_path, label))

        rng = random.Random(f"{SEED}:{folder}")
        tr, va = split_block(labeled, VAL_RATIO, EMBARGO, rng)
        train_samples.extend(tr)
        val_samples.extend(va)

    return train_samples, val_samples


def split_block(samples, val_ratio, embargo, rng):
    """
    Split per classe con finestra contigua di validation a offset casuale.

    - La finestra di validation e' un blocco contiguo di `val_ratio` campioni,
      posizionato a un offset estratto dal `rng` (seed per classe): rimuove il
      bias sistematico della "coda" e mantiene il vantaggio anti-adiacenza.
    - `embargo` frame su entrambi i lati della finestra vengono esclusi dal train
      per non avere frame adiacenti a cavallo del confine.
    - Non restituisce mai un train vuoto: se l'embargo lo svuoterebbe, l'embargo
      viene annullato per quella classe.
    """
    n = len(samples)
    if n < 2:
        return list(samples), []

    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1))
    max_start = n - n_val
    start = rng.randint(0, max_start)
    val_positions = set(range(start, start + n_val))

    train = []
    for emb in (embargo, 0):
        drop = set(range(max(0, start - emb), start)) | \
            set(range(start + n_val, min(n, start + n_val + emb)))
        train = [s for i, s in enumerate(samples)
                 if i not in val_positions and i not in drop]
        if train:
            break
    val = [samples[i] for i in sorted(val_positions)]
    return train, val


# ==============================================================================
# DATASET
# ==============================================================================
class POIDataset(Dataset):
    def __init__(self, samples, processor, is_train):
        self.samples = samples
        self.processor = processor
        self.is_train = is_train

        if is_train:
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(
                    IMG_SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.25)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
                ),
                transforms.RandomRotation(degrees=10),
            ])
            self.erase = transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        else:
            self.augment = None
            self.erase = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.is_train:
            image = self.augment(image)

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.is_train:
            pixel_values = self.erase(pixel_values)

        return pixel_values, torch.tensor(label, dtype=torch.long)


def classification_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1].item() for item in batch], dtype=torch.long)
    return pixel_values, labels


# ==============================================================================
# MODELLO
# ==============================================================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes, unfreeze_last_n=0, dropout=DROPOUT):
        super().__init__()
        print(f"--> Caricamento Backbone {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
                layers = self.backbone.model.layer
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                layers = self.backbone.encoder.layer
            else:
                raise AttributeError("ModuleList dei layer non trovata nel backbone")
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            final_norm = getattr(self.backbone, "norm", None) or getattr(self.backbone, "layernorm", None)
            if final_norm is not None:
                for param in final_norm.parameters():
                    param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_class_weights(samples, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, label in samples:
        counts[label] += 1
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


# ==============================================================================
# TRAINING
# ==============================================================================
def train():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}")
    if USE_AMP and device.type != "cuda":
        print("--> AMP disattivato (nessuna GPU)")

    # --- Class map (57 POI + background = 58 classi) ---
    class_map = create_class_map(TRAIN_DIR)
    num_classes = len(class_map)
    background_id = class_map["Negatives"]
    print(f"--> {num_classes} classi (POI = 0..{num_classes - 2}, background = {background_id})")

    with open(os.path.join(CHECKPOINT_DIR, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=4)

    # Mappa dei soli 57 POI (senza background) per i consumer YOLO: gli id 0..56
    # coincidono con quelli della class_map storica, quindi le label GT restano valide.
    class_map_poi = {k: v for k, v in class_map.items() if k != "Negatives"}
    with open(os.path.join(CHECKPOINT_DIR, "class_map_poi.json"), "w") as f:
        json.dump(class_map_poi, f, indent=4)

    # --- Processor ---
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.size = {"height": IMG_SIZE, "width": IMG_SIZE}

    # --- Split ---
    train_samples, val_samples = build_samples(TRAIN_DIR, class_map)
    n_bg_train = sum(1 for _, l in train_samples if l == background_id)
    n_bg_val = sum(1 for _, l in val_samples if l == background_id)
    print(f"--> Train: {len(train_samples)} (background {n_bg_train}) | "
          f"Val: {len(val_samples)} (background {n_bg_val})")
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("Split vuoto: controlla i percorsi del dataset.")

    train_dataset = POIDataset(train_samples, processor, is_train=True)
    val_dataset = POIDataset(val_samples, processor, is_train=False)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=classification_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=classification_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    # --- Modello ---
    model = DINOClassifier(MODEL_NAME, num_classes, unfreeze_last_n=UNFREEZE_LAST_N).to(device)

    # --- Loss ---
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_samples, num_classes).to(device)
        print("--> Uso class weights (frequenza inversa)")
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING, weight=class_weights
    )

    # --- Optimizer con LR differenziato ---
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": LEARNING_RATE}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": LEARNING_RATE * BACKBONE_LR_FACTOR}
        )
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler(
        device=device.type, enabled=USE_AMP and device.type == "cuda"
    )

    # --- Metriche ---
    train_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    print("\n=== INIZIO TRAINING (v2) ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_acc_metric.reset()

        pbar = tqdm(train_loader, desc=f"Epoca [{epoch + 1}/{NUM_EPOCHS}] (Train)")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            train_acc_metric.update(logits.detach(), labels)
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{train_acc_metric.compute().item() * 100:.1f}%",
            })

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_train_acc = train_acc_metric.compute().item()

        # --- Validation ---
        model.eval()
        val_acc_metric.reset()
        val_f1_metric.reset()
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                    logits = model(pixel_values)
                val_acc_metric.update(logits, labels)
                val_f1_metric.update(logits, labels)

        val_acc = val_acc_metric.compute().item()
        val_f1 = val_f1_metric.compute().item()

        print(f"  Epoca {epoch + 1}: train_loss={epoch_loss:.4f} "
              f"train_acc={epoch_train_acc * 100:.2f}% | "
              f"val_acc={val_acc * 100:.2f}% val_f1_macro={val_f1:.4f}")
        history.append({
            "epoch": epoch + 1, "train_loss": epoch_loss,
            "train_acc": epoch_train_acc, "val_acc": val_acc, "val_f1_macro": val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"--> Early stopping all'epoca {epoch + 1} (nessun miglioramento da {PATIENCE} epoche)")
                break

    # --- Salvataggio best ---
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    checkpoint = {
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "best_val_acc": history[best_epoch - 1]["val_acc"] if best_epoch > 0 else 0.0,
        "val_f1": best_f1,
        "class_map": class_map,
        "config": {
            "model_name": MODEL_NAME, "img_size": IMG_SIZE,
            "num_classes": num_classes, "background_id": background_id,
            "val_ratio": VAL_RATIO, "embargo": EMBARGO, "seed": SEED,
            "unfreeze_last_n": UNFREEZE_LAST_N, "label_smoothing": LABEL_SMOOTHING,
        },
    }
    torch.save(checkpoint, best_path)

    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    info = (
        f"BEST epoch={best_epoch} val_acc={checkpoint['best_val_acc'] * 100:.2f}% "
        f"val_f1_macro={best_f1:.4f}\n"
        f"train={len(train_samples)} val={len(val_samples)} classes={num_classes}\n"
    )
    with open(os.path.join(CHECKPOINT_DIR, "info.txt"), "w") as f:
        f.write(info)

    print(f"\n[+] Miglior epoca: {best_epoch} | val_f1_macro={best_f1:.4f} | val_acc={checkpoint['best_val_acc'] * 100:.2f}%")
    print(f"[+] Checkpoint: {best_path}")
    print(f"[+] Class map:  {os.path.join(CHECKPOINT_DIR, 'class_map.json')}")


if __name__ == "__main__":
    train()
