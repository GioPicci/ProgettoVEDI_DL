"""
Valutazione del classificatore DINOv3 (57 classi POI).

Calcola e salva:
  1. Metriche globali (Accuracy Top-1, Accuracy Top-5, Macro/Weighted F1, Loss).
  2. Metriche disaggregate per classe (Support, Accuracy/Recall, Precision, F1, Tier, Area Media) -> CSV & JSON.
  3. Metriche disaggregate per Tier dimensionale (Tiny, Small, Macro, Full-Frame) -> CSV & JSON.
  4. Matrice di confusione 57x57 -> CSV e grafico ad alta risoluzione (PNG & PDF).

Uso:
    .\\venv\\Scripts\\python.exe test_classification_dinov3.py               # valuta sul Test set (default)
    .\\venv\\Scripts\\python.exe test_classification_dinov3.py --split val  # valuta sul Validation set di training
    .\\venv\\Scripts\\python.exe test_classification_dinov3.py --split both # valuta sia su Val che su Test
"""

import os
import re
import sys
import glob
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import VEDIDataset

# Aree medie relative calcolate per ciascuna delle 57 classi (CLASS_AREAS)
CLASS_AREAS = {
    0: 0.998, 1: 0.998, 2: 0.998, 3: 0.891, 4: 0.739, 5: 0.890, 6: 0.629, 7: 0.688,
    8: 0.998, 9: 0.486, 10: 0.612, 11: 0.998, 12: 0.998, 13: 0.998, 14: 0.396, 15: 0.632,
    16: 0.998, 17: 0.648, 18: 0.459, 19: 0.757, 20: 0.661, 21: 0.072, 22: 0.571, 23: 0.051,
    24: 0.045, 25: 0.751, 26: 0.536, 27: 0.219, 28: 0.576, 29: 0.632, 30: 0.964, 31: 0.031,
    32: 0.272, 33: 0.315, 34: 0.736, 35: 0.827, 36: 0.895, 37: 0.925, 38: 0.813, 39: 0.897,
    40: 0.701, 41: 0.746, 42: 0.998, 43: 0.484, 44: 0.591, 45: 0.998, 46: 0.536, 47: 0.940,
    48: 0.855, 49: 0.548, 50: 0.647, 51: 0.817, 52: 0.786, 53: 0.297, 54: 0.876, 55: 0.658, 56: 0.841
}

def get_tier(area_ratio):
    if area_ratio <= 0.10:
        return "Tiny"
    elif area_ratio < 0.30:
        return "Small"
    elif area_ratio < 0.80:
        return "Macro"
    return "Full-Frame"


def stem_key(stem):
    """Normalizza il nome file per tollerare padding diverso (Test1.0 0024 vs Test1.0 00024)."""
    m = re.match(r"^(.*?)(\d+)$", stem)
    if not m:
        return (stem, -1)
    return (m.group(1), int(m.group(2)))


# ==========================================
# 1. DATASET DI TEST A SESSIONI
# ==========================================
class VEDITestDataset(Dataset):
    """
    Test set a sessioni: Test/Test1..Test7/{images,labels}.
    Legge la classe del POI dalla PRIMA colonna della prima riga del file .txt YOLO.
    I frame senza etichetta nel test sono transizioni e vengono saltati.
    """
    def __init__(self, test_dir, valid_ids, processor):
        self.processor = processor
        self.valid_ids = set(valid_ids)
        self.samples = []

        for session in sorted(os.listdir(test_dir)):
            sdir = os.path.join(test_dir, session)
            if not os.path.isdir(sdir):
                continue
            img_dir = os.path.join(sdir, "images")
            lbl_dir = os.path.join(sdir, "labels")

            label_by_key = {}
            if os.path.isdir(lbl_dir):
                for name in os.listdir(lbl_dir):
                    if name.lower().endswith(".txt"):
                        label_by_key[stem_key(os.path.splitext(name)[0])] = os.path.join(lbl_dir, name)

            images = sorted(
                glob.glob(os.path.join(img_dir, "*.jpg"))
                + glob.glob(os.path.join(img_dir, "*.png"))
            )
            for img_path in images:
                key = stem_key(os.path.splitext(os.path.basename(img_path))[0])
                label_path = label_by_key.get(key)

                class_id = None
                if label_path and os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        rows = [line.split() for line in f if len(line.split()) >= 5]
                    if rows:
                        cid = int(float(rows[0][0]))
                        if cid in self.valid_ids:
                            class_id = cid

                if class_id is not None:
                    self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(class_id, dtype=torch.long)


def custom_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1].item() for item in batch], dtype=torch.long)
    return pixel_values, labels


def val_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1]["labels"][0].item() for item in batch], dtype=torch.long)
    return pixel_values, labels


# ==========================================
# 2. ARCHITETTURA MODELLO
# ==========================================
class DINOClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


# ==========================================
# 3. VALUTAZIONE E CALCOLO METRICHE
# ==========================================
def evaluate_split(model, dataloader, device, num_classes, idx_to_class, split_name, output_dir):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    all_preds = []
    all_targets = []
    all_top5_hits = []

    print(f"\n--> Inizio valutazione su split: {split_name.upper()} ({len(dataloader.dataset)} campioni)...")
    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc=f"Valutazione {split_name}"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits = model(pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = logits.argmax(dim=-1)
            top5 = logits.topk(min(5, num_classes), dim=-1).indices
            top5_hits = (top5 == labels.unsqueeze(-1)).any(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_top5_hits.extend(top5_hits.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_top5_hits = np.array(all_top5_hits)
    total_samples = len(all_targets)

    avg_loss = total_loss / max(1, total_samples)
    global_acc = float((all_preds == all_targets).mean()) * 100.0
    global_top5_acc = float(all_top5_hits.mean()) * 100.0

    # Calcolo Matrice di Confusione
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Calcolo metriche per classe
    per_class_records = []
    for c in range(num_classes):
        c_name = idx_to_class.get(c, f"Class_{c}")
        area = CLASS_AREAS.get(c, 0.5)
        tier = get_tier(area)

        support = int((all_targets == c).sum())
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)

        rec = (tp / support * 100.0) if support > 0 else None
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else (0.0 if support > 0 else None)

        per_class_records.append({
            "class_id": c,
            "class_name": c_name,
            "size_tier": tier,
            "mean_area_ratio": round(area, 4),
            "support": support,
            "correct": tp,
            "accuracy_recall_pct": round(rec, 2) if rec is not None else None,
            "precision_pct": round(prec, 2) if prec is not None else None,
            "f1_score_pct": round(f1, 2) if f1 is not None else None,
        })

    df_classes = pd.DataFrame(per_class_records)

    # Classi valutate (con support > 0)
    eval_classes = df_classes[df_classes["support"] > 0]
    macro_acc = float(eval_classes["accuracy_recall_pct"].mean())
    macro_prec = float(eval_classes["precision_pct"].mean())
    macro_f1 = float(eval_classes["f1_score_pct"].mean())

    weighted_f1 = float((eval_classes["f1_score_pct"] * eval_classes["support"]).sum() / eval_classes["support"].sum())

    # Calcolo metriche aggregate per Tier
    tier_summary = []
    tier_order = ["Tiny", "Small", "Macro", "Full-Frame"]
    for t in tier_order:
        sub = df_classes[df_classes["size_tier"] == t]
        sub_eval = sub[sub["support"] > 0]
        n_classes = len(sub)
        n_eval_classes = len(sub_eval)
        t_samples = int(sub["support"].sum())
        t_correct = int(sub["correct"].sum())

        t_acc_macro = float(sub_eval["accuracy_recall_pct"].mean()) if n_eval_classes > 0 else 0.0
        t_prec_macro = float(sub_eval["precision_pct"].mean()) if n_eval_classes > 0 else 0.0
        t_f1_macro = float(sub_eval["f1_score_pct"].mean()) if n_eval_classes > 0 else 0.0
        t_acc_weighted = float(t_correct / t_samples * 100.0) if t_samples > 0 else 0.0

        tier_summary.append({
            "tier": t,
            "num_classes_total": n_classes,
            "num_classes_evaluated": n_eval_classes,
            "total_samples": t_samples,
            "accuracy_macro_pct": round(t_acc_macro, 2),
            "accuracy_weighted_pct": round(t_acc_weighted, 2),
            "precision_macro_pct": round(t_prec_macro, 2),
            "f1_macro_pct": round(t_f1_macro, 2),
        })

    df_tiers = pd.DataFrame(tier_summary)

    # ==========================================
    # SALVATAGGIO FILE CSV & JSON
    # ==========================================
    split_dir = os.path.join(output_dir, f"{split_name}_results")
    os.makedirs(split_dir, exist_ok=True)

    csv_classes_path = os.path.join(split_dir, f"classification_{split_name}_per_class.csv")
    df_classes.to_csv(csv_classes_path, index=False)

    csv_tiers_path = os.path.join(split_dir, f"classification_{split_name}_per_tier.csv")
    df_tiers.to_csv(csv_tiers_path, index=False)

    # Matrice di confusione CSV
    cm_csv_path = os.path.join(split_dir, f"confusion_matrix_{split_name}.csv")
    df_cm = pd.DataFrame(cm, index=[idx_to_class[i] for i in range(num_classes)],
                         columns=[idx_to_class[i] for i in range(num_classes)])
    df_cm.to_csv(cm_csv_path)

    # Matrice di confusione Plot (PNG & PDF)
    plot_cm_path = os.path.join(split_dir, f"confusion_matrix_{split_name}.png")
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Matrice di Confusione - DINOv3 ({split_name.capitalize()} Set)", fontsize=14, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Classe Predetta (Indice ID)", fontsize=11)
    plt.ylabel("Classe Reale Ground Truth (Indice ID)", fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_cm_path, dpi=200)
    plt.savefig(plot_cm_path.replace(".png", ".pdf"))
    plt.close()

    # JSON di riepilogo
    summary = {
        "split": split_name,
        "total_samples": total_samples,
        "num_classes_total": num_classes,
        "num_classes_in_split": len(eval_classes),
        "loss": round(avg_loss, 4),
        "accuracy_top1_pct": round(global_acc, 2),
        "accuracy_top5_pct": round(global_top5_acc, 2),
        "macro_accuracy_pct": round(macro_acc, 2),
        "macro_precision_pct": round(macro_prec, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "weighted_f1_pct": round(weighted_f1, 2),
        "tier_summary": tier_summary,
    }
    json_path = os.path.join(split_dir, f"classification_{split_name}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # ==========================================
    # STAMPA A CONSOLE
    # ==========================================
    print("\n" + "=" * 80)
    print(f"        RISULTATI CLASSIFICATORE DINOv3 — SPLIT: {split_name.upper()}")
    print("=" * 80)
    print(f"  Campioni totali:        {total_samples}")
    print(f"  Classi rappresentate:   {len(eval_classes)} / {num_classes}")
    print(f"  Loss:                   {avg_loss:.4f}")
    print(f"  Top-1 Accuracy:         {global_acc:.2f}%")
    print(f"  Top-5 Accuracy:         {global_top5_acc:.2f}%")
    print(f"  Macro Recall / Acc:     {macro_acc:.2f}%")
    print(f"  Macro Precision:        {macro_prec:.2f}%")
    print(f"  Macro F1:               {macro_f1:.2f}%")
    print(f"  Weighted F1:            {weighted_f1:.2f}%")
    print("-" * 80)
    print("  RISULTATI PER TIER DIMENSIONALE:")
    for t_row in tier_summary:
        print(f"    - {t_row['tier']:10s} (N={t_row['num_classes_evaluated']:2d}/{t_row['num_classes_total']:2d} classi, "
              f"Campioni={t_row['total_samples']:4d}): "
              f"Acc Macro = {t_row['accuracy_macro_pct']:5.2f}% | "
              f"Acc Weighted = {t_row['accuracy_weighted_pct']:5.2f}% | "
              f"F1 Macro = {t_row['f1_macro_pct']:5.2f}%")
    print("=" * 80)
    print(f"[+] Dati dettagliati salvati in:")
    print(f"    - CSV per classe:         {csv_classes_path}")
    print(f"    - CSV per tier:           {csv_tiers_path}")
    print(f"    - JSON riepilogo:         {json_path}")
    print(f"    - Matrice di confusione:  {plot_cm_path}")
    print("=" * 80 + "\n")

    return summary, df_classes, df_tiers


# ==========================================
# 4. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Valutazione del classificatore DINOv3 su Test o Val set")
    parser.add_argument("--split", choices=["test", "val", "both"], default="test",
                        help="Split da valutare: 'test' (default), 'val' o 'both'")
    parser.add_argument("--checkpoint", default="./checkpoints_classification/best_model.pth",
                        help="Percorso del checkpoint .pth")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Dimensione del batch (default: 32)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Caricamento modello da: {checkpoint_path}")
    print(f"--> Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_map = checkpoint["class_map"]
    num_classes = len(class_map)
    idx_to_class = {v: k for k, v in class_map.items()}

    print(f"--> Modello a {num_classes} classi caricato.")
    if "epoch" in checkpoint:
        print(f"--> Epoca checkpoint: {checkpoint['epoch']} | Val Acc memorizzata: {checkpoint.get('best_val_acc', 0.0)*100:.2f}%")

    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"height": 384, "width": 384}

    model = DINOClassifier(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    splits_to_eval = ["test"] if args.split == "test" else (["val"] if args.split == "val" else ["test", "val"])

    for sp in splits_to_eval:
        if sp == "test":
            test_dir = "./data/UNICT-VEDI_Dataset_bbox/Test"
            test_dataset = VEDITestDataset(test_dir, list(class_map.values()), processor)
            loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
            )
            evaluate_split(model, loader, device, num_classes, idx_to_class, "test", output_dir)
        elif sp == "val":
            train_dir = "./data/UNICT-VEDI_Dataset_bbox/Training"
            from src.dataset import VEDIDataset
            val_dataset = VEDIDataset(train_dir, class_map, processor, split="val")
            loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=val_collate_fn, num_workers=4, pin_memory=True
            )
            evaluate_split(model, loader, device, num_classes, idx_to_class, "val", output_dir)


if __name__ == "__main__":
    main()
