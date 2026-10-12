import os
import glob
import random
import re

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageEnhance
from transformers import AutoImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_fn(batch):
    """
    batch: lista di tuple (pixel_values, target) ritornate da __getitem__
    """
    # 1. Impila le immagini in un unico Tensor [Batch_Size, 3, H, W]
    pixel_values = torch.stack([item[0] for item in batch])

    # 2. Mantieni i target come lista di dizionari (lunghezza = Batch_Size)
    targets = [item[1] for item in batch]

    return pixel_values, targets


class VEDIDataset(Dataset):
    def __init__(self, root_dir, class_map, processor, split="train", val_ratio=0.2):
        self.root_dir = root_dir
        self.class_map = class_map
        self.processor = processor
        self.split = split

        all_samples = []

        # Scansiona tutte le sottocartelle
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            class_id = class_map[folder_name]
            img_dir = os.path.join(folder_path, "images")
            label_dir = os.path.join(folder_path, "labels")

            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + sorted(
                glob.glob(os.path.join(img_dir, "*.png")))
            has_labels = os.path.exists(label_dir)

            # Sub-split deterministico per ogni cartella/POI (1 ogni 5 va in val)
            for idx, img_path in enumerate(img_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt") if has_labels else None

                is_val = (idx % int(1 / val_ratio) == 0)  # Esempio: 1 frame ogni 5 è Validation

                sample = {
                    "img_path": img_path,
                    "label_path": label_path,
                    "default_class_id": class_id
                }

                if split == "val" and is_val:
                    all_samples.append(sample)
                elif split == "train" and not is_val:
                    all_samples.append(sample)

        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample["img_path"]).convert("RGB")
        except Exception as e:
            # Se l'immagine è illeggibile/corrotta, creiamo un'immagine nera sostitutiva
            # così il training non crasha e il worker continua
            image = Image.new("RGB", (518, 518), (0, 0, 0))

        boxes = []
        labels = []

        # 1. Legge le etichette YOLO
        if sample["label_path"] and os.path.exists(sample["label_path"]):
            with open(sample["label_path"], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, cx, cy, w, h = map(float, parts)
                        boxes.append([cx, cy, w, h])
                        labels.append(int(c))

        # 2. Se non ci sono box, usa l'intero frame (regola del README)
        if len(boxes) == 0:
            boxes.append([0.5, 0.5, 1.0, 1.0])
            labels.append(sample["default_class_id"])

        if self.split == "train" and random.random() < 0.5:  # split disponibile come self.split, salvalo in __init__
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for b in boxes:
                b[0] = 1.0 - b[0]  # cx flip, w resta invariato

            # Color jitter leggero (solo in training)
        if self.split == "train":
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),  # [M, 4]
            "labels": torch.tensor(labels, dtype=torch.int64)  # [M]
        }

        # 3. Processa l'immagine con il Processor di DINOv3
        # L'output ha forma [1, 3, H, W], usiamo .squeeze(0) per averlo [3, H, W]
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        return pixel_values, target


class VEDIClassificationDataset(Dataset):
    def __init__(self, root_dir, processor, split="train", val_ratio=0.2, num_poi_classes=57):
        self.root_dir = root_dir
        self.processor = processor
        self.split = split

        # La classe 'Negatives' è l'indice 57 (la 58-esima classe)
        self.negatives_class_id = num_poi_classes

        all_samples = []

        # Scansione di TUTTE le cartelle di sessione/video
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            img_dir = os.path.join(folder_path, "images")
            label_dir = os.path.join(folder_path, "labels")

            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + sorted(
                glob.glob(os.path.join(img_dir, "*.png")))

            for idx, img_path in enumerate(img_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")

                real_class_id = None

                # 1. Leggiamo la VERA classe dal file .txt YOLO (prima colonna)
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            parts = lines[0].strip().split()
                            if len(parts) >= 1:
                                real_class_id = int(parts[0])

                # 2. Se non c'è etichetta nel .txt, è un NEGATIVO (muro, finestra, pavimento)
                if real_class_id is None:
                    real_class_id = self.negatives_class_id

                # Sub-split deterministico per mantenere la sincronia train/val
                is_val = (idx % int(1 / val_ratio) == 0)

                sample = {
                    "img_path": img_path,
                    "class_id": real_class_id
                }

                if split == "val" and is_val:
                    all_samples.append(sample)
                elif split == "train" and not is_val:
                    all_samples.append(sample)

        self.samples = all_samples
        print(f"[{split.upper()}] Caricati {len(self.samples)} campioni reali (POI + Negatives).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample["img_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 384), (0, 0, 0))

        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        return pixel_values, torch.tensor(sample["class_id"], dtype=torch.long)


def natural_sort_key(s):
    """
    Scompone la stringa in testo e numeri interrotti.
    Permette di ordinare 3.1, 3.2 ... 3.9, 3.10, 3.11 nel modo numerico corretto.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def create_class_map(training_dir):
    # 1. Prendiamo le cartelle escludendo 'Negatives' se già presente come cartella fisica
    folders = [
        f for f in os.listdir(training_dir)
        if os.path.isdir(os.path.join(training_dir, f)) and f != "Negatives"
    ]

    # 2. Ordiniamo le cartelle con l'ORDINAMENTO NATURALE (numerico)
    folders = sorted(folders, key=natural_sort_key)

    # 3. Mappiamo le 57 cartelle da 0 a 56
    class_map = {folder_name: idx for idx, folder_name in enumerate(folders)}

    # 4. Aggiungiamo 'Negatives' SEMPRE come ultima classe (indice 57)
    class_map["Negatives"] = len(folders)

    return class_map

if __name__ == "__main__":
    DATASET_DIR = "../data/UNICT-VEDI_Dataset_bbox/Training"
    class_map = create_class_map(DATASET_DIR)

    print(class_map)

    MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    #train_dataset = VEDIDataset(DATASET_DIR, class_map, processor, split="train")
    val_dataset = VEDIDataset(DATASET_DIR, class_map, processor, split="val")

    print(val_dataset[10000])