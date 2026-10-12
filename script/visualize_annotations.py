import os
import random
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFile
from transformers import AutoImageProcessor

# Importa le classi/funzioni dal tuo file originale (es. dataset.py)
# Se incolli questo codice nello stesso file del dataset, non ti servono gli import sotto
from src.dataset import VEDIDataset, create_class_map

ImageFile.LOAD_TRUNCATED_IMAGES = True


def visualize_and_save_samples(
        dataset_dir,
        output_dir="./annotated_samples",
        samples_per_class=5,
        seed=42
):
    random.seed(seed)

    # 1. Inizializzazione Class Map e Processor
    class_map = create_class_map(dataset_dir)
    idx_to_class = {v: k for k, v in class_map.items()}

    # Processor fittizio per istanziare il dataset
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

    # Carichiamo solo lo split di train
    train_dataset = VEDIDataset(dataset_dir, class_map, processor, split="train")
    print(f"Totale campioni nel train dataset: {len(train_dataset)}")

    # 2. Raggruppamento per classe (Stratificazione)
    class_to_samples = defaultdict(list)
    for sample in train_dataset.samples:
        class_id = sample["default_class_id"]
        class_to_samples[class_id].append(sample)

    print(f"Trovate {len(class_to_samples)} classi distillate.")

    os.makedirs(output_dir, exist_ok=True)

    # 3. Processamento e Disegno dei Box
    for class_id, samples in class_to_samples.items():
        class_name = idx_to_class[class_id]

        # Crea la sottocartella per la classe corrente
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Campionamento casuale stratificato
        k = min(samples_per_class, len(samples))
        selected_samples = random.sample(samples, k)

        for i, sample in enumerate(selected_samples):
            img_path = sample["img_path"]
            label_path = sample["label_path"]

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Errore caricamento immagine {img_path}: {e}")
                continue

            w_img, h_img = img.size
            draw = ImageDraw.Draw(img)

            boxes = []
            labels = []

            # 3a. Leggi le etichette reali
            if label_path and os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c, cx, cy, w, h = map(float, parts)
                            boxes.append([cx, cy, w, h])
                            labels.append(int(c))

            # 3b. Fallback: Se non ci sono box, usa l'intero frame
            if len(boxes) == 0:
                boxes.append([0.5, 0.5, 1.0, 1.0])
                labels.append(sample["default_class_id"])

            # 3c. Disegnamo ogni bounding box
            for box, label_id in zip(boxes, labels):
                cx, cy, bw, bh = box

                # Convertiamo da [cx, cy, w, h] normalizzati (0-1) a pixel [xmin, ymin, xmax, ymax]
                xmin = (cx - bw / 2) * w_img
                ymin = (cy - bh / 2) * h_img
                xmax = (cx + bw / 2) * w_img
                ymax = (cy + bh / 2) * h_img

                label_name = idx_to_class.get(label_id, f"Class_{label_id}")

                # Disegna il rettangolo (spessore 4px)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)

                # Testo della classe sopra il box
                text = f"{label_name} ({label_id})"

                # Se disponibile usiamo un font più grande, altrimenti quello base di PIL
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except IOError:
                    font = ImageFont.load_default()

                # Disegna uno sfondo nero dietro il testo per renderlo ben leggibile
                text_position = (xmin + 5, max(0, ymin - 20))
                draw.rectangle(
                    [text_position[0] - 2, text_position[1] - 2, text_position[0] + len(text) * 9,
                     text_position[1] + 16],
                    fill="black"
                )
                draw.text(text_position, text, fill="yellow", font=font)

            # Salva l'immagine con i box annotati
            base_filename = os.path.basename(img_path)
            save_path = os.path.join(class_output_dir, f"sample_{i + 1}_{base_filename}")
            img.save(save_path)

        print(f"Salvati {k} campioni per la classe '{class_name}' -> {class_output_dir}")

    print(f"\nCompletato! Controlla le immagini salvate nella cartella '{output_dir}'.")


if __name__ == "__main__":
    DATASET_DIR = "../data/UNICT-VEDI_Dataset_bbox/Training"

    visualize_and_save_samples(
        dataset_dir=DATASET_DIR,
        output_dir="./annotation_check",
        samples_per_class=5,  # Puoi aumentare se vuoi più immagini per classe
        seed=42
    )