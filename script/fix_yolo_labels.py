import os
import glob
from PIL import Image

# Cartelle da correggere
TARGET_DIRS = [
    #r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\dataset_yolo_distilled",
    # r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Test",
    r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training",
]


def fix_yolo_line(line, img_w=None, img_h=None):
    """
    Corregge e normalizza una singola riga di annotazione YOLO (class_id xc yc w h).
    Garantisce che tutte le coordinate siano strettamente nel range [0.0, 1.0].
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = parts[0]
    coords = [float(x) for x in parts[1:5]]
    xc, yc, w, h = coords

    # 1. Se le coordinate sono in pixel (superano nettamente 1.5), normalizzale usando le dimensioni dell'immagine
    if (xc > 1.5 or yc > 1.5 or w > 1.5 or h > 1.5) and img_w and img_h:
        xc /= img_w
        yc /= img_h
        w /= img_w
        h /= img_h

    # 2. Converti da YOLO (xc, yc, w, h) a coordinate angolari (xmin, ymin, xmax, ymax)
    xmin = xc - (w / 2.0)
    ymin = yc - (h / 2.0)
    xmax = xc + (w / 2.0)
    ymax = yc + (h / 2.0)

    # 3. Clampa (taglia) tutte le coordinate nel range valido [0.0, 1.0]
    xmin_clamped = max(0.0, min(1.0, xmin))
    ymin_clamped = max(0.0, min(1.0, ymin))
    xmax_clamped = max(0.0, min(1.0, xmax))
    ymax_clamped = max(0.0, min(1.0, ymax))

    # 4. Ricalcola width, height e centri normalizzati
    new_w = xmax_clamped - xmin_clamped
    new_h = ymax_clamped - ymin_clamped
    new_xc = xmin_clamped + (new_w / 2.0)
    new_yc = ymin_clamped + (new_h / 2.0)

    # Scarta bbox non valide / degenerate
    if new_w <= 0.001 or new_h <= 0.001:
        return None

    # Formatta in stringa con 6 cifre decimali
    return f"{class_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"


def sanitize_dataset(base_dir):
    print(f"\n[+] Analisi e correzione della cartella: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"    [!] Cartella non trovata, salto: {base_dir}")
        return

    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)

    files_modified = 0
    lines_fixed = 0
    lines_removed = 0

    for txt_path in txt_files:
        # Ignora file di supporto come train.txt, val.txt, README.txt, ecc.
        if os.path.basename(txt_path) in [
            "train.txt",
            "val.txt",
            "README.txt",
            "gt_test_images.txt",
            "classes.txt",
        ]:
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            continue

        # Cerca l'immagine corrispondente se serve recuperare larghezza e altezza
        img_w, img_h = None, None
        img_candidates = glob.glob(os.path.splitext(txt_path)[0] + ".*")
        for cand in img_candidates:
            if cand.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    with Image.open(cand) as img:
                        img_w, img_h = img.size
                    break
                except Exception:
                    pass

        new_lines = []
        file_was_modified = False

        for line in lines:
            fixed_line = fix_yolo_line(line, img_w=img_w, img_h=img_h)

            if fixed_line is None:
                lines_removed += 1
                file_was_modified = True
            else:
                if fixed_line.strip() != line.strip():
                    lines_fixed += 1
                    file_was_modified = True
                new_lines.append(fixed_line + "\n")

        if file_was_modified:
            files_modified += 1
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

    print(
        f"    -> Completato: {files_modified} file modificati, "
        f"{lines_fixed} bbox rientrate in [0, 1], {lines_removed} bbox degenerate rimosse."
    )


if __name__ == "__main__":
    for target in TARGET_DIRS:
        sanitize_dataset(target)
    print("\n[✔] Sanitizzazione completata con successo su tutti i dataset!")