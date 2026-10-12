import os
from pathlib import Path

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "data" / "UNICT-VEDI_Dataset_bbox" / "Test"

# Se True: NON tocca nessun file su disco, stampa a video i dettagli delle conversioni
DRY_RUN = False

# Quanti esempi mostrare a video per ogni casistica intercettata
SAMPLE_PREVIEWS_PER_CASE = 5

IMG_W, IMG_H = 1216, 684
# ==============================================================================


def inspect_and_convert_labels(target_dir: Path, dry_run: bool = True):
    txt_files = [p for p in target_dir.rglob("*.txt")
                 if p.name not in ["gt_test_images.txt", "train_gt.txt", "val_gt.txt"]]

    print(f"Scansione: {target_dir}")
    print(f"File totali trovati: {len(txt_files)}")
    print(f"Modalità: {'DRY RUN (NESSUNA MODIFICA SU DISCO)' if dry_run else 'APPLICAZIONE SU DISCO'}\n")

    stats = {
        "pixel_voc": 0,    # [cls, xmin, ymin, xmax, ymax] in pixel assoluti
        "norm_coco": 0,    # [cls, xmin, ymin, w, h] in float normalizzati
        "skipped": 0,
        "files_modified": 0
    }

    previews_voc = []
    previews_coco = []

    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                stats["skipped"] += 1
                new_lines.append(line)
                continue

            cls_id = parts[0]
            c1, c2, c3, c4 = map(float, parts[1:5])

            # -------------------------------------------------------------
            # CASO 1: Pixel Assoluti [cls, xmin, ymin, xmax, ymax]
            # Valori > 1.0 (es. 56 637 2 1212 542)
            # -------------------------------------------------------------
            if any(v > 1.0 for v in [c1, c2, c3, c4]):
                x_min, y_min = c1 / IMG_W, c2 / IMG_H
                x_max, y_max = c3 / IMG_W, c4 / IMG_H

                w = max(0.0, x_max - x_min)
                h = max(0.0, y_max - y_min)
                x_c = x_min + (w / 2.0)
                y_c = y_min + (h / 2.0)

                transformed_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                new_lines.append(transformed_line)
                stats["pixel_voc"] += 1
                file_changed = True

                if len(previews_voc) < SAMPLE_PREVIEWS_PER_CASE:
                    previews_voc.append((txt_path.name, line.strip(), transformed_line.strip()))

            # -------------------------------------------------------------
            # CASO 2: Float Normalizzati COCO [cls, xmin, ymin, w, h]
            # Valori <= 1.0 (es. 4 0.28125 0.00146 0.23355 0.29678)
            # -------------------------------------------------------------
            else:
                x_min, y_min = c1, c2
                w, h = c3, c4

                x_c = x_min + (w / 2.0)
                y_c = y_min + (h / 2.0)

                transformed_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                new_lines.append(transformed_line)
                stats["norm_coco"] += 1
                file_changed = True

                if len(previews_coco) < SAMPLE_PREVIEWS_PER_CASE:
                    previews_coco.append((txt_path.name, line.strip(), transformed_line.strip()))

        if file_changed:
            stats["files_modified"] += 1
            if not dry_run:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

    # Rimozione cache solo in modalità scrittura
    if not dry_run:
        caches = list(target_dir.rglob("*.cache"))
        for c in caches:
            c.unlink()
        print(f"[!] Rimosse {len(caches)} cache Ultralytics (.cache)")

    # -------------------------------------------------------------
    # STAMPA REPORT DETTAGLIATO
    # -------------------------------------------------------------
    print("=" * 80)
    print("ESEMPI CONVERSIONE: PIXEL ASSOLUTI (VOC -> YOLO)")
    print("=" * 80)
    for fname, orig, conv in previews_voc:
        print(f"File: {fname}")
        print(f"  ORIG: {orig}")
        print(f"  YOLO: {conv}\n")

    print("=" * 80)
    print("ESEMPI CONVERSIONE: FLOAT NORMALIZZATI (COCO -> YOLO)")
    print("=" * 80)
    for fname, orig, conv in previews_coco:
        print(f"File: {fname}")
        print(f"  ORIG: {orig}")
        print(f"  YOLO: {conv}\n")

    print("=" * 80)
    print("STATISTICHE GLOBALI")
    print("=" * 80)
    print(f"File analizzati:         {len(txt_files)}")
    print(f"File che verrebbero modificati: {stats['files_modified']}")
    print(f"Box Pixel VOC convertiti:       {stats['pixel_voc']}")
    print(f"Box COCO float convertiti:      {stats['norm_coco']}")
    print(f"Box ignorati:                   {stats['skipped']}")
    print("=" * 80)

    if dry_run:
        print("\n--> Nessun dato è stato alterato. Controlla le anteprime sopra.")
        print("--> Se i calcoli tornano, imposta DRY_RUN = False ed esegui nuovamente.")


if __name__ == "__main__":
    inspect_and_convert_labels(TARGET_DIR, dry_run=DRY_RUN)