"""
Estrae, per ogni POI del Training, alcuni frame SENZA POI (nessun file .txt
oppure .txt vuoto) e li copia in data/no_poi_samples/<classe>/.

Serve a ispezionare visivamente se quei frame sono davvero "negativi" (nessun
oggetto/POI visibile), utile per decidere se il training di classificazione /
WSOL stia etichettando come POI cose che POI non sono.

Uso (da qualsiasi directory):
    .\\venv\\Scripts\\python.exe script\\extract_no_poi_samples.py
"""

import os
import glob
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "UNICT-VEDI_Dataset_bbox" / "Training"
OUT_DIR = ROOT / "data" / "no_poi_samples"

SAMPLES_PER_CLASS = 5
THUMB_W = 400
COLS = 5


def frame_has_poi(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    if label_path.stat().st_size == 0:
        return False
    with open(label_path, "r") as f:
        return any(len(line.split()) == 5 for line in f)


def even_sample(items, n):
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def make_grid(image_paths, out_path: Path):
    thumbs = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        ratio = THUMB_W / im.width
        thumbs.append((im.resize((THUMB_W, max(1, int(im.height * ratio)))), p.name))
    if not thumbs:
        return
    cell_h = max(t.height for t, _ in thumbs) + 20
    rows = (len(thumbs) + COLS - 1) // COLS
    canvas = Image.new("RGB", (THUMB_W * COLS, cell_h * rows), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for i, (thumb, name) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        x, y = c * THUMB_W, r * cell_h
        canvas.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 2), name, fill=(255, 255, 255))
    canvas.save(out_path)


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training non trovato: {TRAIN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
    total_no_poi = 0
    print(f"{'classe':45s} {'frame':>7s} {'senza-POI':>10s} {'estratti':>9s}")
    print("-" * 76)

    for cls in classes:
        cls_dir = TRAIN_DIR / cls
        img_dir = cls_dir / "images"
        lbl_dir = cls_dir / "labels"

        images = sorted(
            glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))
        )
        no_poi = []
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if not frame_has_poi(lbl_dir / f"{stem}.txt"):
                no_poi.append(Path(img_path))

        total_no_poi += len(no_poi)
        picked = even_sample(no_poi, SAMPLES_PER_CLASS)

        out_cls = OUT_DIR / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        for p in picked:
            shutil.copy2(p, out_cls / p.name)
        if picked:
            make_grid(picked, out_cls / "_grid.png")

        print(f"{cls:45s} {len(images):>7d} {len(no_poi):>10d} {len(picked):>9d}")

    print("-" * 76)
    print(f"Totale frame senza POI: {total_no_poi}")
    print(f"Output: {OUT_DIR}")
    print("Apri le _grid.png per l'ispezione visiva rapida.")


if __name__ == "__main__":
    main()
