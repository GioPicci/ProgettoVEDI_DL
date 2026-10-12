import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "UNICT-VEDI_Dataset_bbox" / "Test"
RUNS_DIR = REPO_ROOT / "runs" / "detect" / "runs_yolo_distilled"
CLASS_MAP_PATH = REPO_ROOT / "checkpoints_classification" / "class_map.json"
OUTPUT_TEST_RUNS = Path(__file__).resolve().parent / "runs_test_distill_on_gt_results"

MODELS = ["yolov8n", "yolo11n", "yolo26n"]
IMG_SIZE, BATCH_SIZE = 640, 16

# Aree medie rilevate nel training set (per class_id)
# Calcolate con "script/check_bbox_sizes_classes.py
CLASS_AREAS = {
    0: 0.998, 1: 0.998, 2: 0.998, 3: 0.891, 4: 0.739, 5: 0.890, 6: 0.629, 7: 0.688,
    8: 0.998, 9: 0.486, 10: 0.612, 11: 0.998, 12: 0.998, 13: 0.998, 14: 0.396, 15: 0.632,
    16: 0.998, 17: 0.648, 18: 0.459, 19: 0.757, 20: 0.661, 21: 0.072, 22: 0.571, 23: 0.051,
    24: 0.045, 25: 0.751, 26: 0.536, 27: 0.219, 28: 0.576, 29: 0.632, 30: 0.964, 31: 0.031,
    32: 0.272, 33: 0.315, 34: 0.736, 35: 0.827, 36: 0.895, 37: 0.925, 38: 0.813, 39: 0.897,
    40: 0.701, 41: 0.746, 42: 0.998, 43: 0.484, 44: 0.591, 45: 0.998, 46: 0.536, 47: 0.940,
    48: 0.855, 49: 0.548, 50: 0.647, 51: 0.817, 52: 0.786, 53: 0.297, 54: 0.876, 55: 0.658, 56: 0.841
}

TIERS = {
    "Small (<0.30)": [c for c, a in CLASS_AREAS.items() if a < 0.30],
    "Macro (0.30-0.79)": [c for c, a in CLASS_AREAS.items() if 0.30 <= a < 0.80],
    "Full-Frame (>=0.80)": [c for c, a in CLASS_AREAS.items() if a >= 0.80]
}


# ==========================================
# 2. GENERAZIONE DATASET YAML
# ==========================================
def setup_test_dataset(test_dir: Path, class_map_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(class_map_path, "r") as f:
        class_map = json.load(f)
    names = {int(v): k for k, v in class_map.items()}

    # Scansione estensioni supportate
    exts = ("*.jpg", "*.jpeg", "*.png")
    test_images = [str(p.resolve()) for ext in exts for p in test_dir.rglob(ext)]
    if not test_images:
        raise RuntimeError(f"Nessuna immagine trovata in: {test_dir}")

    txt_path = out_dir / "test_images.txt"
    txt_path.write_text("\n".join(test_images), encoding="utf-8")

    yaml_path = out_dir / "test_data.yaml"
    data = {
        "path": str(test_dir.resolve()),
        "train": str(txt_path.resolve()),
        "val": str(txt_path.resolve()),
        "names": names
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path


# ==========================================
# 3. VALUTAZIONE E METRICHE STRATIFICATE
# ==========================================
def compute_tier_metrics(metrics):
    """Calcola mAP50 e mAP50-95 aggregati per i 3 gruppi di classi."""
    eval_classes = list(metrics.box.ap_class_index)
    tier_results = {}

    for tier_name, target_classes in TIERS.items():
        # Indici corrispondenti nei tensori delle metriche
        indices = [i for i, c in enumerate(eval_classes) if c in target_classes]
        if indices:
            tier_results[tier_name] = {
                "classes_count": len(indices),
                "map50": float(np.mean([metrics.box.ap50[i] for i in indices])),
                "map50_95": float(np.mean([metrics.box.ap[i] for i in indices]))
            }
        else:
            tier_results[tier_name] = {"classes_count": 0, "map50": 0.0, "map50_95": 0.0}

    return tier_results


def evaluate_models():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--> Device: {device} | Output dir: {OUTPUT_TEST_RUNS}")

    yaml_path = setup_test_dataset(TEST_DIR, CLASS_MAP_PATH, OUTPUT_TEST_RUNS)
    summary = {}

    for tag in MODELS:
        print(f"\n{'='*70}\nVALUTAZIONE: {tag}\n{'='*70}")
        weight_candidates = list(RUNS_DIR.glob(f"**/{tag}/**/best.pt"))
        if not weight_candidates:
            print(f"[!] Checkpoint 'best.pt' per {tag} non trovato in {RUNS_DIR}. Salto.")
            continue

        model = YOLO(str(weight_candidates[0]))
        t0 = time.time()
        res = model.val(
            data=str(yaml_path),
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            project=str(OUTPUT_TEST_RUNS),
            name=f"eval_{tag}",
            exist_ok=True,
            split="val"
        )
        eval_time = round(time.time() - t0, 2)

        tier_metrics = compute_tier_metrics(res)

        summary[tag] = {
            "weights": str(weight_candidates[0]),
            "global": {
                "map50": float(res.box.map50),
                "map50_95": float(res.box.map),
                "precision": float(res.box.mp),
                "recall": float(res.box.mr),
                "speed_ms": float(res.speed["inference"]),
                "eval_sec": eval_time
            },
            "tiers": tier_metrics
        }

    # ==========================================
    # 4. REPORT A TERMINALE E SALVATAGGIO
    # ==========================================
    print("\n" + "=" * 92)
    print("                    RISULTATI COMPARATIVI STRATIFICATI (mAP@50)")
    print("=" * 92)
    print(f"{'MODELLO':<12} | {'OVERALL':<9} | {'SMALL (<0.3)':<14} | {'MACRO (0.3-0.8)':<16} | {'FULL-FRAME':<12}")
    print("-" * 92)
    for tag, d in summary.items():
        g_map = d["global"]["map50"] * 100
        s_map = d["tiers"]["Small (<0.30)"]["map50"] * 100
        m_map = d["tiers"]["Macro (0.30-0.79)"]["map50"] * 100
        f_map = d["tiers"]["Full-Frame (>=0.80)"]["map50"] * 100
        print(f"{tag:<12} | {g_map:>6.2f}%   | {s_map:>10.2f}%    | {m_map:>12.2f}%    | {f_map:>9.2f}%")
    print("=" * 92)

    json_path = OUTPUT_TEST_RUNS / "stratified_test_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Report salvato: {json_path}")


if __name__ == "__main__":
    evaluate_models()