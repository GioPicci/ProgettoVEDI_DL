import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURAZIONE PERCORSI E PARAMETRI
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Cartella in cui sono salvati i file di appoggio del benchmark
BENCHMARK_DIR = SCRIPT_DIR / "runs_sample_efficiency"
TEST_YAML = BENCHMARK_DIR / "test_eval.yaml"
OUTPUT_DIR = BENCHMARK_DIR / "stratified_test_results"

# (Opzionale) pesi della YOLO addestrata sul dataset distillato (WSOL).
# Se il file esiste viene valutato con la stessa procedura e salvato in distilled_per_class.csv.
# Metti None per saltare.
DISTILLED_WEIGHTS = Path(
    r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\runs\detect\runs_yolo_distilled\yolov8n\weights\best.pt"
)

IMG_SIZE = 640
BATCH_SIZE = 16
TARGET_DINO_MAP = 37.39

# Aree medie relative calcolate per class_id
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

# class_id -> nome del tier
CLASS_TO_TIER = {c: tier for tier, classes in TIERS.items() for c in classes}

PER_CLASS_FIELDS = [
    "class_id", "class_name", "mean_area_ratio", "size_tier",
    "ap50", "ap50_95", "precision", "recall"
]


# ==========================================
# 2. FUNZIONI DI CALCOLO E METRICHE
# ==========================================
def extract_metadata(folder_name: str):
    match = re.search(r"(\d+)pct_s(\d+)", folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def compute_tier_metrics(metrics):
    """Calcola mAP@50 e mAP@50-95 per ogni tier in percentuale (0-100)."""
    eval_classes = list(metrics.box.ap_class_index)
    tier_results = {}

    for tier_name, target_classes in TIERS.items():
        indices = [i for i, c in enumerate(eval_classes) if c in target_classes]
        if indices:
            m50 = float(np.mean([metrics.box.ap50[i] for i in indices])) * 100.0
            mall = float(np.mean([metrics.box.ap[i] for i in indices])) * 100.0
            tier_results[tier_name] = {
                "classes_count": len(indices),
                "map50": round(m50, 2),
                "map50_95": round(mall, 2)
            }
        else:
            tier_results[tier_name] = {"classes_count": 0, "map50": 0.0, "map50_95": 0.0}

    return tier_results


def extract_per_class(metrics):
    """Metriche per classe (in %), con area media e tier. Solo classi presenti nel test set."""
    names = getattr(metrics, "names", {}) or {}
    rows = []
    for i, c in enumerate(metrics.box.ap_class_index):
        c = int(c)
        rows.append({
            "class_id": c,
            "class_name": names.get(c, str(c)),
            "mean_area_ratio": CLASS_AREAS.get(c),
            "size_tier": CLASS_TO_TIER.get(c),
            "ap50": round(float(metrics.box.ap50[i]) * 100.0, 2),
            "ap50_95": round(float(metrics.box.ap[i]) * 100.0, 2),
            "precision": round(float(metrics.box.p[i]) * 100.0, 2),
            "recall": round(float(metrics.box.r[i]) * 100.0, 2),
        })
    return rows


def run_validation(pt_path, device):
    model = YOLO(str(pt_path))
    t0 = time.time()
    res = model.val(
        data=str(TEST_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        split="val",
        verbose=False,
        plots=False
    )
    return res, round(time.time() - t0, 2)


# ==========================================
# 3. VALUTAZIONE RUN SUL TEST SET
# ==========================================
def evaluate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"

    if not TEST_YAML.exists():
        raise FileNotFoundError(f"[!] File di test mancante: {TEST_YAML.resolve()}")

    print(f"[*] Ricerca checkpoint best.pt a partire da: {SCRIPT_DIR.resolve()}")
    all_best_pts = list(SCRIPT_DIR.rglob("best.pt"))

    run_checkpoints = {}
    for pt in all_best_pts:
        run_name = pt.parent.parent.name
        pct, seed = extract_metadata(run_name)
        if pct is not None and seed is not None:
            run_checkpoints[run_name] = (pct, seed, pt)

    if not run_checkpoints:
        print("[!] Nessun modello valido trovato corrispondente al pattern {pct}pct_s{seed}.")
        return

    print(f"[+] Trovati {len(run_checkpoints)} modelli da valutare sul test set.\n")

    single_runs_data = []

    for run_name, (pct, seed, pt_path) in sorted(run_checkpoints.items(), key=lambda x: (x[1][0], x[1][1])):
        print(f"\n{'='*75}\nVALUTAZIONE TEST: {run_name} (Frazione: {pct}%, Seed: {seed})\n{'='*75}")
        res, eval_time = run_validation(pt_path, device)

        tiers = compute_tier_metrics(res)
        per_class = extract_per_class(res)
        overall_map50 = round(float(res.box.map50) * 100.0, 2)
        overall_map_all = round(float(res.box.map) * 100.0, 2)

        run_info = {
            "run_name": run_name,
            "fraction_pct": pct,
            "seed": seed,
            "weights": str(pt_path.resolve()),
            "eval_time_sec": eval_time,
            "overall": {
                "map50": overall_map50,
                "map50_95": overall_map_all
            },
            "tiers": tiers,
            "per_class": per_class
        }
        single_runs_data.append(run_info)

        print(f"--> Completato in {eval_time}s | Overall mAP@50: {overall_map50:.2f}%")
        print(f"    Small: {tiers['Small (<0.30)']['map50']:.2f}% | Macro: {tiers['Macro (0.30-0.79)']['map50']:.2f}% | Full-Frame: {tiers['Full-Frame (>=0.80)']['map50']:.2f}%")

    # ==========================================
    # 4. AGGREGAZIONE (MEDIA +/- STD)
    # ==========================================
    summary_grouped = {}
    fractions = sorted(list({r["fraction_pct"] for r in single_runs_data}))

    def get_stats(vals):
        return {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}

    # Aggregazione per classe (media/std tra i seed di ciascuna frazione)
    class_summary_rows = []

    for f in fractions:
        f_runs = [r for r in single_runs_data if r["fraction_pct"] == f]
        label = f"{f}%"

        per_class_vals = defaultdict(lambda: {"ap50": [], "ap50_95": [], "meta": None})
        for r in f_runs:
            for row in r["per_class"]:
                d = per_class_vals[row["class_id"]]
                d["ap50"].append(row["ap50"])
                d["ap50_95"].append(row["ap50_95"])
                d["meta"] = row
        for c_id in sorted(per_class_vals):
            d = per_class_vals[c_id]
            s50, s5095 = get_stats(d["ap50"]), get_stats(d["ap50_95"])
            class_summary_rows.append({
                "fraction_pct": f,
                "n_runs": len(d["ap50"]),
                "class_id": c_id,
                "class_name": d["meta"]["class_name"],
                "mean_area_ratio": d["meta"]["mean_area_ratio"],
                "size_tier": d["meta"]["size_tier"],
                "ap50_mean": s50["mean"], "ap50_std": s50["std"],
                "ap50_95_mean": s5095["mean"], "ap50_95_std": s5095["std"],
            })

        summary_grouped[label] = {
            "fraction_pct": f,
            "n_runs": len(f_runs),
            "status_vs_dino": "SUPERATO" if np.mean([r["overall"]["map50"] for r in f_runs]) >= TARGET_DINO_MAP else "INFERIORE",
            "overall_map50": get_stats([r["overall"]["map50"] for r in f_runs]),
            "overall_map50_95": get_stats([r["overall"]["map50_95"] for r in f_runs]),
            "small_map50": get_stats([r["tiers"]["Small (<0.30)"]["map50"] for r in f_runs]),
            "small_map50_95": get_stats([r["tiers"]["Small (<0.30)"]["map50_95"] for r in f_runs]),
            "macro_map50": get_stats([r["tiers"]["Macro (0.30-0.79)"]["map50"] for r in f_runs]),
            "macro_map50_95": get_stats([r["tiers"]["Macro (0.30-0.79)"]["map50_95"] for r in f_runs]),
            "full_frame_map50": get_stats([r["tiers"]["Full-Frame (>=0.80)"]["map50"] for r in f_runs]),
            "full_frame_map50_95": get_stats([r["tiers"]["Full-Frame (>=0.80)"]["map50_95"] for r in f_runs]),
            "runs": f_runs
        }

    # ==========================================
    # 5. SALVATAGGIO OUTPUT
    # ==========================================
    # JSON (ora include anche per_class dentro ogni run)
    json_path = OUTPUT_DIR / "stratified_sample_efficiency_test.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_grouped, f, indent=4, ensure_ascii=False)

    # CSV Dettagliato
    csv_detail = OUTPUT_DIR / "runs_stratified_detail.csv"
    with open(csv_detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Fraction (%)", "Seed", "Run Name",
            "Overall mAP50", "Overall mAP50-95",
            "Small mAP50", "Small mAP50-95",
            "Macro mAP50", "Macro mAP50-95",
            "FullFrame mAP50", "FullFrame mAP50-95",
            "Checkpoint"
        ])
        for r in single_runs_data:
            writer.writerow([
                r["fraction_pct"], r["seed"], r["run_name"],
                r["overall"]["map50"], r["overall"]["map50_95"],
                r["tiers"]["Small (<0.30)"]["map50"], r["tiers"]["Small (<0.30)"]["map50_95"],
                r["tiers"]["Macro (0.30-0.79)"]["map50"], r["tiers"]["Macro (0.30-0.79)"]["map50_95"],
                r["tiers"]["Full-Frame (>=0.80)"]["map50"], r["tiers"]["Full-Frame (>=0.80)"]["map50_95"],
                r["weights"]
            ])

    # CSV Aggregato
    csv_summary = OUTPUT_DIR / "fractions_stratified_summary.csv"
    with open(csv_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Fraction", "Runs", "Status vs DINO",
            "Overall mAP50 (Mean +/- Std)",
            "Small mAP50 (Mean +/- Std)",
            "Macro mAP50 (Mean +/- Std)",
            "FullFrame mAP50 (Mean +/- Std)",
            "Overall mAP50-95 (Mean +/- Std)",
            "Small mAP50-95 (Mean +/- Std)",
            "Macro mAP50-95 (Mean +/- Std)",
            "FullFrame mAP50-95 (Mean +/- Std)"
        ])
        for label, st in summary_grouped.items():
            writer.writerow([
                label, st["n_runs"], st["status_vs_dino"],
                f"{st['overall_map50']['mean']:.2f} +/- {st['overall_map50']['std']:.2f}",
                f"{st['small_map50']['mean']:.2f} +/- {st['small_map50']['std']:.2f}",
                f"{st['macro_map50']['mean']:.2f} +/- {st['macro_map50']['std']:.2f}",
                f"{st['full_frame_map50']['mean']:.2f} +/- {st['full_frame_map50']['std']:.2f}",
                f"{st['overall_map50_95']['mean']:.2f} +/- {st['overall_map50_95']['std']:.2f}",
                f"{st['small_map50_95']['mean']:.2f} +/- {st['small_map50_95']['std']:.2f}",
                f"{st['macro_map50_95']['mean']:.2f} +/- {st['macro_map50_95']['std']:.2f}",
                f"{st['full_frame_map50_95']['mean']:.2f} +/- {st['full_frame_map50_95']['std']:.2f}"
            ])

    # CSV per classe, una riga per (run, classe)
    csv_per_class = OUTPUT_DIR / "runs_per_class_detail.csv"
    with open(csv_per_class, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fraction_pct", "seed", "run_name"] + PER_CLASS_FIELDS)
        writer.writeheader()
        for r in single_runs_data:
            for row in r["per_class"]:
                writer.writerow({"fraction_pct": r["fraction_pct"], "seed": r["seed"],
                                 "run_name": r["run_name"], **row})

    # CSV per classe aggregato sui seed, una riga per (frazione, classe)
    csv_class_summary = OUTPUT_DIR / "fractions_per_class_summary.csv"
    with open(csv_class_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_summary_rows)

    # Stampa a terminale della tabella riassuntiva finale (mAP@50)
    print("\n" + "=" * 98)
    print("                RISULTATI FINALI SUL TEST SET - STRATIFICATI (mAP@50)")
    print("=" * 98)
    print(f"{'FRAZIONE':<10} | {'OVERALL (Media +/- Std)':<25} | {'SMALL (<0.3)':<18} | {'MACRO (0.3-0.8)':<18} | {'FULL-FRAME':<15}")
    print("-" * 98)
    for label, st in summary_grouped.items():
        ov = f"{st['overall_map50']['mean']:.2f}% +/- {st['overall_map50']['std']:.2f}%"
        sm = f"{st['small_map50']['mean']:.2f}% +/- {st['small_map50']['std']:.2f}%"
        mc = f"{st['macro_map50']['mean']:.2f}% +/- {st['macro_map50']['std']:.2f}%"
        ff = f"{st['full_frame_map50']['mean']:.2f}% +/- {st['full_frame_map50']['std']:.2f}%"
        print(f"{label:<10} | {ov:<25} | {sm:<18} | {mc:<18} | {ff:<15}")
    print("=" * 98)

    # ==========================================
    # 6. (OPZIONALE) YOLO DISTILLATA, STESSA PROCEDURA
    # ==========================================
    if DISTILLED_WEIGHTS is not None:
        if Path(DISTILLED_WEIGHTS).exists():
            print(f"\n[*] Valutazione YOLO distillata: {DISTILLED_WEIGHTS}")
            res, eval_time = run_validation(DISTILLED_WEIGHTS, device)
            dist_rows = extract_per_class(res)
            dist_tiers = compute_tier_metrics(res)
            csv_dist = OUTPUT_DIR / "distilled_per_class.csv"
            with open(csv_dist, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PER_CLASS_FIELDS)
                writer.writeheader()
                writer.writerows(dist_rows)
            with open(OUTPUT_DIR / "distilled_summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "weights": str(Path(DISTILLED_WEIGHTS).resolve()),
                    "eval_time_sec": eval_time,
                    "overall": {"map50": round(float(res.box.map50) * 100.0, 2),
                                "map50_95": round(float(res.box.map) * 100.0, 2)},
                    "tiers": dist_tiers,
                    "per_class": dist_rows,
                }, f, indent=4, ensure_ascii=False)
            print(f"--> Distillata | Overall mAP@50: {float(res.box.map50) * 100.0:.2f}%")
        else:
            print(f"[!] DISTILLED_WEIGHTS non trovato, salto: {DISTILLED_WEIGHTS}")

    print(f"\n[OK] Tutti i file sono stati salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate_all()