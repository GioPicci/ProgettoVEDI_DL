import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ==============================================================================
# CONFIGURAZIONE SPERIMENTAZIONE
# ==============================================================================
TRAIN_DIR = Path(r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Training")
TEST_DIR = Path(r"C:\Users\Workstation\PycharmProjects\ProgettoVEDI_DL\data\UNICT-VEDI_Dataset_bbox\Test")
CLASS_MAP_PATH = Path("../checkpoints_classification/class_map.json")
OUTPUT_BENCHMARK = Path("./runs_sample_efficiency")

# Percentuali di Ground Truth da testare
FRACTIONS = [0.01, 0.02] # [0.05, 0.10, 0.20, 0.40]

# 3 run per calcolare media +/- deviazione standard.
# Stessi seed riusati su ogni frazione -> confronto appaiato (piu' potenza statistica a parita' di n).
SEEDS = [42, 1337, 2026]

MODEL_BASE = "yolov8n.pt"
EPOCHS = 50
PATIENCE = 15          # alzata da 8: con subset piccoli la validation interna e' piu' rumorosa,
                        # patience basse rischiano di fermare il training per rumore, non convergenza
IMG_SIZE = 640
BATCH_SIZE = 8         # su 6GB con yolov8n@640 dovrebbe starci comodo; se va in OOM prova batch=8 o batch=-1 (auto)
WORKERS = 4             # Windows: il dataloader multiprocess puo' dare problemi/rallentamenti, abbassa a 0 se serve

# Frazione del subset di training riservata a validation INTERNA (mai il test set vero)
VAL_INTERNAL_RATIO = 0.15

# Monitoraggio SOLO osservativo della curva di apprendimento sul test set reale.
# Non influenza early stopping ne' la scelta del checkpoint: serve solo a vedere il trend
# e per eventualmente abortire manualmente un run chiaramente senza speranza (risparmio GPU).
ENABLE_TEST_MONITOR = True
EVAL_TEST_EVERY = 10    # ogni quante epoche fare il probe sul test set; alza il valore se l'overhead pesa troppo

# Benchmark target ottenuto da DINOv3 (baseline di pareggio) - confermato: e' mAP@50
TARGET_DINO_MAP = 37.39


# ==============================================================================
# 1. SETUP DATASET & CAMPIONAMENTO STRATIFICATO (con split train/val interno)
# ==============================================================================
def get_class_names():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        cmap = json.load(f)
    return {int(v): k for k, v in cmap.items()}


def create_test_yaml(names):
    """Genera lo YAML fisso per il Test Set. Usato SOLO per:
    (a) il probe periodico di monitoraggio (osservativo), e
    (b) l'UNICA valutazione finale ufficiale per ogni run."""
    test_imgs = [str(p.resolve()) for ext in ("*.jpg", "*.png") for p in TEST_DIR.rglob(ext)]
    test_txt = OUTPUT_BENCHMARK / "test_images.txt"
    test_txt.write_text("\n".join(test_imgs), encoding="utf-8")

    test_yaml = OUTPUT_BENCHMARK / "test_eval.yaml"
    data = {
        "path": str(TEST_DIR.resolve()),
        "train": str(test_txt.resolve()),
        "val": str(test_txt.resolve()),
        "names": names,
    }
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(test_yaml.resolve())


def sample_train_subset(fraction, seed, names, val_ratio=VAL_INTERNAL_RATIO):
    """Estrae una frazione bilanciata per classe, campionando per NUMERO DI IMMAGINI
    (scelta intenzionale: risponde a 'quante immagini deve annotare una persona',
    non 'quanti bounding box'). Dal sottoinsieme estratto ritaglia una validation
    INTERNA stratificata, usata per early stopping e scelta del checkpoint.
    Il Test Set vero non viene mai toccato in questa funzione.

    NB: con frazioni molto piccole (es. 5%) alcune classi con pochissime immagini
    totali possono contribuire 0 immagini alla validation interna (tutto va in
    train). E' un limite intrinseco della granularita' dei dati, non un bug:
    le altre classi coprono comunque la validation interna complessiva.
    """
    random.seed(seed)
    train_images, val_images = [], []

    for folder in sorted(TRAIN_DIR.iterdir()):
        if not folder.is_dir():
            continue
        img_folder = folder / "images"
        if not img_folder.exists():
            continue

        imgs = list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
        if not imgs:
            continue

        n_samples = max(1, int(len(imgs) * fraction))
        sampled = random.sample(imgs, min(n_samples, len(imgs)))

        if len(sampled) >= 2:
            n_val = max(1, round(len(sampled) * val_ratio))
            n_val = min(n_val, len(sampled) - 1)  # lascia sempre almeno 1 immagine in train
            val_part = sampled[:n_val]
            train_part = sampled[n_val:]
        else:
            val_part = []
            train_part = sampled

        train_images.extend([str(p.resolve()) for p in train_part])
        val_images.extend([str(p.resolve()) for p in val_part])

    run_tag = f"train_{int(fraction*100)}pct_seed{seed}"

    train_txt = OUTPUT_BENCHMARK / f"{run_tag}_train.txt"
    train_txt.write_text("\n".join(train_images), encoding="utf-8")

    val_txt = OUTPUT_BENCHMARK / f"{run_tag}_valinternal.txt"
    val_txt.write_text("\n".join(val_images), encoding="utf-8")

    yaml_path = OUTPUT_BENCHMARK / f"{run_tag}.yaml"
    data = {
        "path": str(TRAIN_DIR.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),  # <-- validation INTERNA, MAI il test set
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return str(yaml_path.resolve()), len(train_images), len(val_images)


# ==============================================================================
# 2. MONITOR OSSERVATIVO SUL TEST SET (non influenza training ne' checkpoint)
# ==============================================================================
def make_test_monitor_callback(test_yaml_path, img_size, device, eval_every, target_map, trajectory_out):
    """Ogni `eval_every` epoche carica i pesi appena salvati (last.pt) e li valuta
    sul Test Set SOLO per costruire la curva di apprendimento e vedere quando/se
    la soglia viene sfiorata durante il training.

    Questo NON tocca patience e NON sceglie il checkpoint finale: quello resta
    deciso unicamente dalla validation interna, come farebbe Ultralytics di
    default. Fermarsi appena si supera la soglia introdurrebbe un bias da
    'optional stopping': si selezionerebbe la fluttuazione positiva di un
    singolo controllo rumoroso invece della vera performance a convergenza,
    e il 'delta di quanto si e' superato DINO' risulterebbe falsato."""

    def _callback(trainer):
        epoch = trainer.epoch + 1
        if epoch % eval_every != 0:
            return
        last_ckpt = getattr(trainer, "last", None)
        if last_ckpt is None or not Path(last_ckpt).exists():
            return

        probe = None
        try:
            probe = YOLO(str(last_ckpt))
            with torch.no_grad():
                m = probe.val(
                    data=test_yaml_path,
                    imgsz=img_size,
                    device=device,
                    split="val",
                    batch=8,  # batch ridotto per il probe: minimizza il rischio di OOM mentre il training e' attivo
                    verbose=False,
                    plots=False,
                )
            map50 = float(m.box.map50) * 100.0
            trajectory_out.append({"epoch": epoch, "test_map50": map50})
            note = " (soglia sfiorata - solo osservativo)" if map50 >= target_map else ""
            print(f"    [monitor] epoca {epoch}: test mAP@50 = {map50:.2f}%{note}")
        except RuntimeError as e:
            print(f"    [monitor] probe epoca {epoch} saltato per errore runtime: {e}")
        finally:
            del probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _callback


# ==============================================================================
# 3. RUN BENCHMARK ITERATIVO
# ==============================================================================
def run_benchmark():
    OUTPUT_BENCHMARK.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"
    names = get_class_names()
    test_yaml_path = create_test_yaml(names)

    results = {}

    for frac in FRACTIONS:
        pct_label = f"{int(frac * 100)}%"
        results[pct_label] = []

        for seed in SEEDS:
            run_name = f"yolov8n_{int(frac*100)}pct_s{seed}"
            print(f"\n{'='*75}\nAVVIO: {run_name} (Dataset: {pct_label})\n{'='*75}")

            subset_yaml, n_train, n_val = sample_train_subset(frac, seed, names)
            print(f"Immagini training: {n_train} | Immagini validation interna: {n_val}")

            model = YOLO(MODEL_BASE)

            test_trajectory = []
            if ENABLE_TEST_MONITOR:
                model.add_callback(
                    "on_fit_epoch_end",
                    make_test_monitor_callback(
                        test_yaml_path, IMG_SIZE, device, EVAL_TEST_EVERY, TARGET_DINO_MAP, test_trajectory
                    ),
                )

            # Training: early stopping e scelta del best.pt basati SOLO sulla validation interna
            model.train(
                data=subset_yaml,
                epochs=EPOCHS,
                patience=PATIENCE,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                device=device,
                seed=seed,
                workers=WORKERS,
                cache="ram",
                project=str(OUTPUT_BENCHMARK / "runs"),
                name=run_name,
                exist_ok=True,
                verbose=False,
            )

            # Valutazione UFFICIALE, UNICA, sul Test Set vero (best.pt scelto via validation interna)
            metrics = model.val(
                data=test_yaml_path,
                imgsz=IMG_SIZE,
                device=device,
                split="val",
                verbose=False,
            )

            map50 = float(metrics.box.map50) * 100.0
            map_all = float(metrics.box.map) * 100.0

            first_cross = next((t["epoch"] for t in test_trajectory if t["test_map50"] >= TARGET_DINO_MAP), None)

            results[pct_label].append({
                "seed": seed,
                "map50": map50,
                "map50_95": map_all,
                "n_train_images": n_train,
                "n_val_internal_images": n_val,
                "test_trajectory_monitor_only": test_trajectory,
                "first_epoch_target_crossed_monitor_only": first_cross,
            })
            print(f"--> Completato {run_name} | mAP@50 UFFICIALE (test, best.pt): {map50:.2f}%")

    # ==============================================================================
    # 4. TABELLA CONCLUSIVA E ANALISI DEL PAREGGIO
    # ==============================================================================
    print("\n" + "=" * 65)
    print("      SAMPLE EFFICIENCY: GT ANNOTATO vs DINOv3 (37.39% mAP@50)")
    print("=" * 65)
    print(f"{'% DATI GT':<12} | {'mAP@50 (Media +/- Std)':<25} | {'STATUS':<15}")
    print("-" * 65)

    summary_export = {}

    for pct_label, runs in results.items():
        maps = [r["map50"] for r in runs]
        mean_val = float(np.mean(maps))
        std_val = float(np.std(maps))
        status = "SUPERATO" if mean_val >= TARGET_DINO_MAP else "INFERIORE"

        summary_export[pct_label] = {
            "mean_map50": round(mean_val, 2),
            "std_map50": round(std_val, 2),
            "runs": runs,
        }
        print(f"{pct_label:<12} | {mean_val:>6.2f}% +/- {std_val:<6.2f}%       | {status:<15}")

    print("=" * 65)
    print("NB: la traiettoria sul test set e 'first_epoch_target_crossed_monitor_only'")
    print("sono SOLO osservative (curva di apprendimento / diagnostica). Il numero")
    print("ufficiale per ogni run e' 'map50': un'UNICA valutazione del best.pt scelto")
    print("tramite la validation interna, mai tramite il test set.")

    with open(OUTPUT_BENCHMARK / "sample_efficiency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=4)
    print(f"\n[+] Risultati salvati in: {OUTPUT_BENCHMARK / 'sample_efficiency_summary.json'}")


if __name__ == "__main__":
    run_benchmark()