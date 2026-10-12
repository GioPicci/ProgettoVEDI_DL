import os
import glob
from PIL import Image, ImageFile
from tqdm import tqdm

# Disabilitiamo il tolleramento dei file troncati per fare una scansione rigorosa
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Percorso principale del dataset (risolto indipendentemente dalla cartella di lancio)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(repo_root, "data", "UNICT-VEDI_Dataset_bbox")


def find_all_images(root_dir):
    """ Trova ricorsivamente tutte le immagini .jpg, .jpeg, .png """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(image_paths)


def check_dataset_integrity(dataset_dir):
    print(f"--> Ricerca immagini in: {os.path.abspath(dataset_dir)}")
    image_paths = find_all_images(dataset_dir)
    total_images = len(image_paths)

    if total_images == 0:
        print(f"[!] Nessuna immagine trovata in {dataset_dir}. Verifica il percorso!")
        return

    print(f"--> Trovate {total_images} immagini totali. Inizio scansione dei pixel...")

    corrupted_files = []
    valid_count = 0

    for path in tqdm(image_paths, desc="Scansione integrità"):
        try:
            # 1. Verifica la struttura/header
            with Image.open(path) as img:
                img.verify()

            # 2. Re-apre e forza il caricamento reale dei pixel in memoria
            with Image.open(path) as img:
                img.load()
                valid_count += 1

        except Exception as e:
            corrupted_files.append((path, str(e)))

    # ==========================================
    # REPORT FINALE
    # ==========================================
    corrupted_count = len(corrupted_files)
    percentage_corrupted = (corrupted_count / total_images) * 100

    print("\n" + "=" * 50)
    print("           REPORT INTEGRITÀ DATASET")
    print("=" * 50)
    print(f"Immagini Totali Scansionate: {total_images}")
    print(f"Immagini Integre / Valide:   {valid_count} ({(valid_count / total_images) * 100:.2f}%)")
    print(f"Immagini Corrotte / Troncate: {corrupted_count} ({percentage_corrupted:.2f}%)")
    print("=" * 50)

    if corrupted_files:
        print(f"\n[!] Trovate {corrupted_count} immagini corrotte.")

        # Salva l'elenco in un file di testo per consultazione
        log_file = "corrupted_images.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            for path, err in corrupted_files:
                f.write(f"{path}\t[Errore: {err}]\n")

        print(f"[+] L'elenco dettagliato dei file corrotti è stato salvato in '{log_file}'.")
        print("\nPrimi 5 file corrotti trovati:")
        for path, err in corrupted_files[:5]:
            print(f" - {path}")

    else:
        print("\n[V] PERFETTO! Nessuna immagine corrotta trovata. Il dataset è integro al 100%.")


if __name__ == "__main__":
    check_dataset_integrity(DATASET_DIR)