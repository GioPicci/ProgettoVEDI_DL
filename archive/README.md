# Archivio Esperimenti e Codice Legacy

Questa cartella raccoglie file e script sperimentali precedenti o rami di sviluppo superati dalla pipeline definitiva del progetto (**DINOv3 Linear Probing v1 -> WSOL 4-Method Benchmark -> Distillazione YOLOv8n -> Sample Efficiency**).

---

## Contenuto dell'Archivio

### 1. `train_classification_dinov3_v2.py` & `checkpoints_classification_v2/`
- **Descrizione**: Secondo approccio sperimentale di classificazione con introduzione di una 58ª classe esplicita (`Negatives`, indice 57) per campioni privi di POI, split temporale con embargo e data augmentation avanzata.
- **Motivo di archiviazione**: Per la pipeline definitiva di distillazione WSOL e la relazione finale è stato adottato l'approccio canonico **v1** a 57 classi (`train_classification_dinov3.py` e `checkpoints_classification/`), che ha ottenuto una maggiore stabilità e coerenza con le metriche del dataset originale.

### 2. `evaluate_wsol_detection.py` & `wsol_benchmark_results.txt`
- **Descrizione**: Prima versione del benchmark di localizzazione weakly-supervised (WSOL), limitata alla sola estrazione tramite **Linear-CAM**.
- **Motivo di archiviazione**: Superata interamente da `evaluate_wsol_detection_v2.py` (e relativo log `wsol_benchmark_results_4_methods.txt`), che estende la valutazione comparativa a **4 metodi CAM** (Linear-CAM, DINO Attention Map, Eigen-CAM, Grad-CAM).

### 3. `train_obj_det_dinov3.py`
- **Descrizione**: Implementazione esplorativa iniziale di un rilevatore object detector DETR-style (Transformer Decoder a 6 layer, bipartite matching ungherese e loss SetCriterion) innestato sopra i feature token multi-scala di DINOv3.
- **Motivo di archiviazione**: Ramo prototipale preliminare. La metodologia cardine della tesi/relazione è incentrata sulla distillazione WSOL verso rilevatori compatti real-time (famiglia YOLOv8/11/26).
