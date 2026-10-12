# Appendice C. Ambiente di Esecuzione, Note Tecniche e Riproducibilità

---

La presente appendice documenta l'infrastruttura computazionale, l'ecosistema software, le convenzioni operative e i vincoli di riproducibilità scientifica adottati per lo sviluppo, l'addestramento e la validazione dell'intero progetto.

---

## C.1 Stack Software e Librerie di Riferimento

L'intero codice sorgente è implementato in linguaggio **Python 3.12**, operando in un ambiente virtuale isolato. I principali framework e librerie scientifiche impiegati includono:

| Libreria / Framework | Versione Principale | Funzione Primaria nella Pipeline |
| :--- | :---: | :--- |
| **PyTorch** | `2.1+` (CUDA 12.x) | Computazione su tensori GPU, differenziazione automatica (`torch.autograd`) e training del classificatore DINOv3 |
| **Hugging Face Transformers** | `5.x` | Caricamento del foundation model `facebook/dinov3-vits16-pretrain-lvd1689m` e gestione dell'attenzione *eager* per l'estrazione delle mappe |
| **Ultralytics** | `8.x` | Addestramento, validazione ed esportazione dei modelli di Object Detection (`yolov8n`, `yolo11n`, `yolo26n`) |
| **OpenCV (`cv2`)** | `4.x` | Sovracampionamento bicubico delle heatmap di salienza, binarizzazione, operazioni morfologiche ellittiche ed estrazione contorni |
| **Pillow (`PIL`)** | `10.x` | Lettura, decodifica ed elaborazione delle immagini RGB grezze |
| **NumPy & SciPy** | `1.26+` / `1.12+` | Manipolazioni tensoriali, calcolo di percentili dinamici, decomposizioni a valori singolari (SVD) e interpolazioni |
| **Torchmetrics** | `1.x` | Calcolo rigoroso delle metriche di classificazione multiclasse (Top-1, Top-5, Macro F1, matrici di confusione $57 \times 57$) |
| **Pandas & Matplotlib** | `2.x` / `3.8+` | Aggregazione tabellare dei risultati statistici, generazione dei *range plot* per tier e rendering delle matrici di confusione |

---

## C.2 Risorse Hardware e Accelerazione di Calcolo

Gli esperimenti sono stati condotti su una workstation dotata di accelerazione grafica dedicata:
* **Unità di Elaborazione Grafica (GPU)**: GPU NVIDIA GeForce RTX 2060 con 6 GB di VRAM (GDDR6).
* **Accelerazione Numerica**: Driver NVIDIA compatibili con CUDA 12 e librerie cuDNN per la parallelizzazione delle primitive convoluzionali e di attenzione.
* **Precisione Mista (AMP)**: Per gli addestramenti dello studente YOLOv8n è stato abilitato il regime *Automatic Mixed Precision* (FP16/FP32), raddoppiando il throughput di elaborazione dei fotogrammi e dimezzando l'ingombro di memoria senza degradare la stabilità numerica delle perdite di localizzazione.

---

## C.3 Protocollo di Riproducibilità e Controllo del Determinismo

Per assicurare la replicabilità dei risultati riportati nella relazione:
1. **Semi Pseudocasuali Fissati**: 
   Tutte le operazioni stocastiche sono state inizializzate con semi costanti documentati:
   - Training del classificatore DINOv3 e addestramento della YOLO distillata: `seed = 42`.
   - Benchmark di Sample Efficiency: esecuzione sistematica sui tre semi `seed = 42`, `seed = 1337` e `seed = 2026` con calcolo esplicito di media e deviazione standard ($\mu \pm \sigma$).
2. **Determinismo delle Primitive CUDA**:  
   L'impostazione di `torch.backends.cudnn.deterministic = True` e la disattivazione del benchmarking dinamico non deterministico dei convolutori garantiscono la consistenza bit-a-bit dei gradienti.
3. **Isolamento e Allineamento degli Split**:  
   La suddivisione deterministica 80/20 per cartella-classe (assegnando ad esempio il campione $i$-esimo a validazione se $i \pmod 5 == 0$) è stata applicata in modo identico sia al dataset di classificazione sia al dataset di distillazione per YOLO, eliminando ogni rischio di dispersione di informazioni (*data leakage*) tra le partizioni.

---

## C.4 Condizioni di Licenza e Conformità dei Dati

* **UNICT-VEDI POI Dataset**: Rilasciato dall'Image Processing Laboratory (IPLab) dell'Università degli Studi di Catania per finalità esclusive di ricerca scientifica e didattica accademica.
* **Pesi DINOv3**: Rilasciati da Meta AI Research sotto licenza permissiva per scopi di ricerca e sperimentazione non commerciale.
* **Ultralytics YOLO**: Concesso sotto licenza open-source AGPL-3.0 per implementazioni di studio e ricerca.
