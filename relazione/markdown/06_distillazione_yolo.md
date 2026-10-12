# 6. Distillazione della Conoscenza in YOLO

---

## 6.1 Costruzione del Dataset Pseudo-Etichettato

Per trasferire la capacità di localizzazione debolmente supervisionata dal modello *teacher* (il classificatore DINOv3 con attribuzione **Linear-CAM**) a un rilevatore real-time ad alta efficienza (*student*), è stata implementata una procedura di distillazione del dataset (*Dataset Distillation* o *Pseudo-Labeling*).

![Pipeline di distillazione della conoscenza e pseudo-etichettatura](figure/pipeline_distillazione.png)

*Figura 6.1: Flusso del processo di distillazione del dataset: inferenza del modello Teacher DINOv3, selezione dei fotogrammi ad alta confidenza, estrazione dei bounding box Linear-CAM, formattazione YOLO e addestramento del rilevatore Student YOLOv8n.*

### 6.1.1 Criteri di filtraggio e confidenza
Poiché l'addestramento di architetture di object detection regredisce in presenza di etichette grossolane o incoerenti, la generazione dei metadati di supervisione è vincolata a due criteri di selezione qualitativa:
1. **Soglia di confidenza minima di classe**: Vengono processati esclusivamente i fotogrammi per i quali il classificatore esprime una probabilità a posteriori superiore al $50\%$ ($p(y \mid \mathbf{x}) \ge 0.50$). I fotogrammi incerti vengono scartati per prevenire l'iniezione di rumore semantico.
2. **Validità geometrica del contorno**: Dalla mappa Linear-CAM binarizzata (con soglia $\tau = 0.45$ ed elemento strutturante ellittico $7 \times 7$) deve emergere almeno un contorno chiuso non degenere che racchiuda il picco di attivazione. I fotogrammi privi di contorni validi vengono esclusi.

Per ciascuna immagine ritenuta valida, le coordinate cartesiane estreme $[x, y, w, h]$ del rettangolo minimo circoscritto al contorno principale vengono normalizzate rispetto alle dimensioni native dell'immagine $(W, H)$ e codificate nello standard YOLO:

$$c_x = \frac{x + w/2}{W}, \quad c_y = \frac{y + h/2}{H}, \quad w_n = \frac{w}{W}, \quad h_n = \frac{h}{H}$$

con coordinate vincolate rigorosamente nell'intervallo continuo $[0.0, 1.0]$.

### 6.1.2 Statistiche del dataset generato
L'applicazione della pipeline sull'intero corpus di addestramento ha prodotto le seguenti volumetrie:

| Parametro | Valore numerico | Quota percentuale |
| :--- | :---: | :---: |
| **Fotogrammi sorgente esaminati** | $57.000$ | $100.00\%$ |
| **Pseudo-annotazioni generate con successo** | **$55.964$** | **$98.18\%$** |
| Scartate per bassa confidenza ($< 0.50$) | $1.011$ | $1.77\%$ |
| Scartate per assenza di contorno valido | $23$ | $0.04\%$ |

Oltre il **$98\%$ dei campioni** supera i filtri qualitativi, garantendo un dataset distillato ricco, denso e rappresentativo di tutte le 57 classi del dominio monumentale.

---

## 6.2 Addestramento del Modello Studente (YOLOv8)

### 6.2.1 Architettura dello studente
Come modello studente è stato adottato **YOLOv8n** (*nano*), caratterizzato da circa $3.2$ milioni di parametri. Questa scelta risponde a un duplice requisito:
* **Efficienza computazionale**: Offrire una latenza di inferenza sub-millisecondo idonea all'impiego su dispositivi mobili e wearable in contesti museali interattivi.
* **Robustezza e regolarizzazione**: Rispetto a modelli di taglia superiore, una capacità architetturale compatta funge da filtro naturale contro l'overfitting sul rumore residuo delle pseudo-label.

Esperimenti preliminari condotti anche sulle evoluzioni architetturali successive (*YOLO11n* e *YOLO26n*) hanno evidenziato prestazioni quasi sovrapponibili sul test set (rispettivamente $37.01\%$ e $37.78\%$ mAP@50 a fronte del $37.39\%$ di YOLOv8n), confermando l'invarianza delle conclusioni rispetto alla variante nano specifica e giustificando la focalizzazione su YOLOv8n come riferimento metodologico per il confronto.

### 6.2.2 Partizionamento dei dati coerente
Per preservare la rigorosa separazione sperimentale ed evitare forme di memorizzazione del segnale, il dataset distillato è stato suddiviso in uno split train/val ($80\%$ train, $20\%$ validation deterministico per classe) **perfettamente allineato** a quello adottato durante il training del classificatore DINOv3. In tal modo, le immagini impiegate per la validazione interna dello studente corrispondono a campioni che il modello teacher non ha utilizzato per l'ottimizzazione dei pesi della testa lineare.

### 6.2.3 Iperparametri e regolarizzazione
L'addestramento dello studente è stato condotto con i seguenti iperparametri:
* **Risoluzione di input**: $640 \times 640$ pixel.
* **Batch size**: 16 campioni.
* **Durata**: 100 epoche con meccanismo di arresto anticipato (*early stopping*) con pazienza di 20 epoche.
* **Ottimizzatore**: SGD con momentum stocastico a $0.937$, weight decay $5 \times 10^{-4}$ e tasso di apprendimento iniziale $\text{lr}_0 = 0.01$ con decadimento cosinusoidale ($\text{lr}_f = 0.01$).
* **Fase di riscaldamento**: Warmup lineare per 3 epoche con learning rate del bias a $0.1$.
* **Data Augmentation**: Pipeline stocastica comprendente *Mosaic* ($p=1.0$, disattivato nelle ultime 10 epoche per stabilizzare le coordinate dei box), *Random Horizontal Flip* ($p=0.5$), *Random Erasing* ($p=0.4$), traslazione ($\pm 10\%$), scala ($\pm 50\%$) e distorsione fotometrica nello spazio colore HSV.

---

## 6.3 Convergenza e Validazione Preliminare

Al termine delle 100 epoche, la rete YOLOv8n distillata ha mostrato una rapida e stabile convergenza:

| Set di Valutazione | Riferimento BBox | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50:95 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validazione Interna** | Pseudo-Label (Teacher) | $95.12\%$ | $94.89\%$ | **$96.64\%$** | **$86.66\%$** |
| **Test Set Indipendente** | Ground Truth (Umano) | $50.19\%$ | $40.68\%$ | **$37.39\%$** | **$25.84\%$** |

L'analisi comparata delle due valutazioni evidenzia due aspetti metodologici fondamentali:
1. **Fedeltà di distillazione quasi perfetta ($96.64\%$ mAP@50)**: Lo studente YOLO impara a replicare con estrema fedeltà la politica di localizzazione del teacher DINOv3, dimostrando che una rete convoluzionale compatta è pienamente in grado di interiorizzare la funzione di stima spaziale prodotta dal Vision Transformer.
2. **Capacità di generalizzazione sul Ground Truth reale ($37.39\%$ mAP@50)**: Valutato sul test set indipendente non annotato dal teacher e privo di qualsiasi correlazione, il modello studente conserva una solida capacità di localizzazione reale. Il divario rispetto alla validazione interna rispecchia fedelmente l'errore intrinseco di approssimazione delle mappe CAM rispetto ai confini fisici tracciati manualmente dall'annotatore umano.

Nei capitoli successivi (Capitoli 7 e 8) queste prestazioni saranno messe a confronto diretto con le baseline interamente supervisionate per quantificare l'esatto valore di sample efficiency della pipeline debolmente supervisionata.
