# Appendice A. Iperparametri Completi e Protocollo Sperimentale

---

La presente appendice raccoglie in modo esaustivo e sistematico tutti gli iperparametri, le configurazioni di ottimizzazione e le impostazioni operative impiegate nelle diverse fasi della pipeline sperimentale, a garanzia della piena trasparenza e riproducibilità metodologica.

---

## A.1 Classificatore DINOv3 (Linear Probing)

| Categoria | Iperparametro | Valore Adottato | Note e Rationale |
| :--- | :--- | :---: | :--- |
| **Backbone** | Modello | `dinov3-vits16-pretrain-lvd1689m` | ViT-S/16 preaddestrato da Meta su LVD-1689M |
| | Dimensione latente ($D$) | $384$ | Dimensione del token di embedding |
| | Numero di strati ($L$) | $12$ | Blocchi sequenziali di Transformer Encoder |
| | Teste di attenzione ($h$) | $6$ | Dimensionalità per testa $d_k = 64$ |
| | Dimensione della patch ($P$) | $16 \times 16$ px | Risoluzione nativa del patch embedding |
| | Stato dei pesi backbone | **Frozen** | Pesi congelati; nessun gradiente retropropagato |
| **Testa di Classe** | Architettura | Dropout + Linear | Proiezione lineare diretta dal token CLS |
| | Dropout rate | $0.3$ | Regolarizzazione per contrastare la co-adattamento |
| | Numero di classi di output | $57$ | Corrispondenti ai 57 POI del dataset VEDI |
| **Dati & Runtime** | Risoluzione di ingresso | $384 \times 384$ px | Griglia discreta di patch pari a $24 \times 24$ (576 token) |
| | Normalizzazione | Canali RGB ImageNet | $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$ |
| | Batch size | $16$ | Campioni per iterazione di ottimizzazione |
| | Data Augmentation | Flip + Color Jitter | Random Horizontal Flip ($p=0.5$); Jitter contrasto/luce/saturazione ($p=0.5$) |
| **Ottimizzazione** | Ottimizzatore | AdamW | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| | Learning Rate iniziale ($\eta_0$) | $1 \times 10^{-4}$ | Ottimizzato su testa lineare |
| | Weight Decay | $1 \times 10^{-4}$ | Penalizzazione $L_2$ per regolarizzazione pesi |
| | Learning Rate Scheduler | Cosine Annealing | Decadimento cosinusoidale con riscaldamento (*warmup*) |
| | Epoche totali | $20$ | Checkpoint migliore selezionato all'epoca 13 (Val Acc $98.31\%$) |
| | Criterio di split Train/Val | $80\% / 20\%$ | Split deterministico stratificato per classe (`idx % 5 == 0`) |

---

## A.2 Pipeline di Estrazione Bounding Box (WSOL)

| Componente | Parametro | Valore Adottato | Funzione nel Sistema |
| :--- | :--- | :---: | :--- |
| **Attribuzione** | Metodo Primario | `Linear-CAM` | Proiezione vettoriale: $\text{ReLU}(\mathbf{w}_c^\top \mathbf{z}_{i,j})$ |
| | Metodi di Confronto | `dino_attn`, `eigen_cam`, `grad_cam` | Utilizzati per il benchmark comparativo (Cap. 5) |
| **Preprocessing Heatmap** | Risoluzione di estrazione | $24 \times 24$ celle | Risoluzione nativa della griglia di patch ViT |
| | Normalizzazione | Min-Max $[0.0, 1.0]$ | $\hat{M} = (M - \min M) / (\max M - \min M + \epsilon)$ |
| | Interpolazione Spaziale | Bicubica | Proiezione dalla griglia $24 \times 24$ alla risoluzione nativa $(W, H)$ |
| | Spazio Colore Output | Scala di grigi 8-bit | Mappatura nell'intervallo discreto intero $[0, 255]$ |
| **Binarizzazione** | Soglia `Linear-CAM` / `Eigen-CAM` | $\tau = 0.45 \cdot \max$ | Taglio relativo proporzionale al picco di intensità |
| | Soglia `DINO Attn` / `Grad-CAM` | Percentile $50^\circ$ | Taglio dinamico sulla mediana di densità energetica |
| | Area Minima di Salvaguardia | $5\%$ dell'immagine | Espansione isotropa per prevenire il collasso sub-risoluzione |
| **Morfologia & Geometria** | Elemento Strutturante | Ellittico $7 \times 7$ px | Operazione di apertura: erosione seguita da dilatazione |
| | Criterio di Selezione Contorno | Picco Assoluto | Selezione del poligono che racchiude $\arg\max \mathcal{H}(x, y)$ |
| | Criterio di Fallback | Massima Energia | Poligono con massima intensità integrata interna |
| | Formato Bounding Box | Rettangolo Minimo Circoscritto | Rettangolo asse-allineato normalizzato $[c_x, c_y, w, h] \in [0, 1]^4$ |
| **Filtro Qualitativo** | Confidenza minima di classe | $p(y \mid \mathbf{x}) \ge 0.50$ | Scarto di fotogrammi con classificazione incerta |

---

## A.3 Addestramento del Modello Studente (YOLOv8n Distillato)

| Ambito | Iperparametro | Valore Adottato | Descrizione |
| :--- | :--- | :---: | :--- |
| **Modello** | Architettura | `YOLOv8n` | Variante compatta (*nano*), $\approx 3.2\text{ M}$ parametri |
| | Inizializzazione | Pretrained COCO | Pesi di partenza ufficiali Ultralytics (`yolov8n.pt`) |
| **Dati** | Risoluzione di ingresso (`imgsz`) | $640 \times 640$ px | Risoluzione standard YOLO per rilevamento fine |
| | Dimensione del batch (`batch`) | $16$ | Esempi per iterazione |
| | Numero di classi (`nc`) | $57$ | Mapping coerente con la tassonomia naturale VEDI |
| | Durata del training (`epochs`) | $100$ | Epoche massime previste |
| | Early Stopping (`patience`) | $20$ | Arresto anticipato in caso di stallo su mAP di validazione |
| **Ottimizzazione** | Ottimizzatore | SGD / Auto | Discesa del gradiente stocastica con momentum |
| | Momentum | $0.937$ | Coefficiente di inerzia dell'aggiornamento |
| | Weight Decay | $5 \times 10^{-4}$ | Regolarizzazione pesi |
| | Learning Rate Iniziale (`lr0`) | $0.01$ | Tasso iniziale di apprendimento |
| | Learning Rate Finale (`lrf`) | $0.01$ | Decadimento a frazione dell'iniziale ($\text{lr}_{\min} = 10^{-4}$) |
| | Warmup Epoche | $3.0$ | Transizione iniziale a gradiente controllato |
| | Warmup Bias Learning Rate | $0.1$ | Tasso iniziale specifico per i parametri di bias |
| **Data Augmentation** | Mosaic | $1.0$ (chiuso a -10 ep.) | Composizione di 4 fotogrammi; disattivato all'epoca 90 |
| | Random Horizontal Flip | $0.5$ | Probabilità di specchiatura orizzontale |
| | Random Erasing | $0.4$ | Probabilità di mascheramento rettangolare casuale |
| | Variazione di Scala (`scale`) | $\pm 50\%$ | Alterazione casuale delle dimensioni dei soggetti |
| | Traslazione (`translate`) | $\pm 10\%$ | Spostamento casuale sui piani cartesiani |
| | Tonalità HSV (`hsv_h`) | $0.015$ | Fluttuazione casuale dello spazio tinta |
| | Saturazione HSV (`hsv_s`) | $0.7$ | Fluttuazione della saturazione cromatica |
| | Luminosità HSV (`hsv_v`) | $0.4$ | Fluttuazione del canale di intensità luminosa |

---

## A.4 Protocollo Sperimentale di Sample Efficiency

| Parametro | Valore / Impostazione | Rationale Sperimentale |
| :--- | :---: | :--- |
| **Frazioni di Supervisione ($\mathcal{F}$)** | $\{1\%, 2\%, 5\%, 10\%, 20\%, 40\%, 100\%\}$ | Campionamento su scala logaritmica/geometrica per mappare l'andamento asintotico |
| **Criterio di Campionamento** | Stratificato per classe per **numero di frame** | Rispecchia il costo reale del tempo di ispezione di un operatore umano |
| **Semi Casuali per Frazione ($< 100\%$)** | $\{42, 1337, 2026\}$ (3 seed) | Controllo rigoroso della varianza stocastica su campioni esigui |
| **Disegno Sperimentale** | A blocchi appaiati (*Paired Comparison*) | Gli stessi 3 seed sono riutilizzati lungo tutte le frazioni per minimizzare il rumore |
| **Partizionamento Interno di Validazione** | $15\%$ del sottoinsieme campionato | Riservato unicamente a early stopping e scelta checkpoint; non tocca il Test set |
| **Pazienza Early Stopping Baseline** | $15$ epoche | Elevata per gestire la rumorosità intrinseca della validazione su micro-dataset |
| **Valutazione Test Ufficiale** | Eseguita su checkpoint migliore | Valutazione su $5.462$ immagini del Test set indipendente con sole etichette GT umane |
