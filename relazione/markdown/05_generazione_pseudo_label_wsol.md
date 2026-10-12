# 5. Generazione delle Pseudo-Label (WSOL)

---

## 5.1 Dalle Mappe di Attivazione alle Bounding Box

Nel contesto della *Weakly-Supervised Object Localization* (WSOL), l'obiettivo fondamentale consiste nell'estrarre coordinate geometriche di localizzazione (*bounding box*) disponendo unicamente di etichette a livello di immagine, senza alcuna annotazione supervisionata delle coordinate. Come introdotto nel Capitolo 3, le rappresentazioni spaziali di DINOv3 permettono di estrarre mappe bidimensionali di salienza $M \in \mathbb{R}^{H \times W}$ sfruttando quattro distinti principi di attribuzione: la proiezione lineare delle feature sui pesi della testa di classe (**Linear-CAM**), i coefficienti di auto-attenzione del token di classificazione (**DINO Self-Attention**), la decomposizione spettrale delle attivazioni spaziali (**Eigen-CAM**) e i gradienti di retropropagazione rispetto agli stati latenti intermedi (**Grad-CAM**).

Per convertire una mappa di attivazione continua in un bounding box discreto $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, è stata definita una pipeline deterministica articolata in quattro stadi consecutivi:

![Pipeline WSOL per estrazione di bounding box](figure/pipeline_wsol.png)

*Figura 5.1: Pipeline sequenziale di estrazione dei bounding box a partire dalle mappe di salienza dei token: sovracampionamento bicubico, binarizzazione differenziata, regolarizzazione morfologica e circoscrizione del contorno attorno al picco.*

---

### 5.1.1 Normalizzazione e sovracampionamento spaziale
La mappa grezza estratta a livello di griglia di patch ($24 \times 24$ celle) viene normalizzata nell'intervallo unitario $[0.0, 1.0]$:

$$\hat{M}(i, j) = \frac{M(i, j) - \min(M)}{\max(M) - \min(M) + \epsilon}$$

La matrice normalizzata viene quindi proiettata alla risoluzione nativa dell'immagine ($W \times H$) mediante interpolazione bicubica continua, generando una superficie di densità di confidenza $\mathcal{H}(x, y) \in [0, 255]$ espressa in livelli di grigio a 8 bit.

---

### 5.1.2 Strategie di binarizzazione differenziata e risoluzione dell'anomalia sui gradienti
La trasformazione della mappa continua in una maschera binaria $B(x, y) \in \{0, 255\}$ richiede l'imposizione di una soglia di attivazione. L'analisi sperimentale ha evidenziato che l'adozione di un'unica strategia convenzionale penalizza drasticamente specifici paradigmi di attribuzione:

#### 1. Metodi a risposta distribuita (`Linear-CAM` ed `Eigen-CAM`)
Questi metodi generano mappe con transizioni graduali e ampie regioni di attivazione. La binarizzazione adotta la classica **soglia relativa standard** proporzionale al valore di picco:
$$T_{\text{rel}} = \tau \cdot \max_{(x,y)} \mathcal{H}(x, y), \quad \text{con } \tau = 0.45$$
I pixel con intensità superiore a $T_{\text{rel}}$ vengono posti a $255$, preservando fedelmente il corpo centrale dell'oggetto.

#### 2. Metodi a risposta piccata (`DINO Attention` e `Grad-CAM`)
Nei primi esperimenti comparativi, l'applicazione della soglia standard $\tau = 0.45$ su `DINO Attention` e `Grad-CAM` causava un collasso catastrofico delle prestazioni, registrando un mAP prossimo allo zero ($0.09\%$ e $0.01\%$). Questo fenomeno è determinato da due proprietà intrinseche dei Vision Transformer:
* **Spiccata concentrazione dell'attenzione e dei gradienti**: I pesi di attenzione uscenti dal token CLS e i gradienti di classe tendono ad accumularsi su un numero esiguo di patch prominenti. Una soglia proporzionale al massimo riduce la maschera a pochi pixel isolati, generando riquadri minuscoli che falliscono sistematicamente la soglia di IoU ($\ge 0.5$) rispetto alle annotazioni di Ground Truth.
* **Assenza di gradienti sui patch token all'ultimo strato**: Poiché la testa lineare riceve in ingresso esclusivamente il token CLS, il gradiente dell'output rispetto ai patch token dell'ultimo blocco è analiticamente nullo.

Per sanare questa criticità e consentire un confronto equo, sono state introdotte due modifiche architetturali:
1. **Calcolo dei gradienti su stati latenti intermedi**: In `Grad-CAM`, le derivate parziali sono calcolate rispetto all'output del penultimo strato del Transformer ($\mathbf{Z}^{(L-1)}$), dove i token di patch influenzano ancora la rappresentazione del token CLS attraverso l'ultimo blocco di attenzione.
2. **Binarizzazione robusta basata su percentile di energia**: La soglia viene calcolata dinamicamente come il **$50^\circ$ percentile** della distribuzione di intensità della mappa ($\text{Percentile}_{50}$), trattenendo la metà superiore dell'energia complessiva. A valle della delimitazione, è applicato un vincolo di **area minima** pari al $5\%$ della superficie del fotogramma, espandendo isotropicamente i box che collasserebbero su frammenti sub-risoluzione.

Questa riformulazione ha permesso di recuperare appieno la validità di entrambi i metodi, portando il mAP@50 dal valore degenere iniziale a oltre il $26\%$ per l'attenzione e il $15.7\%$ per Grad-CAM.

---

### 5.1.3 Filtraggio morfologico e selezione del contorno saliente
Sulla maschera binaria viene applicata un'operazione di **apertura morfologica** (*Morphological Opening*) con elemento strutturante ellittico di dimensione $7 \times 7$ pixel:
$$B_{\text{clean}} = (B \ominus K) \oplus K$$
L'erosione seguita da dilatazione elimina frammenti spuri e rumore ad alta frequenza ai bordi, disconnettendo ponti sottili tra l'oggetto e lo sfondo.

Tra tutti i contorni chiusi esterni identificati sulla maschera, viene selezionato il contorno ottimale $\mathcal{C}^*$ adottando il **criterio del picco di confidenza**:
1. Si individua il punto di massima attivazione globale $(x_{\max}, y_{\max}) = \arg\max_{(x,y)} \mathcal{H}(x, y)$.
2. Si verifica l'inclusione spaziale del picco all'interno dei contorni candidati: $\mathcal{C}^*$ è il poligono che racchiude le coordinate del picco.
3. Se nessun contorno include strettamente il picco, viene selezionato il poligono che massimizza l'energia di attivazione integrata:
   $$\mathcal{C}^* = \arg\max_{\mathcal{C}} \iint_{\text{interior}(\mathcal{C})} \mathcal{H}(x, y) \, dx \, dy$$

Il bounding box asse-allineato finale $[x, y, w, h]$ viene calcolato come il rettangolo minimo circoscritto (*bounding rectangle*) al contorno $\mathcal{C}^*$.

---

## 5.2 Benchmark Comparativo tra i Metodi WSOL

### 5.2.1 Protocollo di valutazione
La valutazione comparativa è stata condotta sull'intero volume delle immagini del dataset di addestramento. Per ciascun metodo e per ciascun fotogramma, il classificatore DINOv3 frozen genera la predizione di classe e la mappa di salienza corrispondente, da cui viene estratto un singolo bounding box candidato associato allo score di confidenza del modello.

Le predizioni sono state confrontate con le annotazioni di Ground Truth impiegando il protocollo standard di Object Detection COCO tramite tre metriche chiave:
* **mAP@50**: Mean Average Precision calcolato a soglia fissa di intersezione $\text{IoU} \ge 0.50$.
* **mAP@50:95**: Media del mAP calcolata su 10 soglie di IoU progressive da $0.50$ a $0.95$ con passo $0.05$.
* **Mean Average Recall (mAR@100)**: Recall medio per classe considerando fino a 100 rilevamenti per immagine.

---

### 5.2.2 Risultati sperimentali globali
I risultati globali conseguiti dai quattro metodi sull'intero corpus di valutazione sono riassunti nella seguente tabella:

| Metodo WSOL | Tipologia di Attribuzione | mAP@50 (%) | mAP@50:95 (%) | Recall Medio (%) |
| :--- | :--- | :---: | :---: | :---: |
| **`linear_cam`** | **Supervisionata per classe (pesi testa lineare)** | **64.65%** | **41.56%** | **53.39%** |
| **`dino_attn`** | Class-agnostic (Self-Attention token CLS) | $26.02\%$ | $12.03\%$ | $25.15\%$ |
| **`grad_cam`** | Supervisionata per classe (gradienti penultimo strato) | $15.70\%$ | $7.06\%$ | $16.22\%$ |
| **`eigen_cam`** | Class-agnostic (SVD primo autovettore feature) | $12.94\%$ | $5.63\%$ | $15.81\%$ |

Il metodo **Linear-CAM supera nettamente tutti i concorrenti**, distaccando la seconda miglior tecnica (`dino_attn`) di circa **$38.6$ punti percentuali di mAP@50** e quadruplicando le prestazioni rispetto a `eigen_cam` e `grad_cam`.

---

### 5.2.3 Disaggregazione delle prestazioni per Tier dimensionale
La scomposizione delle metriche lungo i tier dimensionali del dataset chiarisce le dinamiche di localizzazione al variare della scala degli elementi:

| Tier Dimensionale | Metrica | `linear_cam` | `dino_attn` | `eigen_cam` | `grad_cam` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Small** | **mAP@50** | **18.32%** | $5.53\%$ | $10.81\%$ | $6.77\%$ |
| ($< 0.30$) | **mAP@50:95** | **8.56%** | $1.33\%$ | $5.75\%$ | $1.64\%$ |
| | **Recall** | **12.28%** | $4.56\%$ | $11.36\%$ | $6.66\%$ |
| **Macro** | **mAP@50** | **68.04%** | $25.87\%$ | $18.47\%$ | $19.34\%$ |
| ($0.30 - 0.79$) | **mAP@50:95** | **38.07%** | $11.62\%$ | $7.68\%$ | $9.35\%$ |
| | **Recall** | **51.42%** | $24.77\%$ | $19.88\%$ | $19.54\%$ |
| **Full-Frame** | **mAP@50** | **74.79%** | $32.45\%$ | $7.09\%$ | $14.14\%$ |
| ($\ge 0.80$) | **mAP@50:95** | **55.71%** | $15.78\%$ | $3.20\%$ | $6.02\%$ |
| | **Recall** | **68.21%** | $31.87\%$ | $12.37\%$ | $15.24\%$ |

L'analisi evidenzia due evidenze cardine:
1. **Prestazioni crescenti con l'estensione dell'oggetto**: Per tutti i metodi, e per `linear_cam` in modo peculiare, l'accuratezza geometrica cresce monotonicamente con l'area media del target: si passa da un modesto $18.32\%$ sul tier Small a un solido $68.04\%$ su Macro, fino al $74.79\%$ sulle strutture Full-Frame.
2. **Superiorità trasversale di Linear-CAM**: In ciascuna delle fasce dimensionali considerate, `linear_cam` preserva una dominanza sistematica rispetto a tutti i metodi alternativi.

---

## 5.3 Criterio di Scelta del Metodo e Giustificazione di Linear-CAM

### 5.3.1 Giustificazione teorico-sperimentale di Linear-CAM come metodo unico
La pipeline di distillazione adotta conseguentemente **`linear_cam` come unico algoritmo universale di generazione delle pseudo-etichette** per tutte le 57 classi del dataset. Tale decisione poggia su due solide motivazioni:

1. **Allineamento analitico con la funzione decisionale**: A differenza dei metodi *class-agnostic* (`eigen_cam` e `dino_attn`), che evidenziano indistintamente qualsiasi regione ad alto contrasto o salienza visiva nel fotogramma (inclusi visitatori di passaggio, pavimenti riflettenti o porte di servizio), `linear_cam` interroga specificamente il gradiente di pesi associato alla categoria predetta. Le attivazioni riflettono esattamente le caratteristiche visive che hanno motivato la classificazione del POI.
2. **Dominanza statistica robusta**: Su scala globale, `linear_cam` non presenta anomalie o instabilità locali: ottiene il punteggio mAP@50 più alto sul $91\%$ delle classi censite, fornendo pseudo-box coerenti sia per elementi parziali sia per intere visuali architettoniche.

---

## 5.4 Comportamento Spaziale delle Heatmap e Analisi Qualitativa

L'osservazione comparativa delle mappe di salienza generate dai quattro approcci chiarisce i fondamenti geometrici che determinano le discrepanze quantitative registrate:

| Metodo WSOL | Struttura dell'Attivazione | Rischio Principale Identificato |
| :--- | :--- | :--- |
| **Linear-CAM** | Continua, densa, legata ai pesi di classe | Lieve dilatazione sul contesto contiguo |
| **DINO Self-Attention** | Puntiforme, focalizzata sul nucleo | Sottostima del perimetro di grandi scene |
| **Eigen-CAM** | Bimodale, guidata dal contrasto visivo | Attivazione su elementi di foreground errati |
| **Grad-CAM** | Sparsa, frammentata sulle patch | Incertezza ai bordi ad alta frequenza |

![Esempi di localizzazione riuscita con Linear-CAM](figure/wsol_esempi_successo.png)

*Figura 5.2: Esempi di convergenza e localizzazione qualitativa mediante Linear-CAM: (a) elemento architettonico asimmetrico nel tier Macro con bounding box aderente; (b) struttura monumentale continua; (c) prospettiva full-frame con saturazione omogenea dell'invaso visivo.*

* **Linear-CAM**: Produce una copertura compatta e geometricamente omogenea sull'intero corpo del POI. Nei soggetti ampi (affreschi, portali, corridoi) copre con continuità sia le zone interne sia i margini strutturali. L'unico limite emerge su elementi circoscritti, dove la mappa tende a diffondersi lievemente sulla porzione di parete adiacente.
* **DINO Self-Attention**: Risulta estremamente efficace nell'identificare il punto focale del soggetto (ad esempio il volto di un busto o il rilievo scultoreo centrale), ma tende a concentrare la probabilità su un nucleo compatto, faticando a espandersi verso le periferie di oggetti estesi o strutture architettoniche a piena inquadratura.
* **Eigen-CAM**: Identifica con rapidità i gradienti di contrasto dominanti dell'immagine. Tuttavia, trattandosi di un'analisi spettrale non condizionata dalla classe, attiva frequentemente elementi estranei al POI che presentano forte contrasto cromatico (come riflessi luminosi sul pavimento o sagome di passanti).
* **Grad-CAM**: Manifesta mappe con distribuzioni di energia fortemente frammentate su patch sparse, che anche con la binarizzazione a percentile faticano a formare un poligono compatto regolare privo di concavità indesiderate.

Le pseudo-box generate tramite **Linear-CAM** forniscono pertanto il miglior compromesso tra copertura semantica, compattezza geometrica e stabilità di scala, costituendo la base dati per la successiva fase di distillazione nel modello rilevatore YOLO.
