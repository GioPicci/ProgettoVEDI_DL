# 4. Training e Prestazioni del Classificatore DINOv3

---

## 4.1 Setup Sperimentale e Metodologia di Addestramento

### 4.1.1 Configurazione dell'architettura e preprocessing di runtime
Come descritto nel Capitolo 3, il modello adotta come estrattore di caratteristiche il backbone pre-addestrato **ViT-S/16** di DINOv3 ($D=384$, 12 strati di auto-attenzione, 6 teste di attenzione per strato, circa $22\text{ M}$ parametri). Il backbone è mantenuto a pesi congelati (*frozen*), operando secondo il paradigma del *Linear Probing*. Sul token di classificazione globale $\mathbf{z}_{\text{CLS}}^{(L)} \in \mathbb{R}^{384}$ viene innestata una testa lineare formata da uno strato di regolarizzazione *Dropout* ($p=0.3$) e da uno strato completamente connesso proiettato sulle $57$ classi del dataset:

$$\hat{\mathbf{y}} = \mathbf{W} \cdot \text{Dropout}(\mathbf{z}_{\text{CLS}}^{(L)}) + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{57 \times 384}, \; \mathbf{b} \in \mathbb{R}^{57}$$

In fase di runtime, ciascun fotogramma RGB viene elaborato secondo la seguente pipeline deterministica:
1. Ridimensionamento bilineare alla risoluzione fissa di $H \times W = 384 \times 384$ pixel, corrispondente alla griglia di token $24 \times 24$.
2. Normalizzazione dei canali cromatici tramite media e deviazione standard del corpus ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

### 4.1.2 Strategie di Data Augmentation stocastica
Durante la fase di addestramento, per irrobustire il modello contro la forte eterogeneità fotometrica e prospettica caratteristica delle riprese egocentriche indoor (variazioni di luce naturale nei chiostri, zone d'ombra nei corridoi e oscillazioni rapide della testa del visitatore), le immagini di training sono state sottoposte alle seguenti trasformazioni stocastiche applicate in sequenza:

* **Riflessione orizzontale casuale (*Random Horizontal Flip*)**: applicata con probabilità $p = 0.5$.
* **Perturbazione fotometrica (*Color Jitter*)**: alterazioni casuali e congiunte di luminosità, contrasto e saturazione cromatica nell'intervallo $[0.7, 1.3]$ (probabilità $p = 0.5$ per ciascun fattore), e lieve perturbazione della tinta (probabilità $p = 0.3$).

In fase di validazione e inferenza sul test set, tutte le trasformazioni casuali sono disattivate, garantendo la deterministica riproducibilità delle predizioni.

---

### 4.1.3 Iperparametri di addestramento e ottimizzazione
L'addestramento è stato condotto minimizzando la funzione di perdita di *Cross-Entropy* standard per la classificazione multiclasse:

$$\mathcal{L}_{\text{CE}}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{c=1}^{57} y_c \log \left( \frac{\exp(\hat{y}_c)}{\sum_{j=1}^{57} \exp(\hat{y}_j)} \right)$$

I parametri operativi del protocollo sperimentale sono riassunti nella seguente tabella:

| Parametro | Valore | Descrizione / Motivazione |
| :--- | :---: | :--- |
| **Ottimizzatore** | AdamW | Con parametro di decadimento del peso $\beta_1=0.9, \beta_2=0.999$ |
| **Learning Rate iniziale ($\eta_0$)** | $1 \times 10^{-4}$ | Tasso di apprendimento sulla testa lineare |
| **Weight Decay** | $1 \times 10^{-4}$ | Regolarizzazione $L_2$ sui pesi della matrice di proiezione |
| **Dimensione del Batch** | $16$ | Bilanciamento tra stabilità del gradiente e consumo di VRAM |
| **Numero di epoche** | $20$ | Criterio di arresto globale |
| **Learning Rate Scheduler** | Cosine con Warmup | Warmup lineare sui primi step seguito da decadimento a coseno |
| **Dropout rate** | $0.3$ | Regolarizzazione applicata prima della proiezione lineare |
| **Selezione modello migliore** | Epoca 13 | Checkpoint selezionato sulla massima accuratezza di validazione |

Il modello ha raggiunto la massima accuratezza di validazione ($98.31\%$) in corrispondenza dell'**epoca 13**, checkpoint che è stato congelato e adottato per tutte le valutazioni sul Test set e per le fasi a valle della pipeline (WSOL e distillazione).

---

## 4.2 Risultati di Classificazione sul Test Set Ufficiale

### 4.2.1 Metriche globali di valutazione
La valutazione sperimentale è stata condotta sul **Test Set ufficiale** ($5.462$ immagini distribuite su 7 sessioni video continue di visita reale), analizzando i **$3.673$ fotogrammi** contenenti annotazioni di Ground Truth associate a POI. Delle 57 classi totali del dataset, **56 classi risultano rappresentate nel set di test** (la classe `3.8_StanzaAbate` non presenta campioni registrati nelle 7 sessioni di test considerate).

Le prestazioni complessive ottenute dal modello sono riassunte di seguito:

| Metrica Globale | Valore | Note metodologiche |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **70.24%** | Frazione globale di predizioni esatte su 3.673 campioni |
| **Top-5 Accuracy** | **84.32%** | La classe reale è inclusa tra i 5 logit più alti del modello |
| **Macro Recall** | **72.10%** | Media aritmetica del recall calcolata uniformemente sulle 56 classi |
| **Macro Precision** | **73.54%** | Media aritmetica della precisione sulle classi valutate |
| **Macro F1-Score** | **70.00%** | Media non pesata della media armonica precision-recall |
| **Weighted F1-Score** | **70.23%** | F1 pesato sul numero di campioni per classe |

Il raggiungimento di oltre il **$70.2\%$ di Top-1 Accuracy** e dell'**$84.3\%$ di Top-5 Accuracy** su un compito a grana fine con 57 classi all'interno dello stesso complesso monumentale — sotto condizioni di ripresa egocentrica non controllata — dimostra l'efficacia descrittiva delle rappresentazioni estratte da DINOv3.

---

### 4.2.2 Analisi disaggregata delle prestazioni per Tier dimensionale
Per indagare il comportamento del classificatore alle differenti scale degli elementi storico-artistici, le prestazioni sul test set sono state scomposte lungo i 4 tier dimensionali introdotti nel Capitolo 2:

| Tier Dimensionale | Classi Valutate | Campioni Test | Accuracy Macro | Accuracy Pesata | Precision Macro | F1 Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiny** ($\bar{A}_c \le 0.10$) | $4 / 4$ | $98$ | **94.38%** | **90.82%** | $79.48\%$ | **83.86%** |
| **Small** ($0.10 < \bar{A}_c < 0.30$) | $3 / 3$ | $104$ | **84.71%** | **85.58%** | $78.10\%$ | **80.66%** |
| **Macro** ($0.30 \le \bar{A}_c < 0.80$) | $26 / 27$ | $1.490$ | **66.10%** | **60.20%** | $78.44\%$ | **69.02%** |
| **Full-Frame** ($\bar{A}_c \ge 0.80$) | $23 / 23$ | $1.981$ | **73.38%** | **75.97%** | $66.37\%$ | **67.31%** |
| **Complessivo** | $56 / 57$ | $3.673$ | **72.10%** | **70.24%** | $73.54\%$ | **70.00%** |

---

### 4.2.3 Il "Paradosso" del Tier Tiny e il supporto campionario ridotto
Il fatto che il classificatore ottenga le prestazioni più elevate proprio sul Tier Tiny (**$94.38\%$** di accuratezza macro e **$90.82\%$** pesata) delinea un apparente paradosso:
* `5.5_Cupola`: $100.0\%$ Recall ($14 / 14$ corretti)
* `5.7_S.Agata`: $100.0\%$ Recall ($32 / 32$ corretti)
* `5.8_S.Scolastica`: $100.0\%$ Recall ($12 / 12$ corretti)
* `6.5_BassorilieviMancanti`: $77.5\%$ Recall ($31 / 40$ corretti)

Questo andamento va tuttavia interpretato considerando due fattori chiave:
1. **Supporto campionario ridotto**: Il tier comprende appena 4 classi e **98 campioni complessivi nel test set** (meno del $2.7\%$ delle annotazioni totali, a fronte dei $1.490$ di Macro e $1.981$ di Full-Frame). La scarsa numerosità limita la variabilità delle inquadrature e favorisce percentuali elevate rispetto a classi con centinaia di campioni distribuiti su prospettive più eterogenee.
2. **Riconoscimento del contesto vs localizzazione del POI**: Come approfondito nel Capitolo 8, l'elevata accuratezza di DINOv3 non implica che la rete isoli il minuscolo manufatto in sé, ma che riconosca con efficacia l'imponente contesto architettonico che lo ospita (l'arco, la volta, la parete stuccata). Se questo ancoraggio contestuale è sufficiente a classificare correttamente la scena a livello globale, nella successiva fase di localizzazione WSOL determinerà uno spostamento del picco di attivazione verso la parete circostante, impedendo la corretta delimitazione geometrica della bounding box.

---

### 4.2.4 Analisi Topologica degli Errori: Matrice di Confusione e Scomposizione Spaziale

Data la dimensionalità elevata della matrice di confusione completa a $57 \times 57$ classi — riportata integralmente nell'**Appendice B** per completezza documentale —, l'analisi degli errori è stata strutturata secondo una scomposizione gerarchica basata sui **9 macro-ambienti** del complesso monumentale.

#### Scomposizione dell'Errore: Intra-Ambiente vs Inter-Ambiente
Classificando ciascuna predizione in tre esiti mutuamente esclusivi (predizione esatta di classe, errore intra-ambiente ed errore inter-ambiente), emerge chiaramente la natura topologica degli scambi:

| Macro-Ambiente | Campioni ($N$) | Predizione Esatta (Top-1) | Errore Intra-Ambiente | Errore Inter-Ambiente | Accuratezza Macro-Area |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Area 1 — Ingresso** | $53$ | $92.5\%$ | — | $7.5\%$ | **92.5%** |
| **Area 2 — Scalone** | $189$ | $74.1\%$ | $23.3\%$ | $2.6\%$ | **97.4%** |
| **Area 3 — Chiostro e Corridoi** | $1.461$ | $53.8\%$ | $34.5\%$ | $11.7\%$ | **88.3%** |
| **Area 4 — Cantoria e Chiesa** | $384$ | $89.8\%$ | — | $10.2\%$ | **89.8%** |
| **Area 5 — Antirefettorio** | $324$ | $76.2\%$ | $18.5\%$ | $5.2\%$ | **94.8%** |
| **Area 6 — Coro di Notte** | $294$ | $76.9\%$ | $19.4\%$ | $3.7\%$ | **96.3%** |
| **Area 7 — Refettorio Piccolo** | $393$ | $79.1\%$ | $18.6\%$ | $2.3\%$ | **97.7%** |
| **Area 8 — Cucine e Sotterranei** | $449$ | $83.7\%$ | $14.9\%$ | $1.3\%$ | **98.7%** |
| **Area 9 — Giardino dei Novizi** | $126$ | $79.4\%$ | $20.6\%$ | **0.0%** | **100.0%** |
| **Totale Globale** | **3.673** | **70.24%** | **22.62%** | **7.13%** | **92.87%** |

Dall'analisi della distribuzione complessiva emergono tre esiti distinti sui $3.673$ fotogrammi di test:
* **$70.24\%$** delle predizioni: il modello individua **esattamente l'opera specifica** (predizione corretta).
* **$22.62\%$** delle predizioni: il modello sbaglia l'opera specifica, ma **riconosce comunque l'ambiente corretto** (errore intra-ambiente, es. scambia due rampe dello stesso scalone o due quadri nella stessa sala).
* **$7.13\%$** delle predizioni: il modello commette un errore grossolano, confondendo sale diverse (errore inter-ambiente).

Sommando le predizioni esatte ($70.24\%$) e gli errori locali ($22.62\%$), il modello colloca il visitatore nel **macro-ambiente corretto nel $92.87\%$ dei casi totali** ($100\% - 7.13\%$). 
Considerando infine il solo insieme degli errori commessi ($29.76\%$ del totale), si nota che circa **tre errori su quattro ($76.0\%$, pari a $22.62 / 29.76$)** sono semplici ambiguità locali tra elementi vicini della medesima stanza, mentre solo un errore su quattro ($24.0\%$, pari a $7.13 / 29.76$) rappresenta un reale sconfinamento verso ambienti estranei.

---

#### Matrice di Confusione tra Macro-Ambienti ($9 \times 9$)
Aggregando i 57 POI nelle 9 macro-aree, la matrice di confusione assume una forma compatta e interpretabile, con una diagonale nettamente dominante:

| Macro-Area Reale | Area 1 | Area 2 | Area 3 | Area 4 | Area 5 | Area 6 | Area 7 | Area 8 | Area 9 | Recall Area |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Ingresso** | **49** | $0$ | $1$ | $1$ | $0$ | $0$ | $0$ | $0$ | $2$ | **92.5%** |
| **2. Scalone** | $1$ | **184** | $4$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | **97.4%** |
| **3. Chiostro/Corr.** | $62$ | $44$ | **1290** | $5$ | $14$ | $9$ | $12$ | $13$ | $12$ | **88.3%** |
| **4. Cantoria/Chiesa**| $0$ | $1$ | $7$ | **345** | $0$ | $31$ | $0$ | $0$ | $0$ | **89.8%** |
| **5. Antirefettorio** | $0$ | $3$ | $10$ | $1$ | **307** | $3$ | $0$ | $0$ | $0$ | **94.8%** |
| **6. Coro di Notte** | $0$ | $0$ | $5$ | $2$ | $2$ | **283** | $1$ | $1$ | $0$ | **96.3%** |
| **7. Ref. Piccolo**  | $0$ | $0$ | $4$ | $0$ | $4$ | $1$ | **384** | $0$ | $0$ | **97.7%** |
| **8. Cucine/Sott.**  | $0$ | $0$ | $1$ | $1$ | $0$ | $0$ | $4$ | **443** | $0$ | **98.7%** |
| **9. Giardino**      | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | **126** | **100.0%** |

Anche la frazione minoritaria di scambi inter-ambiente riflette una chiara contiguità architettonica:
- L'Area 3 (Chiostro e Corridoi) scambia campioni prevalentemente con l'Area 1 ($62$ frame) e l'Area 2 ($44$ frame), che rappresentano i varchi fisici in cui i corridoi convergono direttamente nell'Ingresso e nello Scalone monumentale.
- L'Area 4 (Cantoria sulla Chiesa) scambia $31$ campioni esclusivamente con l'Area 6 (Coro di Notte), due spazi sopraelevati che condividono il medesimo scorcio prospettico sulla navata della chiesa abbaziale.

---

#### Analisi dei Sotto-Ambienti Critici con Residuo Esterno
Per visualizzare nel dettaglio i meccanismi di ambiguità locale senza appesantire la lettura, vengono riportate di seguito le matrici di confusione ristrette per due sotto-ambienti rappresentativi, integrando una colonna e una riga sintetica per l'errore verso **Altri Ambienti (Esterno)**:

##### 1. Area 2 — Scalone Monumentale (Simmetria Architettonica)
Le due rampe gemelle dello scalone condividono identici materiali in marmo e pietra lavica, geometria speculare e illuminazione analoga:

| Classe Reale | `2.1_RampaS.Nicola` | `2.2_RampaS.Benedetto` | Esterno (Altri Ambienti) | Totale |
| :--- | :---: | :---: | :---: | :---: |
| `2.1_RampaS.Nicola` | **84** ($83.2\%$) | $15$ ($14.9\%$) | $2$ ($2.0\%$) | **101** |
| `2.2_RampaS.Benedetto` | $29$ ($33.0\%$) | **56** ($63.6\%$) | $3$ ($3.4\%$) | **88** |
| *Predetti da Esterno* | $1$ | $0$ | — | — |

La dispersione verso ambienti esterni è pressoché nulla ($\le 3.4\%$), a conferma del fatto che l'incertezza risiede interamente nella discriminazione fine tra le due strutture gemelle.

##### 2. Area 9 — Giardino dei Novizi (Spazio Aperto)
L'area comprende tre POI collocati nel cortile esterno del monastero:

| Classe Reale            | `9.1_NicchiaePavimento` | `9.2_TraccePalestra` | `9.3_PergolatoNovizi` | Totale |
|:------------------------|:-----------------------:|:--------------------:|:---------------------:|:------:|
| `9.1_NicchiaePavimento` |    **22** ($66.7\%$)    |    $1$ ($3.0\%$)     |    $10$ ($30.3\%$)    | **33** |
| `9.2_TraccePalestra`    |      $0$ ($0.0\%$)      |  **40** ($74.1\%$)   |    $14$ ($25.9\%$)    | **54** |
| `9.3_PergolatoNovizi`   |      $0$ ($0.0\%$)      |    $1$ ($2.6\%$)     |   **38** ($97.4\%$)   | **39** |
| *Predetti da Esterno*   |           $0$           |         $0$          |          $0$          |   —    |

In questo ambiente, la confusione verso l'esterno è identicamente pari a **$0.0\%$**: il modello riconosce l'ambiente del giardino con precisione categorica ($100.0\%$), manifestando una parziale sovrapposizione unicamente tra le tracce della palestra e il pergolato a causa dell'inquadratura ad ampio raggio del visitatore.
