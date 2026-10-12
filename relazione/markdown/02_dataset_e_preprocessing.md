# 2. Il Dataset UNICT-VEDI POI e Sanitizzazione delle Annotazioni

---

## 2.1 Struttura e origine del dataset UNICT-VEDI POI

### 2.1.1 Origine scientifica e contesto applicativo
Il dataset di riferimento impiegato in questo lavoro è **UNICT-VEDI POI** (*Vision Exploitation for Data Interpretation - Points of Interest*), sviluppato dall'*Image Processing Laboratory* (IPLab) dell'Università degli Studi di Catania nell'ambito del progetto di ricerca VEDI e introdotto formalmente in letteratura da Ragusa et al. (2019)[^1].

Il contesto applicativo rientra nell'ambito della **First-Person Vision (FPV)** — o visione egocentrica — applicata ai beni culturali. Le acquisizioni video sono state condotte tramite dispositivi indossabili (*wearable cameras*) all'interno del complesso monumentale del **Monastero dei Benedettini di San Nicolò l'Arena a Catania**, uno dei complessi benedettini più estesi d'Europa e sede del Dipartimento di Scienze Umanistiche (DISUM) dell'Università di Catania.

L'obiettivo applicativo consiste nell'individuazione e localizzazione automatica dei **Punti di Interesse (POI)** verso cui l'utente rivolge l'attenzione visiva (*attended objects*) durante la visita. Il dataset include un'ampia varietà di elementi storico-artistici: affreschi, tele, sculture, busti, portali monumentali, epigrafi e interi ambienti architettonici.

| Proprietà del Dataset | Valore / Dettaglio |
| :--- | :--- |
| **Sito monumentale** | Monastero dei Benedettini di San Nicolò l'Arena (Catania) |
| **Tipologia dati** | Fotogrammi estratti da sequenze video egocentriche |
| **Risoluzione nativa** | $1216 \times 684$ pixel |
| **Numero di classi POI** | 57 classi distribuite su 9 macro-ambienti storici |
| **Volume Training Set** | 67.050 immagini (organizzate per cartella di classe) |
| **Volume Test Set** | 5.462 immagini (distribuite su 7 sessioni continuative di visita) |
| **Formato annotazioni** | Coordinate bounding box normalizzate ($x_c, y_c, w, h$) |

---

### 2.1.2 Tassonomia delle classi e articolazione spaziale
I 57 POI sono organizzati gerarchicamente all'interno di 9 macro-ambienti del complesso monumentale, identificati dalla notazione `X.Y_NomePOI`, dove $X \in [1, 9]$ rappresenta l'area monumentale e $Y$ il singolo elemento d'interesse censito:

* **Area 1 — Ingresso Monumentale**: `1.1_Ingresso`
* **Area 2 — Scalone Monumentale**: `2.1_RampaS.Nicola`, `2.2_RampaS.Benedetto`
* **Area 3 — Chiostro di Levante e Corridoi**: `3.1_SimboloTreBiglie`, `3.2_ChiostroLevante`, `3.3_Plastico`, `3.4_Affresco`, `3.5_FinestraChiostroLevante`, `3.6_PortaCorodiNotte`, `3.7_TracciaPortone`, `3.8_StanzaAbate`, `3.9_CorridoiodiLevante`, `3.10_CorridoioCorodiNotte`, `3.11_CorridoioOrologio`
* **Area 4 — Cantoria e Affaccio sulla Chiesa**: `4.1_Quadro`, `4.2_PavimentoOrig.Altare`, `4.3_BalconeChiesa`
* **Area 5 — Antirefettorio e Museo della Fabbrica**: `5.1_PortaAulaS.Mazzarino`, `5.2_PortaIngressoMuseoFabbrica`, `5.3_PortaAntirefettorio`, `5.4_PortaIngressoRef.Piccolo`, `5.5_Cupola`, `5.6_AperturaPavimento`, `5.7_S.Agata`, `5.8_S.Scolastica`, `5.9_ArcoconFirma`, `5.10_BustoVaccarini`
* **Area 6 — Coro di Notte e Aula Santo Mazzarino**: `6.1_QuadroSantoMazzarino`, `6.2_Affresco`, `6.3_PavimentoOriginale`, `6.4_PavimentoRestaurato`, `6.5_BassorilieviMancanti`, `6.6_LavamaniSx`, `6.7_LavamaniDx`, `6.8_TavoloRelatori`, `6.9_Poltrone`
* **Area 7 — Refettorio Piccolo e Ambienti di Servizio**: `7.1_Edicola`, `7.2_PavimentoA`, `7.3_PavimentoB`, `7.4_Passavivande`, `7.5_AperturaPavimento`, `7.6_Scala`, `7.7_SalaMetereologica`
* **Area 8 — Cucine del Vaccarini e Sotterranei**: `8.1_Doccione`, `8.2_VanoRaccoltaCenere`, `8.3_SalaRossa`, `8.4_ScalaCucina`, `8.5_CucinaProvv`, `8.6_Ghiacciaia`, `8.7_Latrina`, `8.8_OssaeScarti`, `8.9_Pozzo`, `8.10_Cisterna`, `8.11_BustoPietroTacchini`
* **Area 9 — Giardino dei Novizi e Spazi Esterni**: `9.1_NicchiaePavimento`, `9.2_TraccePalestra`, `9.3_PergolatoNovizi`

---

### 2.1.3 Organizzazione tra Training e Test Set
La ripartizione dei dati su disco presenta una marcata asimmetria strutturale:

* **Training Set (67.050 immagini)**: I fotogrammi sono ripartiti in 57 directory distinte, ciascuna intitolata alla specifica classe POI.
* **Test Set (5.462 immagini)**: I dati non sono catalogati per classe, bensì distribuiti lungo **7 sessioni continuative di visita reale** (`Test1` – `Test7`). Questa organizzazione riproduce fedelmente una fruizione reale in sequenza temporale. Nel set di test sono presenti **3.673 file di annotazione**: i fotogrammi sprovvisti di etichetta corrispondono a fasi transitorie di spostamento del visitatore (corridoi anonimi, svolte, scale prive di opere) in cui non è inquadrato alcun POI.

La valutazione delle predizioni sul Test Set non può dunque avvenire tramite il nome della cartella genitrice, ma richiede l'interrogazione diretta delle annotazioni di Ground Truth associate a ogni singolo fotogramma all'interno delle sottodirectory `labels/`.

---

## 2.2 Sanitizzazione delle annotazioni del dataset

L'ispezione preliminare delle etichette fornite a corredo del dataset grezzo ha evidenziato diverse anomalie geometriche e sintattiche, che hanno reso necessaria una fase mirata di sanitizzazione per garantire la corretta convergenza delle funzioni di costo e la stabilità delle metriche di valutazione.

### 2.2.1 Correzione delle anomalie geometriche
Il formato standard adottato per la localizzazione (formato YOLO) definisce ciascuna bounding box mediante la quadrupla normalizzata $[x_c, y_c, w, h] \in [0.0, 1.0]^4$, dove $(x_c, y_c)$ indica il centro del riquadro e $(w, h)$ le rispettive dimensioni relative. Sono state corrette sistematicamente tre anomalie:

1. **Riconduzione a coordinate normalizzate**:  
   Alcune annotazioni esprimevano erroneamente le coordinate nello spazio assoluto dei pixel ($1216 \times 684$). Tali valori (identificati da coordinate $> 1.5$) sono stati riscalati dividendo per la risoluzione nativa:
   $$\hat{x}_c = \frac{x_c}{W_{\text{img}}}, \quad \hat{y}_c = \frac{y_c}{H_{\text{img}}}, \quad \hat{w} = \frac{w}{W_{\text{img}}}, \quad \hat{h} = \frac{h}{H_{\text{img}}}$$

2. **Troncamento ai margini dell'immagine (*clamping*)**:  
   Diversi box presentavano vertici esterni al campo visivo (valori inferiori a $0.0$ o superiori a $1.0$). Le coordinate sono state convertite nei vertici estremi $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, limitate rigidamente all'intervallo unitario ammissibile e riconvertite nel formato centro-ampiezza:
   $$x_{\min}^{\text{clamp}} = \max(0.0, \min(1.0, x_{\min})), \quad x_{\max}^{\text{clamp}} = \max(0.0, \min(1.0, x_{\max}))$$

3. **Eliminazione di bounding box degenerate**:  
   Tutte le annotazioni con ampiezza o altezza inferiori a una soglia minima ($\epsilon = 10^{-3}$) sono state scartate, prevenendo instabilità numeriche ed errori di divisione per zero nel calcolo delle funzioni di perdita basate su IoU (*GIoU*, *CIoU*).

---

### 2.2.2 Trattamento dei fotogrammi privi di bounding box: Convenzione Full-Frame
Nel training set, **3.106 fotogrammi** (circa il $4.6\%$ del totale) risultano privi di un'annotazione esplicita o associati a file di testo vuoti.

In conformità con le specifiche ufficiali del dataset rilasciate dagli autori nel file `README.txt`:
> *"Some folders have not the two \*.csv files, in this case for each related RGB frame you have to consider a bounding box which contains the whole frame."*

quando per un fotogramma non è presente una bounding box esplicita nel file di etichetta, viene adottata la convenzione standard prevista:
- Assegnazione della classe corrispondente alla cartella del POI di appartenenza.
- Generazione di un bounding box fittizio a tutto schermo, coincidente con l'intera area visibile del fotogramma:
  $$[x_c = 0.5, \; y_c = 0.5, \; w = 1.0, \; h = 1.0]$$

Tale scelta rispecchia fedelmente la natura del contesto egocentrico, nel quale ampi elementi architettonici o inquadrature a distanza ravvicinata saturano per intero il campo ottico del sensore.

---

## 2.3 Analisi dell'area media per classe e definizione dei tier dimensionali

### 2.3.1 Motivazione metodologica e natura non ufficiale dei tier
Nel dataset UNICT-VEDI POI, le dimensioni fisiche e apparenti dei soggetti esibiscono un'elevata variabilità: si spazia da piccoli rilievi che occupano una frazione trascurabile della scena fino a interi corridoi monumentali.

Una valutazione incentrata unicamente sul dato aggregato globale di mAP (*mean Average Precision*) tenderebbe a mascherare i comportamenti e i limiti operativi delle architetture alle differenti scale. Questa distinzione è particolarmente rilevante per modelli *Vision Transformer* (come DINOv3), che elaborano l'immagine su una griglia discreta di patch spaziali e generano mappe di attenzione a risoluzione intrinsecamente ridotta.

> [!NOTE] Nota metodologica sulla definizione dei Tier
> La suddivisione delle classi in fasce dimensionali (**Tier**) introdotta in questa sezione **non appartiene alla specifica ufficiale del dataset UNICT-VEDI né costituisce una tassonomia standard di letteratura**. È stata definita appositamente in questo lavoro come strumento analitico per consentire un'indagine disaggregata e rigorosa del comportamento dei modelli al variare della taglia relativa dell'oggetto.

---

### 2.3.2 Formulazione e distribuzione nei 4 tier dimensionali
I tier sono assegnati **a livello di classe**, calcolando per ciascuna delle 57 categorie l'area media normalizzata $\bar{A}_c$ occupata dalla bounding box rispetto alla superficie totale dell'immagine su tutte le occorrenze di Ground Truth:

$$\bar{A}_c = \frac{1}{N_c} \sum_{i=1}^{N_c} (w_{c,i} \cdot h_{c,i}) \in [0.0, 1.0]$$

In funzione di $\bar{A}_c$, lo spazio delle classi è stato segmentato in quattro gruppi dimensionali:

| Tier | Soglia area media ($\bar{A}_c$) | N. Classi | Quota classi | Area media del tier | Range area ($\min - \max$) | Esempi rappresentativi |
|:---|:---:|:---:|:---:|:---:|:---:|---|
| **Tiny** | $\bar{A}_c \le 0.10$ | **4** | $7.0\%$ | **0.0497** ($4.97\%$) | $0.0310 - 0.0720$ | `6.5_BassorilieviMancanti`, `5.8_S.Scolastica` |
| **Small** | $0.10 < \bar{A}_c < 0.30$ | **3** | $5.3\%$ | **0.2627** ($26.27\%$) | $0.2190 - 0.2970$ | `6.1_QuadroSantoMazzarino`, `6.6_LavamaniSx` |
| **Macro** | $0.30 \le \bar{A}_c < 0.80$ | **27** | $47.4\%$ | **0.6119** ($61.19\%$) | $0.3150 - 0.7860$ | Portali monumentali, grandi affreschi, tavoli |
| **Full-Frame** | $\bar{A}_c \ge 0.80$ | **23** | $40.3\%$ | **0.9309** ($93.09\%$) | $0.8130 - 0.9980$ | Corridoi prospettici, scalinate, sale storiche |
| **Totale** | — | **57** | $100\%$ | **0.7075** ($70.75\%$) | $0.0310 - 0.9980$ | — |

*Per la consultazione della tabella esaustiva con i valori esatti di area media e la ripartizione dettagliata per ciascuna delle 57 classi, si rimanda all'**Appendice B**.*

---

### 2.3.3 Caratteristiche e implicazioni dei singoli tier

#### 1. Tier Tiny (4 classi — area media $< 10\%$)
Raggruppa elementi decorativi e scultorei circoscritti: `6.5_BassorilieviMancanti` ($3.1\%$), `5.8_S.Scolastica` ($4.5\%$), `5.7_S.Agata` ($5.1\%$) e `5.5_Cupola` ($7.2\%$).
* **Implicazioni per la localizzazione**: Oggetti che occupano una frazione così ridotta del fotogramma rappresentano la sfida più complessa per i metodi debolmente supervisionati, trovandosi sia al limite del supporto spaziale della griglia di token, sia esposti a una forte interferenza da parte dell'imponente contesto architettonico in cui sono incastonati.

#### 2. Tier Small (3 classi — area media $10\% - 30\%$)
Comprende manufatti artistici con contorni circoscritti rispetto alle pareti: `6.1_QuadroSantoMazzarino` ($21.9\%$), `6.6_LavamaniSx` ($27.2\%$) e `8.11_BustoPietroTacchini` ($29.7\%$).
* **Implicazioni per la localizzazione**: Sebbene questi soggetti presentino contorni definiti, in uno scenario debolmente supervisionato rimangono problematici: occupando tra il $10\%$ e il $30\%$ dell'inquadratura, lasciano dal $70\%$ al $90\%$ dell'area visiva al contesto murario o architettonico. In assenza di supervisione geometrica, le mappe CAM tendono frequentemente a espandersi sulla parete o a disperdersi, collocando questo raggruppamento in una fascia intermedia complessa in cui le metriche di rilevazione debole rimangono sensibilmente distanti da quelle conseguite su target Macro o Full-Frame.

#### 3. Tier Macro (27 classi — area media $30\% - 80\%$)
Rappresenta la quota maggioritaria del dataset (circa il $47\%$ delle categorie totali). Include grandi portali monumentali (es. `5.3_PortaAntirefettorio`, $75.7\%$), ampi affreschi (`3.4_Affresco`, $62.9\%$), porzioni di pavimentazione e scalinate storiche. L'attenzione del visitatore copre gran parte dell'inquadratura, pur mantenendo margini di contesto visibili.

#### 4. Tier Full-Frame (23 classi — area media $\ge 80\%$)
Raggruppa prospettive di corridoi (`3.9_CorridoiodiLevante`), rampe d'accesso monumentali (`1.1_Ingresso`, `2.1_RampaS.Nicola`) e ampi spazi aperti in cui l'area media registrata è prossima al valore saturato ($99.8\%$).
* Per questo gruppo, il problema geometrico della localizzazione converge concettualmente verso una **classificazione della scena**: identificare correttamente il target equivale a predire la classe semantica associata, poiché il box ottimale collassa sull'intero fotogramma.

---

[^1]: F. Ragusa, A. Furnari, S. Battiato, G. Signorello, G. M. Farinella, *"Egocentric Point of Interest Recognition in Cultural Sites"*, in *Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP/VISAPP)*, 2019.