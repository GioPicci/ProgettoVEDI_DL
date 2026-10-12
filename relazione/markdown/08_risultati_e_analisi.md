# 8. Risultati e Analisi

---

## 8.1 Risultati Globali e Curva di Sample Efficiency

La valutazione complessiva condotta sul test set indipendente ($5.462$ fotogrammi non visti) consente di confrontare direttamente il rilevatore debolmente supervisionato (**YOLOv8n distillato**, addestrato a costo geometrico zero) con la famiglia delle baseline supervisionate addestrate con frazioni crescenti di bounding box reali.

| Modello / Configurazione | Supervisione Box (GT) | mAP@50 (%) | mAP@50:95 (%) | Ritenzione su 100% GT (mAP@50) |
| :--- | :---: | :---: | :---: | :---: |
| **Supervisionato (1%)** | $1\%$ (3 seed) | $32.57 \pm 0.44$ | $26.51 \pm 0.12$ | $53.83\%$ |
| **WSOL Distillato (Nostro)** | **0% (Solo CAM)** | **$37.39$** | **$25.84$** | **$61.79\%$** |
| **Supervisionato (2%)** | $2\%$ (3 seed) | $43.76 \pm 0.58$ | $36.48 \pm 0.61$ | $72.32\%$ |
| **Supervisionato (5%)** | $5\%$ (3 seed) | $52.20 \pm 0.79$ | $44.64 \pm 0.56$ | $86.27\%$ |
| **Supervisionato (10%)** | $10\%$ (3 seed) | $56.64 \pm 0.20$ | $48.77 \pm 0.25$ | $93.60\%$ |
| **Supervisionato (20%)** | $20\%$ (3 seed) | $58.54 \pm 0.09$ | $50.66 \pm 0.09$ | $96.74\%$ |
| **Supervisionato (40%)** | $40\%$ (3 seed) | $59.29 \pm 0.38$ | $51.51 \pm 0.39$ | $97.98\%$ |
| **Supervisionato (100% Oracolo)** | $100\%$ (Pieno GT) | **$60.51$** | **$52.97$** | **$100.00\%$** |

L'interpretazione dei risultati beneficia di una duplice prospettiva metodologica:

1. **Punto di pareggio campionario (*Breakeven Point*)**:
   A livello globale, il modello distillato ($37.39\%$ mAP@50) supera chiaramente la baseline all'$1\%$ ($32.57\%$) e si attesta al di sotto di quella al $2\%$ ($43.76\%$). Per interpolazione lineare, la pipeline debolmente supervisionata equivale a circa il **$1.43\%$ di dati annotati manualmente**.
2. **Ritenzione relativa delle capacità (*Capacity Retention*)**:
   Se invece si valuta la quota percentuale di prestazioni trattenute rispetto al limite teorico dell'oracolo supervisionato al $100\%$, il modello distillato raggiunge il **$61.79\%$ dell'efficacia complessiva**. Ottenere quasi i due terzi delle capacità di un modello supervisionato completo senza aver mai disegnato un singolo riquadro geometrico costituisce un solido indicatore dell'efficacia del trasferimento di conoscenza da DINOv3.

---

## 8.2 Analisi Disaggregata per Scala Dimensionale (Tier)

La media globale nasconde tuttavia una marcata asimmetria legata alle dimensioni fisiche dei punti di interesse. La tabella seguente illustra il comportamento delle baseline e del modello distillato suddiviso sui diversi scaglioni dimensionali:

| Gruppo Dimensionale | N. Classi | Baseline 1% mAP@50 | Distillato (0% GT) mAP@50 | Oracolo 100% mAP@50 | Pareggio Stimato (% GT) | Ritenzione Relativa (% di 100% GT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full-Frame ($\ge 0.80$)** | 23 | $23.50 \pm 2.15\%$ | **$45.67\%$** | $54.28\%$ | **$\approx 6.0\%$** | **$84.14\%$** |
| **Macro ($0.30 - 0.79$)** | 26 | $32.42 \pm 1.73\%$ | **$37.71\%$** | $59.29\%$ | **$\approx 1.4\%$** | **$63.60\%$** |
| **Small ($< 0.30$)** | 7 | $62.92 \pm 4.44\%$ | **$9.01\%$** | $85.49\%$ | $< 1\%$ | **$10.54\%$** |
| *— di cui Tiny ($\le 0.10$)* | 4 | $65.81 \pm 3.12\%$ | **$0.00\%$** | $88.15\%$ | $0\%$ | **$0.00\%$** |

![Ritenzione relativa delle prestazioni per gruppo dimensionale](figure/ritenzione_per_tier.png)

*Figura 8.1: Ritenzione relativa di mAP@50 del modello distillato (0% bounding box di Ground Truth) rispetto all'oracolo supervisionato al 100% per ciascun raggruppamento dimensionale (tier).*

### 8.2.1 Il Tier Macro come reale indicatore di valore applicativo
Sebbene il tier *Full-Frame* esprima formalmente il pareggio più alto ($\approx 6\%$ di dati annotati e l'$84.14\%$ di ritenzione), la sua natura geometrica è degenere: occupando l'intero schermo, la regressione delle coordinate si sovrappone concettualmente a un problema di pura classificazione d'immagine (approfondito in § 8.5), per il quale il risparmio di tempo umano associato al tracciamento dei box è trascurabile.

Il risultato di maggior rilievo scientifico e pratico risiede pertanto nel **Tier Macro** (26 classi, area media tra il $30\%$ e il $79\%$ del frame):
* Include oggetti architettonici complessi e asimmetrici (altari, portali, affreschi, sculture, cisterne, elementi decorativi) che richiedono una reale regressione dei confini spaziali rispetto allo sfondo.
* In questo scenario, delimitare a mano decine di migliaia di fotogrammi comporta un costo di annotazione imponente. La pipeline proposta riesce a trattenere il **$63.60\%$ delle prestazioni del modello pienamente supervisionato** e supera di oltre $5$ punti percentuali la baseline supervisionata all'$1\%$ ($37.71\%$ vs $32.42\%$), dimostrando una reale capacità di generare localizzazioni geometriche informative a costo umano nullo.

---

## 8.3 Analisi Puntuale per Classe (Range Plot)

L'ispezione della distribuzione delle prestazioni per singola classe (evidenziata nei *range plot* per tier in Figura 8.2) rivela l'eterogeneità del comportamento del modello distillato (rappresentato dal rombo) rispetto all'escursione delle frazioni supervisionate:

![Range plot per classe e scaglione dimensionale](figure/range_plot_grid.png)

*Figura 8.2: Distribuzione delle prestazioni AP@50 per singola classe suddivisa per tier dimensionale. I punti e le barre orizzontali indicano l'escursione delle frazioni supervisionate (dall'1% al 40% di Ground Truth), mentre il rombo indica la prestazione conseguita dal modello distillato a costo di annotazione zero.*

1. **Eccellenze nel Tier Full-Frame**:
   Per svariati elementi panoramici o strutturali, il modello distillato supera perfino la supervisione al $40\%$:
   * *Balcone Chiesa* ($4.3$): AP@50 pari al **$94.5\%$** (superando il $40\%$ supervisionato a $91.8\%$).
   * *Corridoio Coro di Notte* ($3.10$): AP@50 pari all'**$88.4\%$** (superiore al $40\%$ supervisionato a $84.7\%$).
   * *Sala Meteorologica* ($7.7$): AP@50 pari al **$74.2\%$** (competitiva con il $20\%$ e $40\%$ supervisionato).
2. **Successi Localizzati nel Tier Macro**:
   In numerose classi macro, la distillata batte le baseline all'$1\%$ e $2\%$ e rivaleggia con il $5-10\%$:
   * *Apertura Pavimento* ($5.6$): AP@50 pari all'**$86.3\%$**, a ridosso della supervisione al $5\%$ ($87.8\%$).
   * *Scala* ($7.6$): AP@50 pari al **$70.4\%$**, nettamente superiore all'$1\%$ ($25.1\%$) e vicina al $2\%$ ($56.3\%$).
   * *Arco con Firma* ($5.9$): AP@50 pari al **$56.4\%$** (superiore all'$1\%$ a $18.9\%$).
   * *Porta Refettorio Piccolo* ($5.4$): AP@50 pari al **$31.1\%$**, battendo le frazioni $1\%$, $2\%$, $5\%$ e $10\%$.
3. **Comportamento nel Tier Small**:
   Anche nel tier Small compaiono casi virtuosi isolati, come il *Lavamani Sinistro* ($6.6$, area $0.27$), dove la distillata tocca il **$46.0\%$** di AP@50, superando l'$1\%$ ($33.8\%$) e avvicinando il $2\%$ ($56.7\%$).

---

## 8.4 Analisi Qualitativa del Tier Tiny: Bias Contestuale e Disallineamento Discriminativo

Il fallimento sistematico nel tier Tiny ($0.00\%$ mAP@50 su tutte e 4 le classi con area media inferiore al $10\%$: *Sant'Agata*, *Cupola*, *Santa Scolastica*, *Bassorilievi Mancanti*) rappresenta il limite più marcato dell'approccio debolmente supervisionato. 

A prima vista, si potrebbe imputare tale collasso a un mero limite di risoluzione fisica della griglia di patch ($24 \times 24$ token) o all'incapacità del Vision Transformer di risolvere oggetti sub-risoluzione. Tuttavia, **l'ispezione qualitativa diretta delle mappe di salienza (`Linear-CAM`) smentisce questa interpretazione ed evidenzia una dinamica visiva e teorica ben più profonda**.

![Evidenza visiva del bias contestuale e dell'attribution leakage nei punti di interesse Tiny](figure/bias_contestuale_tiny.png)

*Figura 8.3: Evidenza empirica del disallineamento discriminativo su oggetti Tiny: (a) Sant'Agata (area 5.1%) e (b) Cupola (area 7.2%). La mappa di salienza isola visivamente il profilo del reperto, ma il picco di calore globale (mirino giallo) si sposta sulla cornice architettonica contigua, provocando la generazione di uno pseudo-box errato (rosso) e un IoU nullo rispetto al Ground Truth (verde).*

### 8.4.1 L'evidenza visiva: l'oggetto viene segmentato, ma non eletto a picco
Esaminando le mappe d'attenzione sovrapposte ai fotogrammi reali (es. *Sant'Agata*, area $5.1\%$, e *Cupola*, area $7.2\%$):
* **Capacità di segmentazione latente**: Il modello DINOv3 dimostra una sorprendente sensibilità spaziale. L'oggetto di Ground Truth (es. il busto marmoreo di Sant'Agata o il cerchio prospettico della Cupola) appare chiaramente isolato come un'entità spaziale a sé stante, con valori di attivazione uniformi e bordi marcatamente discontinui rispetto alle pareti e agli elementi adiacenti. Le feature latenti del backbone auto-supervisionato riescono dunque a "vedere" e ritagliare il reperto con nitidezza.
* **Spostamento dell'evidenza discriminativa (*Attribution Leakage*)**: Nonostante il reperto sia segmentato, **non viene identificato come l'elemento più discriminativo della scena**. Il picco globale di calore (il punto di massima intensità $\arg\max \mathcal{H}(x, y)$, evidenziato dal mirino di calibrazione) non cade all'interno del box verde di Ground Truth, ma si colloca sistematicamente su strutture architettoniche limitrofe:
  * Nel caso di *Sant'Agata*, la massima attivazione si concentra sull'ampio arco trionfale in stucco sottostante o sulla cornice della volta superiore.
  * Nel caso della *Cupola*, l'area circolare della cupola vera e propria registra un'attivazione secondaria, mentre il massimo assoluto si estende a raggiera sulla volta e sull'intonaco perimetrale circostante.

### 8.4.2 La causa teorica: il Bias Contestuale nella Weakly-Supervised Learning
Questo disallineamento è una conseguenza diretta della formulazione matematica della classificazione debolmente supervisionata:
1. Il modello è stato ottimizzato esclusivamente tramite una Cross-Entropy Loss a livello d'immagine, priva di qualunque vincolo di localizzazione cartesiana. L'obiettivo dell'ottimizzatore è massimizzare la confidenza di classe sfruttando qualsiasi correlazione predittiva affidabile nei pixel.
2. In un bene culturale monumentale, un reperto di dimensioni minuscole è permanentemente incastonato in un ambiente architettonico statico. Per distinguere la classe *Sant'Agata* da tutte le altre 56 classi del monastero, l'imponente arco barocco e le modanature del soffitto offrono una superficie decine di volte più estesa e una firma visiva univoca e priva di ambiguità.
3. Di conseguenza, i pesi della testa lineare $\mathbf{w}_c$ assegnano il contributo più rilevante ai token del contesto architettonico co-occorrente piuttosto che al piccolo manufatto in sé.

### 8.4.3 Conseguenze sull'estrazione dei box
Poiché la pipeline WSOL individua la bounding box ricercando il poligono che include il picco assoluto di energia, l'algoritmo seleziona il contorno dell'elemento architettonico errato oppure espande la maschera all'intero settore della parete, generando un box finale con coordinate del tutto disallineate.

Questa analisi chiarisce in modo definitivo il contrasto evidenziato rispetto al Capitolo 4:
* L'elevatissima accuratezza del classificatore sui Tiny (**$94.38\%$**) è reale, ma è sostenuta dal **riconoscimento del contesto monumentale associato** al reperto.
* Nella localizzazione senza coordinate, questo "scorciatoia predittiva" (*clever Hans effect* o *context bias*) impedisce l'allineamento con il Ground Truth umano, dove l'annotatore ha isolato strettamente il solo busto.

Al contrario, le baseline supervisionate a cui vengono fornite coordinate esplicite rompono immediatamente questo bias (superando il $65\%$ di mAP già con l'$1\%$ di dati), confermando che per target sub-areali con forte ancoraggio contestuale, la supervisione geometrica umana rimane imprescindibile.

---

## 8.5 Baseline "Box a Tutta Immagine" e Ruolo del Prior Spaziale

Per quantificare in modo rigoroso quanta parte delle prestazioni ottenute nel tier Full-Frame e Macro derivi da un'effettiva capacità di localizzazione anziché dalla sola classificazione unita a un prior geometrico fisso, è utile considerare la baseline teorica della **"box a tutto fotogramma"** ($[c_x, c_y, w, h] = [0.5, 0.5, 1.0, 1.0]$):

* Se un oggetto occupa in media il $93\%$ dell'immagine (come nel tier Full-Frame), l'intersezione su unione (IoU) tra la scatola reale e l'intero fotogramma è analiticamente pari a:
  $$\text{IoU}_{\text{full}} = \frac{\text{Area}(\text{GT})}{\text{Area}(\text{Frame})} \approx 0.93 \gg 0.50$$
  Qualsiasi predizione che classifichi correttamente il fotogramma e generi una scatola sufficientemente ampia supera automaticamente la soglia di IoU $\ge 0.50$. Ciò spiega per quale motivo il tier Full-Frame conservi l'$84.14\%$ delle capacità dell'oracolo: in questo regime la sfida algoritmica è dominata dal riconoscimento semantico, in cui DINOv3 eccelle.
* Nel **Tier Macro**, al contrario, l'area media è del $61\%$ con un'ampia dispersione ($31\% - 78\%$) e una collocazione asimmetrica nel fotogramma. La baseline a tutto fotogramma produrrebbe un IoU frequentemente al di sotto di $0.50$ a causa dell'eccesso di sfondo incluso. Il fatto che la YOLO distillata raggiunga il $37.71\%$ di mAP@50 (e il $63.60\%$ di ritenzione relativa) dimostra che per questo tier il modello studente sta eseguendo una stima geometrica attiva, ritagliando contorni coerenti e non banali.

---

## 8.6 Qualità delle Pseudo-Label e Trasferimento allo Studente

Il confronto tra le prestazioni della pipeline di generazione (teacher DINOv3 su immagini di addestramento) e le prestazioni del rilevatore finale (studente YOLOv8n su test set indipendente) evidenzia le dinamiche del processo di distillazione:

1. **Fedeltà di apprendimento intra-dominio**: Sul set di validazione interno (immagini con pseudo-label del teacher), lo studente converge a un mAP@50 del **$96.64\%$** e a una precisione del **$95.12\%$**. Ciò prova che YOLOv8n possiede una capacità espressiva ideale per mimare la funzione di localizzazione del teacher senza subire sottodimensionamenti architetturali.
2. **Gap di generalizzazione sul test set reale**: Quando valutato rispetto alle annotazioni umane del test set continuo, il mAP@50 scende al **$37.39\%$** ($25.84\%$ mAP@50:95). Questa discrepanza riflette la combinazione di due fattori fisiologici:
   * Lo scarto tra i contorni continui stimati dalla CAM e i rettangoli discreti tracciati dagli annotatori umani.
   * Il *domain shift* tra le immagini statiche di training e le 7 sessioni video continue di test, caratterizzate da variazioni dinamiche di illuminazione, inclinazione e prospettiva.

In sintesi, la distillazione si dimostra un meccanismo altamente efficiente: pur registrando un divario rispetto alla supervisione completa sui target a piccola scala, consente di recuperare fino all'**$84\%$ delle prestazioni sui target estesi** e quasi i **due terzi ($64\%$) sui target intermedi**, abbattendo totalmente il costo del tracciamento manuale delle annotazioni per la maggioranza delle classi del dominio.
