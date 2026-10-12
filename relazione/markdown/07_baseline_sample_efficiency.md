# 7. Baseline Supervisionate e Protocollo di Sample Efficiency

---

## 7.1 Obiettivo e Disegno Sperimentale

La domanda di ricerca centrale posta all'inizio del lavoro è:
> *A quale frazione di dati annotati manualmente con bounding box equivale un modello addestrato senza alcuna annotazione geometrica supervisionata?*

Per rispondere in modo quantitativo e riproducibile, è stato progettato un protocollo comparativo di **Sample Efficiency**. Tale protocollo confronta il rilevatore distillato dalle sole etichette d'immagine (**YOLOv8n distillato**) con una serie di modelli di riferimento (*baseline supervisionate*) addestrati con percentuali crescenti di bounding box reali (*Ground Truth*).

![Disegno sperimentale e protocollo di sample efficiency](figure/protocollo_sample_efficiency.png)

*Figura 7.1: Disegno sperimentale del benchmark di Sample Efficiency: partizionamento stratificato del budget di annotazione umana su 7 frazioni (con repliche stocastiche e validazione interna), a confronto diretto con il modello YOLOv8n distillato (0% bounding box di Ground Truth) sul test set ufficiale non visto.*

### 7.1.1 Frazioni di campionamento e repliche stocastiche
Per tracciare la curva di apprendimento sono state selezionate sette frazioni di supervisione:
$$\mathcal{F} = \{1\%, \, 2\%, \, 5\%, \, 10\%, \, 20\%, \, 40\%, \, 100\%\}$$

Per ciascuna frazione compresa tra $1\%$ e $40\%$, il campionamento dal dataset di addestramento è stato condotto secondo i seguenti criteri:
* **Campionamento stratificato per classe**: La frazione $\alpha \in \mathcal{F}$ viene applicata al numero di immagini di ciascuna delle 57 cartelle di classe. La scelta di campionare per *numero di fotogrammi* (anziché per numero di riquadri) rispecchia fedelmente il processo reale di annotazione umana, dove il costo di annotazione scala proporzionalmente al numero di frame ispezionati.
* **Controllo della varianza (3 seed stocastici)**: Per mitigare le fluttuazioni dovute alla selezione fortunata o sfortunata dei frame a regimi di scarsità di dati, ciascun esperimento sotto il $100\%$ viene replicato su 3 distinti seed casuali ($\text{seed} \in \{42, 1337, 2026\}$). I risultati finali sono riportati come media e deviazione standard ($\mu \pm \sigma$). L'adozione dei medesimi semi per tutte le frazioni configura un disegno sperimentale a blocchi (*paired comparison*).
* **Partizionamento interno per la validazione**: Da ciascun sottoinsieme estratto viene ritagliata una quota interna del $15\%$ riservata alla validazione (*early stopping* e selezione del miglior checkpoint). In tal modo, l'ottimizzazione degli iperparametri e la regolarizzazione non contaminano mai il test set esterno.

---

## 7.2 Metriche e Protocollo di Valutazione

Tutti i modelli supervisionati e il modello distillato vengono valutati esclusivamente sul **Test Set ufficiale** non visto ($5.462$ fotogrammi ripartiti nelle 7 sessioni continue di test). 

La valutazione adotta il formalismo standard COCO basato su:
* **mAP@50**: Precisione media calcolata alla soglia di sovrapposizione $\text{IoU} \ge 0.50$, metrica di riferimento primaria per quantificare la rilevazione corretta del punto di interesse.
* **mAP@50:95**: Precisione media integrata su dieci soglie di IoU progressive da $0.50$ a $0.95$ con passo di $0.05$, fondamentale per valutare la precisione geometrica millimetrica dei contorni predetti.
* **Disaggregazione per Tier Dimensionale**: Per evidenziare la dipendenza tra scala dell'oggetto e qualità della localizzazione, le metriche vengono calcolate sia a livello globale sia stratificate sui 4 tier dimensionali:
  * *Tiny* (area relativa $\le 0.10$)
  * *Small* ($0.10 < \text{area} < 0.30$)
  * *Macro* ($0.30 \le \text{area} < 0.80$)
  * *Full-Frame* ($\text{area} \ge 0.80$)

---

## 7.3 Equità Metodologica del Confronto (*Fairness*)

Per garantire che la stima del valore di sample efficiency sia scientificamente solida e non viziata da asimmetrie di modellazione, il disegno sperimentale adotta una rigida politica di controllo delle variabili.

### 7.3.1 Condizioni identiche tra le varianti
Tra il modello distillato e tutti i modelli delle baseline supervisionate sussiste una perfetta parità metodologica:
1. **Architettura identica**: Tutti i modelli utilizzano la dorsale convoluzionale compatta **YOLOv8n** ($3.2\text{M}$ parametri).
2. **Inizializzazione equivalente**: Tutti i training partono dai pesi preaddestrati ufficiali su COCO (`yolov8n.pt`).
3. **Risoluzione e batch size**: Tutti i modelli operano a risoluzione di input di $640 \times 640$ pixel con batch size pari a 16 (o ridotto a 8 per i sottoinsiemi più piccoli garantendo invarianza nei gradienti).
4. **Data Augmentation**: Stessa politica di regolarizzazione stocastica (Mosaic con chiusura nelle ultime epoche, Random Horizontal Flip, Random Erasing, perturbazioni HSV).
5. **Ambiente di test indipendente**: Tutti i modelli vengono misurati sullo stesso identico insieme di etichette umane del test set.

### 7.3.2 L'asimmetria ontologica: il costo dell'annotazione
L'unico elemento asimmetrico risiede nella **natura e nel costo della supervisione**:
* Le **baseline supervisionate** ricevono annotazioni cartesiane complete $[c_x, c_y, w, h]$, il cui costo di annotazione manuale è elevato e richiede l'intervento di operatori umani per delimitare manualmente i quattro vertici dell'oggetto in ciascun fotogramma.
* Il **modello distillato**, al contrario, è stato addestrato a costo geometrico zero: non ha mai avuto accesso a un singolo riquadro umano, traendo l'intera informazione di delimitazione spaziale dalle rappresentazioni dense apprese in modo auto-supervisionato da DINOv3 e proiettate via Linear-CAM.

Il capitolo successivo (Capitolo 8) analizza i risultati empirici derivanti da questo confronto, determinando il punto di pareggio (*breakeven point*) tra le due curve.
