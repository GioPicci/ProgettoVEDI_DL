# 10. Conclusioni e Sviluppi Futuri

---

## 10.1 Risposta alla Domanda di Ricerca

Il presente lavoro ha indagato quantitativamente una questione centrale per l'applicazione delle tecniche di visione artificiale ai beni culturali:
> *A quale frazione di dati annotati manualmente con bounding box equivale una pipeline di Weakly-Supervised Object Localization basata su Vision Transformer, addestrata senza alcuna informazione geometrica a priori?*

L'indagine condotta sul complesso dataset monumentale UNICT-VEDI (57 classi di punti di interesse distribuite su 9 macro-ambienti storici) consente di trarre conclusioni nette e articolate su due livelli:

### 1. Il valore campionario (Breakeven Point)
Dal punto di vista dell'efficienza campionaria pura, il rilevatore compatto **YOLOv8n distillato** a partire dalle mappe Linear-CAM di DINOv3 ($37.39\%$ mAP@50 sul test set indipendente) si colloca a un livello equivalente a circa l'**$1.43\%$ di annotazioni umane complete** con coordinate cartesiane. Questo risultato dimostra che è possibile superare una baseline supervisionata all'$1\%$ senza che un operatore umano debba delimitare manualmente anche un solo riquadro.

### 2. La ritenzione delle capacità ottimali (Capacity Retention)
Valutando il sistema rispetto al tetto teorico espresso dall'oracolo supervisionato al $100\%$ ($60.51\%$ mAP@50), la pipeline a costo geometrico zero recupera il **$61.79\%$ delle massime capacità del modello**:
* **Nel Tier Macro (il vero valore applicativo)**: Su 26 classi eterogenee di oggetti intermedi (portali, altari, affreschi parziali, sculture, cisterne), per le quali l'annotazione manuale di decine di migliaia di fotogrammi richiederebbe centinaia di ore-uomo, il modello debolmente supervisionato trattiene il **$63.60\%$ delle prestazioni dell'oracolo** ($37.71\%$ vs $59.29\%$). Si dimostra così che la combinazione DINOv3 + Linear-CAM non esegue una banale predizione a tutto schermo, ma produce una reale e informativa stima dei confini geometrici.
* **Nel Tier Full-Frame**: Il modello raggiunge l'**$84.14\%$ di ritenzione** e pareggia il $6\%$ di annotazioni supervisionate, sebbene la natura geometrica dell'inquadratura riconduca il problema a una classificazione d'immagine.
* **Nel Tier Tiny**: Emerge il limite dell'approccio debolmente supervisionato ($0.00\%$ mAP@50). L'analisi visiva ha tuttavia dimostrato che il fallimento non è dovuto a un'incapacità visiva di DINOv3 di segmentare il piccolo oggetto, bensì all'insorgere di un **bias contestuale**: la funzione di costo di classificazione concentra l'evidenza predittiva sulle imponenti strutture architettoniche circostanti (archi, volte, cornici) anziché sul reperto specifico, spostando il picco di calore della CAM al di fuori del riquadro umano.

---

## 10.2 Sviluppi Futuri

I risultati e i limiti emersi suggeriscono diverse direttrici di ricerca metodologica per estendere e potenziare la pipeline proposta:

### 10.2.1 Moduli di Grounding sull'Attention di DINOv3 per Rilevazioni Multi-Oggetto
Il limite più severo dell'approccio attuale è l'assunzione di una singola bounding box per fotogramma, derivante dall'estrazione attorno all'unico picco globale della CAM.
Un'evoluzione naturale consiste nell'innestare un modulo di **Visual Grounding / Prompt-based Querying** (architettura cross-attention in stile DETR o Grounding DINO) direttamente sopra i token di patch e le matrici di auto-attenzione del Vision Transformer:
* Anziché dipendere da una sogliatura euristica, il modello riceve come query semantica una lista di "candidati" (es. i nomi dei reperti attesi nella sala o descrizioni testuali dei dettagli).
* Il modulo di cross-attention può interrogare simultaneamente le rappresentazioni dense di DINOv3, identificando molteplici raggruppamenti di token salienti e disaccoppiando istanze distinte coesistenti nella medesima inquadratura. In tal modo, le proprietà emergenti di segmentazione di DINOv3 verrebbero sfruttate per la detection multi-oggetto debolmente supervisionata.

### 10.2.2 Decomposizione Gerarchica (Macro-Area $\to$ POI) e Formulazione Multi-Label
La Linear-CAM opera calcolando il prodotto scalare tra il vettore dei pesi di classe $\mathbf{w}_c$ e ciascun token spaziale $\mathbf{z}_p$. Attualmente, il modello è addestrato come un classificatore piatto a 57 classi con normalizzazione Softmax mutuamente esclusiva.
Due modifiche architetturali potrebbero risolvere alla radice il bias contestuale:
1. **Pipeline gerarchica a cascata**: Poiché DINOv3 identifica la Macro-Area con un'accuratezza del $92.87\%$, è possibile strutturare l'inferenza in due livelli:
   * Un primo stadio predice l'ambiente monumentale (ancorandosi legittimamente alle pareti, alle volte e all'architettura globale).
   * Un secondo stadio, condizionato alla macro-area individuata, discrimina specificamente i POI locali. Sgravando i pesi del POI dal dover riconoscere anche l'ambiente circostante, la proiezione $\mathbf{w}_{\text{poi}}^\top \mathbf{z}_p$ si concentrerebbe in modo molto più selettivo sulle peculiarità del reperto piccolo.
2. **Formulazione Multi-Labeling (Sigmoid / BCE)**: Sostituendo la Softmax con attivazioni Sigmoid indipendenti per classe, i vettori di peso non competerebbero più per l'energia complessiva del fotogramma. Se in una sala sono presenti contemporaneamente una statua e un arco decorato, la Linear-CAM potrebbe attivarsi contemporaneamente e indipendentemente su entrambe le regioni spaziali.

### 10.2.3 Regolarizzazione Anti-Bias Contestuale e Risoluzione Flessibile
Per contrastare l'attrazione gravitazionale dello sfondo nei target di piccola taglia:
* **Perdite di contrasto tra patch (*Patch-level Contrastive Loss*)**: Introduzione di una penalizzazione durante il training della testa lineare che scoraggi l'attivazione dei pesi di un POI ristretto su token di sfondo ricorrenti in più classi della medesima sala.
* **Aumento della granularità di campionamento**: Utilizzo di risoluzioni di ingresso superiori (es. $512 \times 512$ o $768 \times 768$) o di varianti ViT con *patch stride* ridotto (es. 8 o 14 px), per fornire un numero quintuplicato di token su elementi del tier Tiny, agevolando la separazione tra contorno del manufatto e modanatura muraria.

### 10.2.4 Adozione di Metriche di Valutazione Disaccoppiate
Infine, per separare formalmente l'accuratezza semantica dalla qualità della delimitazione geometrica, futuri benchmark potranno affiancare al mAP metriche come:
* **CorLoc (*Correct Localization*)**: Percentuale di immagini per le quali la box predetta ha $\text{IoU} \ge 0.50$ con il Ground Truth, valutata assumendo la classe come nota.
* **Class-Agnostic mAP**: Misura la pura capacità del modello di ritagliare la sagoma saliente indipendentemente dall'etichetta associata.

Questo disaccoppiamento consentirà di isolare con precisione millimetrica l'errore di pura regressione dei contorni rispetto alle ambiguità di classificazione tassonomica.
