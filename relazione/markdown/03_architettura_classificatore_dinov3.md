# 3. DINOv3 e Architettura del Classificatore

---

## 3.1 I Vision Transformer e il paradigma DINOv3

### 3.1.1 Fondamenti dei Vision Transformer (ViT)
I Vision Transformer (ViT)[^1] hanno ridefinito lo stato dell'arte nella visione artificiale, superando su numerosi fronti il paradigma storicamente dominante delle reti neurali convoluzionali (CNN). Mentre le CNN si basano su un forte bias induttivo locale, modellato tramite filtri convoluzionali invarianti per traslazione a campo recettivo limitato, l'architettura ViT elabora l'informazione visiva trattando l'immagine come una sequenza discreta di token unidimensionali, mutuando direttamente il meccanismo di auto-attenzione (*Self-Attention*) introdotto originariamente nell'elaborazione del linguaggio naturale[^2].

Data un'immagine in ingresso $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, essa viene suddivisa in una griglia regolare di $N = \frac{H \cdot W}{P^2}$ patch bidimensionali non sovrapposte di risoluzione $P \times P$:
$$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

Ciascuna patch viene linearmente proiettata in uno spazio di incorporamento (*embedding space*) di dimensione latente $D$ mediante una matrice di proiezione apprendibile $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$. Alla sequenza di token risultante viene anteposto un token speciale apprendibile di classificazione, denominato **token CLS** ($\mathbf{z}_{\text{CLS}} \in \mathbb{R}^D$), deputato ad aggregare una rappresentazione sintetica globale dell'intera immagine. Per preservare l'informazione geometrica bidimensionale, alla sequenza viene sommato un vettore di incorporamento posizionale (*Positional Embedding*) $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$:

$$\mathbf{z}_0 = \left[ \mathbf{z}_{\text{CLS}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}$$

La sequenza $\mathbf{z}_0$ attraversa quindi $L$ blocchi identici composti da meccanismi di *Multi-Head Self-Attention* (MSA) e blocchi *Multi-Layer Perceptron* (MLP), intervallati da normalizzazione di livello (*LayerNorm*, LN) e connessioni residue:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1, \dots, L$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

Il nucleo computazionale dell'auto-attenzione proietta i token in tre matrici distinte: Query ($\mathbf{Q}$), Key ($\mathbf{K}$) e Value ($\mathbf{V}$). La matrice dei coefficienti di attenzione $\mathbf{A}$, che quantifica la mutua rilevanza tra ogni coppia di token lungo l'intera immagine, è formalizzata come:

$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right), \quad \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \mathbf{V}$$

dove $d_k = D / h$ rappresenta la dimensionalità di ciascuna delle $h$ teste di attenzione. A differenza delle convoluzioni, questo meccanismo garantisce un **campo recettivo globale istantaneo** fin dal primo strato di computazione, consentendo al modello di correlare elementi visivi distanti nello spazio bidimensionale.

---

### 3.1.2 Il pretraining self-supervised di DINOv3
L'assenza di un bias induttivo rigido rende i Vision Transformer estremamente dipendenti dal volume di dati durante l'addestramento. L'approccio **DINO** (*Self-distillation with no labels*)[^3] e le sue evoluzioni architetturali fino a **DINOv3**[^4] risolvono questo vincolo mediante un paradigma di pre-addestramento auto-supervisionato (*self-supervised learning*) su dataset visivi su larghissima scala (il corpus LVD-1689M, comprendente circa 1.7 miliardi di immagini).

![Paradigma di pre-addestramento self-supervised DINO](figure/schema_dino_pretraining.png)

*Figura 3.1: Schema concettuale dell'apprendimento auto-supervisionato DINO. La rete Student viene addestrata su crop locali e globali tramite discesa del gradiente, mentre la rete Teacher è aggiornata per media mobile esponenziale (EMA) con regolarizzazione tramite centering e sharpening.*

La formulazione poggia su una dinamica di distillazione della conoscenza tra due reti omologhe:
1. Una rete **Student** con parametri $\theta_s$, aggiornata tramite discesa del gradiente.
2. Una rete **Teacher** con parametri $\theta_t$, la cui architettura è identica ma i cui pesi evolvono secondo una media mobile esponenziale (*Exponential Moving Average*, EMA) dei parametri dello Student:
   $$\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s, \quad \lambda \in [0.99, 1.0]$$

Durante il training, la rete Student riceve molteplici ritagli locali a risoluzione ridotta (*local crops*), mentre la rete Teacher elabora viste globali (*global crops*). La funzione di perdita minimizza l'entropia incrociata tra le distribuzioni di probabilità di output, spingendo lo Student a ricostruire la coerenza globale a partire da evidenze spaziali parziali. Per evitare il collasso della rappresentazione verso soluzioni banali o degeneri, l'uscita del Teacher è sottoposta congiuntamente a *centering* (sottrazione della media empirica dei vettori) e *sharpening* (temperatura ridotta sulla funzione Softmax).

Una delle proprietà emergenti più rilevanti del pre-addestramento DINO risiede nella struttura semantica delle sue rappresentazioni interne: **le mappe di attenzione dell'ultimo strato segmentano spontaneamente e senza alcuna etichetta di supervisione i soggetti in primo piano rispetto allo sfondo**. I token di patch conservano un'eccellente separabilità e coerenza semantica locale, rendendo DINOv3 una base ideale (*foundation model*) per compiti di localizzazione debolmente supervisionata (*Weakly-Supervised Object Localization*, WSOL).

---

## 3.2 Backbone ViT-S/16 e testa di classificazione lineare

### 3.2.1 Configurazione architetturale del backbone
Nel presente lavoro è stato adottato come backbone di estrazione delle caratteristiche il modello pre-addestrato **ViT-S/16** (Vision Transformer Small con patch da $16 \times 16$ pixel). I parametri architetturali fondamentali del modello sono riassunti nella seguente tabella:

| Iperparametro | Valore | Descrizione |
| :--- | :---: | :--- |
| **Dimensione latente ($D$)** | $384$ | Dimensione del vettore di embedding di ciascun token |
| **Numero di strati ($L$)** | $12$ | Blocchi consecutivi di Transformer Encoder |
| **Teste di attenzione ($h$)** | $6$ | Numero di teste parallele nel modulo Multi-Head Self-Attention |
| **Dimensione per testa ($d_k$)** | $64$ | Spazio di proiezione di ciascuna testa ($384 / 6$) |
| **Dimensione patch ($P$)** | $16$ | Risoluzione spaziale di ciascuna patch elementare ($16 \times 16$ px) |
| **Numero di parametri** | $\approx 22\text{ M}$ | Complessità parametrica complessiva del backbone |

---

### 3.2.2 Strategia di addestramento: Linear Probing e Backbone congelato
Per valorizzare appieno le caratteristiche semantiche e spaziali apprese da DINOv3 durante il pretraining, è stata adottata la metodologia del **Linear Probing**:
- **Pesi del backbone congelati (*frozen*)**: I parametri $\theta_{\text{backbone}}$ non ricevono aggiornamenti di gradiente durante l'addestramento ($\nabla_{\theta_{\text{backbone}}} \mathcal{L} = 0$). Questa scelta garantisce che le proprietà di auto-segmentazione e l'omogeneità dello spazio latente restino intatte, evitando fenomeni di *catastrophic forgetting* su un dataset specialistico come UNICT-VEDI.
- **Testa di classificazione lineare**: Sul token globale $\mathbf{z}_{\text{CLS}}^{(L)} \in \mathbb{R}^{384}$, estratto all'uscita del dodicesimo strato, viene innestata una testa di predizione composta da un layer di regolarizzazione *Dropout* ($p = 0.3$) seguito da un singolo strato lineare completamente connesso (*Dense Layer*):
  $$\hat{\mathbf{y}} = \mathbf{W} \cdot \text{Dropout}(\mathbf{z}_{\text{CLS}}^{(L)}) + \mathbf{b}$$
  dove la matrice dei pesi $\mathbf{W} \in \mathbb{R}^{57 \times 384}$ mappa direttamente lo spazio delle feature di DINOv3 sui logits associati alle 57 classi di POI del dataset, con vettore di bias $\mathbf{b} \in \mathbb{R}^{57}$.

![Architettura del classificatore DINOv3 con Linear Probing e Linear-CAM](figure/architettura_classificatore.png)

*Figura 3.2: Flusso computazionale del classificatore DINOv3: il backbone ViT-S/16 congelato elabora il tensore d'ingresso fornendo sia il token $[\text{CLS}]$ per la classificazione categoriale (ramo superiore), sia i token spaziali di patch impiegati per l'attribuzione Linear-CAM (ramo inferiore).*

Questa formulazione lineare offre un vantaggio strutturale determinante per la successiva localizzazione: poiché la proiezione è strettamente lineare, **la riga $\mathbf{w}_c \in \mathbb{R}^{384}$ della matrice $\mathbf{W}$ rappresenta la direzione ottimale nello spazio latente che massimizza il punteggio di appartenenza alla classe $c$**. Tale proprietà permette di valutare analiticamente il contributo di ciascuna area dell'immagine proiettando $\mathbf{w}_c$ sui token di patch locali.

---

## 3.3 Discretizzazione in patch e risoluzione intrinseca della CAM

### 3.3.1 Relazione spaziale tra risoluzione di ingresso e griglia dei token
La risoluzione spaziale delle immagini somministrate al modello è impostata a $H = W = 384$ pixel. Poiché ciascuna patch elementare copre una superficie di $P \times P = 16 \times 16$ pixel, il piano bidimensionale viene discretizzato in una griglia regolare:

$$H_p = \frac{384}{16} = 24, \quad W_p = \frac{384}{16} = 24$$

All'uscita dell'ultimo strato, oltre al token di classificazione $\mathbf{z}_{\text{CLS}}$, la rete produce una mappa tensoriale di token spaziali:
$$\mathbf{Z}_{\text{patch}} \in \mathbb{R}^{24 \times 24 \times 384}$$
in cui ciascun vettore $\mathbf{z}_{i,j} \in \mathbb{R}^{384}$ (con $i, j \in \{1, \dots, 24\}$) riassume le caratteristiche visive della corrispondente regione locale dell'immagine.

A differenza delle reti convoluzionali tradizionali (come ResNet-50), che riducono la risoluzione mediante stadi di pooling e stride successivi fino a un fattore di sottocampionamento di $32\times$ (producendo ad esempio una mappa $7 \times 7$ per input $224 \times 224$), il Vision Transformer conserva uno stride fisso pari a $16$, generando una matrice di attivazione sensibilmente più densa ($576$ celle).

---

### 3.3.2 Il limite di risoluzione per il Tier Tiny
Nonostante la griglia $24 \times 24$ fornisca una risoluzione superiore rispetto alle CNN convenzionali, la dimensione minima della cella di campionamento ($16 \times 16$ pixel originali) impone un limite teorico alla capacità di localizzare dettagli di taglia ridotta.

Correlando la griglia dei token con i tier dimensionali definiti nel Capitolo 2, emergono differenze strutturali profonde:

| Tier | Area media occupata ($\bar{A}_c$) | Numero equivalente di patch ($24 \times 24 \times \bar{A}_c$) | Dimensione su griglia token | Risoluzione teorica della CAM |
|:---|:---:|:---:|:---:|:---|
| **Tiny** | $\approx 4.97\%$ | **$\approx 28$ patch** | $\approx 5 \times 5$ celle | **Critica**: forte effetto di bordo e inclusione sfondo |
| **Small** | $\approx 26.27\%$ | **$\approx 151$ patch** | $\approx 12 \times 12$ celle | **Intermedia**: risoluzione discreta ma dispersione sul contesto |
| **Macro** | $\approx 61.19\%$ | **$\approx 352$ patch** | $\approx 19 \times 19$ celle | **Ampia**: localizzazione stabile del nucleo dell'oggetto |
| **Full-Frame** | $\approx 93.09\%$ | **$\approx 536$ patch** | $\approx 23 \times 23$ celle | **Saturata**: attivazione estesa a quasi l'intera griglia |

Per le classi appartenenti al Tier Tiny (es. `6.5_BassorilieviMancanti`, con area media del $3.1\%$, pari a circa 18 patch complessive), una singola patch al contorno contiene contestualmente sia una porzione dell'oggetto d'interesse sia un'ampia porzione dello sfondo circostante, inducendo una dispersione dell'energia di attivazione. Similmente, per il Tier Small, sebbene la discretizzazione offra un centinaio di celle, la preponderanza dell'ambiente murario (che occupa dal $70\%$ al $90\%$ dell'inquadratura) espone le mappe CAM al rischio di dilatazione sul contesto architettonico circostante, anticipando le complessità di localizzazione che emergeranno nelle valutazioni empiriche.

---

## 3.4 Connessione concettuale verso la localizzazione debole (WSOL)

L'architettura del classificatore non costituisce soltanto uno strumento di inferenza dell'etichetta semantica, ma funge da **generatore primario delle evidenze spaziali** necessarie per estrarre le coordinate di bounding box senza supervisione geometrica. 

I token spaziali $\mathbf{z}_{i,j}$, i pesi della testa lineare $\mathbf{W}$ e le matrici di attenzione $\mathbf{A}$ contengono informazioni complementari sulla salienza visiva, che possono essere interrogate attraverso diverse formulazioni matematiche:

1. **Linear-CAM (Class Activation Mapping lineare)**:  
   Sfrutta la natura lineare del classificatore calcolando il prodotto scalare tra il vettore dei pesi della classe predetta $\mathbf{w}_c$ e ciascun token spaziale $\mathbf{z}_{i,j}$:
   $$M_{\text{Linear}}(i, j) = \mathbf{w}_c^T \mathbf{z}_{i,j}$$
   Fornisce una mappa selettiva per classe direttamente legata alla decisione del modello.

2. **DINO Self-Attention Maps**:  
   Estrae la distribuzione dei pesi di attenzione dell'ultimo strato dal token CLS verso ciascun token spaziale, mediata sulle teste di attenzione:
   $$M_{\text{Attn}}(i, j) = \frac{1}{h} \sum_{k=1}^h \mathbf{A}_{k, \, \text{CLS} \to (i,j)}^{(L)}$$
   Sfrutta la tendenza naturale del pretraining DINOv3 a raggruppare le regioni in primo piano in modo indipendente dall'etichetta (*class-agnostic*).

3. **Grad-CAM per Vision Transformer**:  
   Pesa l'attivazione dei singoli token mediante il gradiente del punteggio di classe $y_c$ rispetto agli stati latenti dell'ultimo strato:
   $$\alpha_{i,j}^c = \frac{\partial y_c}{\partial \mathbf{z}_{i,j}}, \quad M_{\text{GradCAM}}(i, j) = \text{ReLU}\left( \sum_{d=1}^D \alpha_{i,j,d}^c \cdot \mathbf{z}_{i,j,d} \right)$$
   Identifica le componenti latenti la cui variazione influenza maggiormente la confidenza del POI bersaglio.

4. **Eigen-CAM**:  
   Applica la decomposizione a valori singolari (SVD) sulla matrice dei token spaziali $\mathbf{Z}_{\text{patch}}$, proiettando le feature sul primo autovettore principale:
   $$M_{\text{Eigen}}(i, j) = \mathbf{z}_{i,j} \cdot \mathbf{v}_1, \quad \text{con } \mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \|\mathbf{Z}_{\text{patch}} \mathbf{v}\|$$
   Isola la componente dominante di energia visiva presente nel fotogramma.

La trattazione formale, l'analisi comparativa empirica e l'algoritmo di conversione da mappa bidimensionale a coordinate di bounding box costituiscono il fulcro metodologico del **Capitolo 5**.

---

[^1]: A. Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, in *International Conference on Learning Representations (ICLR)*, 2021.
[^2]: A. Vaswani et al., *"Attention Is All You Need"*, in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
[^3]: M. Caron et al., *"Emerging Properties in Self-Supervised Vision Transformers"*, in *International Conference on Computer Vision (ICCV)*, 2021.
[^4]: M. Oquab et al., *"DINOv2: Learning Robust Visual Features without Supervision"*, in *Transactions on Machine Learning Research (TMLR)*, 2023.
