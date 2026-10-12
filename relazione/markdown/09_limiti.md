# 9. Limiti della Metodologia

---

## 9.1 Assunzione di Singola Istanza per Immagine

Il limite metodologico più vincolante dell'intera pipeline risiede nell'**assunzione a priori di una singola bounding box per fotogramma**.

La procedura deterministica di conversione da heatmap a coordinate seleziona esclusivamente il contorno chiuso che racchiude il picco globale di attivazione $\arg\max \mathcal{H}(x, y)$. Sebbene questa euristica sia efficace per la maggior parte dei punti di interesse isolati, manifesta evidenti criticità negli ambienti museali complessi:
* **Presenza di istanze multiple**: In numerose inquadrature del sito monumentale sono visibili contemporaneamente più elementi di interesse distinti (es. busti decorativi multipli, sequenze di paraste, rampe prospettiche o porte multiple lungo il medesimo corridoio).
* **Effetto di fusione (*Bounding Box Merging*) o soppressione**: Quando due entità appartenenti alla medesima macro-area coesistono nel frame, la pipeline estrae unicamente il riquadro circoscritto al picco dominante, sopprimendo la seconda istanza oppure, se la mappa di attivazione genera un "ponte" a bassa densità tra i due fuochi, fondendo entrambi gli oggetti in un unico macro-box fittizio che include un'ampia porzione di sfondo non annotato.

Questo limite si riflette direttamente sul dataset distillato, che trasmette allo studente YOLO una supervisione a singola istanza per frame, penalizzando il recall in scene ad alta densità di reperti.

---

## 9.2 Bias Contestuale e Disallineamento Discriminativo

Come evidenziato dall'indagine qualitativa condotta nel Capitolo 8 sui punti di interesse a scala ridotta (*Tier Tiny*, quali *Sant'Agata* e *Cupola*), l'assenza di supervisione geometrica espone il sistema al fenomeno del **bias contestuale** (*Contextual Co-occurrence Bias*):

* La funzione di costo del classificatore (Cross-Entropy a livello d'immagine) premia l'estrazione di qualunque correlazione visiva in grado di minimizzare l'errore di classificazione, senza alcuna distinzione tra il manufatto d'interesse e lo scenario architettonico circostante.
* Nei contesti museali storici, un reperto è fisicamente e permanentemente incorporato all'interno di una sala specifica (es. sopra un arco in stucco, incastonato in una volta a botte o collocato su un pavimento caratteristico). Poiché l'architettura circostante è decine di volte più estesa dell'oggetto e sempre co-occorrente, il modello proietta la maggior parte dei pesi della testa lineare sulle feature dell'ambiente.
* Di conseguenza, sebbene il Vision Transformer sia capace di segmentare l'oggetto a livello latente, il **picco assoluto di energia della CAM viene attirato dal contesto monumentale**, portando la pipeline di localizzazione a generare box disallineati rispetto alle delimitazioni stabilite dagli annotatori umani.

---

## 9.3 Rigidità della Binarizzazione con Soglia Costante

Il passaggio dalla distribuzione di calore continua alla maschera binaria si affida a una soglia relativa uniforme ($\tau = 0.45$ del valore di picco):

* **Eterogeneità geometrica non considerata**: Questa soglia presuppone che l'energia di attivazione decada rapidamente al di fuori del perimetro dell'oggetto secondo una distribuzione unimodale isotropa. Tuttavia, elementi architettonici concavi, sottili o allungati (quali balaustre, gradinate, stipiti di portali o grandi tele verticali) esibiscono gradienti di confidenza asimmetrici.
* **Compromesso tra sotto-segmentazione e sovra-segmentazione**: Una soglia fissa produce inevitabilmente un compromesso rigido: per classi con attivazione molto concentrata genera box troppo stretti che tagliano porzioni del manufatto; per classi a forte dispersione energetica tende a includere ampie aree adiacenti, penalizzando la precisione geometrica alle soglie IoU più severe (mAP@50:95).

---

## 9.4 Approssimazione Rettangolare di Maschere Irregolari

La formulazione del task secondo lo standard di Object Detection impone la delimitazione degli oggetti tramite **bounding box asse-allineati** $[x, y, w, h]$:

* Per elementi architettonici con orientamento diagonale (es. rampe di scale prospettiche come la *Rampa S. Benedetto*, archi inclinati o balaustre riprese di scorcio), il rettangolo minimo circoscritto a un contorno obliquo racchiude un volume considerevole di spazio vuoto e pixel non pertinenti.
* Questa approssimazione introduce una discrepanza strutturale: anche in presenza di una mappa di salienza che segmenta fedelmente la sola sagoma dell'oggetto, la scatola cartesiana circoscritta vede diminuire il proprio IoU rispetto al poligono reale, ponendo un limite intrinseco alla precisione geometrica ottenibile rispetto a una pipeline di Weakly-Supervised Semantic Segmentation.

---

## 9.5 Domain Shift Dinamico tra Training Statico e Test Continuo

Un ultimo fattore critico riguarda la discrepanza cinematica e prospettica tra le due partizioni del dataset:

* **Partizione di addestramento (Teacher)**: Le immagini sorgente su cui DINOv3 genera le pseudo-label provengono da collezioni fotografiche organizzate per singolo ambiente/reperto, con inquadrature generalmente stabili, frontali e a distanza standard.
* **Partizione di test (Student)**: Le 7 sessioni continue di test simulano la fruizione reale di un visitatore che percorre il museo con una videocamera mobile, introducendo repentini cambi di scala, rotazioni sull'asse ottico, occlusioni parziali temporanee e sfocature da movimento (*motion blur*).

Tale *domain shift* implica che lo studente YOLOv8n, addestrato su coordinate estratte da prospettive prevalentemente stabili, debba generalizzare su angolazioni dinamiche mai affrontate dal teacher durante la fase di pseudo-etichettatura.
