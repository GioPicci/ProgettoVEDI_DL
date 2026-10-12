# Titolo del progetto

**Rilevamento di POI senza bounding box annotate: pseudo-label da DINOv3 (WSOL) e distillazione in YOLOv8**

Corso: ___ · Autori: ___ · Anno accademico: ___

> Legenda: `TODO` = da scrivere · `[FIG]` = figura da inserire · `[TAB]` = tabella da inserire · `[FILE]` = file da cui prendere i dati

---

## Abstract

*(Scrivere per ultimo. 5-6 righe: problema, metodo, risultato principale, limite principale.)*

TODO

---

## 1. Introduzione e definizione del problema

### 1.1 Contesto
- Riconoscimento di punti di interesse (POI) nel museo, dataset UNICT-VEDI
- Costo dell'annotazione con bounding box rispetto alle sole label a livello di immagine

TODO

### 1.2 Domanda di ricerca
> Quanto vale, in termini di annotazioni supervisionate, una pipeline addestrata senza nessuna bounding box annotata?

TODO

### 1.3 Panoramica della pipeline
- [FIG] Schema a blocchi: classificatore DINOv3 → CAM → pseudo-box → dataset distillato → YOLOv8

TODO

### 1.4 Cenni a WSOL, CAM e distillazione del dataset
*(Mezza pagina, 3-4 riferimenti bibliografici.)*

TODO

---

## 2. Dataset e preprocessing
> *Documento dedicato completo: vedi [`02_dataset_e_preprocessing.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/02_dataset_e_preprocessing.md)*

### 2.1 Struttura del dataset
- 57 classi POI (distribuite sui 9 macro-ambienti storici del Monastero dei Benedettini), split train/val/test, formato YOLO

### 2.2 Preprocessing, correzione delle etichette e convenzione Full-Frame
- Sanitizzazione coordinate (pixel normalizzati, clamping, eliminazione box degenerate)
- Ordinamento naturale delle cartelle (`natural_sort_key`) per coerenza con le classi YOLO
- Regola ufficiale per i frame non etichettati: bounding box a tutto fotogramma ([0.5, 0.5, 1.0, 1.0])

### 2.3 Area media per classe e definizione dei tier
- Area media della GT box / area dell'immagine calcolata su tutte le label (`CLASS_AREAS`)
- *Nota metodologica: la suddivisione in tier dimensionali è un'introduzione non ufficiale, adottata ad-hoc per comodità d'indagine*
- [TAB] Tier, numero di classi, area media, soglie

| Tier | Soglia area media | N. classi | Area media | Range (min - max) |
|---|---|---|---|---|
| Tiny | ≤ 0.10 | 4 | 4.97% | 3.1% – 7.2% |
| Small | 0.10 – 0.30 | 3 | 26.27% | 21.9% – 29.7% |
| Macro | 0.30 – 0.79 | 27 | 61.19% | 31.5% – 78.6% |
| Full-Frame | ≥ 0.80 | 23 | 93.09% | 81.3% – 99.8% |
| **Totale** | — | **57** | **70.75%** | **3.1% – 99.8%** |

*(Nota: Preprocessing in ingresso, normalizzazione e Data Augmentation stocastica sono trattati specificamente nei rispettivi capitoli metodologici di training: Cap. 4 per DINOv3 e Cap. 6 per YOLO).*

---

## 3. DINOv3 e architettura del classificatore
> *Documento dedicato completo: vedi [`03_architettura_classificatore_dinov3.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/03_architettura_classificatore_dinov3.md)*

### 3.1 Vision Transformer e DINOv3 in breve
- Fondamenti del Vision Transformer: patch di 16 px, token CLS, multi-head self-attention, campo recettivo globale.
- Pretraining self-supervised (DINO/DINOv3, LVD-1689M, distillazione Student-Teacher EMA) e proprietà emergenti di segmentazione dello sfondo.

### 3.2 Backbone ViT-S/16 e testa lineare
- Backbone ViT-S/16 ($D=384$, 12 blocchi, 6 teste) con pesi congelati (*linear probing*).
- Testa di classificazione: Dropout ($p=0.3$) + Linear Layer ($384 \to 57$). Proprietà di interpretabilità dei pesi per la WSOL.

### 3.3 Griglia di patch e risoluzione della CAM
- Input a $384 \times 384$ px $\to$ griglia di token $24 \times 24$ (576 celle).
- Analisi del limite teorico di risoluzione per il Tier Tiny (oggetti con $3-7\%$ dell'area coperti da sole 17-28 patch).

### 3.4 Connessione concettuale verso la localizzazione debole (WSOL)
- Breve introduzione ai meccanismi di attribuzione e salienza: Linear-CAM, Self-Attention Maps, Grad-CAM ed Eigen-CAM.

---

## 4. Training del classificatore
> *Documento dedicato completo: vedi [`04_training_e_risultati_classificatore.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/04_training_e_risultati_classificatore.md)*

### 4.1 Setup
- Backbone ViT-S/16 frozen ($D=384$) + Dropout ($0.3$) + Linear Head ($384 \to 57$).
- Preprocessing ($384 \times 384$, normalizzazione ImageNet), Data Augmentation (Horizontal Flip $p=0.5$, Color Jitter).
- Ottimizzatore AdamW (lr $10^{-4}$, weight decay $10^{-4}$), scheduler Cosine con warmup, Cross-Entropy Loss, batch size 16, 20 epoche (miglior modello all'epoca 13 con val acc $98.31\%$).

### 4.2 Risultati di classificazione sul Test Set
- [TAB] Prestazioni globali (Top-1 Accuracy $70.24\%$, Top-5 Accuracy $84.32\%$, Macro Recall $72.10\%$, Macro F1 $70.00\%$) e per tier (Tiny $94.38\%$, Small $84.71\%$, Macro $66.10\%$, Full-Frame $73.38\%$).
- [FIG] Matrice di confusione tra macro-ambienti $9 \times 9$ e scomposizione errore intra/inter-ambiente (il $92.87\%$ ricade nella macro-area corretta, il $76\%$ degli errori è intra-ambiente; matrice completa $57 \times 57$ in Appendice B).

---

## 5. Generazione delle pseudo-label (WSOL)
> *Documento dedicato completo: vedi [`05_generazione_pseudo_label_wsol.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/05_generazione_pseudo_label_wsol.md)*

### 5.1 Dalle mappe di attivazione alle bounding box
- Pipeline deterministica in 4 stadi: interpolazione bicubica, normalizzazione, binarizzazione differenziata, apertura morfologica $7 \times 7$ ed estrazione contorno attorno al picco globale.
- Risoluzione dell'anomalia di `dino_attn` e `grad_cam` (collasso a zero): binarizzazione al $50^\circ$ percentile di energia, soglia di area minima $5\%$, e calcolo dei gradienti sullo strato penultimo del ViT.

### 5.2 Protocollo di valutazione comparativa
- Valutazione quantitativa esaustiva su $57.000$ immagini del training set annotato.
- Confronto tra 4 metodi di attribuzione: `linear_cam`, `dino_attn`, `eigen_cam`, `grad_cam`.
- Metriche: mAP@50, mAP@50:95, Recall globale e disaggregazione per tier dimensionale.

### 5.3 Risultati sperimentali e selezione del metodo
- [TAB] Metriche comparative globali (`linear_cam`: $64.65\%$ mAP@50, $41.56\%$ mAP@50:95; `dino_attn`: $26.02\%$; `grad_cam`: $15.70\%$; `eigen_cam`: $12.94\%$).
- [TAB] Prestazioni disaggregate per tier dimensionale (`linear_cam`: Full-Frame $74.79\%$, Macro $68.04\%$, Small $18.32\%$, Tiny $0.00\%$).
- Analisi critica: motivazione della rinuncia alla selezione combinatoria per-classe in favore dell'adozione universale di `linear_cam` per la distillazione.

---

## 6. Distillazione in YOLOv8
> *Documento dedicato completo: vedi [`06_distillazione_yolo.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/06_distillazione_yolo.md)*

### 6.1 Costruzione del dataset distillato
- Pipeline di pseudo-labeling con Teacher DINOv3 e Linear-CAM: filtraggio confidenza $\ge 0.50$ ed esclusione contorni degeneri.
- Statistiche: $55.964$ immagini etichettate ($98.18\%$), $1.011$ scartate per bassa confidenza ($1.77\%$), $23$ scartate per assenza contorno ($0.04\%$).
- Normalizzazione e formattazione secondo lo standard YOLO.

### 6.2 Addestramento del modello studente (YOLOv8n)
- Architettura compatta YOLOv8n ($3.2\text{M}$ parametri) per efficienza real-time e regolarizzazione intrinseca sul rumore.
- Split train/val ($80/20$) rigorosamente allineato a DINOv3.
- Iperparametri: 100 epoche, batch 16, SGD (momentum 0.937, weight decay 0.0005, lr 0.01), Mosaic e data augmentation.

### 6.3 Convergenza e validazione preliminare
- Validazione interna su pseudo-label: $96.64\%$ mAP@50, $86.66\%$ mAP@50:95.
- Valutazione su test set indipendente (Ground Truth reale): $37.39\%$ mAP@50, $25.84\%$ mAP@50:95.

---

## 7. Baseline supervisionate e protocollo di sample efficiency
> *Documento dedicato completo: vedi [`07_baseline_sample_efficiency.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/07_baseline_sample_efficiency.md)*

### 7.1 Setup delle frazioni e repliche stocastiche
- Frazioni di Ground Truth supervisionato: $\mathcal{F} = \{1\%, 2\%, 5\%, 10\%, 20\%, 40\%, 100\%\}$.
- Campionamento stratificato per classe calcolato sul numero di fotogrammi (costo reale di annotazione umana).
- Repliche stocastiche su 3 seed distinti ($\{42, 1337, 2026\}$) con disegno sperimentale a blocchi ($\mu \pm \sigma$). Split interno del $15\%$ per validazione ed early stopping.

### 7.2 Metriche e valutazione su Test Set indipendente
- Valutazione esclusiva sulle 7 sessioni continue del test set ($5.462$ immagini).
- Metriche standard COCO: mAP@50, mAP@50:95, e disaggregazione sui 4 tier dimensionali (Tiny, Small, Macro, Full-Frame).

### 7.3 Equità del confronto (Fairness) e asimmetria della supervisione
- Elementi identici: architettura YOLOv8n ($3.2\text{M}$ parametri), pesi preaddestrati COCO, risoluzione $640 \times 640$, batch size, data augmentation e test set.
- Asimmetria ontologica: bounding box rettangolari umane (alto costo di etichettatura per frame) vs supervisione debole a livello d'immagine a costo geometrico zero via DINOv3 + Linear-CAM.

---

## 8. Risultati e analisi
> *Documento dedicato completo: vedi [`08_risultati_e_analisi.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/08_risultati_e_analisi.md)*

### 8.1 Risultati globali: pareggio campionario e ritenzione relativa
- [TAB] Confronto tra baseline ($1\%, 2\%, 5\%, 10\%, 20\%, 40\%, 100\%$) e YOLOv8n distillato ($37.39\%$ mAP@50, $25.84\%$ mAP@50:95).
- **Punto di pareggio (Breakeven Point)**: equivalenza a circa l'**$1.43\%$** di annotazioni manuali con bounding box.
- **Ritenzione relativa (Capacity Retention)**: la distillata trattiene il **$61.79\%$** dell'efficacia massima dell'oracolo supervisionato al $100\%$ ($60.51\%$), a costo geometrico nullo.

### 8.2 Analisi disaggregata per scala (Tier dimensionale)
- [TAB] Prestazioni disaggregate su Full-Frame, Macro, Small e Tiny.
- *Full-Frame*: pareggio a $\approx 6.0\%$ di dati e ritenzione dell'**$84.14\%$** ($45.67\%$ vs $54.28\%$). Riconduzione a un problema di classificazione d'immagine.
- *Macro*: pareggio a $\approx 1.4\%$ di dati e ritenzione del **$63.60\%$** ($37.71\%$ vs $59.29\%$). Risultato di maggior valore applicativo: risparmio del tracciamento di coordinate per 26 classi complesse conservando i due terzi della capacità del modello supervisionato.
- *Small e Tiny*: crollo delle prestazioni ($9.01\%$ e $0.00\%$) rispetto a una supervisione che satura rapidamente oltre l'$85\%$.

### 8.3 Analisi per classe (Range Plot)
- [FIG] Range plot per classe (`by_tier/fullframe.png`, `by_tier/macro.png`, `by_tier/small.png`, `by_tier/tiny.png` e `per_class_range_grid.png`).
- Casi di eccellenza in Full-Frame (Balcone Chiesa a $94.5\%$, Coro di Notte a $88.4\%$) e in Macro (Apertura Pavimento a $86.3\%$, Scala a $70.4\%$, Arco con Firma a $56.4\%$).

### 8.4 Analisi qualitativa del Tier Tiny: bias contestuale e disallineamento discriminativo
- Evidenza visiva da `visual_sample_dino_vs_gt`: DINOv3 risolve e segmenta l'oggetto (busto di S. Agata e Cupola hanno contorni nitidi), ma non lo elegge a picco globale di calore.
- Il modello impara a discriminare la classe tramite il contesto architettonico circostante co-occorrente (archi barocchi, cornici, volte). Il picco cade all'esterno del GT e l'algoritmo WSOL genera un box errato ($\text{IoU} < 0.10$), spiegando il contrasto tra l'alta accuratezza di classificazione ($94.38\%$) e lo $0.00\%$ di localizzazione.

### 8.5 Baseline "Box a Tutta Immagine" e disaccoppiamento dei compiti
- Studio dell'IoU geometrico intrinseco: nel Full-Frame il box fisso garantisce analiticamente $\text{IoU} \approx 0.93 \gg 0.50$, mentre nel Tier Macro la variabilità dei contorni convalida una reale capacità di stima spaziale della rete.

### 8.6 Qualità delle pseudo-label e trasferimento allo studente
- Fedeltà di distillazione intra-dominio ($96.64\%$ val mAP@50 dello studente sulle pseudo-label del teacher) vs generalizzazione sul test set ($37.39\%$). Gap ascrivibile alla transizione tra maschere continue e box rigidi e al domain shift dinamico del test video.

---

## 9. Limiti della metodologia
> *Documento dedicato completo: vedi [`09_limiti.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/09_limiti.md)*

### 9.1 Assunzione di singola istanza per immagine
- L'estrazione attorno al picco globale seleziona un solo contorno per frame, sopprimendo istanze multiple coesistenti nella scena o fondendole in un unico macro-box fittizio.

### 9.2 Bias contestuale e disallineamento discriminativo
- La sola Cross-Entropy loss a livello d'immagine privilegia elementi architettonici co-occorrenti molto estesi (archi, volte, cornici), attirando il picco della CAM all'esterno del reperto sui target a piccola scala.

### 9.3 Rigidità della binarizzazione con soglia costante
- Soglia relativa fissa ($\tau = 0.45$) che non si adatta a geometrie allungate, concave o asimmetriche, generando un compromesso rigido tra sotto-segmentazione e sovra-segmentazione.

### 9.4 Approssimazione rettangolare di maschere irregolari
- Proiezione di contorni continui su bounding box asse-allineati $[x, y, w, h]$: per strutture diagonali o curve l'inclusione di pixel di sfondo riduce l'aderenza geometrica alle soglie IoU elevate (mAP@50:95).

### 9.5 Domain shift dinamico tra training statico e test continuo
- Discrepanza cinematica tra le immagini stabili usate dal teacher per la distillazione e le 7 sessioni continue video ad altezza uomo (movimenti rapidi, motion blur, cambi continui di prospettiva).

---

## 10. Conclusioni e sviluppi futuri
> *Documento dedicato completo: vedi [`10_conclusioni_e_sviluppi_futuri.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/10_conclusioni_e_sviluppi_futuri.md)*

### 10.1 Risposta alla domanda di ricerca
- **Valore campionario**: la pipeline senza bounding box equivale a circa l'**$1.43\%$** di annotazioni manuali con coordinate cartesiane, superando la baseline supervisionata all'$1\%$.
- **Ritenzione delle capacità ottimali**: il modello recupera il **$61.79\%$** dell'oracolo supervisionato al $100\%$ a livello globale, il **$63.60\%$ nel fondamentale Tier Macro** (oggetti intermedi non banali) e l'**$84.14\%$ nel Full-Frame**, azzerando il costo di annotazione su gran parte del patrimonio monumentale.
- **Lezione appresa sui target Tiny**: il fallimento a scala ridotta non deriva da cecità visiva, ma dall'insorgere del bias contestuale insito nella supervisione puramente semantica a livello d'immagine.

### 10.2 Sviluppi futuri
- **Visual Grounding su attention di DINOv3**: moduli cross-attention basati su prompt semantici per abilitare la rilevazione simultanea di oggetti multipli nella stessa inquadratura.
- **Decomposizione gerarchica (Macro-Area $\to$ POI) e Multi-Labeling**: disaccoppiare il riconoscimento dell'ambiente dall'identificazione del reperto e adottare attivazioni Sigmoid indipendenti per consentire alla Linear-CAM attivazioni multiple.
- **Regolarizzazione anti-bias contestuale**: introduzione di Patch-level Contrastive Loss e uso di risoluzioni maggiori o patch stride ridotto.
- **Metriche disaccoppiate**: adozione di CorLoc e Class-Agnostic mAP per isolare l'errore geometrico puro dalla classificazione tassonomica.

---

## Bibliografia
> *Documento dedicato completo: vedi [`bibliografia.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/bibliografia.md)*
- Riferimenti scientifici formali: UNICT-VEDI POI (Ragusa et al., VISAPP 2019), Vision Transformers (Dosovitskiy et al., 2021; Vaswani et al., 2017), DINO/DINOv2/DINOv3 (Caron et al., 2021; Oquab et al., 2023), CAM (Zhou et al., 2016), Grad-CAM (Selvaraju et al., 2017), Eigen-CAM (Muhammad & Yeasin, 2020), Ultralytics YOLOv8 (Jocher et al., 2023), WSOL benchmark e survey (Choe et al., 2020; Zhang et al., 2018; Bilgi & Cinbis, 2021).

---

## Appendici

### A. Iperparametri completi e protocollo sperimentale
> *Documento dedicato completo: vedi [`appendice_a_iperparametri.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/appendice_a_iperparametri.md)*
- Tabelle esaustive degli iperparametri per: training classificatore DINOv3 (Linear Probing), pipeline di estrazione CAM e binarizzazione, addestramento YOLOv8n distillato, data augmentation e protocollo delle baseline di sample efficiency.

### B. Tabella completa per classe
> *Documento dedicato completo: vedi [`appendice_b_classi.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/appendice_b_classi.md)*
- Tabella esaustiva delle 57 classi con nome, tier dimensionale, area media relativa ($\bar{A}_c$) e copertura percentuale.
- Risultati per classe su frazioni di training e distillazione (`fractions_per_class_summary.csv`, `distilled_per_class.csv`).

### C. Ambiente di esecuzione, note tecniche e riproducibilità
> *Documento dedicato completo: vedi [`appendice_c_note_tecniche.md`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/appendice_c_note_tecniche.md)*
- Stack software e librerie scientifiche (Python 3.12, PyTorch 2.1+, HuggingFace Transformers, Ultralytics, OpenCV, Torchmetrics).
- Specifiche hardware GPU, accelerazione CUDA/cuDNN e precisione mista (AMP).
- Garanzie di determinismo e riproducibilità sperimentale, conformità licenze.