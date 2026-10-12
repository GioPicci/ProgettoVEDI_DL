# AGENTS.md — Cartella `relazione/`

Relazione accademica di progetto scritta in LaTeX per il corso di Deep Learning / Computer Vision.  
Il documento completo (`main.pdf`) conta 43 pagine, con il corpo principale (Capitoli 1--10) calibrato esattamente sul budget di **25 pagine** (Appendici escluse).

---

## Compilazione del PDF

- **Script automatico (Windows)**: eseguire [`compile.bat`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/compile.bat).  
  Esegue la sequenza completa: `pdflatex` $\to$ `bibtex` $\to$ `pdflatex` $\to$ `pdflatex` con MiKTeX.
- **Da terminale / PowerShell**:
  ```powershell
  cd relazione
  pdflatex -interaction=nonstopmode --disable-installer main.tex
  bibtex main
  pdflatex -interaction=nonstopmode --disable-installer main.tex
  pdflatex -interaction=nonstopmode --disable-installer main.tex
  ```
- **Output generato**: [`main.pdf`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/main.pdf).

---

## Struttura della Cartella

### File Principali
- [`main.tex`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/main.tex): Master file LaTeX con configurazione del layout (`report`, A4, margini 2.8/3.0 cm, font `lmodern`, palette cromatica accademica, comandi personalizzati).
- [`bibliography.bib`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/bibliography.bib): Riferimenti bibliografici (DINOv2/v3, ViT, YOLO, WSOL, CAM, UNICT-VEDI).
- [`compile.bat`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/compile.bat): Script batch per compilazione pulita in ambiente Windows.

### Sottocartelle
- **[`sections/`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/sections/)**: Capitoli del corpo principale:
  - `00_abstract.tex`: Sommario e abstract.
  - `01_introduzione.tex`: Introduzione e definizione del problema.
  - `02_dataset_e_preprocessing.tex`: Dataset UNICT-VEDI POI, sanitizzazione e tier dimensionali.
  - `03_architettura_classificatore_dinov3.tex`: ViT-S/16, linear probing e CAM.
  - `04_training_e_risultati_classificatore.tex`: Risultati di classificazione e analisi errori per area.
  - `05_generazione_pseudo_label_wsol.tex`: Pipeline WSOL e benchmark 4 metodi CAM.
  - `06_distillazione_yolo.tex`: Pseudo-labeling, studente YOLOv8n e gap di generalizzazione.
  - `07_baseline_sample_efficiency.tex`: Protocollo sperimentale di Sample Efficiency (7 frazioni, 3 seed).
  - `08_risultati_e_analisi.tex`: Breakeven (1.43%), capacity retention (61.79%), tier macro e bias tiny.
  - `09_limiti.tex`: I 5 limiti architetturali e metodologici.
  - `10_conclusioni_e_sviluppi_futuri.tex`: Conclusioni e direttrici di ricerca.
- **[`appendices/`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/appendices/)**: Materiale tecnico di consultazione (intoccabile nelle sintesi ordinarie):
  - `appendice_a_iperparametri.tex`: Iperparametri completi e tabelle di training.
  - `appendice_b_classi.tex`: Tassonomia completa delle 57 classi con aree medie e tier.
  - `appendice_c_note_tecniche.tex`: Hardware, GPU RTX 2060 (6 GB), versioni librerie e riproducibilità.
- **[`figure/`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/figure/)**: Figure PNG/JPG incluse nella relazione (schemi architetturali, grafici di ritenzione, confronti qualitativi).
- **[`markdown/`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/markdown/)**: Copia di backup dei testi originali in Markdown prima della conversione in LaTeX.
- **[`skills/`](file:///c:/Users/Workstation/PycharmProjects/ProgettoVEDI_DL/relazione/skills/)**: Skill `sintesi-relazione.md` contenente le regole di compressione, stile e budget di pagina.

---

## Convenzioni per gli Agenti

1. **Tipografia Tabelle**: per colonne descrittive a larghezza automatica con allineamento a sinistra usare sempre la colonna `L` (definita in `main.tex` come `>{\raggedright\arraybackslash}X`).
2. **Dimensioni Figure**: per evitare salti di pagina indesiderati o galleggiamento fuori capitolo, mantenere le figure tra `0.52\textwidth` e `0.72\textwidth` con specifier `[!htbp]`.
3. **Verifica Compilazione**: dopo ogni modifica ai file `.tex`, compilare con `pdflatex` e verificare l'assenza di `Overfull \hbox` e warning su label mancanti nel file `main.log`.
