"""
Generatore dell'intera struttura LaTeX per la relazione accademica di progetto.
Genera main.tex, tutte le sezioni, le appendici e verifica la compilazione.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELAZIONE_DIR = REPO_ROOT / "relazione"
SECTIONS_DIR = RELAZIONE_DIR / "sections"
APPENDICES_DIR = RELAZIONE_DIR / "appendices"

SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
APPENDICES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MAIN.TEX
# ==============================================================================
MAIN_TEX = r"""\documentclass[11pt,a4paper,titlepage,openany]{report}

% --- Pacchetti Fondamentali di Sistema e Lingua ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[italian]{babel}
\usepackage{lmodern}
\usepackage[a4paper,top=2.8cm,bottom=2.8cm,left=3.0cm,right=2.8cm]{geometry}

% --- Matematica e Simboli ---
\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}

% --- Grafica e Tabelle ---
\usepackage{graphicx}
\graphicspath{{figure/}{../figure/}}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{longtable}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

% --- Tipografia e Stile ---
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{cite}
\usepackage{url}
\usepackage{hyperref}

% --- Palette Cromatica Accademica ---
\definecolor{primaryblue}{RGB}{27, 79, 114}     % #1b4f72
\definecolor{accentblue}{RGB}{41, 128, 185}     % #2980b9
\definecolor{darkcharcoal}{RGB}{44, 62, 80}     % #2c3e50
\definecolor{darkgray}{RGB}{60, 60, 60}
\definecolor{lightgray}{RGB}{245, 247, 248}

% --- Configurazione Collegamenti Ipertestuali ---
\hypersetup{
    colorlinks=true,
    linkcolor=primaryblue,
    citecolor=accentblue,
    urlcolor=accentblue,
    pdftitle={Rilevamento di POI senza Bounding Box Annotate: Pseudo-Label da DINOv3 (WSOL) e Distillazione in YOLOv8},
    pdfauthor={Relazione di Progetto},
    pdfsubject={Deep Learning per la Computer Vision - UNICT-VEDI POI}
}

% --- Formattazione Didascalie ---
\captionsetup{
    font=small,
    labelfont={bf,color=primaryblue},
    format=hang,
    margin=10pt
}

% --- Intestazioni e Piè di Pagina ---
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{darkcharcoal}\nouppercase{\leftmark}}
\fancyhead[R]{\small\color{darkcharcoal}\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}

% Stile pulito per inizio capitoli
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[C]{\small\color{darkcharcoal}\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

% Formattazione elegante dei titoli di capitolo (standard LaTeX)
\makeatletter
\def\@makechapterhead#1{%
  \vspace*{20\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \large\bfseries\color{accentblue} CAPITOLO \thechapter
        \par\nobreak
        \vskip 6\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries\color{primaryblue} #1\par\nobreak
    \vskip 8\p@
    \color{accentblue}\hrule height 1.2pt
    \vskip 20\p@
  }}
\def\@makeschapterhead#1{%
  \vspace*{20\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries\color{primaryblue} #1\par\nobreak
    \vskip 8\p@
    \color{accentblue}\hrule height 1.2pt
    \vskip 20\p@
  }}
\makeatother

\begin{document}

% ==============================================================================
% FRONTESPIZIO ACCADEMICO
% ==============================================================================
\begin{titlepage}
    \centering
    \vspace*{0.5cm}
    
    {\Large\textbf{\textsc{Università degli Studi di Catania}}}\\[0.3cm]
    {\large\textbf{\textsc{Dipartimento di Matematica e Informatica}}}\\[0.2cm]
    {\normalsize Corso di Laurea Magistrale in Informatica / Data Science}\\[1.0cm]
    
    \includegraphics[width=0.22\textwidth]{figure/pipeline_wsol.png}\\[1.2cm] % Sostituire con logo ateneo se disponibile
    
    {\huge\bfseries\color{primaryblue} Rilevamento di POI senza Bounding Box Annotate}\\[0.4cm]
    {\LARGE\bfseries\color{accentblue} Pseudo-Label da DINOv3 (WSOL) e Distillazione in YOLOv8}\\[0.6cm]
    {\large Studio di Sample Efficiency e Capacity Retention sul Dataset Monumentale UNICT-VEDI}\\[1.8cm]
    
    \begin{minipage}{0.45\textwidth}
        \flushleft
        \textbf{\color{darkcharcoal}Candidato / Studente:}\\[0.15cm]
        Autore del Progetto
    \end{minipage}
    \hfill
    \begin{minipage}{0.45\textwidth}
        \flushright
        \textbf{\color{darkcharcoal}Corso:}\\[0.15cm]
        Deep Learning / Computer Vision\\[0.3cm]
        \textbf{\color{darkcharcoal}Docente:}\\[0.15cm]
        Prof. Incaricato
    \end{minipage}
    
    \vfill
    {\small\color{darkcharcoal} Anno Accademico 2025/2026}
\end{titlepage}

% ==============================================================================
% SOMMARIO E INDICI
% ==============================================================================
\pagenumbering{roman}

\input{sections/00_abstract.tex}

\tableofcontents
\cleardoublepage

\listoffigures
\cleardoublepage

\listoftables
\cleardoublepage

% ==============================================================================
% CAPITOLI DELLA RELAZIONE
% ==============================================================================
\pagenumbering{arabic}

\input{sections/01_introduzione.tex}
\input{sections/02_dataset_e_preprocessing.tex}
\input{sections/03_architettura_classificatore_dinov3.tex}
\input{sections/04_training_e_risultati_classificatore.tex}
\input{sections/05_generazione_pseudo_label_wsol.tex}
\input{sections/06_distillazione_yolo.tex}
\input{sections/07_baseline_sample_efficiency.tex}
\input{sections/08_risultati_e_analisi.tex}
\input{sections/09_limiti.tex}
\input{sections/10_conclusioni_e_sviluppi_futuri.tex}

% ==============================================================================
% BIBLIOGRAFIA
% ==============================================================================
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{Bibliografia}
\bibliographystyle{ieeetr}
\bibliography{bibliography}

% ==============================================================================
% APPENDICI
% ==============================================================================
\appendix
\input{appendices/appendice_a_iperparametri.tex}
\input{appendices/appendice_b_classi.tex}
\input{appendices/appendice_c_note_tecniche.tex}

\end{document}
"""

with open(RELAZIONE_DIR / "main.tex", "w", encoding="utf-8") as f:
    f.write(MAIN_TEX)
print("[OK] Generato: main.tex")
