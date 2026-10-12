"""
Script di generazione delle figure per la relazione accademica del progetto VEDI_DL.
Genera diagrammi architetturali, diagrammi di flusso e composizioni visive
ad alta risoluzione (300 DPI) per sostituire i diagrammi ASCII.
"""

import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

# Configurazione directory di output
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "relazione" / "figure"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Impostazioni grafiche globali per Matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


# ==============================================================================
# 1. SCHEMA PRE-TRAINING DINO (Capitolo 3)
# ==============================================================================
def generate_dino_pretraining():
    fig, ax = plt.subplots(figsize=(15.8, 4.6), dpi=300)
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    # Titolo Generale
    ax.text(7.9, 4.35, "Paradigma di Pre-Addestramento Self-Supervised DINOv3 (Meta AI, 2025)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1a252f')

    # =========================================================================
    # COLONNA 1: INPUT DATA (x: 0.5 -> 2.7, center: 1.6)
    # =========================================================================
    box_input = patches.FancyBboxPatch((0.5, 1.45), 2.2, 1.5, boxstyle="round,pad=0.08", 
                                       facecolor='#f4f6f7', edgecolor='#2c3e50', linewidth=1.6)
    ax.add_patch(box_input)
    ax.text(1.6, 2.45, r"Immagine $x$", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(1.6, 2.08, "Corpus LVD-1689M\n(1.69B immagini)", ha='center', va='center', fontsize=8.5, color='#34495e')
    ax.text(1.6, 1.68, "Multi-Crop & Masking", ha='center', va='center', fontsize=8, color='#7f8c8d', style='italic')

    # Frecce da Input a Teacher e Student
    # Verso Teacher (in alto)
    ax.annotate("", xy=(4.0, 3.40), xytext=(2.7, 2.70),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#2980b9'))
    ax.text(3.35, 3.45, "Global Crops\n" + r"($224 \times 224$)", ha='center', va='bottom', 
            fontsize=8.5, color='#2980b9', fontweight='bold')

    # Verso Student (in basso)
    ax.annotate("", xy=(4.0, 0.95), xytext=(2.7, 1.65),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#d35400'))
    ax.text(3.25, 0.65, "Local Crops & Masking\n" + r"($96 \times 96$ / MIM)", ha='center', va='top', 
            fontsize=8.5, color='#d35400', fontweight='bold')

    # =========================================================================
    # COLONNA 2: RETI TEACHER E STUDENT (x: 4.0 -> 6.8, center: 5.4)
    # =========================================================================
    # Teacher Box (Top: y=2.60 -> 4.00, height=1.4)
    box_teacher = patches.FancyBboxPatch((4.0, 2.60), 2.8, 1.4, boxstyle="round,pad=0.08", 
                                         facecolor='#ebf5fb', edgecolor='#2980b9', linewidth=1.8)
    ax.add_patch(box_teacher)
    ax.text(5.4, 3.68, r"Teacher $\theta_t$", ha='center', va='center', fontsize=11, fontweight='bold', color='#1b4f72')
    ax.text(5.4, 3.38, "ViT Backbone (Frozen)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2874a6')
    ax.text(5.4, 3.05, "Pesi aggiornati via EMA:", ha='center', va='center', fontsize=8, color='#2874a6')
    ax.text(5.4, 2.78, r"$\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$", 
            ha='center', va='center', fontsize=8.5, color='#1b4f72')

    # Student Box (Bottom: y=0.40 -> 1.80, height=1.4)
    box_student = patches.FancyBboxPatch((4.0, 0.40), 2.8, 1.4, boxstyle="round,pad=0.08", 
                                         facecolor='#fef5e7', edgecolor='#d35400', linewidth=1.8)
    ax.add_patch(box_student)
    ax.text(5.4, 1.48, r"Student $\theta_s$", ha='center', va='center', fontsize=11, fontweight='bold', color='#7e3200')
    ax.text(5.4, 1.18, "ViT Backbone (Trainable)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#a04000')
    ax.text(5.4, 0.85, "Ottimizzato via Gradiente:", ha='center', va='center', fontsize=8, color='#a04000')
    ax.text(5.4, 0.58, r"$\nabla_{\theta_s} \mathcal{L}_{\mathrm{tot}}$", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#7e3200')

    # Freccia EMA da Student verso Teacher
    ax.annotate("", xy=(5.4, 2.60), xytext=(5.4, 1.80),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#2980b9', linestyle='--'))
    ax.text(5.55, 2.20, r"EMA ($\lambda \approx 0.999$)", ha='left', va='center', 
            fontsize=8, color='#2980b9', fontweight='bold')

    # =========================================================================
    # COLONNA 3: GLI OBIETTIVI DINOv3 (x: 7.9 -> 11.9, center: 9.9)
    # =========================================================================
    # 1. Ramo DINO Loss (Top: y=2.85 -> 3.95, height=1.1)
    box_loss_dino = patches.FancyBboxPatch((7.9, 2.85), 4.0, 1.1, boxstyle="round,pad=0.08",
                                           facecolor='#eafaf1', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(box_loss_dino)
    ax.text(9.9, 3.65, r"$\mathcal{L}_{\mathrm{DINO}}$ — Invarianza Globale ([CLS])", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#196f3d')
    ax.text(9.9, 3.30, "Distillazione cross-entropy con Teacher Centering & Sharpening", 
            ha='center', va='center', fontsize=7.8, color='#1e8449')
    ax.text(9.9, 3.03, r"$P_t = \mathrm{Softmax}((g_t-c)/\tau_t) \quad \longleftrightarrow \quad P_s = \mathrm{Softmax}(g_s/\tau_s)$", 
            ha='center', va='center', fontsize=8, color='#145a32')

    # 2. Ramo iBOT Loss (Center: y=1.65 -> 2.65, height=1.0)
    box_loss_ibot = patches.FancyBboxPatch((7.9, 1.65), 4.0, 1.0, boxstyle="round,pad=0.08",
                                           facecolor='#fcf3cf', edgecolor='#f39c12', linewidth=1.5)
    ax.add_patch(box_loss_ibot)
    ax.text(9.9, 2.38, r"$\mathcal{L}_{\mathrm{iBOT}}$ — Masked Image Modeling (Patch)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#9a7d0a')
    ax.text(9.9, 2.05, "Predizione dei token delle patch mascherate dello Student", 
            ha='center', va='center', fontsize=7.8, color='#7d6608')
    ax.text(9.9, 1.80, r"Allineamento denso: $\sum_{j \in \mathrm{mask}} H(P_{t, j}^{\mathrm{patch}}, P_{s, j}^{\mathrm{patch}})$", 
            ha='center', va='center', fontsize=8, color='#7d6608')

    # 3. Ramo Gram Anchoring (Bottom: y=0.45 -> 1.45, height=1.0)
    box_loss_gram = patches.FancyBboxPatch((7.9, 0.45), 4.0, 1.0, boxstyle="round,pad=0.08",
                                           facecolor='#f4ecf7', edgecolor='#8e44ad', linewidth=1.5)
    ax.add_patch(box_loss_gram)
    ax.text(9.9, 1.18, r"$\mathcal{L}_{\mathrm{Gram}}$ — Gram Anchoring (Novità DINOv3)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#5b2c6f')
    ax.text(9.9, 0.85, "Preserva le correlazioni spaziali a coppie tra patch", 
            ha='center', va='center', fontsize=7.8, color='#6c3483')
    ax.text(9.9, 0.60, r"$\|\mathbf{G}_s - \mathbf{G}_t\|_F^2 \quad (\mathbf{G} = \mathbf{Z}_{\mathrm{patch}} \mathbf{Z}_{\mathrm{patch}}^\top) \Rightarrow$ Previene feature collapse", 
            ha='center', va='center', fontsize=7.8, color='#512e5f')

    # Frecce pulite da Teacher verso i rami
    ax.annotate("", xy=(7.9, 3.50), xytext=(6.8, 3.50),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#2980b9'))
    ax.annotate("", xy=(7.9, 2.30), xytext=(6.8, 3.10),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#2980b9'))
    ax.annotate("", xy=(7.9, 1.05), xytext=(6.8, 2.80),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#2980b9', linestyle='--'))

    # Frecce pulite da Student verso i rami
    ax.annotate("", xy=(7.9, 0.75), xytext=(6.8, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#d35400'))
    ax.annotate("", xy=(7.9, 1.90), xytext=(6.8, 1.25),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#d35400'))
    ax.annotate("", xy=(7.9, 3.15), xytext=(6.8, 1.55),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#d35400', linestyle='--'))

    # =========================================================================
    # COLONNA 4: LOSS TOTALE & REGOLARIZZAZIONE (x: 12.8 -> 15.5, center: 14.15)
    # =========================================================================
    box_loss_tot = patches.FancyBboxPatch((12.8, 1.2), 2.7, 2.0, boxstyle="round,pad=0.08",
                                          facecolor='#e8f8f5', edgecolor='#16a085', linewidth=1.8)
    ax.add_patch(box_loss_tot)
    ax.text(14.15, 2.85, "Loss Totale DINOv3", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0e6655')
    ax.text(14.15, 2.45, r"$\mathcal{L}_{\mathrm{tot}} = \mathcal{L}_{\mathrm{DINO}}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#117a65')
    ax.text(14.15, 2.18, r"$+ \; \mathcal{L}_{\mathrm{iBOT}}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#117a65')
    ax.text(14.15, 1.91, r"$+ \; \lambda_g \mathcal{L}_{\mathrm{Gram}} + \mathcal{L}_{\mathrm{KoLeo}}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#117a65')
    ax.text(14.15, 1.50, "KoLeo: regolarizzatore\ndi dispersione token", ha='center', va='center', fontsize=7.8, color='#148f77', style='italic')

    # Frecce dai 3 rami verso Loss Totale
    ax.annotate("", xy=(12.8, 2.5), xytext=(11.9, 3.4),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#27ae60'))
    ax.annotate("", xy=(12.8, 2.2), xytext=(11.9, 2.15),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#f39c12'))
    ax.annotate("", xy=(12.8, 1.9), xytext=(11.9, 0.95),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#8e44ad'))

    out_path = FIGURES_DIR / "schema_dino_pretraining.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 2. ARCHITETTURA DEL CLASSIFICATORE E PROBING (Capitolo 3)
# ==============================================================================
def generate_classifier_architecture():
    fig, ax = plt.subplots(figsize=(12.8, 3.8), dpi=300)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 3.9)
    ax.axis('off')

    # Titolo minimale ed elegante
    ax.text(6.4, 3.65, "Architettura del Classificatore DINOv3 ed Estrazione Linear-CAM", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1a252f')

    # =========================================================================
    # 1. IMMAGINE INPUT (x: 0.4 -> 2.0, y: 1.15 -> 2.65, center: 1.2, 1.9)
    # =========================================================================
    box_in = patches.FancyBboxPatch((0.4, 1.15), 1.6, 1.5, boxstyle="round,pad=0.06", 
                                   facecolor='#f8f9fa', edgecolor='#34495e', linewidth=1.4)
    ax.add_patch(box_in)
    ax.text(1.2, 2.15, "Immagine Input", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(1.2, 1.78, r"$384 \times 384$ px", ha='center', va='center', fontsize=8.5, color='#495057')
    ax.text(1.2, 1.45, r"$24 \times 24$ patch", ha='center', va='center', fontsize=8, color='#6c757d', style='italic')

    # Freccia verso Backbone
    ax.annotate("", xy=(2.6, 1.90), xytext=(2.0, 1.90),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    # =========================================================================
    # 2. BACKBONE DINOv3 (x: 2.6 -> 5.0, y: 0.55 -> 3.25, center: 3.8, 1.9)
    # =========================================================================
    box_bb = patches.FancyBboxPatch((2.6, 0.55), 2.4, 2.7, boxstyle="round,pad=0.08", 
                                    facecolor='#eaf2f8', edgecolor='#2980b9', linewidth=1.6)
    ax.add_patch(box_bb)
    ax.text(3.8, 2.85, "DINOv3 ViT-S/16", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72')
    ax.text(3.8, 2.50, "(Backbone Congelato)", ha='center', va='center', fontsize=8.5, color='#2874a6', style='italic')
    ax.text(3.8, 1.90, "12 Strati Transformer\nDimensione $D = 384$\n6 Teste di Attenzione", 
            ha='center', va='center', fontsize=8, color='#34495e')
    ax.text(3.8, 1.05, r"Griglia: $24 \times 24$ token" + "\n(576 patch spaziali)", 
            ha='center', va='center', fontsize=7.8, color='#1b4f72', style='italic')

    # =========================================================================
    # RAMO SUPERIORE: CLASSIFICAZIONE (y = 2.70 costante)
    # =========================================================================
    # Freccia da Backbone a Dropout
    ax.annotate("", xy=(6.2, 2.70), xytext=(5.0, 2.70),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#6c3483'))
    ax.text(5.6, 2.92, "Token [CLS]", ha='center', va='bottom', fontsize=7.8, color='#6c3483', fontweight='bold')
    ax.text(5.6, 2.78, r"$\mathbf{z}_{\mathrm{CLS}} \in \mathbb{R}^{384}$", ha='center', va='bottom', fontsize=7.8, color='#6c3483')

    # Box Dropout
    box_drop = patches.FancyBboxPatch((6.2, 2.25), 1.4, 0.9, boxstyle="round,pad=0.06", 
                                      facecolor='#f4ecf7', edgecolor='#8e44ad', linewidth=1.3)
    ax.add_patch(box_drop)
    ax.text(6.9, 2.78, "Dropout", ha='center', va='center', fontsize=9, fontweight='bold', color='#512e5f')
    ax.text(6.9, 2.48, r"($p = 0.3$)", ha='center', va='center', fontsize=8, color='#7d3c98')

    # Freccia da Dropout a Linear Head
    ax.annotate("", xy=(8.1, 2.70), xytext=(7.6, 2.70),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#6c3483'))

    # Box Linear Head
    box_head = patches.FancyBboxPatch((8.1, 2.15), 2.3, 1.1, boxstyle="round,pad=0.06", 
                                      facecolor='#e8f8f5', edgecolor='#16a085', linewidth=1.5)
    ax.add_patch(box_head)
    ax.text(9.25, 2.90, "Linear Head (Probing)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0e6655')
    ax.text(9.25, 2.60, r"$\mathbf{W} \in \mathbb{R}^{57 \times 384}, \; \mathbf{b} \in \mathbb{R}^{57}$", ha='center', va='center', fontsize=8, color='#117a65')
    ax.text(9.25, 2.33, r"Logits $\hat{\mathbf{y}} = \mathbf{W}\mathbf{z} + \mathbf{b}$", ha='center', va='center', fontsize=8, color='#117a65', fontweight='bold')

    # Freccia verso Output Predizione
    ax.annotate("", xy=(11.0, 2.70), xytext=(10.4, 2.70),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#16a085'))
    
    # Box Output Predizione
    box_out = patches.FancyBboxPatch((11.0, 2.25), 1.4, 0.9, boxstyle="round,pad=0.06", 
                                     facecolor='#e8f6f3', edgecolor='#1abc9c', linewidth=1.3)
    ax.add_patch(box_out)
    ax.text(11.7, 2.78, "Predizione", ha='center', va='center', fontsize=9, fontweight='bold', color='#117a65')
    ax.text(11.7, 2.48, r"Classe $\hat{c} \in [0, 56]$", ha='center', va='center', fontsize=7.8, color='#0e6655')

    # =========================================================================
    # RAMO INFERIORE: LOCALIZZAZIONE LINEAR-CAM (y = 1.10 costante)
    # =========================================================================
    # Freccia da Backbone a Linear-CAM
    ax.annotate("", xy=(6.2, 1.10), xytext=(5.0, 1.10),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#d35400'))
    ax.text(5.6, 1.32, "Token Patch", ha='center', va='bottom', fontsize=7.8, color='#d35400', fontweight='bold')
    ax.text(5.6, 1.18, r"$\mathbf{z}_{i,j} \in \mathbb{R}^{384}$", ha='center', va='bottom', fontsize=7.8, color='#d35400')

    # Box Linear-CAM
    box_cam = patches.FancyBboxPatch((6.2, 0.55), 4.2, 1.1, boxstyle="round,pad=0.06", 
                                     facecolor='#fef5e7', edgecolor='#d35400', linewidth=1.5)
    ax.add_patch(box_cam)
    ax.text(8.3, 1.35, "Estrazione Mappa di Attivazione (Linear-CAM)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#7e3200')
    ax.text(8.3, 1.05, r"$M_c(i,j) = \mathrm{ReLU}\left( \mathbf{w}_c^\top \mathbf{z}_{i,j} \right) \in \mathbb{R}^{24 \times 24}$", 
            ha='center', va='center', fontsize=9, color='#a04000', fontweight='bold')
    ax.text(8.3, 0.75, "Proiezione dei token spaziali sui pesi della classe target", ha='center', va='center', fontsize=7.8, color='#873600', style='italic')

    # Freccia verticale dai pesi W della Linear Head a Linear-CAM
    ax.annotate("", xy=(9.25, 1.65), xytext=(9.25, 2.15),
                arrowprops=dict(arrowstyle="->", lw=1.3, linestyle="--", color='#d35400'))
    ax.text(9.35, 1.90, r"Vettore pesi $\mathbf{w}_c$", ha='left', va='center', fontsize=8, color='#d35400', fontweight='bold')

    # Freccia da Linear-CAM a Heatmap
    ax.annotate("", xy=(11.0, 1.10), xytext=(10.4, 1.10),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#d35400'))

    # Box Output Heatmap
    box_cam_out = patches.FancyBboxPatch((11.0, 0.65), 1.4, 0.9, boxstyle="round,pad=0.06", 
                                         facecolor='#fdebd0', edgecolor='#e67e22', linewidth=1.3)
    ax.add_patch(box_cam_out)
    ax.text(11.7, 1.18, "Heatmap", ha='center', va='center', fontsize=9, fontweight='bold', color='#7e3200')
    ax.text(11.7, 0.88, r"$M_c \in \mathbb{R}^{24 \times 24}$", ha='center', va='center', fontsize=7.8, color='#a04000')

    out_path = FIGURES_DIR / "architettura_classificatore.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 3. PIPELINE WSOL (Capitolo 5)
# ==============================================================================
def generate_wsol_pipeline():
    fig, ax = plt.subplots(figsize=(13.2, 3.4), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 3.4)
    ax.axis('off')

    # Titolo pulito ed elegante
    ax.text(6.6, 3.12, "Pipeline WSOL: Dalla Mappa di Salienza al Bounding Box Discreto", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1a252f')

    stages = [
        {
            'x': 0.4, 'w': 2.15,
            'tag': 'STADIO 1', 'title': 'Salienza Grezza',
            'lines': [r"Mappa patch $24 \times 24$", r"Feature DINOv3", r"Linear-CAM / Attn"],
            'color': '#ebf5fb', 'edge': '#2980b9'
        },
        {
            'x': 2.95, 'w': 2.15,
            'tag': 'STADIO 2', 'title': 'Upsampling',
            'lines': ["Interp. Bicubica", r"$H \times W$ continuo", r"Norm. su $[0, 255]$"],
            'color': '#e8f8f5', 'edge': '#16a085'
        },
        {
            'x': 5.5, 'w': 2.15,
            'tag': 'STADIO 3', 'title': 'Binarizzazione',
            'lines': ["Soglia Differenziata", r"$\tau = 0.45 \cdot \max$", r"Percentile $85\mathrm{--}92\%$"],
            'color': '#fef9e7', 'edge': '#f39c12'
        },
        {
            'x': 8.05, 'w': 2.15,
            'tag': 'STADIO 4', 'title': 'Filtro Morfologico',
            'lines': ["Apertura Morfologica", r"Kernel Ellittico $7 \times 7$", "Eliminazione rumore"],
            'color': '#f8f9f9', 'edge': '#7f8c8d'
        },
        {
            'x': 10.6, 'w': 2.15,
            'tag': 'STADIO 5', 'title': 'Estrazione Box',
            'lines': [r"Contorno su Picco", r"$(x^*, y^*) = \arg\max M$", r"Riquadro $[x, y, w, h]$"],
            'color': '#fadbd8', 'edge': '#e74c3c'
        }
    ]

    for i, st in enumerate(stages):
        # Box con bordo pulito e meno padding superfluo
        box = patches.FancyBboxPatch((st['x'], 0.35), st['w'], 2.35, boxstyle="round,pad=0.06", 
                                     facecolor=st['color'], edgecolor=st['edge'], linewidth=1.6)
        ax.add_patch(box)
        cx = st['x'] + st['w'] / 2.0
        
        # Tag e Titolo
        ax.text(cx, 2.38, st['tag'], ha='center', va='center', fontsize=8.5, fontweight='bold', color=st['edge'])
        ax.text(cx, 2.08, st['title'], ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1a252f')
        
        # Linea divisoria sottile
        ax.plot([st['x'] + 0.15, st['x'] + st['w'] - 0.15], [1.82, 1.82], color=st['edge'], lw=1.0, alpha=0.7)
        
        # Testo descrittivo compatto e ben leggibile
        ax.text(cx, 1.48, st['lines'][0], ha='center', va='center', fontsize=9.2, color='#2c3e50', fontweight='semibold')
        ax.text(cx, 1.15, st['lines'][1], ha='center', va='center', fontsize=9.0, color='#34495e')
        ax.text(cx, 0.78, st['lines'][2], ha='center', va='center', fontsize=8.8, color='#566573')

        # Freccia di connessione allo stadio successivo
        if i < len(stages) - 1:
            next_x = stages[i+1]['x']
            ax.annotate("", xy=(next_x, 1.52), xytext=(st['x'] + st['w'], 1.52),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color='#2c3e50'))

    out_path = FIGURES_DIR / "pipeline_wsol.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")



# ==============================================================================
# 4. PIPELINE DISTILLAZIONE (Capitolo 6)
# ==============================================================================
def generate_distillation_pipeline():
    fig, ax = plt.subplots(figsize=(15.2, 4.3), dpi=300)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 4.3)
    ax.axis('off')

    # Titolo principale
    ax.text(7.6, 4.02, "Pipeline di Distillazione del Dataset: da Teacher DINOv3 a Student YOLOv8n", 
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1a252f')

    # y range dei 6 blocchi principali
    y_top = 1.40
    h_box = 2.25

    stages = [
        {
            'x': 0.35, 'w': 2.05,
            'tag': 'INPUT', 'title': 'Dataset Training',
            'lines': ["57.000 Fotogrammi", "Etichetta di classe", "0 annotazioni manuali"],
            'color': '#f8f9fa', 'edge': '#495057'
        },
        {
            'x': 2.75, 'w': 2.05,
            'tag': 'TEACHER', 'title': 'Teacher DINOv3',
            'lines': ["ViT-S/16 congelato", "Linear Head (Probing)", "Mappe Linear-CAM"],
            'color': '#ebf5fb', 'edge': '#2980b9'
        },
        {
            'x': 5.15, 'w': 2.10,
            'tag': 'FILTRO 1', 'title': 'Filtro Confidenza',
            'lines': [r"$p(y \mid \mathbf{x}) \geq 0.50$", "Scarto predizioni incerte", "Validazione classe"],
            'color': '#fef9e7', 'edge': '#f39c12'
        },
        {
            'x': 7.75, 'w': 2.15,
            'tag': 'FILTRO 2 & BBOX', 'title': 'Estrazione WSOL',
            'lines': [r"BBox da picco ($\tau = 0.45$)", "Controllo contorno", r"Format YOLO $[c_x, c_y, w, h]$"],
            'color': '#e8f8f5', 'edge': '#16a085'
        },
        {
            'x': 10.40, 'w': 2.15,
            'tag': 'DATASET WSOL', 'title': 'Dataset Distillato',
            'lines': ["55.964 Frame validati", "Tasso accettazione 98.18%", "Split 80% Tr / 20% Val"],
            'color': '#eafaf1', 'edge': '#27ae60'
        },
        {
            'x': 12.90, 'w': 1.95,
            'tag': 'STUDENT', 'title': 'Student YOLOv8n',
            'lines': ["3.2M Parametri", "Supervisionato", "Detector real-time"],
            'color': '#ebf5fb', 'edge': '#2980b9'
        }
    ]

    for i, st in enumerate(stages):
        box = patches.FancyBboxPatch((st['x'], y_top), st['w'], h_box, boxstyle="round,pad=0.06", 
                                     facecolor=st['color'], edgecolor=st['edge'], linewidth=1.6)
        ax.add_patch(box)
        cx = st['x'] + st['w'] / 2.0
        
        # Tag e Titolo
        ax.text(cx, y_top + h_box - 0.26, st['tag'], ha='center', va='center', fontsize=8.2, fontweight='bold', color=st['edge'])
        ax.text(cx, y_top + h_box - 0.56, st['title'], ha='center', va='center', fontsize=9.8, fontweight='bold', color='#1a252f')
        
        # Linea divisoria sottile
        ax.plot([st['x'] + 0.12, st['x'] + st['w'] - 0.12], [y_top + h_box - 0.78, y_top + h_box - 0.78], 
                color=st['edge'], lw=0.9, alpha=0.6)
        
        # Testo descrittivo
        ax.text(cx, y_top + 1.05, st['lines'][0], ha='center', va='center', fontsize=8.6, color='#2c3e50', fontweight='bold')
        ax.text(cx, y_top + 0.68, st['lines'][1], ha='center', va='center', fontsize=8.3, color='#34495e')
        ax.text(cx, y_top + 0.32, st['lines'][2], ha='center', va='center', fontsize=8.0, color='#566573')

    # Frecce orizzontali tra blocchi
    # 1 -> 2
    ax.annotate("", xy=(stages[1]['x'], y_top + h_box/2), xytext=(stages[0]['x'] + stages[0]['w'], y_top + h_box/2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#2c3e50'))

    # 2 -> 3
    ax.annotate("", xy=(stages[2]['x'], y_top + h_box/2), xytext=(stages[1]['x'] + stages[1]['w'], y_top + h_box/2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#2c3e50'))

    # 3 -> 4 (Accettati confidenza: 98.22%)
    x3_end = stages[2]['x'] + stages[2]['w']
    x4_start = stages[3]['x']
    ax.annotate("", xy=(x4_start, y_top + h_box/2), xytext=(x3_end, y_top + h_box/2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#27ae60'))
    ax.text((x3_end + x4_start)/2, y_top + h_box/2 + 0.16, "Sì (98.2%)", 
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle="round,pad=0.18", facecolor='white', edgecolor='none', alpha=0.88))

    # 4 -> 5 (Accettati geometria: 98.18% finale)
    x4_end = stages[3]['x'] + stages[3]['w']
    x5_start = stages[4]['x']
    ax.annotate("", xy=(x5_start, y_top + h_box/2), xytext=(x4_end, y_top + h_box/2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#27ae60'))
    ax.text((x4_end + x5_start)/2, y_top + h_box/2 + 0.16, "Validi (98.2%)", 
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle="round,pad=0.18", facecolor='white', edgecolor='none', alpha=0.88))

    # 5 -> 6
    ax.annotate("", xy=(stages[5]['x'], y_top + h_box/2), xytext=(stages[4]['x'] + stages[4]['w'], y_top + h_box/2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color='#2980b9'))

    # Allineamento inferiore uniforme per i due box di scarto
    y_deck = 0.18
    h_deck = 0.82

    # 1. SCARTO 1 (Sotto a Filtro Confidenza, stage 2)
    cx_f1 = stages[2]['x'] + stages[2]['w'] / 2.0
    w_drop1 = 2.25
    x_drop1 = cx_f1 - w_drop1 / 2.0
    box_drop1 = patches.FancyBboxPatch((x_drop1, y_deck), w_drop1, h_deck, boxstyle="round,pad=0.06", 
                                       facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=1.4)
    ax.add_patch(box_drop1)
    ax.text(cx_f1, y_deck + 0.56, "Scarto Confidenza", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#922b21')
    ax.text(cx_f1, y_deck + 0.32, "1.013 frame (1.78%)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#c0392b')
    ax.text(cx_f1, y_deck + 0.14, r"$p(y \mid \mathbf{x}) < 0.50$ (incertezza)", ha='center', va='center', fontsize=7.2, color='#78281f')

    # Freccia scarto 1
    ax.annotate("", xy=(cx_f1, y_deck + h_deck), xytext=(cx_f1, y_top),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#c0392b'))
    ax.text(cx_f1 + 0.08, (y_top + y_deck + h_deck)/2, "No (1.78%)", 
            ha='left', va='center', fontsize=7.5, fontweight='bold', color='#c0392b')

    # 2. SCARTO 2 (Sotto a Estrazione WSOL, stage 3)
    cx_f2 = stages[3]['x'] + stages[3]['w'] / 2.0
    w_drop2 = 2.25
    x_drop2 = cx_f2 - w_drop2 / 2.0
    box_drop2 = patches.FancyBboxPatch((x_drop2, y_deck), w_drop2, h_deck, boxstyle="round,pad=0.06", 
                                       facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=1.4)
    ax.add_patch(box_drop2)
    ax.text(cx_f2, y_deck + 0.56, "Scarto Geometrico", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#922b21')
    ax.text(cx_f2, y_deck + 0.32, "23 frame (0.04%)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#c0392b')
    ax.text(cx_f2, y_deck + 0.14, "Contorno degenere o assente", ha='center', va='center', fontsize=7.2, color='#78281f')

    # Freccia scarto 2
    ax.annotate("", xy=(cx_f2, y_deck + h_deck), xytext=(cx_f2, y_top),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#c0392b'))
    ax.text(cx_f2 + 0.08, (y_top + y_deck + h_deck)/2, "No (0.04%)", 
            ha='left', va='center', fontsize=7.5, fontweight='bold', color='#c0392b')

    out_path = FIGURES_DIR / "pipeline_distillazione.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 5. PROTOCOLLO SAMPLE EFFICIENCY (Capitolo 7)
# ==============================================================================
def generate_sample_efficiency_protocol():
    fig, ax = plt.subplots(figsize=(13.6, 5.0), dpi=300)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    # Titolo
    ax.text(6.8, 4.75, "Disegno Sperimentale e Protocollo del Benchmark di Sample Efficiency", 
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1a252f')

    # =========================================================================
    # 1. RAMO SUPERVISIONATO (Sinistra: x = 0.4 -> 9.6)
    # =========================================================================
    box_gt = patches.FancyBboxPatch((0.4, 3.45), 9.3, 0.95, boxstyle="round,pad=0.06", 
                                    facecolor='#f8f9fa', edgecolor='#34495e', linewidth=1.5)
    ax.add_patch(box_gt)
    ax.text(5.05, 4.08, "Budget di Annotazione Manuale (Ground Truth con Box Reali)", 
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#2c3e50')
    ax.text(5.05, 3.70, "Campionamento stratificato per frame su 57 classi (15% val interna) | 3 seed (42, 1337, 2026)", 
            ha='center', va='center', fontsize=8.2, color='#566573')

    # 7 Frazioni
    fractions = [
        {"name": "1%", "w": 1.15, "sub": "3 seed"},
        {"name": "2%", "w": 1.15, "sub": "3 seed"},
        {"name": "5%", "w": 1.15, "sub": "3 seed"},
        {"name": "10%", "w": 1.15, "sub": "3 seed"},
        {"name": "20%", "w": 1.15, "sub": "3 seed"},
        {"name": "40%", "w": 1.15, "sub": "3 seed"},
        {"name": "100%", "w": 1.35, "sub": "Oracolo"}
    ]

    start_x = 0.4
    y_f = 2.15
    h_f = 0.85
    f_centers = []

    for i, frac in enumerate(fractions):
        w = frac["w"]
        cx = start_x + w / 2.0
        f_centers.append(cx)
        
        is_or = (frac["name"] == "100%")
        fc = '#fadbd8' if is_or else '#fef9e7'
        ec = '#c0392b' if is_or else '#f39c12'
        tc = '#922b21' if is_or else '#7d6608'

        box = patches.FancyBboxPatch((start_x, y_f), w, h_f, boxstyle="round,pad=0.05",
                                     facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(box)
        ax.text(cx, y_f + 0.54, frac["name"], ha='center', va='center', fontsize=10.5, fontweight='bold', color=tc)
        ax.text(cx, y_f + 0.22, frac["sub"], ha='center', va='center', fontsize=7.8, color=tc)

        # Freccia dal budget in alto
        ax.annotate("", xy=(cx, y_f + h_f), xytext=(cx, 3.45),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#7f8c8d'))
        
        start_x += w + 0.18

    # Collector Bus (senza sovrapposizioni)
    y_bus = 1.70
    x_bus_start = f_centers[0]
    x_bus_end = f_centers[-1]
    for cx in f_centers:
        ax.plot([cx, cx], [y_f, y_bus], color='#d35400', lw=1.2)
    ax.plot([x_bus_start, x_bus_end], [y_bus, y_bus], color='#d35400', lw=1.5)

    # Freccia singola verticale dal bus al Test Set
    cx_sup = 5.05
    ax.annotate("", xy=(cx_sup, 1.05), xytext=(cx_sup, y_bus),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='#d35400'))
    ax.text(cx_sup, 1.38, "19 Modelli Supervisionati (18 frazioni stocastiche + 1 oracolo)", 
            ha='center', va='center', fontsize=7.8, color='#d35400', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='none', alpha=0.9))

    # =========================================================================
    # 2. RAMO DISTILLATO (Destra: x = 10.2 -> 13.2)
    # =========================================================================
    box_dist = patches.FancyBboxPatch((10.2, 2.15), 3.0, 2.25, boxstyle="round,pad=0.06", 
                                      facecolor='#ebf5fb', edgecolor='#2980b9', linewidth=1.6)
    ax.add_patch(box_dist)
    ax.text(11.7, 4.08, "YOLOv8n Distillato", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72')
    ax.plot([10.5, 12.9], [3.78, 3.78], color='#2980b9', lw=0.8, alpha=0.5)
    ax.text(11.7, 3.42, "0% Box Ground Truth", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#2874a6')
    ax.text(11.7, 3.00, "Teacher DINOv3 + WSOL\n55.964 Frame Pseudo-Etichettati", ha='center', va='center', fontsize=8.0, color='#34495e')
    ax.text(11.7, 2.45, "Performance: 37.39% mAP@50", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1b4f72')

    # Freccia singola verticale dal Distillato al Test Set
    cx_dist = 11.7
    ax.annotate("", xy=(cx_dist, 1.05), xytext=(cx_dist, 2.15),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='#2980b9'))
    ax.text(cx_dist, 1.55, "Modello Proposto", 
            ha='center', va='center', fontsize=7.8, color='#2980b9', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='none', alpha=0.9))

    # =========================================================================
    # 3. VALUTAZIONE COMUNE: TEST SET UFFICIALE (In basso, a tutta larghezza)
    # =========================================================================
    box_test = patches.FancyBboxPatch((0.4, 0.20), 12.8, 0.85, boxstyle="round,pad=0.06", 
                                      facecolor='#eafaf1', edgecolor='#27ae60', linewidth=1.8)
    ax.add_patch(box_test)
    ax.text(6.8, 0.68, "Test Set Ufficiale Non Visto (5.462 fotogrammi, 7 sessioni video continue)", 
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#145a32')
    ax.text(6.8, 0.38, "Valutazione rigorosa mAP@50 e confronto appaiato (paired benchmark) per stimare l'equivalenza di annotazione", 
            ha='center', va='center', fontsize=8.2, color='#196f3d')

    out_path = FIGURES_DIR / "protocollo_sample_efficiency.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 6. RITENZIONE PER TIER (Capitolo 8)
# ==============================================================================
def generate_retention_chart():
    tiers = ['Full-Frame (≥0.80)', 'Macro (0.30 - 0.79)', 'Overall (Totale)', 'Small (<0.30)', 'Tiny (≤0.10)']
    retention = [84.14, 63.60, 61.79, 10.54, 0.00]
    map50_dist = [45.67, 37.71, 37.39, 9.01, 0.00]
    map50_oracle = [54.28, 59.29, 60.51, 85.49, 88.15]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    y = np.arange(len(tiers))
    height = 0.55

    # Barre di ritenzione
    colors = ['#27ae60', '#2980b9', '#8e44ad', '#e67e22', '#c0392b']
    bars = ax.barh(y, retention, height=height, color=colors, alpha=0.85, edgecolor='#2c3e50', linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(tiers, fontsize=10.5, fontweight='semibold')
    ax.invert_yaxis()  # Primo in alto
    ax.set_xlabel('Ritenzione Relativa rispetto all\'Oracolo 100% GT (%)', fontsize=11, fontweight='semibold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Etichette sulle barre
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ret_val = retention[i]
        d_val = map50_dist[i]
        o_val = map50_oracle[i]

        label_text = f" {ret_val:.1f}%  (Dist: {d_val:.1f}% vs GT: {o_val:.1f}%)"
        if w > 15:
            ax.text(w - 2.0, bar.get_y() + bar.get_height()/2.0, label_text, 
                    ha='right', va='center', color='white', fontweight='bold', fontsize=9.5)
        else:
            ax.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, label_text, 
                    ha='left', va='center', color='#2c3e50', fontweight='bold', fontsize=9.5)

    ax.set_title("Efficienza di Distillazione per Scala Dimensionale (Tier)\nmAP@50 Modello Distillato rapportato a 100% Supervisionato", 
                 fontsize=12, fontweight='bold', pad=15)

    out_path = FIGURES_DIR / "ritenzione_per_tier.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 7. EVIDENZA VISIVA BIAS CONTESTUALE TINY (Capitolo 8)
# ==============================================================================
def generate_tiny_bias_figure():
    # Seleziona campioni significativi per tutte e 4 le classi del Tier Tiny
    samples = [
        {
            "path": REPO_ROOT / "visual_sample_dino_vs_gt" / "5.7_S.Agata" / "frame_000115_cam_overlay.jpg",
            "title": "(a) Classe 5.7: Sant'Agata (Area: 5.1% - Tier Tiny)\nPicco di salienza sull'arco barocco contiguo"
        },
        {
            "path": REPO_ROOT / "visual_sample_dino_vs_gt" / "5.5_Cupola" / "frame_000196_cam_overlay.jpg",
            "title": "(b) Classe 5.5: Cupola (Area: 7.2% - Tier Tiny)\nPicco saliente sulla volta perimetrale"
        },
        {
            "path": REPO_ROOT / "visual_sample_dino_vs_gt" / "5.8_S.Scolastica" / "frame_000151_cam_overlay.jpg",
            "title": "(c) Classe 5.8: Santa Scolastica (Area: 4.5% - Tier Tiny)\nDispersione del picco sulla cornice dell'altare"
        },
        {
            "path": REPO_ROOT / "visual_sample_dino_vs_gt" / "6.5_BassorilieviMancanti" / "frame_000127_cam_overlay.jpg",
            "title": "(d) Classe 6.5: Bassorilievi Mancanti (Area: 3.1% - Tier Tiny)\nAttivazione diffusa sulla parete muraria"
        }
    ]

    for s in samples:
        if not s["path"].exists():
            print(f"[WARNING] File non trovato: {s['path']}, salto composizione.")
            return

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), dpi=300)
    axes = axes.flatten()

    for i, s in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(str(s["path"])), cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(s["title"], fontsize=10.5, fontweight='bold', pad=8)
        axes[i].axis('off')

    out_path = FIGURES_DIR / "bias_contestuale_tiny.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 8. ESEMPI QUALITATIVI LOCALIZZAZIONE WSOL (Capitolo 5)
# ==============================================================================
def generate_success_samples_figure():
    sample_dir = REPO_ROOT / "visual_sample_dino_vs_gt"

    samples = [
        # Riga 1: Casi di successo (Macro e Full-Frame)
        {
            "cls": "5.6_AperturaPavimento", 
            "fname": "frame_000239_cam_overlay.jpg", 
            "title": "(a) Apertura Pavimento (Macro, IoU 0.90)\nContorno ellittico ad elevata aderenza",
            "row": 0, "col": 0
        },
        {
            "cls": "7.6_Scala", 
            "fname": "frame_000361_cam_overlay.jpg", 
            "title": "(b) Scala Monumentale (Macro, IoU 0.79)\nAttivazione compatta sui gradini",
            "row": 0, "col": 1
        },
        {
            "cls": "3.10_CorridoioCorodiNotte", 
            "fname": "frame_000103_cam_overlay.jpg", 
            "title": "(c) Corridoio Coro di Notte (Full-Frame, IoU 0.98)\nCopertura integrale della prospettiva",
            "row": 0, "col": 2
        },
        # Riga 2: Casi di fallimento (Tiny)
        {
            "cls": "5.7_S.Agata", 
            "fname": "frame_000115_cam_overlay.jpg", 
            "title": "(d) Sant'Agata (Tiny, Area 5%, IoU 0.06)\nAttribution leakage sull'arco barocco",
            "row": 1, "col": 0
        },
        {
            "cls": "5.5_Cupola", 
            "fname": "frame_000196_cam_overlay.jpg", 
            "title": "(e) Cupola (Tiny, Area 7%, IoU 0.09)\nDislocamento del picco sulla volta",
            "row": 1, "col": 1
        },
        {
            "cls": "6.5_BassorilieviMancanti", 
            "fname": "frame_000127_cam_overlay.jpg", 
            "title": "(f) Bassorilievi (Tiny, Area 3%, IoU 0.01)\nAttivazione diffusa e dispersa sulla parete",
            "row": 1, "col": 2
        }
    ]

    for s in samples:
        img_p = sample_dir / s["cls"] / s["fname"]
        if not img_p.exists():
            print(f"[WARNING] File non trovato: {img_p}, salto composizione.")
            return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), dpi=300)

    for s in samples:
        img_p = sample_dir / s["cls"] / s["fname"]
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        ax = axes[s["row"], s["col"]]
        ax.imshow(img)
        ax.set_title(s["title"], fontsize=9.8, fontweight='bold', pad=7, color='#1a252f')
        ax.axis('off')

    out_path = FIGURES_DIR / "wsol_esempi_successo.png"
    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Salvata figura: {out_path.name}")


# ==============================================================================
# 9. COPIA PLOT SAMPLE EFFICIENCY
# ==============================================================================
def copy_sample_efficiency_plots():
    src_grid = REPO_ROOT / "yolo" / "runs_sample_efficiency" / "stratified_test_results" / "plots" / "per_class_range_grid.png"
    src_macro = REPO_ROOT / "yolo" / "runs_sample_efficiency" / "stratified_test_results" / "plots" / "by_tier" / "macro.png"

    if src_grid.exists():
        dst_grid = FIGURES_DIR / "range_plot_grid.png"
        shutil.copy2(src_grid, dst_grid)
        print(f"[OK] Copiato: {dst_grid.name}")

    if src_macro.exists():
        dst_macro = FIGURES_DIR / "range_plot_macro.png"
        shutil.copy2(src_macro, dst_macro)
        print(f"[OK] Copiato: {dst_macro.name}")


if __name__ == "__main__":
    print("--> Inizio generazione figure per relazione accademica...")
    generate_dino_pretraining()
    generate_classifier_architecture()
    generate_wsol_pipeline()
    generate_distillation_pipeline()
    generate_sample_efficiency_protocol()
    generate_retention_chart()
    generate_tiny_bias_figure()
    generate_success_samples_figure()
    copy_sample_efficiency_plots()
    print("[COMPLETATO] Tutte le figure sono state create con successo in relazione/figure/")
