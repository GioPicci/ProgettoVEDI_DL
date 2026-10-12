"""
Range plot per classe: per ogni classe una linea da min a max AP ottenuta negli esperimenti,
con un punto per ogni frazione di training supervised e un rombo per la YOLO distillata (WSOL).
Classi ordinate per area media crescente (piccole in alto).

Produce, in <outdir>:
  per_class_range_full.(png|pdf)   tutte le classi in un unico grafico (AP50 | AP50-95)
  per_class_range_grid.(png|pdf)   un pannello per (gruppo dimensionale x metrica)
  by_tier/<gruppo>.(png|pdf)       una figura per gruppo (AP50 | AP50-95), comoda per la relazione

Uso:
    python plot_per_class_range.py              # 4 gruppi: Tiny (<=0.10), Small, Macro, Full-Frame
    python plot_per_class_range.py --tiers 3    # 3 gruppi come nello script YOLO
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path("runs_sample_efficiency/stratified_test_results")
METRICS = ["ap50", "ap50_95"]
METRIC_LABELS = {"ap50": "AP@50 (%)", "ap50_95": "AP@50:95 (%)"}
DIST_COLOR = "#d62728"


# ---------------------------------------------------------------- dati
def build_table(sup_csv, dist_csv, metric):
    sup = pd.read_csv(sup_csv)
    dist = pd.read_csv(dist_csv)

    meta = (sup[["class_id", "class_name", "mean_area_ratio", "size_tier"]]
            .drop_duplicates("class_id").set_index("class_id"))
    wide = sup.pivot(index="class_id", columns="fraction_pct", values=f"{metric}_mean")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    dist_vals = dist.set_index("class_id")[metric].rename("distilled")

    df = meta.join(wide).join(dist_vals)
    df = df.sort_values("mean_area_ratio").reset_index()
    return df, list(wide.columns)


def assign_groups(df, n_tiers):
    def group(r):
        if n_tiers == 4 and r.mean_area_ratio <= 0.10:
            return "Tiny (≤0.10)"
        if r.size_tier.startswith("Small"):
            return "Small (0.10–0.30)" if n_tiers == 4 else "Small (<0.30)"
        if r.size_tier.startswith("Macro"):
            return "Macro (0.30–0.79)"
        return "Full-Frame (≥0.80)"
    df = df.copy()
    df["group"] = [group(r) for r in df.itertuples()]
    return df


def slug(group):
    return group.split()[0].lower().replace("-", "")


# ---------------------------------------------------------------- disegno
def draw_panel(ax, df, frac_cols, metric, colors, show_ylabels=True,
               show_xlabel=True, bands=False, band_labels=False):
    n = len(df)
    y = np.arange(n)
    value_cols = frac_cols + ["distilled"]

    if bands:
        groups = df["group"].tolist()
        start, band = 0, 0
        for i in range(1, n + 1):
            if i == n or groups[i] != groups[start]:
                if band % 2 == 0:
                    ax.axhspan(start - 0.5, i - 0.5, color="0.95", zorder=0)
                if band_labels:
                    ax.text(1.01, (start + i - 1) / 2, groups[start], rotation=270,
                            va="center", ha="left", fontsize=9, color="0.3",
                            transform=ax.get_yaxis_transform())
                start = i
                band += 1

    vals = df[value_cols].to_numpy(dtype=float)
    ax.hlines(y, np.nanmin(vals, axis=1), np.nanmax(vals, axis=1),
              color="0.65", lw=2.2, zorder=1)

    for f, c in zip(frac_cols, colors):
        ax.scatter(df[f], y, s=26, color=c, edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(df["distilled"], y, marker="D", s=44, color=DIST_COLOR,
               edgecolor="black", linewidth=0.7, zorder=4)

    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-2, 102)
    if show_xlabel:
        ax.set_xlabel(METRIC_LABELS[metric])
    ax.grid(axis="x", color="0.85", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([f"{r.class_name}  ({r.mean_area_ratio:.2f})" for r in df.itertuples()],
                           fontsize=7)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def legend_handles(frac_cols, colors):
    handles = [Line2D([0], [0], marker="o", ls="", color=c, markeredgecolor="white",
                      markersize=7, label=f"{int(f)}% annotazioni")
               for f, c in zip(frac_cols, colors)]
    handles.append(Line2D([0], [0], marker="D", ls="", color=DIST_COLOR, markeredgecolor="black",
                          markersize=7, label="WSOL distillata (0 box annotate)"))
    return handles


def save(fig, path_no_ext):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_no_ext}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {path_no_ext}.png / .pdf")


def add_legend(fig, frac_cols, colors, y=1.0):
    h = legend_handles(frac_cols, colors)
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- figure
def fig_full(tables, frac_cols, colors, outdir):
    n = len(tables[METRICS[0]])
    fig, axes = plt.subplots(1, 2, figsize=(16.5, max(6, 0.24 * n + 1.8)), squeeze=False)
    for i, (ax, m) in enumerate(zip(axes[0], METRICS)):
        draw_panel(ax, tables[m], frac_cols, m, colors, show_ylabels=(i == 0),
                   bands=True, band_labels=(i == 1))
    add_legend(fig, frac_cols, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir / "per_class_range_full")


def fig_grid(tables, group_order, frac_cols, colors, outdir):
    counts = [int((tables[METRICS[0]]["group"] == g).sum()) for g in group_order]
    fig, axes = plt.subplots(
        len(group_order), 2, figsize=(14, 0.26 * sum(counts) + 1.1 * len(group_order) + 1.2),
        gridspec_kw={"height_ratios": [c + 1.5 for c in counts]},
        sharex="col", squeeze=False)
    for r, g in enumerate(group_order):
        for c, m in enumerate(METRICS):
            sub = tables[m][tables[m]["group"] == g].reset_index(drop=True)
            ax = axes[r][c]
            draw_panel(ax, sub, frac_cols, m, colors, show_ylabels=(c == 0),
                       show_xlabel=(r == len(group_order) - 1))
            ax.set_title(f"{g}  |  {METRIC_LABELS[m]}", loc="left", fontsize=10, fontweight="bold")
    add_legend(fig, frac_cols, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, outdir / "per_class_range_grid")


def fig_by_tier(tables, group_order, frac_cols, colors, outdir):
    tdir = outdir / "by_tier"
    tdir.mkdir(parents=True, exist_ok=True)
    for g in group_order:
        n = int((tables[METRICS[0]]["group"] == g).sum())
        fig, axes = plt.subplots(1, 2, figsize=(14, max(2.6, 0.3 * n + 1.9)), squeeze=False)
        for c, (ax, m) in enumerate(zip(axes[0], METRICS)):
            sub = tables[m][tables[m]["group"] == g].reset_index(drop=True)
            draw_panel(ax, sub, frac_cols, m, colors, show_ylabels=(c == 0))
            ax.set_title(f"{g}  |  {METRIC_LABELS[m]}", loc="left", fontsize=10, fontweight="bold")
        add_legend(fig, frac_cols, colors)
        fig.tight_layout(rect=(0, 0, 1, 0.9 if n <= 4 else 0.94))
        save(fig, tdir / slug(g))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--supervised", default=str(BASE / "fractions_per_class_summary.csv"))
    p.add_argument("--distilled", default=str(BASE / "distilled_per_class.csv"))
    p.add_argument("--outdir", default=str(BASE / "plots"))
    p.add_argument("--tiers", type=int, choices=[3, 4], default=4,
                   help="4 = Tiny(<=0.10)/Small/Macro/Full-Frame, 3 = Small/Macro/Full-Frame")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables, frac_cols = {}, None
    for m in METRICS:
        df, fc = build_table(args.supervised, args.distilled, m)
        tables[m] = assign_groups(df, args.tiers)
        frac_cols = fc
    group_order = list(tables[METRICS[0]]["group"].unique())
    colors = plt.cm.YlGnBu(np.linspace(0.35, 1.0, len(frac_cols)))

    fig_full(tables, frac_cols, colors, outdir)
    fig_grid(tables, group_order, frac_cols, colors, outdir)
    fig_by_tier(tables, group_order, frac_cols, colors, outdir)


if __name__ == "__main__":
    main()