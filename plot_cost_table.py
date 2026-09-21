"""
Cost summary table rendered as a publication-quality figure.
Saves: reports/cost_summary_table.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = [
    ("CIFAR-100",   "Vanilla",  243,  2,     0.00,  0.38),
    ("CIFAR-100",   "EWC",      694,  3,   251.48,  0.39),
    ("CIFAR-100",   "MAS",      327,  2,    25.15,  0.38),
    ("CIFAR-100",   "OGD",      283,  2,   125.74,  0.38),
    ("CIFAR-100",   "ER",       461,  0,    11.73,  0.38),
    ("CIFAR-100",   "A-GEM",   1139,  1,   117.26,  0.48),
    ("Rot. MNIST",  "Vanilla",    9,  1,     0.00,  2.46),
    ("Rot. MNIST",  "EWC",       42,  4,    15.53,  2.47),
    ("Rot. MNIST",  "MAS",       23,  2,     1.55,  2.46),
    ("Rot. MNIST",  "OGD",       18,  1,     7.76,  2.46),
    ("Rot. MNIST",  "ER",        23,  2,     3.00,  2.46),
    ("Rot. MNIST",  "A-GEM",     20,  0,    29.98,  2.56),
    ("EMNIST",      "Vanilla",   70,  6,     0.00, 20.42),
    ("EMNIST",      "EWC",      249, 21,     8.02, 20.43),
    ("EMNIST",      "MAS",      839, 75,     1.60, 20.41),
    ("EMNIST",      "OGD",       93,  9,     1.60, 20.42),
    ("EMNIST",      "ER",        69,  5,     3.00, 22.69),
    ("EMNIST",      "A-GEM",    139, 13,    14.99, 20.52),
]

METHOD_COLORS = {
    "Vanilla": "#e74c3c", "EWC": "#e67e22", "MAS": "#f1c40f",
    "OGD": "#2ecc71",     "ER":  "#3498db", "A-GEM": "#9b59b6",
}
DATASET_BG = {
    "CIFAR-100":  "#eaf4fb",
    "Rot. MNIST": "#eafaf1",
    "EMNIST":     "#fef9e7",
}

n_rows = len(DATA)
fig_h  = 0.45 * n_rows + 2.2
fig, ax = plt.subplots(figsize=(13, fig_h))
ax.set_xlim(0, 13)
ax.set_ylim(0, fig_h)
ax.axis("off")

fig.suptitle(
    "Computational Cost Summary  (mean ± std, 3 seeds, buffer / memory = 1,000)",
    fontsize=13, fontweight="bold", y=0.99
)

# column x positions and widths
COL_X     = [0.2,  2.2,  4.2,  9.0]
COL_ALIGN = ["left", "left", "right", "right"]
HEADERS   = ["Dataset", "Method", "Training Time (s)", "Extra Storage (MB)"]

ROW_H   = 0.42
HEADER_Y = fig_h - 0.55

# ── draw header ───────────────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (0.05, HEADER_Y - 0.05), 12.9, ROW_H,
    boxstyle="round,pad=0.05",
    facecolor="#2c3e50", edgecolor="none"
))
for hdr, cx, align in zip(HEADERS, COL_X, COL_ALIGN):
    ax.text(cx, HEADER_Y + ROW_H * 0.38, hdr,
            fontsize=10, fontweight="bold", color="white",
            ha=align, va="center")

# ── draw rows ─────────────────────────────────────────────────────────────────
prev_ds = None
ds_start_y = {}
ds_row_count = {}
for ds, *_ in DATA:
    ds_row_count[ds] = ds_row_count.get(ds, 0) + 1

ds_order = list(dict.fromkeys(d[0] for d in DATA))
ds_y_start = {}
y = HEADER_Y - ROW_H * 0.15
for ds in ds_order:
    ds_y_start[ds] = y
    y -= ROW_H * ds_row_count[ds]

# draw dataset background bands
for ds in ds_order:
    n = ds_row_count[ds]
    band_y = ds_y_start[ds] - ROW_H * n
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, band_y), 12.9, ROW_H * n,
        boxstyle="round,pad=0.05",
        facecolor=DATASET_BG[ds], edgecolor="#cccccc", linewidth=0.6
    ))

# draw rows
ds_idx = {ds: 0 for ds in ds_order}
for dataset, method, time_s, time_std, storage, ram in DATA:
    idx   = ds_idx[dataset]
    row_y = ds_y_start[dataset] - ROW_H * (idx + 0.5)
    ds_idx[dataset] += 1

    color = METHOD_COLORS[method]

    # dataset label (first row of group only)
    if idx == 0:
        mid_y = ds_y_start[dataset] - ROW_H * ds_row_count[dataset] / 2
        ax.text(COL_X[0], mid_y,
                dataset, fontsize=9, color="#555555",
                style="italic", ha="left", va="center")

    # method colour dot
    ax.add_patch(plt.Circle(
        (COL_X[1] - 0.35, row_y), 0.12,
        color=color, zorder=3
    ))

    # method name
    ax.text(COL_X[1], row_y, method,
            fontsize=9.5, fontweight="bold", color=color,
            ha="left", va="center")

    # training time
    time_str = f"{time_s:,} ± {time_std}"
    # highlight slowest per dataset in red
    max_time = max(d[2] for d in DATA if d[0] == dataset)
    t_color  = "#c0392b" if time_s == max_time else "#222222"
    ax.text(COL_X[2], row_y, time_str,
            fontsize=9, ha="right", va="center", color=t_color,
            fontweight="bold" if time_s == max_time else "normal")

    # extra storage
    store_str = f"{storage:.2f}"
    if storage == 0:
        s_color = "#888888"
    elif method == "ER":
        s_color = "#27ae60"    # green = best (lowest non-zero replay method)
    elif storage > 100:
        s_color = "#c0392b"    # red = very high
    else:
        s_color = "#222222"
    ax.text(COL_X[3], row_y, store_str,
            fontsize=9, ha="right", va="center", color=s_color,
            fontweight="bold" if s_color != "#222222" else "normal")

    # row separator
    sep_y = ds_y_start[dataset] - ROW_H * (idx + 1)
    ax.plot([0.05, 12.95], [sep_y, sep_y], color="#dddddd", linewidth=0.5)

# ── legend + footnote ─────────────────────────────────────────────────────────
bottom_y = min(ds_y_start[ds] - ROW_H * ds_row_count[ds] for ds in ds_order) - 0.28

legend_items = [
    mpatches.Patch(color="#27ae60", label="ER storage (lowest non-zero replay)"),
    mpatches.Patch(color="#c0392b", label="Slowest time / storage > 100 MB"),
    mpatches.Patch(color="#888888", label="No extra storage (Vanilla)"),
]
ax.legend(handles=legend_items, loc="lower left",
          bbox_to_anchor=(0.0, 0), fontsize=8,
          framealpha=0.8, ncol=3)

ax.text(6.5, bottom_y - 0.05,
        "Extra Storage = Fisher matrices (EWC) / gradient subspace (OGD) / episodic buffer (ER, A-GEM).  "
        "Rotated MNIST: 1 epoch/task (unified setting).",
        fontsize=7.5, ha="center", va="top", color="#666666", style="italic")

plt.tight_layout()
fname = "reports/cost_summary_table.png"
plt.savefig(fname, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {fname}")
