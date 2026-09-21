"""
Buffer / memory size sensitivity analysis for ER and A-GEM.
Plots ACC / BWT / Forgetting vs memory size for all 3 datasets.
Both methods shown on the same axes for easy comparison.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS      = [0, 1, 2]
BUF_SIZES  = [200, 500, 1000, 2000]
OUT_DIR    = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ── File templates (format with buf=<int> and seed=<int>) ─────────────────────
ER_TEMPLATES = {
    "CIFAR-100":     "cifar100/reports/acc_matrix_ER_10tasks_buf{buf}_seed{seed}.csv",
    "Rotated MNIST": "reports/acc_matrix_ER_10tasks_buf{buf}_seed{seed}.csv",
    "EMNIST":        "emnist/reports/acc_matrix_EMNIST_ER_5tasks_buf{buf}_seed{seed}.csv",
}

AGEM_TEMPLATES = {
    "CIFAR-100":     "cifar100/reports/acc_matrix_AGEM_10tasks_mem{buf}_seed{seed}.csv",
    "Rotated MNIST": "reports/acc_matrix_AGEM_10tasks_mem{buf}_seed{seed}.csv",
    "EMNIST":        "emnist/reports/acc_matrix_EMNIST_AGEM_5tasks_mem{buf}_seed{seed}.csv",
}

DATASET_COLORS = {
    "CIFAR-100":     "#3498db",
    "Rotated MNIST": "#e74c3c",
    "EMNIST":        "#2ecc71",
}

METHOD_STYLE = {
    "ER":    {"linestyle": "-",  "marker": "o"},
    "A-GEM": {"linestyle": "--", "marker": "s"},
}


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    arr = df.values.astype(float)
    if str(df.index[0]).startswith("T"):
        arr = arr.T
    n = arr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            arr[i, j] = np.nan
    return arr


def get_metrics(tmpl, buf):
    acc_vals, bwt_vals, fgt_vals = [], [], []
    for s in SEEDS:
        p = tmpl.format(buf=buf, seed=s)
        if not os.path.exists(p):
            continue
        R = load_matrix(p)
        n = R.shape[0]
        diag = [R[i, i] for i in range(n)]
        last = []
        for j in range(n):
            col = R[:, j]
            valid = col[~np.isnan(col)]
            last.append(valid[-1] if len(valid) > 0 else 0.0)
        acc_vals.append(np.mean(last))
        bwt_vals.append(np.mean(np.array(last[:-1]) - np.array(diag[:-1])))
        fgt_vals.append(np.mean(np.array(diag[:-1]) - np.array(last[:-1])))
    if not acc_vals:
        return None
    return {
        "ACC":        (np.mean(acc_vals), np.std(acc_vals, ddof=0)),
        "BWT":        (np.mean(bwt_vals), np.std(bwt_vals, ddof=0)),
        "Forgetting": (np.mean(fgt_vals), np.std(fgt_vals, ddof=0)),
    }


# ── Plot 1: 3 metrics × (3 datasets × 2 methods) line chart ──────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Replay Method Memory Size Sensitivity  (mean ± std, 3 seeds)\n"
    "Solid = ER  ·  Dashed = A-GEM",
    fontsize=13, fontweight='bold'
)

metric_titles = [
    "ACC  (↑ higher is better)",
    "BWT  (↑ closer to 0 is better)",
    "Forgetting  (↓ lower is better)",
]

all_templates = [
    ("ER",    ER_TEMPLATES),
    ("A-GEM", AGEM_TEMPLATES),
]

for ax, metric, title in zip(axes, ["ACC", "BWT", "Forgetting"], metric_titles):
    for dataset in ["CIFAR-100", "Rotated MNIST", "EMNIST"]:
        color = DATASET_COLORS[dataset]
        for method, templates in all_templates:
            tmpl = templates[dataset]
            style = METHOD_STYLE[method]
            means, stds = [], []
            for buf in BUF_SIZES:
                r = get_metrics(tmpl, buf)
                if r:
                    means.append(r[metric][0])
                    stds.append(r[metric][1])
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds  = np.array(stds)

            # only plot if we have at least one valid point
            if np.all(np.isnan(means)):
                continue

            label = f"{dataset} ({method})"
            ax.plot(BUF_SIZES, means,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.2,
                    label=label,
                    color=color,
                    alpha=0.85 if method == "ER" else 0.6)
            ax.fill_between(BUF_SIZES, means - stds, means + stds,
                            alpha=0.08, color=color)

            # annotate only ER (to keep plot readable)
            if method == "ER":
                for x, y in zip(BUF_SIZES, means):
                    if not np.isnan(y):
                        ax.annotate(f"{y:.3f}", (x, y),
                                    textcoords="offset points", xytext=(0, 7),
                                    ha='center', fontsize=6.5, color=color)

    ax.set_xlabel("Buffer / Memory Size (samples per task stored at most)", fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(BUF_SIZES)
    ax.set_xticklabels(BUF_SIZES)
    ax.grid(True, alpha=0.3)
    if metric == "BWT":
        ax.axhline(0, color='black', lw=0.8, linestyle='--')

# ── Compact legend: datasets × line style ────────────────────────────────────
legend_dataset = [
    Line2D([0],[0], color=DATASET_COLORS[d], linewidth=2, label=d)
    for d in ["CIFAR-100", "Rotated MNIST", "EMNIST"]
]
legend_method = [
    Line2D([0],[0], color='gray', linestyle="-",  marker="o", linewidth=2, label="ER"),
    Line2D([0],[0], color='gray', linestyle="--", marker="s", linewidth=2, label="A-GEM"),
]
fig.legend(handles=legend_dataset + legend_method,
           loc='lower center', ncol=5, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])
fname = f"{OUT_DIR}/buffer_sensitivity.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fname}")


# ── Plot 2: Per-dataset subplots (3 rows × 3 metrics) ────────────────────────

fig2, axes2 = plt.subplots(3, 3, figsize=(16, 13))
fig2.suptitle(
    "Memory Size Sensitivity — per Dataset  (mean ± std, 3 seeds)\n"
    "Solid = ER  ·  Dashed = A-GEM",
    fontsize=13, fontweight='bold'
)

datasets = ["CIFAR-100", "Rotated MNIST", "EMNIST"]
metrics  = ["ACC", "BWT", "Forgetting"]
m_titles = ["ACC (↑)", "BWT (↑ → 0)", "Forgetting (↓)"]

for row, dataset in enumerate(datasets):
    color = DATASET_COLORS[dataset]
    for col, (metric, m_title) in enumerate(zip(metrics, m_titles)):
        ax = axes2[row, col]
        for method, templates in all_templates:
            tmpl = templates[dataset]
            style = METHOD_STYLE[method]
            means, stds = [], []
            for buf in BUF_SIZES:
                r = get_metrics(tmpl, buf)
                if r:
                    means.append(r[metric][0])
                    stds.append(r[metric][1])
                else:
                    means.append(np.nan)
                    stds.append(0)
            means = np.array(means)
            stds  = np.array(stds)
            if np.all(np.isnan(means)):
                continue
            ax.plot(BUF_SIZES, means,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.2, color=color,
                    alpha=0.9 if method == "ER" else 0.6,
                    label=method)
            ax.fill_between(BUF_SIZES, means - stds, means + stds,
                            alpha=0.12, color=color)
            for x, y in zip(BUF_SIZES, means):
                if not np.isnan(y):
                    offset = 7 if method == "ER" else -13
                    ax.annotate(f"{y:.3f}", (x, y),
                                textcoords="offset points",
                                xytext=(0, offset),
                                ha='center', fontsize=6.5, color=color,
                                alpha=0.9 if method == "ER" else 0.65)

        if col == 0:
            ax.set_ylabel(dataset, fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title(m_title, fontsize=10, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(BUF_SIZES)
        ax.set_xticklabels(BUF_SIZES, fontsize=8)
        ax.set_xlabel("Memory Size", fontsize=8)
        ax.grid(True, alpha=0.25)
        if metric == "BWT":
            ax.axhline(0, color='black', lw=0.8, linestyle='--')
        ax.legend(fontsize=8)

plt.tight_layout()
fname2 = f"{OUT_DIR}/buffer_sensitivity_by_dataset.png"
plt.savefig(fname2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fname2}")


# ── Print summary table ────────────────────────────────────────────────────────

print("\n" + "="*80)
print(f"{'Method':<8} {'Dataset':<16} {'Memory':>8} {'ACC':>14} {'BWT':>14} {'Forgetting':>14}")
print("="*80)

for method, templates in all_templates:
    for dataset, tmpl in templates.items():
        for buf in BUF_SIZES:
            r = get_metrics(tmpl, buf)
            if r:
                print(f"{method:<8} {dataset:<16} {buf:>8} "
                      f"{r['ACC'][0]:>8.4f}±{r['ACC'][1]:.4f} "
                      f"{r['BWT'][0]:>8.4f}±{r['BWT'][1]:.4f} "
                      f"{r['Forgetting'][0]:>9.4f}±{r['Forgetting'][1]:.4f}")
            else:
                print(f"{method:<8} {dataset:<16} {buf:>8}  (pending)")
print("="*80)
