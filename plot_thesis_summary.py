"""
Thesis summary figure — one clean overview chart for the paper.
3 rows (datasets) × 3 cols (ACC bar / forgetting curve / cost scatter)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

SEEDS = [0, 1, 2]
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_COLORS = {
    "Vanilla": "#e74c3c", "EWC": "#e67e22", "MAS": "#f1c40f",
    "OGD":     "#2ecc71", "ER":  "#3498db", "A-GEM": "#9b59b6",
}
ALL_METHODS = ["Vanilla", "EWC", "MAS", "OGD", "ER", "A-GEM"]

EXPERIMENTS = {
    "CIFAR-100\n(Task-Incremental)": [
        ("Vanilla", "cifar100/reports/acc_matrix_Vanilla_10tasks_seed{seed}.csv"),
        ("EWC",     "cifar100/reports/acc_matrix_EWC_10tasks_seed{seed}.csv"),
        ("MAS",     "cifar100/reports/acc_matrix_MAS_10tasks_seed{seed}.csv"),
        ("OGD",     "cifar100/reports/acc_matrix_OGD_10tasks_seed{seed}.csv"),
        # ER and A-GEM both at memory/buffer=1000 for fair comparison
        ("ER",      "cifar100/reports/acc_matrix_ER_10tasks_buf1000_seed{seed}.csv"),
        ("A-GEM",   "cifar100/reports/acc_matrix_AGEM_10tasks_mem1000_seed{seed}.csv"),
    ],
    "Rotated MNIST\n(Rotation Drift)": [
        ("Vanilla", "reports/acc_matrix_Vanilla_10tasks_seed{seed}.csv"),
        ("EWC",     "reports/acc_matrix_EWC_10tasks_seed{seed}.csv"),
        ("MAS",     "reports/acc_matrix_MAS_10tasks_seed{seed}.csv"),
        ("OGD",     "reports/acc_matrix_OGD_10tasks_seed{seed}.csv"),
        # ER and A-GEM both at memory/buffer=1000 for fair comparison
        ("ER",      "reports/acc_matrix_ER_10tasks_buf1000_seed{seed}.csv"),
        ("A-GEM",   "reports/acc_matrix_AGEM_10tasks_mem1000_seed{seed}.csv"),
    ],
    "EMNIST\n(Visual Style Drift)": [
        ("Vanilla", "emnist/reports/acc_matrix_EMNIST_Vanilla_5tasks_seed{seed}.csv"),
        ("EWC",     "emnist/reports/acc_matrix_EMNIST_EWC_5tasks_seed{seed}.csv"),
        ("MAS",     "emnist/reports/mas_acc_log_seed{seed}.csv"),
        ("OGD",     "emnist/reports/ogd_acc_log_seed{seed}.csv"),
        # ER and A-GEM both at memory/buffer=1000 for fair comparison
        ("ER",      "emnist/reports/acc_matrix_EMNIST_ER_5tasks_buf1000_seed{seed}.csv"),
        ("A-GEM",   "emnist/reports/acc_matrix_EMNIST_AGEM_5tasks_mem1000_seed{seed}.csv"),
    ],
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


def get_metrics(tmpl):
    vals = {"ACC": [], "BWT": [], "LA": []}
    curves = []
    for s in SEEDS:
        p = tmpl.format(seed=s)
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
        vals["ACC"].append(np.mean(last))
        vals["BWT"].append(np.mean(np.array(last[:-1]) - np.array(diag[:-1])))
        vals["LA"].append(np.mean(diag))
        # forgetting curve: task-0 accuracy at each step
        curve = []
        for i in range(n):
            v = R[i, 0]
            if np.isnan(v):
                col = R[:i+1, 0]
                valid = col[~np.isnan(col)]
                v = valid[-1] if len(valid) > 0 else 0.0
            curve.append(v)
        curves.append(curve)
    if not vals["ACC"]:
        return None
    result = {k: (np.mean(v), np.std(v, ddof=0)) for k, v in vals.items()}
    if curves:
        arr = np.array(curves)
        result["curve_mean"] = arr.mean(axis=0)
        result["curve_std"]  = arr.std(axis=0, ddof=0)
    return result


# ── Build figure ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    "Continual Learning under Visual Drift\n"
    "Replay vs. Regularization vs. Optimization Methods  (mean ± std, 3 seeds)",
    fontsize=13, fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35,
                       left=0.07, right=0.97, top=0.92, bottom=0.07)

col_titles = ["Final ACC per Method", "Task-1 Forgetting Curve", "BWT per Method"]

for row, (dataset, exps) in enumerate(EXPERIMENTS.items()):
    dataset_label = dataset.replace("\n", " ")
    methods_data = {}
    for method, tmpl in exps:
        r = get_metrics(tmpl)
        if r:
            methods_data[method] = r

    # ── Col 0: ACC bar chart ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 0])
    if row == 0:
        ax.set_title(col_titles[0], fontsize=10, fontweight='bold', pad=6)
    ax.set_ylabel("Final ACC", fontsize=9, fontweight='bold')
    ax.text(
        -0.12, 0.5, dataset_label,
        transform=ax.transAxes,
        rotation=90,
        va='center',
        ha='right',
        fontsize=9,
        fontweight='bold',
    )

    methods = [m for m in ALL_METHODS if m in methods_data]
    x = np.arange(len(methods))
    acc_vals  = [methods_data[m]["ACC"][0] for m in methods]
    acc_stds  = [methods_data[m]["ACC"][1] for m in methods]
    colors    = [METHOD_COLORS[m] for m in methods]

    bars = ax.bar(x, acc_vals, yerr=acc_stds, color=colors,
                  alpha=0.85, capsize=4, edgecolor='white', linewidth=0.8)
    # highlight best
    best_i = int(np.argmax(acc_vals))
    bars[best_i].set_edgecolor('black'); bars[best_i].set_linewidth(2)

    for i, (v, s) in enumerate(zip(acc_vals, acc_stds)):
        ax.text(i, v + s + 0.01, f"{v:.2f}", ha='center', fontsize=7,
                fontweight='bold' if i == best_i else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(0, color='black', lw=0.6)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(axis='y', labelsize=8)

    # ── Col 1: Forgetting curve ────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 1])
    if row == 0:
        ax.set_title(col_titles[1], fontsize=10, fontweight='bold', pad=6)

    for method in methods:
        d = methods_data[method]
        if "curve_mean" not in d:
            continue
        cm = d["curve_mean"]
        cs = d["curve_std"]
        xs = np.arange(1, len(cm) + 1)
        ax.plot(xs, cm, marker='o', markersize=3, label=method,
                color=METHOD_COLORS[method], linewidth=1.8)
        ax.fill_between(xs, cm - cs, cm + cs,
                        alpha=0.12, color=METHOD_COLORS[method])

    ax.set_xlabel("After Task #", fontsize=8)
    ax.set_ylabel("Accuracy on Task 1", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    if row == 0:
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    # ── Col 2: BWT bar chart ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[row, 2])
    if row == 0:
        ax.set_title(col_titles[2], fontsize=10, fontweight='bold', pad=6)
    ax.set_ylabel("BWT", fontsize=8)

    bwt_vals = [methods_data[m]["BWT"][0] for m in methods]
    bwt_stds = [methods_data[m]["BWT"][1] for m in methods]
    bar_colors = [METHOD_COLORS[m] for m in methods]

    bars = ax.bar(x, bwt_vals, yerr=bwt_stds, color=bar_colors,
                  alpha=0.85, capsize=4, edgecolor='white', linewidth=0.8)
    # highlight best (closest to 0 or most positive)
    best_bwt_i = int(np.argmax(bwt_vals))
    bars[best_bwt_i].set_edgecolor('black'); bars[best_bwt_i].set_linewidth(2)

    for i, (v, s) in enumerate(zip(bwt_vals, bwt_stds)):
        offset = s + 0.01 if v >= 0 else -(s + 0.04)
        ax.text(i, v + offset, f"{v:.2f}", ha='center', fontsize=7,
                fontweight='bold' if i == best_bwt_i else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=1.2)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(axis='y', labelsize=8)

# ── Method legend at bottom ────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], color=METHOD_COLORS[m], marker='s', linestyle='None',
           markersize=9, label=m) for m in ALL_METHODS
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

fname = f"{OUT_DIR}/thesis_summary.png"
plt.savefig(fname, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {fname}")
