"""
Line chart visualization with mean ± std shading across 3 seeds.
Generates:
  1. Final accuracy per task (after all training) — one line per method
  2. Accuracy on Task-1 over time (forgetting curve) — one line per method
  3. Combined 2×3 grid for all datasets
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEEDS = [0, 1, 2]
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_COLORS = {
    "Vanilla": "#e74c3c", "EWC": "#e67e22", "MAS": "#f1c40f",
    "OGD":     "#2ecc71", "ER":  "#3498db", "A-GEM": "#9b59b6",
}
ALL_METHODS = ["Vanilla", "EWC", "MAS", "OGD", "ER", "A-GEM"]

EXPERIMENTS = {
    "CIFAR-100": [
        ("Vanilla", "cifar100/reports/acc_matrix_Vanilla_10tasks_seed{seed}.csv"),
        ("EWC",     "cifar100/reports/acc_matrix_EWC_10tasks_seed{seed}.csv"),
        ("MAS",     "cifar100/reports/acc_matrix_MAS_10tasks_seed{seed}.csv"),
        ("OGD",     "cifar100/reports/acc_matrix_OGD_10tasks_seed{seed}.csv"),
        # ER and A-GEM at buffer/memory=1000 for fair comparison
        ("ER",      "cifar100/reports/acc_matrix_ER_10tasks_buf1000_seed{seed}.csv"),
        ("A-GEM",   "cifar100/reports/acc_matrix_AGEM_10tasks_mem1000_seed{seed}.csv"),
    ],
    "Rotated MNIST": [
        ("Vanilla", "reports/acc_matrix_Vanilla_10tasks_seed{seed}.csv"),
        ("EWC",     "reports/acc_matrix_EWC_10tasks_seed{seed}.csv"),
        ("MAS",     "reports/acc_matrix_MAS_10tasks_seed{seed}.csv"),
        ("OGD",     "reports/acc_matrix_OGD_10tasks_seed{seed}.csv"),
        # ER and A-GEM at buffer/memory=1000 for fair comparison
        ("ER",      "reports/acc_matrix_ER_10tasks_buf1000_seed{seed}.csv"),
        ("A-GEM",   "reports/acc_matrix_AGEM_10tasks_mem1000_seed{seed}.csv"),
    ],
    "EMNIST": [
        ("Vanilla", "emnist/reports/acc_matrix_EMNIST_Vanilla_5tasks_seed{seed}.csv"),
        ("EWC",     "emnist/reports/acc_matrix_EMNIST_EWC_5tasks_seed{seed}.csv"),
        ("MAS",     "emnist/reports/mas_acc_log_seed{seed}.csv"),
        ("OGD",     "emnist/reports/ogd_acc_log_seed{seed}.csv"),
        # ER and A-GEM at buffer/memory=1000 for fair comparison
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


def load_all_seeds(tmpl):
    """Load acc matrices for all seeds. Returns list of arrays."""
    matrices = []
    for s in SEEDS:
        p = tmpl.format(seed=s)
        if os.path.exists(p):
            matrices.append(load_matrix(p))
    return matrices


def final_acc_per_task(matrices):
    """For each task j, take the last valid accuracy across seeds → mean, std."""
    n = matrices[0].shape[0]
    per_task = []
    for j in range(n):
        vals = []
        for R in matrices:
            col = R[:, j]
            valid = col[~np.isnan(col)]
            vals.append(valid[-1] if len(valid) > 0 else 0.0)
        per_task.append(vals)
    means = [np.mean(v) for v in per_task]
    stds  = [np.std(v, ddof=0) for v in per_task]
    return np.array(means), np.array(stds)


def task1_forgetting_curve(matrices):
    """Accuracy on Task 0 after each training step → mean, std across seeds."""
    n = matrices[0].shape[0]
    per_step = []
    for i in range(n):
        vals = []
        for R in matrices:
            v = R[i, 0]
            if np.isnan(v):
                # fill with last known
                col = R[:i+1, 0]
                valid = col[~np.isnan(col)]
                v = valid[-1] if len(valid) > 0 else 0.0
            vals.append(v)
        per_step.append(vals)
    means = [np.mean(v) for v in per_step]
    stds  = [np.std(v, ddof=0) for v in per_step]
    return np.array(means), np.array(stds)


def acc_over_training(matrices):
    """Average accuracy on all seen tasks after each training step → mean, std."""
    n = matrices[0].shape[0]
    per_step = []
    for i in range(n):
        vals = []
        for R in matrices:
            row_vals = [R[i, j] for j in range(i+1) if not np.isnan(R[i, j])]
            vals.append(np.mean(row_vals) if row_vals else 0.0)
        per_step.append(vals)
    means = [np.mean(v) for v in per_step]
    stds  = [np.std(v, ddof=0) for v in per_step]
    return np.array(means), np.array(stds)


# ── Per-dataset detailed plots ─────────────────────────────────────────────────

for dataset, exps in EXPERIMENTS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset} — Multi-seed Line Charts (seeds 0/1/2)",
                 fontsize=14, fontweight='bold')

    ax0, ax1, ax2 = axes

    for method, tmpl in exps:
        matrices = load_all_seeds(tmpl)
        if not matrices:
            continue
        color = METHOD_COLORS.get(method, 'gray')
        n = matrices[0].shape[0]
        xs = np.arange(1, n + 1)

        # — Plot 1: Final accuracy per task —
        means, stds = final_acc_per_task(matrices)
        ax0.plot(xs, means, marker='o', label=method, color=color, linewidth=2)
        ax0.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color)

        # — Plot 2: Task-1 forgetting curve —
        means2, stds2 = task1_forgetting_curve(matrices)
        ax1.plot(xs, means2, marker='s', label=method, color=color, linewidth=2)
        ax1.fill_between(xs, means2 - stds2, means2 + stds2, alpha=0.15, color=color)

        # — Plot 3: Average accuracy on all seen tasks —
        means3, stds3 = acc_over_training(matrices)
        ax2.plot(xs, means3, marker='^', label=method, color=color, linewidth=2)
        ax2.fill_between(xs, means3 - stds3, means3 + stds3, alpha=0.15, color=color)

    for ax, title, ylabel in [
        (ax0, "Final Accuracy per Task\n(after all training)", "Accuracy"),
        (ax1, "Task-1 Accuracy over Training\n(forgetting curve)", "Accuracy on Task 1"),
        (ax2, "Avg Accuracy on Seen Tasks\n(over training steps)", "Avg Accuracy"),
    ]:
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, n + 1))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{OUT_DIR}/linechart_{dataset.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ── Combined 3-dataset overview (1 row per dataset, 3 plots) ──────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Continual Learning — All Datasets & Methods (mean ± std, 3 seeds)",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

plot_titles = [
    "Final Accuracy per Task",
    "Task-1 Forgetting Curve",
    "Avg Accuracy over Training",
]

for row, (dataset, exps) in enumerate(EXPERIMENTS.items()):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"{dataset}\n{plot_titles[col]}", fontsize=9, fontweight='bold')

        for method, tmpl in exps:
            matrices = load_all_seeds(tmpl)
            if not matrices:
                continue
            color = METHOD_COLORS.get(method, 'gray')
            n = matrices[0].shape[0]
            xs = np.arange(1, n + 1)

            if col == 0:
                means, stds = final_acc_per_task(matrices)
            elif col == 1:
                means, stds = task1_forgetting_curve(matrices)
            else:
                means, stds = acc_over_training(matrices)

            ax.plot(xs, means, marker='o', markersize=3, label=method,
                    color=color, linewidth=1.5)
            ax.fill_between(xs, means - stds, means + stds,
                            alpha=0.12, color=color)

        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, n + 1))
        ax.set_xlabel("Task", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.legend(fontsize=6, loc='upper right')

fname = f"{OUT_DIR}/linechart_all_datasets.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fname}")
