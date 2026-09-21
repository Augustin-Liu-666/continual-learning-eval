"""
Cost visualization across all datasets and methods.
Generates:
  1. Per-dataset: training time / extra storage growth / RAM bar charts
  2. Cost-vs-ACC scatter plot (efficiency trade-off)
  3. Combined overview
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

# ── cost file registry ────────────────────────────────────────────────────────
COST_FILES = {
    "CIFAR-100": {
        "Vanilla": "cifar100/reports/cost_Vanilla_10tasks_seed{seed}.csv",
        "EWC":     "cifar100/reports/cost_EWC_10tasks_seed{seed}.csv",
        "MAS":     "cifar100/reports/cost_MAS_10tasks_seed{seed}.csv",
        "OGD":     "cifar100/reports/cost_OGD_10tasks_seed{seed}.csv",
        # ER and A-GEM at buffer/memory=1000 for fair comparison
        "ER":      "cifar100/reports/cost_ER_10tasks_buf1000_seed{seed}.csv",
        "A-GEM":   "cifar100/reports/cost_AGEM_10tasks_mem1000_seed{seed}.csv",
    },
    "Rotated MNIST": {
        "Vanilla": "reports/cost_Vanilla_10tasks_seed{seed}.csv",
        "EWC":     "reports/cost_EWC_10tasks_seed{seed}.csv",
        "MAS":     "reports/cost_MAS_10tasks_seed{seed}.csv",
        "OGD":     "reports/cost_OGD_10tasks_seed{seed}.csv",
        # ER and A-GEM at buffer/memory=1000 for fair comparison
        "ER":      "reports/cost_ER_10tasks_buf1000_seed{seed}.csv",
        "A-GEM":   "reports/cost_AGEM_10tasks_mem1000_seed{seed}.csv",
    },
    "EMNIST": {
        "Vanilla": "emnist/reports/cost_Vanilla_5tasks_seed{seed}.csv",
        "EWC":     "emnist/reports/cost_EWC_5tasks_seed{seed}.csv",
        "MAS":     "emnist/reports/mas_cost_log_seed{seed}.csv",
        "OGD":     "emnist/reports/ogd_cost_log_seed{seed}.csv",
        # ER and A-GEM at buffer/memory=1000 for fair comparison
        "ER":      "emnist/reports/cost_ER_5tasks_buf1000_seed{seed}.csv",
        "A-GEM":   "emnist/reports/cost_AGEM_5tasks_mem1000_seed{seed}.csv",
    },
}

# ACC values (multi-seed means) for scatter plot — from summary_multiseed.csv
# CIFAR-100: task-incremental, 20 epochs/task
# Rotated MNIST: 1 epoch/task (fair unified setting)
# EMNIST: 1 epoch/task
ACC_MEANS = {
    "CIFAR-100":     {"Vanilla":0.174, "EWC":0.171, "MAS":0.403, "OGD":0.175, "ER":0.465, "A-GEM":0.414},
    "Rotated MNIST": {"Vanilla":0.596, "EWC":0.607, "MAS":0.639, "OGD":0.591, "ER":0.833, "A-GEM":0.712},
    "EMNIST":        {"Vanilla":0.839, "EWC":0.831, "MAS":0.877, "OGD":0.564, "ER":0.839, "A-GEM":0.843},
}


def load_cost_seeds(tmpl):
    dfs = []
    for s in SEEDS:
        p = tmpl.format(seed=s)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    return dfs


def agg_cost(dfs, col):
    """Mean and std of a column's sum across seeds."""
    vals = [df[col].sum() for df in dfs]
    return np.mean(vals), np.std(vals, ddof=0)


def storage_curve(dfs):
    """Extra storage at each task step → mean, std per task."""
    n = len(dfs[0])
    per_task = []
    for i in range(n):
        vals = [df["extra_storage_mb"].iloc[i] for df in dfs]
        per_task.append((np.mean(vals), np.std(vals, ddof=0)))
    return per_task


# ── Per-dataset cost plots ────────────────────────────────────────────────────

for dataset, method_tmpls in COST_FILES.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset} — Computational Cost Comparison",
                 fontsize=13, fontweight='bold')
    ax_time, ax_store, ax_ram = axes

    methods_present = [m for m in ALL_METHODS if method_tmpls.get(m)]
    x = np.arange(len(methods_present))
    colors = [METHOD_COLORS[m] for m in methods_present]

    time_means, time_stds   = [], []
    store_means, store_stds = [], []
    ram_means,   ram_stds   = [], []
    store_curves = {}

    for method in methods_present:
        dfs = load_cost_seeds(method_tmpls[method])
        if not dfs:
            time_means.append(0); time_stds.append(0)
            store_means.append(0); store_stds.append(0)
            ram_means.append(0); ram_stds.append(0)
            continue
        tm, ts = agg_cost(dfs, "train_time_s")
        sm, ss = [df["extra_storage_mb"].iloc[-1] for df in dfs], 0
        sm, ss = np.mean(sm), np.std(sm, ddof=0)
        rm, rs = agg_cost(dfs, "peak_ram_mb")

        time_means.append(tm); time_stds.append(ts)
        store_means.append(sm); store_stds.append(ss)
        ram_means.append(rm); ram_stds.append(rs)
        store_curves[method] = storage_curve(dfs)

    # Bar: total training time
    bars = ax_time.bar(x, time_means, yerr=time_stds, color=colors,
                       alpha=0.85, capsize=5, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, time_means):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_stds)*0.05,
                     f"{val:.0f}s", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_time.set_xticks(x); ax_time.set_xticklabels(methods_present, rotation=30, ha='right', fontsize=9)
    ax_time.set_ylabel("Total Training Time (s)"); ax_time.set_title("Training Time")
    ax_time.grid(True, alpha=0.3, axis='y')

    # Line: extra storage growth over tasks
    for method in methods_present:
        if method not in store_curves:
            continue
        curve = store_curves[method]
        ms = [c[0] for c in curve]
        ss = [c[1] for c in curve]
        t = np.arange(1, len(ms)+1)
        ax_store.plot(t, ms, marker='o', label=method,
                      color=METHOD_COLORS[method], linewidth=2)
        ax_store.fill_between(t,
                              np.array(ms)-np.array(ss),
                              np.array(ms)+np.array(ss),
                              alpha=0.15, color=METHOD_COLORS[method])
    ax_store.set_xlabel("Task"); ax_store.set_ylabel("Extra Storage (MB)")
    ax_store.set_title("Extra Storage Growth per Task")
    ax_store.legend(fontsize=8); ax_store.grid(True, alpha=0.3)
    ax_store.set_xticks(range(1, len(store_curves[methods_present[0]])+1))

    # Bar: peak RAM
    bars = ax_ram.bar(x, ram_means, yerr=ram_stds, color=colors,
                      alpha=0.85, capsize=5, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, ram_means):
        ax_ram.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ram_stds or [0])*0.05 + 0.01,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_ram.set_xticks(x); ax_ram.set_xticklabels(methods_present, rotation=30, ha='right', fontsize=9)
    ax_ram.set_ylabel("Peak RAM (MB)"); ax_ram.set_title("Peak RAM Usage")
    ax_ram.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f"{OUT_DIR}/cost_{dataset.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ── Cost vs ACC scatter (all datasets combined) ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cost–Performance Trade-off (mean across 3 seeds)",
             fontsize=13, fontweight='bold')

dataset_markers = {"CIFAR-100": "o", "Rotated MNIST": "s", "EMNIST": "^"}

for ax, (cost_col, xlabel, title) in zip(axes, [
    ("train_time_s", "Total Training Time (s)", "Time vs ACC"),
    ("extra_storage_mb", "Final Extra Storage (MB)", "Storage vs ACC"),
    ("peak_ram_mb", "Peak RAM (MB)", "RAM vs ACC"),
]):
    for dataset, method_tmpls in COST_FILES.items():
        for method in ALL_METHODS:
            tmpl = method_tmpls.get(method)
            if not tmpl:
                continue
            dfs = load_cost_seeds(tmpl)
            if not dfs:
                continue

            if cost_col == "extra_storage_mb":
                cost_vals = [df[cost_col].iloc[-1] for df in dfs]
            else:
                cost_vals = [df[cost_col].sum() for df in dfs]

            cost_mean = np.mean(cost_vals)
            acc_mean  = ACC_MEANS[dataset].get(method, 0)

            ax.scatter(cost_mean, acc_mean,
                       color=METHOD_COLORS[method],
                       marker=dataset_markers[dataset],
                       s=120, alpha=0.85, zorder=3,
                       edgecolors='white', linewidths=0.8)
            ax.annotate(f"{method}\n({dataset[:3]})",
                        (cost_mean, acc_mean),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6, color=METHOD_COLORS[method])

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("ACC", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)

# Legend for datasets
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='gray', linestyle='None', markersize=8, label='CIFAR-100'),
    Line2D([0],[0], marker='s', color='gray', linestyle='None', markersize=8, label='Rotated MNIST'),
    Line2D([0],[0], marker='^', color='gray', linestyle='None', markersize=8, label='EMNIST'),
]
axes[2].legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
fname = f"{OUT_DIR}/cost_vs_acc_scatter.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fname}")


# ── Summary cost table ────────────────────────────────────────────────────────

print("\n" + "="*85)
print(f"{'Dataset':<16} {'Method':<8} {'Total time (s)':>14} {'Final storage (MB)':>18} {'Peak RAM (MB)':>14}")
print("="*85)
for dataset, method_tmpls in COST_FILES.items():
    for method in ALL_METHODS:
        tmpl = method_tmpls.get(method)
        if not tmpl:
            continue
        dfs = load_cost_seeds(tmpl)
        if not dfs:
            continue
        tm = np.mean([df["train_time_s"].sum() for df in dfs])
        ts = np.std([df["train_time_s"].sum() for df in dfs], ddof=0)
        sm = np.mean([df["extra_storage_mb"].iloc[-1] for df in dfs])
        ss = np.std([df["extra_storage_mb"].iloc[-1] for df in dfs], ddof=0)
        rm = np.mean([df["peak_ram_mb"].max() for df in dfs])
        rs = np.std([df["peak_ram_mb"].max() for df in dfs], ddof=0)
        print(f"{dataset:<16} {method:<8} {tm:>8.1f}±{ts:<4.1f} {sm:>10.2f}±{ss:<5.2f} {rm:>8.2f}±{rs:.2f}")
print("="*85)
