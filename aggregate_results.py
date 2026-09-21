"""
Aggregate multi-seed results → mean ± std for all metrics.
Run after run_multiseed.sh completes.
Outputs:
  reports/summary_multiseed.csv   — full table
  reports/comparison_multiseed_*.png — plots with error bars
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2]
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_COLORS = {
    "Vanilla": "#e74c3c", "EWC": "#e67e22", "MAS": "#f1c40f",
    "OGD": "#2ecc71",     "ER":  "#3498db", "A-GEM": "#9b59b6",
}

# ── path registry ──────────────────────────────────────────────────────────────
# Each entry: (method_display, path_template)  — {seed} is replaced per seed
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


# ── helpers ────────────────────────────────────────────────────────────────────

def load_acc_matrix(path):
    df = pd.read_csv(path, index_col=0)
    arr = df.values.astype(float)
    if str(df.index[0]).startswith("T"):
        arr = arr.T
    n = arr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            arr[i, j] = np.nan
    return arr


def cl_metrics(R):
    n = R.shape[0]
    diag = np.array([R[i, i] for i in range(n)])
    last_row = np.array([R[n-1, j] for j in range(n)])
    for j in range(n):
        if np.isnan(last_row[j]):
            col = R[:, j]; valid = col[~np.isnan(col)]
            last_row[j] = valid[-1] if len(valid) > 0 else 0.0
    return dict(
        ACC=np.nanmean(last_row),
        BWT=np.nanmean(last_row[:-1] - diag[:-1]),
        LA=np.nanmean(diag),
        Forgetting=np.nanmean(diag[:-1] - last_row[:-1]),
    )


# ── aggregate ──────────────────────────────────────────────────────────────────

records = []

for dataset, exps in EXPERIMENTS.items():
    for method, tmpl in exps:
        seed_metrics = []
        for s in SEEDS:
            path = tmpl.format(seed=s)
            if os.path.exists(path):
                R = load_acc_matrix(path)
                seed_metrics.append(cl_metrics(R))

        if not seed_metrics:
            print(f"[skip] {dataset} / {method} — no seed files found")
            continue

        n_seeds = len(seed_metrics)
        for metric in ["ACC", "BWT", "LA", "Forgetting"]:
            vals = [m[metric] for m in seed_metrics]
            records.append(dict(
                Dataset=dataset, Method=method,
                Metric=metric,
                Mean=np.mean(vals),
                Std=np.std(vals, ddof=0) if n_seeds > 1 else 0.0,
                N=n_seeds,
            ))

df_all = pd.DataFrame(records)
df_all.to_csv(f"{OUT_DIR}/summary_multiseed.csv", index=False)
print(f"Saved: {OUT_DIR}/summary_multiseed.csv")


# ── print summary table ────────────────────────────────────────────────────────

pivot = df_all.pivot_table(index=["Dataset", "Method"], columns="Metric",
                            values=["Mean", "Std"], aggfunc="first")
pivot.columns = [f"{m}_{s}" for s, m in pivot.columns]

print("\n" + "="*90)
print(f"{'Dataset':<18} {'Method':<8} {'ACC (mean±std)':>16} {'BWT':>16} {'LA':>16} {'Forgetting':>16}")
print("="*90)
for (dataset, method), row in pivot.iterrows():
    def fmt(m):
        mean = row.get(f"{m}_Mean", float('nan'))
        std  = row.get(f"{m}_Std",  float('nan'))
        return f"{mean:.3f}±{std:.3f}"
    print(f"{dataset:<18} {method:<8} {fmt('ACC'):>16} {fmt('BWT'):>16} {fmt('LA'):>16} {fmt('Forgetting'):>16}")
print("="*90)


# ── bar charts with error bars ─────────────────────────────────────────────────

all_methods = ["Vanilla", "EWC", "MAS", "OGD", "ER", "A-GEM"]
metric_list = ["ACC", "BWT", "LA", "Forgetting"]
metric_colors = {"ACC": "#2ecc71", "BWT": "#e74c3c", "LA": "#3498db", "Forgetting": "#e67e22"}

for dataset in EXPERIMENTS:
    subset = df_all[df_all.Dataset == dataset]
    if subset.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset} — Multi-seed Results (N={SEEDS})", fontsize=13, fontweight='bold')

    # Left: ACC with error bars
    ax = axes[0]
    methods_present = [m for m in all_methods if not subset[(subset.Method == m) & (subset.Metric == "ACC")].empty]
    x = np.arange(len(methods_present))
    for i, method in enumerate(methods_present):
        row = subset[(subset.Method == method) & (subset.Metric == "ACC")]
        mean, std = row["Mean"].values[0], row["Std"].values[0]
        ax.bar(i, mean, yerr=std, color=metric_colors["ACC"],
               alpha=0.85, capsize=5, edgecolor='white')
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(methods_present, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("ACC"); ax.set_title("ACC (mean ± std)")
    ax.axhline(0, color='black', lw=0.8); ax.grid(True, alpha=0.3, axis='y')

    # Right: all 4 metrics grouped
    ax = axes[1]
    width = 0.18
    for i, metric in enumerate(metric_list):
        vals  = [subset[(subset.Method == m) & (subset.Metric == metric)]["Mean"].values[0]
                 if not subset[(subset.Method == m) & (subset.Metric == metric)].empty else 0
                 for m in methods_present]
        errs  = [subset[(subset.Method == m) & (subset.Metric == metric)]["Std"].values[0]
                 if not subset[(subset.Method == m) & (subset.Metric == metric)].empty else 0
                 for m in methods_present]
        xs = np.arange(len(methods_present)) + i * width
        ax.bar(xs, vals, width, yerr=errs, label=metric,
               color=metric_colors[metric], alpha=0.8, capsize=3)

    ax.set_xticks(np.arange(len(methods_present)) + width * 1.5)
    ax.set_xticklabels(methods_present, rotation=30, ha='right', fontsize=9)
    ax.set_title("All Metrics (mean ± std)"); ax.legend(fontsize=9)
    ax.axhline(0, color='black', lw=0.8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f"{OUT_DIR}/comparison_multiseed_{dataset.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")
