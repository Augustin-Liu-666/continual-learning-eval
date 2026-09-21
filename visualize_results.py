"""
Cross-method comparison visualization for all 3 datasets.
Run at any time — skips missing experiments automatically.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_acc_matrix(path):
    """
    Supports two CSV formats:
      - acc_matrix: rows=After_Task_i, cols=Task_j  → R[i,j] = acc on task j after training i
      - acc_log:    rows=T_k,          cols=T_t      → R[k,t] = acc on task k after training t
    Returns a 2D numpy array where R[i,j] = acc on task j after training task i.
    Upper triangle (future tasks not yet trained) is set to NaN.
    """
    df = pd.read_csv(path, index_col=0)
    arr = df.values.astype(float)

    # Detect orientation: acc_log has "T0/T1/..." row labels
    if str(df.index[0]).startswith("T"):
        # acc_log format: R[k,t] → transpose to get R[i,j]
        arr = arr.T

    n = arr.shape[0]
    # Mask upper triangle (j > i): task j was not yet trained when we were at task i
    for i in range(n):
        for j in range(i + 1, n):
            arr[i, j] = np.nan

    return arr


def cl_metrics(R):
    """Compute ACC, BWT, LA, Forgetting from R[i,j] = acc on task j after training task i."""
    n = R.shape[0]
    # fill lower triangle only
    diag = np.array([R[i, i] for i in range(n)])

    # Last row: final accuracy on each task
    last_row = np.array([R[n-1, j] for j in range(n)])
    # Fill NaN in last row with nearest non-nan
    for j in range(n):
        if np.isnan(last_row[j]):
            col = R[:, j]
            valid = col[~np.isnan(col)]
            last_row[j] = valid[-1] if len(valid) > 0 else 0.0

    ACC = np.nanmean(last_row)
    BWT = np.nanmean(last_row[:-1] - diag[:-1])
    LA  = np.nanmean(diag)
    Forgetting = np.nanmean(diag[:-1] - last_row[:-1])
    return dict(ACC=ACC, BWT=BWT, LA=LA, Forgetting=Forgetting)


# ── data registry ─────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "CIFAR-100": [
        ("Vanilla", "cifar100/reports/acc_matrix_Vanilla_10tasks.csv"),
        ("EWC",     "cifar100/reports/acc_matrix_EWC_10tasks.csv"),
        ("MAS",     "cifar100/reports/acc_matrix_MAS_10tasks.csv"),
        ("OGD",     "cifar100/reports/acc_matrix_OGD_10tasks.csv"),
        ("ER",      "cifar100/reports/acc_matrix_ER_10tasks.csv"),
        ("A-GEM",   "cifar100/reports/acc_matrix_AGEM_10tasks.csv"),
    ],
    "Rotated MNIST": [
        ("Vanilla", "reports/acc_matrix_Vanilla_10tasks.csv"),
        ("EWC",     "reports/acc_matrix_EWC_10tasks.csv"),
        ("MAS",     "reports/acc_matrix_MAS_10tasks.csv"),
        ("OGD",     "reports/acc_matrix_OGD_10tasks.csv"),
        ("ER",      "reports/acc_matrix_ER_10tasks.csv"),
        ("A-GEM",   "reports/acc_matrix_AGEM_10tasks.csv"),
    ],
    "EMNIST": [
        ("Vanilla", "emnist/reports/acc_matrix_EMNIST_Vanilla_5tasks.csv"),
        ("EWC",     "emnist/reports/acc_matrix_EMNIST_EWC_5tasks.csv"),
        ("MAS",     "emnist/reports/mas_acc_log.csv"),
        ("OGD",     "emnist/reports/ogd_acc_log.csv"),
        ("ER",      "emnist/reports/acc_matrix_EMNIST_ER_5tasks.csv"),
        ("A-GEM",   "emnist/reports/acc_matrix_EMNIST_AGEM_5tasks.csv"),
    ],
    # IAM removed — insufficient data quality for rigorous CL evaluation
    # (2,802 classes, median 1 sample/class, replaced by CIFAR-100)
}

METHOD_COLORS = {
    "Vanilla": "#e74c3c",
    "EWC":     "#e67e22",
    "MAS":     "#f1c40f",
    "OGD":     "#2ecc71",
    "ER":      "#3498db",
    "A-GEM":   "#9b59b6",
}

# ── per-dataset detail plots ──────────────────────────────────────────────────

for dataset, exps in EXPERIMENTS.items():
    available = [(name, path) for name, path in exps if os.path.exists(path)]
    if not available:
        print(f"[skip] {dataset} — no results yet")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset} — Continual Learning Comparison", fontsize=14, fontweight='bold')

    # ── left: accuracy-per-task line plot ──────────────────────────────────
    ax = axes[0]
    for name, path in available:
        R = load_acc_matrix(path)
        n = R.shape[0]
        final = []
        for j in range(n):
            col = R[:, j]
            valid = col[~np.isnan(col)]
            final.append(valid[-1] if len(valid) > 0 else np.nan)
        ax.plot(range(1, n+1), final, marker='o', label=name,
                color=METHOD_COLORS.get(name, 'gray'), linewidth=2)

    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Final Accuracy", fontsize=11)
    ax.set_title("Final Accuracy per Task (after all training)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── right: metrics bar chart ───────────────────────────────────────────
    ax = axes[1]
    metrics_data = {}
    for name, path in available:
        R = load_acc_matrix(path)
        metrics_data[name] = cl_metrics(R)

    methods = list(metrics_data.keys())
    metric_names = ["ACC", "BWT", "LA", "Forgetting"]
    x = np.arange(len(methods))
    width = 0.2

    for i, m in enumerate(metric_names):
        vals = [metrics_data[name][m] for name in methods]
        bars = ax.bar(x + i*width, vals, width, label=m,
                      color=["#2ecc71","#e74c3c","#3498db","#e67e22"][i], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("CL Metrics per Method", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    fname = f"{OUT_DIR}/comparison_{dataset.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ── combined ACC comparison across all datasets ───────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Final Accuracy (ACC) — All Datasets & Methods", fontsize=14, fontweight='bold')

all_methods = ["Vanilla", "EWC", "MAS", "OGD", "ER", "A-GEM"]

for ax, (dataset, exps) in zip(axes, EXPERIMENTS.items()):
    accs = {}
    for name, path in exps:
        if os.path.exists(path):
            R = load_acc_matrix(path)
            accs[name] = cl_metrics(R)["ACC"]
        else:
            accs[name] = None

    names = [m for m in all_methods]
    vals  = [accs.get(m) for m in names]
    colors = [METHOD_COLORS[m] if vals[i] is not None else "#dddddd" for i, m in enumerate(names)]
    numeric = [v if v is not None else 0 for v in vals]

    bars = ax.bar(names, numeric, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2%}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                    "pending", ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.set_ylabel("ACC" if dataset == "Rotated MNIST" else "")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fname = f"{OUT_DIR}/comparison_all_datasets.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fname}")

# ── print summary table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Dataset':<18} {'Method':<10} {'ACC':>7} {'BWT':>8} {'LA':>7} {'Forgetting':>11}")
print("="*70)
for dataset, exps in EXPERIMENTS.items():
    for name, path in exps:
        if os.path.exists(path):
            R = load_acc_matrix(path)
            m = cl_metrics(R)
            print(f"{dataset:<18} {name:<10} {m['ACC']:>7.3f} {m['BWT']:>8.3f} {m['LA']:>7.3f} {m['Forgetting']:>11.3f}")
        else:
            print(f"{dataset:<18} {name:<10} {'(pending)':>7}")
print("="*70)
