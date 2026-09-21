#!/usr/bin/env bash
PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$ROOT/reports"
mkdir -p "$ROOT/emnist/reports"
mkdir -p "$ROOT/iam/reports"

cd "$ROOT"
export PYTHONPATH="$ROOT"

run_exp() {
    local name="$1"
    local cmd="$2"
    local log="$LOG_DIR/${name}.log"
    echo "========================================"
    echo "▶️  START: $name  $(date '+%H:%M:%S')"
    echo "========================================"
    if eval "$cmd" > "$log" 2>&1; then
        echo "✅ DONE:  $name  $(date '+%H:%M:%S')"
    else
        echo "❌ FAIL:  $name  $(date '+%H:%M:%S')  — see $log"
    fi
}

# ── IAM (8 tasks, 10 epochs, small dataset) ──────────────────────────────────
run_exp "iam_vanilla"  "$PYTHON iam/method_vanilla/iam_vanilla.py"
run_exp "iam_ewc"      "$PYTHON iam/method_ewc/iam_ewc.py"
run_exp "iam_mas"      "$PYTHON iam/method_mas/train_iam_mas.py"
run_exp "iam_ogd"      "$PYTHON iam/method_ogd/train_iam_ogd.py"
run_exp "iam_er"       "$PYTHON iam/method_er/train_iam_er.py"
run_exp "iam_agem"     "$PYTHON iam/method_agem/iam_agem.py"

# ── Rotated MNIST (10 tasks, 5 epochs) ────────────────────────────────────────
run_exp "rmn_vanilla"  "$PYTHON rotated_mnist/method-vanilla/rotated_mnist_vanilla.py --n_tasks 10 --epochs 5"
run_exp "rmn_ewc"      "$PYTHON rotated_mnist/method-ewc/rotated_mnist_ewc.py --n_tasks 10 --epochs 5"
run_exp "rmn_mas"      "$PYTHON rotated_mnist/method-mas/rotated_mnist_mas.py --n_tasks 10 --epochs 5"
run_exp "rmn_ogd"      "$PYTHON rotated_mnist/method-ogd/rotated_mnist_ogd.py --n_tasks 10 --epochs 5"
run_exp "rmn_er"       "$PYTHON rotated_mnist/method-er/rotated_mnist_er.py --n_tasks 10 --epochs 5"
run_exp "rmn_agem"     "$PYTHON rotated_mnist/method-agem/rotated_mnist_agem.py --n_tasks 10 --epochs 5"

# ── EMNIST (5 tasks, 1 epoch, large dataset) ──────────────────────────────────
run_exp "emnist_vanilla" "$PYTHON emnist/method_vanilla/emnist_vanilla.py --n_tasks 5 --epochs 1"
run_exp "emnist_ewc"     "$PYTHON emnist/method_ewc/emnist_ewc.py --n_tasks 5 --epochs 1"
run_exp "emnist_mas"     "$PYTHON emnist/method_mas/mas_trainer.py"
run_exp "emnist_ogd"     "$PYTHON emnist/method_ogd/ogd_trainer.py"
run_exp "emnist_er"      "$PYTHON emnist/method_er/emnist_er.py --n_tasks 5"
run_exp "emnist_agem"    "$PYTHON emnist/method_agem/emnist_agem.py --n_tasks 5 --epochs 1"

echo ""
echo "════════════════════════════════════════"
echo "🎉 ALL EXPERIMENTS FINISHED  $(date '+%H:%M:%S')"
echo "════════════════════════════════════════"
