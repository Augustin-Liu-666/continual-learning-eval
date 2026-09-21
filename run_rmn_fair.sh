#!/usr/bin/env bash
# Re-run ALL Rotated MNIST methods with unified --epochs 1
# Fixes fairness issue: ER default was 5 epochs, all others were 1
# 6 methods × 3 seeds = 18 experiments

PYTHON="${PYTHON:-python3}"
mkdir -p logs

run_exp() {
    local name=$1; shift
    echo "▶️  START: $name  $(date +%H:%M:%S)"
    if PYTHONPATH=. "$@" > "logs/${name}.log" 2>&1; then
        echo "✅ DONE:  $name  $(date +%H:%M:%S)"
    else
        echo "❌ FAIL:  $name  $(date +%H:%M:%S)"
        tail -5 "logs/${name}.log"
    fi
}

echo "════════════════════════════════════════════════"
echo "  Rotated MNIST — Fair Comparison (epochs=1)"
echo "  $(date +%H:%M:%S)"
echo "════════════════════════════════════════════════"

for SEED in 0 1 2; do
    run_exp "rmn_fair_vanilla_s${SEED}" \
        $PYTHON rotated_mnist/method-vanilla/rotated_mnist_vanilla.py \
        --seed $SEED --epochs 1

    run_exp "rmn_fair_ewc_s${SEED}" \
        $PYTHON rotated_mnist/method-ewc/rotated_mnist_ewc.py \
        --seed $SEED --epochs 1

    run_exp "rmn_fair_mas_s${SEED}" \
        $PYTHON rotated_mnist/method-mas/rotated_mnist_mas.py \
        --seed $SEED --epochs 1

    run_exp "rmn_fair_ogd_s${SEED}" \
        $PYTHON rotated_mnist/method-ogd/rotated_mnist_ogd.py \
        --seed $SEED --epochs 1

    run_exp "rmn_fair_er_s${SEED}" \
        $PYTHON rotated_mnist/method-er/rotated_mnist_er.py \
        --buffer_size 1000 --seed $SEED --epochs 1

    run_exp "rmn_fair_agem_s${SEED}" \
        $PYTHON rotated_mnist/method-agem/rotated_mnist_agem.py \
        --memory_size 1000 --seed $SEED --epochs 1
done

echo ""
echo "🏁 RMN fair comparison DONE  $(date +%H:%M:%S)"
