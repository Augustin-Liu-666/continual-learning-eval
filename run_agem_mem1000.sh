#!/usr/bin/env bash
# Rerun A-GEM with memory_size=1000 for fair cross-method comparison
# (ER main comparison uses buffer_size=1000, so A-GEM should match)
# 3 datasets × 3 seeds = 9 experiments

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

echo "════════════════════════════════════════"
echo "  A-GEM memory=1000 (fair comparison)"
echo "  $(date +%H:%M:%S)"
echo "════════════════════════════════════════"

for SEED in 0 1 2; do
    run_exp "rmn_agem_mem1000_s${SEED}" \
        $PYTHON rotated_mnist/method-agem/rotated_mnist_agem.py \
        --memory_size 1000 --seed $SEED

    run_exp "emnist_agem_mem1000_s${SEED}" \
        $PYTHON emnist/method_agem/emnist_agem.py \
        --memory_size 1000 --seed $SEED

    run_exp "cifar100_agem_mem1000_s${SEED}" \
        $PYTHON cifar100/method_agem/cifar100_agem.py \
        --memory_size 1000 --seed $SEED --epochs 20
done

echo ""
echo "🏁 A-GEM mem1000 DONE  $(date +%H:%M:%S)"
