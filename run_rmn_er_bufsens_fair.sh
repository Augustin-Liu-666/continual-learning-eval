#!/usr/bin/env bash
# Re-run ER buffer sensitivity for Rotated MNIST with epochs=1 (fair)
# 4 buffer sizes × 3 seeds = 12 experiments

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

echo "════════════════════════════════════════════════════"
echo "  ER Buffer Sensitivity — Rotated MNIST (epochs=1)"
echo "  $(date +%H:%M:%S)"
echo "════════════════════════════════════════════════════"

for BUF in 200 500 1000 2000; do
    for SEED in 0 1 2; do
        run_exp "rmn_er_buf${BUF}_s${SEED}_fair" \
            $PYTHON rotated_mnist/method-er/rotated_mnist_er.py \
            --buffer_size $BUF --seed $SEED --epochs 1
    done
    echo "🎉 Buffer $BUF done  $(date +%H:%M:%S)"
done

echo ""
echo "🏁 RMN ER buffer sensitivity (fair) DONE  $(date +%H:%M:%S)"
