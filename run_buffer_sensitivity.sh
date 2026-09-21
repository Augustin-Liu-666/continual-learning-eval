#!/usr/bin/env bash
# Buffer size sensitivity analysis for ER
# 4 buffer sizes × 3 datasets × 3 seeds = 36 experiments

PYTHON="${PYTHON:-python3}"
mkdir -p logs

run_exp() {
    local name=$1; shift
    echo "▶️  START: $name  $(date +%H:%M:%S)"
    if PYTHONPATH=. "$@" > "logs/${name}.log" 2>&1; then
        echo "✅ DONE:  $name  $(date +%H:%M:%S)"
    else
        echo "❌ FAIL:  $name  $(date +%H:%M:%S)"
    fi
}

for BUF in 200 500 1000 2000; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  BUFFER SIZE: $BUF  $(date +%H:%M:%S)"
    echo "════════════════════════════════════════"
    for SEED in 0 1 2; do
        run_exp "rmn_er_buf${BUF}_s${SEED}" \
            $PYTHON rotated_mnist/method-er/rotated_mnist_er.py \
            --buffer_size $BUF --seed $SEED

        run_exp "emnist_er_buf${BUF}_s${SEED}" \
            $PYTHON emnist/method_er/emnist_er.py \
            --buffer_size $BUF --seed $SEED

        run_exp "cifar100_er_buf${BUF}_s${SEED}" \
            $PYTHON cifar100/method_er/cifar100_er.py \
            --buffer_size $BUF --seed $SEED --epochs 20
    done
    echo "🎉 Buffer $BUF done  $(date +%H:%M:%S)"
done

echo ""
echo "🏁 ALL BUFFER EXPERIMENTS DONE  $(date +%H:%M:%S)"
