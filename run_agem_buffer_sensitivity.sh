#!/usr/bin/env bash
# A-GEM memory size sensitivity analysis
# 4 memory sizes × 3 datasets × 3 seeds = 36 experiments

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

for MEM in 200 500 1000 2000; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  MEMORY SIZE: $MEM  $(date +%H:%M:%S)"
    echo "════════════════════════════════════════"
    for SEED in 0 1 2; do
        run_exp "rmn_agem_mem${MEM}_s${SEED}" \
            $PYTHON rotated_mnist/method-agem/rotated_mnist_agem.py \
            --memory_size $MEM --seed $SEED

        run_exp "emnist_agem_mem${MEM}_s${SEED}" \
            $PYTHON emnist/method_agem/emnist_agem.py \
            --memory_size $MEM --seed $SEED

        run_exp "cifar100_agem_mem${MEM}_s${SEED}" \
            $PYTHON cifar100/method_agem/cifar100_agem.py \
            --memory_size $MEM --seed $SEED --epochs 20
    done
    echo "🎉 Memory $MEM done  $(date +%H:%M:%S)"
done

echo ""
echo "🏁 ALL A-GEM MEMORY EXPERIMENTS DONE  $(date +%H:%M:%S)"
