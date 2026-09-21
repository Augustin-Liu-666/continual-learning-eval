#!/usr/bin/env bash
# Re-run ALL EMNIST methods with unified --epochs 1
# Fixes fairness issue: MAS default=3 epochs, OGD default=4 epochs, others=1 epoch
# ER has no --epochs arg (single-pass training by design, equivalent to 1 epoch)
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
echo "  EMNIST — Fair Comparison (all epochs=1)"
echo "  MAS: was default 3  →  now 1"
echo "  OGD: was default 4  →  now 1"
echo "  $(date +%H:%M:%S)"
echo "════════════════════════════════════════════════"

for SEED in 0 1 2; do
    run_exp "emnist_fair_vanilla_s${SEED}" \
        $PYTHON emnist/method_vanilla/emnist_vanilla.py \
        --seed $SEED --epochs 1

    run_exp "emnist_fair_ewc_s${SEED}" \
        $PYTHON emnist/method_ewc/emnist_ewc.py \
        --seed $SEED --epochs 1

    run_exp "emnist_fair_mas_s${SEED}" \
        $PYTHON emnist/method_mas/mas_trainer.py \
        --seed $SEED --epochs 1

    run_exp "emnist_fair_ogd_s${SEED}" \
        $PYTHON emnist/method_ogd/ogd_trainer.py \
        --seed $SEED --epochs 1

    # ER has no --epochs argument; trains one pass per task by design
    run_exp "emnist_fair_er_s${SEED}" \
        $PYTHON emnist/method_er/emnist_er.py \
        --buffer_size 1000 --seed $SEED

    run_exp "emnist_fair_agem_s${SEED}" \
        $PYTHON emnist/method_agem/emnist_agem.py \
        --memory_size 1000 --seed $SEED --epochs 1

    echo "── Seed ${SEED} done  $(date +%H:%M:%S)"
done

echo ""
echo "🏁 EMNIST fair comparison DONE  $(date +%H:%M:%S)"
