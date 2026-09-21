#!/usr/bin/env bash
# Multi-seed runner: seeds 0, 1, 2 for all 18 experiments
# Usage: bash run_multiseed.sh

PYTHON="${PYTHON:-python3}"
mkdir -p logs

run_exp() {
    local name=$1; shift
    echo "▶️  START: $name  $(date +%H:%M:%S)"
    if PYTHONPATH=. "$@" > "logs/${name}.log" 2>&1; then
        echo "✅ DONE:  $name  $(date +%H:%M:%S)"
    else
        echo "❌ FAIL:  $name  $(date +%H:%M:%S)" >&2
    fi
}

for SEED in 0 1 2; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  SEED $SEED  $(date +%H:%M:%S)"
    echo "════════════════════════════════════════"

    # ── Rotated MNIST ──────────────────────────
    run_exp "rmn_vanilla_s${SEED}"  $PYTHON rotated_mnist/method-vanilla/rotated_mnist_vanilla.py --seed $SEED
    run_exp "rmn_ewc_s${SEED}"      $PYTHON rotated_mnist/method-ewc/rotated_mnist_ewc.py         --seed $SEED
    run_exp "rmn_mas_s${SEED}"      $PYTHON rotated_mnist/method-mas/rotated_mnist_mas.py         --seed $SEED
    run_exp "rmn_ogd_s${SEED}"      $PYTHON rotated_mnist/method-ogd/rotated_mnist_ogd.py         --seed $SEED
    run_exp "rmn_er_s${SEED}"       $PYTHON rotated_mnist/method-er/rotated_mnist_er.py           --seed $SEED
    run_exp "rmn_agem_s${SEED}"     $PYTHON rotated_mnist/method-agem/rotated_mnist_agem.py       --seed $SEED

    # ── EMNIST ────────────────────────────────
    run_exp "emnist_vanilla_s${SEED}" $PYTHON emnist/method_vanilla/emnist_vanilla.py  --seed $SEED
    run_exp "emnist_ewc_s${SEED}"     $PYTHON emnist/method_ewc/emnist_ewc.py          --seed $SEED
    run_exp "emnist_mas_s${SEED}"     $PYTHON emnist/method_mas/mas_trainer.py         --seed $SEED
    run_exp "emnist_ogd_s${SEED}"     $PYTHON emnist/method_ogd/ogd_trainer.py         --seed $SEED
    run_exp "emnist_er_s${SEED}"      $PYTHON emnist/method_er/emnist_er.py            --seed $SEED
    run_exp "emnist_agem_s${SEED}"    $PYTHON emnist/method_agem/emnist_agem.py        --seed $SEED

    # ── CIFAR-100 ─────────────────────────────
    run_exp "cifar100_vanilla_s${SEED}" $PYTHON cifar100/method_vanilla/cifar100_vanilla.py --seed $SEED --epochs 20
    run_exp "cifar100_ewc_s${SEED}"     $PYTHON cifar100/method_ewc/cifar100_ewc.py         --seed $SEED --epochs 20
    run_exp "cifar100_mas_s${SEED}"     $PYTHON cifar100/method_mas/cifar100_mas.py         --seed $SEED --epochs 20
    run_exp "cifar100_ogd_s${SEED}"     $PYTHON cifar100/method_ogd/cifar100_ogd.py         --seed $SEED --epochs 20
    run_exp "cifar100_er_s${SEED}"      $PYTHON cifar100/method_er/cifar100_er.py           --seed $SEED --epochs 20
    run_exp "cifar100_agem_s${SEED}"    $PYTHON cifar100/method_agem/cifar100_agem.py       --seed $SEED --epochs 20

    echo "🎉 Seed $SEED done  $(date +%H:%M:%S)"
done

echo ""
echo "🏁 ALL SEEDS DONE  $(date +%H:%M:%S)"
