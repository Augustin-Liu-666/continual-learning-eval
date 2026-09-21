# An Empirical Evaluation of Continual Learning under Visual Drift

This repository contains the experimental code and results for a comparative study of continual learning under visual and semantic drift. It evaluates six methods across three benchmarks, using three random seeds and shared training budgets for the main comparisons.

## Benchmarks and methods

| Benchmark | Drift setting | Tasks |
| --- | --- | ---: |
| Rotated MNIST | Gradual geometric rotation | 10 |
| EMNIST | Appearance transformations | 5 |
| CIFAR-100 | Class-incremental semantic drift | 10 |

The evaluated methods are Vanilla sequential training, Elastic Weight Consolidation (EWC), Memory Aware Synapses (MAS), Orthogonal Gradient Descent (OGD), Experience Replay (ER), and Averaged Gradient Episodic Memory (A-GEM).

## Repository structure

```text
.
|-- rotated_mnist/       # Rotated MNIST data loader and methods
|-- emnist/              # EMNIST data loader, models, and methods
|-- cifar100/            # CIFAR-100 data loader, model, and methods
|-- iam/                 # Additional IAM handwriting experiments
|-- reports/             # Aggregated CSV results, figures, and reports
|-- scripts/             # Utilities used to generate report artifacts
|-- tools/               # Dataset-sample and PDF-generation tools
|-- run_*.sh             # Reproducible experiment runners
|-- aggregate_results.py # Multi-seed aggregation
`-- calc_metrics.py      # Continual-learning metrics
```

Raw datasets, caches, local logs, and environment files are intentionally excluded from version control.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Augustin-Liu-666/continual-learning-eval.git
cd continual-learning-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MNIST, EMNIST, and CIFAR-100 are downloaded automatically by `torchvision`. IAM experiments expect the following local dataset layout:

```text
data/iam/
|-- images/
`-- meta/labels.txt
```

Each line of `labels.txt` must contain `image_name,label`.

## Running experiments

Run commands from the repository root so local imports and output paths resolve correctly.

```bash
# Main multi-seed experiments
bash run_multiseed.sh

# Fair-comparison reruns
bash run_rmn_fair.sh
bash run_emnist_fair.sh

# ER and A-GEM memory-size sensitivity analyses
bash run_buffer_sensitivity.sh
bash run_agem_buffer_sensitivity.sh
```

The scripts use `python3` by default. To select another interpreter:

```bash
PYTHON=/path/to/python bash run_multiseed.sh
```

To run one experiment directly:

```bash
PYTHONPATH=. python3 cifar100/method_er/cifar100_er.py \
  --buffer_size 1000 --seed 0 --epochs 20
```

## Results

Per-run accuracy matrices and cost measurements are stored in each benchmark's `reports/` directory. Cross-benchmark summaries, plots, and English progress reports are available in [`reports/`](reports/). The main aggregate table is [`reports/summary_multiseed.csv`](reports/summary_multiseed.csv).

To regenerate the aggregate summaries and figures:

```bash
python3 aggregate_results.py
python3 plot_thesis_summary.py
python3 plot_buffer_sensitivity.py
python3 plot_costs.py
```

## Reproducibility notes

- Main results use random seeds `0`, `1`, and `2`.
- ER and A-GEM use matched memory budgets in the fair comparisons.
- Training time, peak memory, and method-specific storage are recorded by `CostTracker`.
- Dataset files are not redistributed; follow the original dataset licenses and access terms.
