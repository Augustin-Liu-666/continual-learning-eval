# Progress Report: Continual Learning under Visual Drift

## 1. Project Aim

This project investigates how different continual learning methods perform under visual drift. The main research question is:

**Compared with regularisation-based and optimisation-based methods, can replay-based methods more effectively reduce catastrophic forgetting under visual distribution shift?**

The current study compares six methods:

| Category | Method |
|---|---|
| Baseline | Vanilla fine-tuning |
| Regularisation-based | EWC, MAS |
| Optimisation-based | OGD |
| Replay-based | ER, A-GEM |

The experiments are conducted on three datasets: Rotated MNIST, EMNIST, and CIFAR-100.

## 2. Work Completed

So far, I have completed the following:

- Implemented and organised the experimental pipelines for Rotated MNIST, EMNIST, and CIFAR-100.
- Ran the main comparison across Vanilla, EWC, MAS, OGD, ER, and A-GEM.
- Repeated the experiments with three random seeds: 0, 1, and 2.
- Computed continual learning metrics: ACC, BWT, LA, and Forgetting.
- Added computational cost tracking for training time and extra storage.
- Ran a buffer/memory-size sensitivity study for ER and A-GEM with sizes 200, 500, 1000, and 2000.
- Generated the main result figures for the report and Overleaf draft.

## 3. Experimental Setup

### Rotated MNIST

Rotated MNIST contains 10 tasks, where the digit images are rotated from 0 degrees to 90 degrees. This setting represents smooth geometric drift with a shared label space. All methods are trained for 1 epoch per task using an MLP.

### EMNIST

EMNIST contains 5 tasks with the same 36-class label space, but each task applies a different visual transformation such as rotation, blur, brightness jitter, or mixed augmentation. This setting represents visual style drift with strong feature sharing across tasks. All methods are trained for 1 epoch per task using an MLP.

### CIFAR-100

CIFAR-100 contains 10 tasks, each with 10 fine-grained object classes. This setting represents large semantic drift. The experiments use task-incremental evaluation, and all methods are trained for 20 epochs per task using a CNN.

## 4. Main Results

### Rotated MNIST

ER achieves the best final accuracy and the lowest forgetting.

| Method | ACC | BWT | LA | Forgetting |
|---|---:|---:|---:|---:|
| Vanilla | 0.596 ± 0.003 | -0.331 ± 0.003 | 0.894 ± 0.000 | 0.331 ± 0.003 |
| EWC | 0.607 ± 0.002 | -0.317 ± 0.002 | 0.892 ± 0.000 | 0.317 ± 0.002 |
| MAS | 0.639 ± 0.002 | -0.258 ± 0.003 | 0.870 ± 0.001 | 0.258 ± 0.003 |
| OGD | 0.591 ± 0.010 | -0.333 ± 0.010 | 0.891 ± 0.000 | 0.333 ± 0.010 |
| ER | 0.833 ± 0.002 | -0.059 ± 0.002 | 0.886 ± 0.001 | 0.059 ± 0.002 |
| A-GEM | 0.712 ± 0.003 | -0.202 ± 0.003 | 0.893 ± 0.001 | 0.202 ± 0.003 |

ER outperforms the best regularisation method, MAS, by approximately 19.4 percentage points in ACC.

### EMNIST

EMNIST behaves differently from the other datasets. Most methods show positive BWT, suggesting positive transfer rather than catastrophic forgetting.

| Method | ACC | BWT | LA | Forgetting |
|---|---:|---:|---:|---:|
| Vanilla | 0.839 ± 0.001 | +0.034 ± 0.001 | 0.812 ± 0.001 | -0.034 ± 0.001 |
| EWC | 0.831 ± 0.001 | +0.029 ± 0.001 | 0.808 ± 0.001 | -0.029 ± 0.001 |
| MAS | 0.877 ± 0.001 | +0.020 ± 0.002 | 0.861 ± 0.001 | -0.020 ± 0.002 |
| OGD | 0.564 ± 0.006 | +0.069 ± 0.009 | 0.509 ± 0.006 | -0.069 ± 0.009 |
| ER | 0.839 ± 0.002 | +0.034 ± 0.001 | 0.812 ± 0.001 | -0.034 ± 0.001 |
| A-GEM | 0.843 ± 0.001 | +0.036 ± 0.001 | 0.814 ± 0.001 | -0.036 ± 0.001 |

MAS performs best on EMNIST, reaching 87.7% ACC. ER does not provide a clear advantage because this benchmark shows little or no forgetting.

### CIFAR-100

CIFAR-100 is the most challenging setting due to strong semantic drift. ER again achieves the highest final accuracy.

| Method | ACC | BWT | LA | Forgetting |
|---|---:|---:|---:|---:|
| Vanilla | 0.174 ± 0.007 | -0.511 ± 0.021 | 0.634 ± 0.021 | 0.511 ± 0.021 |
| EWC | 0.171 ± 0.008 | -0.451 ± 0.074 | 0.577 ± 0.070 | 0.451 ± 0.074 |
| MAS | 0.403 ± 0.017 | -0.091 ± 0.014 | 0.484 ± 0.012 | 0.091 ± 0.014 |
| OGD | 0.175 ± 0.005 | -0.510 ± 0.054 | 0.635 ± 0.045 | 0.510 ± 0.054 |
| ER | 0.465 ± 0.010 | -0.215 ± 0.008 | 0.658 ± 0.004 | 0.215 ± 0.008 |
| A-GEM | 0.414 ± 0.011 | -0.279 ± 0.034 | 0.665 ± 0.031 | 0.279 ± 0.034 |

ER reaches 46.5% ACC, followed by A-GEM at 41.4% and MAS at 40.3%. Vanilla, EWC, and OGD remain close to 17%, showing that they struggle under strong semantic drift.

## 5. Buffer/Memory Sensitivity

I also evaluated ER and A-GEM with memory sizes 200, 500, 1000, and 2000.

For ER:

| Buffer Size | CIFAR-100 ACC | Rotated MNIST ACC | EMNIST ACC |
|---:|---:|---:|---:|
| 200 | 0.337 ± 0.012 | 0.738 ± 0.006 | 0.839 ± 0.002 |
| 500 | 0.414 ± 0.009 | 0.799 ± 0.004 | 0.839 ± 0.002 |
| 1000 | 0.465 ± 0.010 | 0.833 ± 0.002 | 0.839 ± 0.002 |
| 2000 | 0.538 ± 0.012 | 0.859 ± 0.003 | 0.839 ± 0.002 |

The results show that ER improves consistently as buffer size increases on Rotated MNIST and CIFAR-100. In contrast, A-GEM does not reliably improve with larger memory and becomes unstable on CIFAR-100 at memory size 2000.

## 6. Cost Analysis

The current cost results show that ER provides a strong trade-off between performance and storage cost. On CIFAR-100, ER achieves the best ACC while requiring only 11.73 MB of extra storage, compared with 251.48 MB for EWC, 125.74 MB for OGD, and 117.26 MB for A-GEM.

| Dataset | Method | Training Time | Extra Storage |
|---|---|---:|---:|
| CIFAR-100 | Vanilla | 243 ± 2 s | 0.00 MB |
| CIFAR-100 | EWC | 694 ± 3 s | 251.48 MB |
| CIFAR-100 | MAS | 327 ± 2 s | 25.15 MB |
| CIFAR-100 | OGD | 283 ± 2 s | 125.74 MB |
| CIFAR-100 | ER | 461 ± 0 s | 11.73 MB |
| CIFAR-100 | A-GEM | 1139 ± 1 s | 117.26 MB |

## 7. Current Interpretation

The current results suggest that replay-based methods are the most effective under genuine visual drift. ER is especially reliable because it directly revisits old samples, which gives it a stable way to preserve previous input-output mappings. Regularisation-based methods depend on parameter-importance estimates, which appear less reliable when the input distribution changes strongly. OGD is often too restrictive and can reduce plasticity.

EMNIST is an important exception. Since the tasks share the same label space and visual structure, later tasks can improve earlier representations. Therefore, EMNIST is better interpreted as a positive-transfer or plasticity benchmark rather than a pure forgetting benchmark.

## 8. Next Steps

The next steps are:

- Polish the Overleaf draft and make the experimental setup clearer.
- Improve figure captions and explicitly explain the evaluation protocol.
- Strengthen the discussion of why ER performs better under visual drift.
- Clarify the special interpretation of EMNIST.
- Incorporate supervisor feedback on the framing and analysis.

## 9. Suggested Figures to Share

The most important figures are:

- `reports/thesis_summary.png`
- `reports/linechart_all_datasets.png`
- `reports/buffer_sensitivity_by_dataset.png`
- `reports/cost_vs_acc_scatter.png`
- `reports/comparison_multiseed_rotated_mnist.png`
- `reports/comparison_multiseed_cifar-100.png`
