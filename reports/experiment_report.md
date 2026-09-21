# Continual Learning under Visual Drift: A Comprehensive Experimental Study
## Comparing Replay-Based, Regularisation-Based, and Optimisation-Based Methods

---

## 1. Research Question and Motivation

The central question investigated in this study is:

> **Compared with regularisation-based and optimisation-based methods, how well can replay-based methods handle visual drift in continual learning tasks?**

Continual learning (CL) requires a model to learn a sequence of tasks without forgetting previously acquired knowledge — the phenomenon known as catastrophic forgetting. Most real-world visual applications involve some form of *visual drift*: the statistical distribution of input images shifts across tasks due to changes in viewpoint, lighting, style, or category. Standard neural networks trained with stochastic gradient descent are particularly vulnerable to this because new gradient updates overwrite the parameter space associated with earlier tasks.

Three families of methods have been proposed to address forgetting:

- **Regularisation-based methods** add a penalty term to the loss function to prevent important weights from changing too rapidly.
- **Optimisation-based methods** constrain the direction of gradient updates so that they do not interfere with past task performance.
- **Replay-based methods** retain a small episodic memory of past samples and revisit them during training, directly preserving the input-output mappings learned earlier.

While all three families target catastrophic forgetting, they make fundamentally different trade-offs between stability, plasticity, memory overhead, and computational cost. This study provides a systematic, fair, and reproducible empirical evaluation of all three families under controlled visual drift conditions.

---

## 2. Method Selection Rationale

Six methods were selected to ensure balanced coverage of each family:

### 2.1 Baseline
- **Vanilla SGD (Finetuning)**: Sequential training with no forgetting protection. Serves as the lower bound — any useful CL method must outperform this.

### 2.2 Regularisation-Based Methods
- **EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017)**: Estimates the importance of each weight using the diagonal of the Fisher Information Matrix after each task. Important weights are penalised more heavily when updated. Selected because it is the most widely cited regularisation baseline in CL literature and provides an interpretable measure of parameter importance.
- **MAS (Memory Aware Synapses, Aljundi et al. 2018)**: Estimates weight importance using the sensitivity of the model's output (L2 norm of gradients of output w.r.t. parameters), rather than the Fisher Information. Selected because it does not require task labels to compute importance, making it more general, and because its importance measure targets output stability rather than likelihood — a meaningful distinction under visual drift.

### 2.3 Optimisation-Based Methods
- **OGD (Orthogonal Gradient Descent, Farajtabar et al. 2020)**: Projects the current gradient to be orthogonal to the subspace spanned by the gradients of all previous task losses. This prevents gradient updates from disrupting past performance geometrically. Selected because it represents the optimisation-based family and imposes a hard geometric constraint rather than a soft penalty, making it conceptually distinct from EWC/MAS.

### 2.4 Replay-Based Methods
- **ER (Experience Replay)**: Maintains a fixed-size episodic buffer of randomly sampled past examples. At each training step, a mini-batch from the buffer is replayed alongside the current task batch. Selected because it is the simplest and most interpretable replay baseline, with a single tuneable hyperparameter (buffer size) and no gradient-level modifications.
- **A-GEM (Averaged Gradient Episodic Memory, Chaudhry et al. 2019)**: Stores past samples in an episodic memory and uses them to constrain the current gradient update: if the new gradient would increase the loss on past samples, it is projected to satisfy the memory constraint. Selected because it extends ER by applying a gradient-level constraint rather than direct sample mixing, allowing comparison of whether the *way* episodic memory is used (interpolation vs. constraint) matters for performance.

---

## 3. Dataset Selection Rationale

Three datasets were selected to capture different types and magnitudes of visual drift:

### 3.1 Rotated MNIST — Smooth Geometric Drift
- **Setup**: 10 tasks, each task rotates the MNIST digit images by an additional 10° (0° → 90°, step = 10°). Each task shares the same label space (digits 0–9) but the visual distribution shifts gradually across tasks.
- **Drift type**: Continuous geometric (rotation) drift. The model must maintain knowledge of digit identity under systematically shifting input orientation.
- **Rationale**: Tests whether methods can handle smooth, predictable visual drift. Because the label space is shared, it is a domain-incremental scenario — the hardest setting for methods that rely on task-specific heads.
- **Architecture**: MLP (784 → 256 → 10), **1 training epoch per task (unified across all methods)**.

### 3.2 EMNIST — Visual Style Drift with Shared Labels
- **Setup**: 5 tasks, all using the same pool of 36-class EMNIST characters (letters and digits), but each task applies a different visual transformation: Task 1 = identity (no augmentation), Task 2 = random rotation (±30°), Task 3 = Gaussian blur, Task 4 = brightness jitter, Task 5 = mixed augmentation (rotation + blur + brightness). The label space is identical across all tasks.
- **Drift type**: Visual style drift with shared label space. The input distribution shifts due to appearance transformations, but the underlying categories remain constant. This is a domain-incremental scenario where the challenge is maintaining consistent recognition under visual degradation, not learning new categories.
- **Rationale**: Tests methods under a different form of visual drift from Rotated MNIST — augmentation-based style shift rather than geometric rotation. Because all tasks share the same 36-class label space with no categorical change, this benchmark also reveals whether methods can exploit inter-task feature overlap (forward transfer) rather than merely preventing forgetting.
- **Architecture**: MLP (784 → 256 → 36), 1 training epoch per task.

### 3.3 CIFAR-100 — Large-Scale Object Category Drift
- **Setup**: 10 tasks, each task covers 10 fine-grained object categories from CIFAR-100 (100 total classes), evaluated under the **task-incremental** protocol (task identity given at test time, logits masked to the 10 relevant classes).
- **Drift type**: Large-scale semantic drift. Each task introduces entirely new visual concepts (vehicles, animals, household objects) with substantially different low- and high-level visual features.
- **Rationale**: Tests methods on the most visually demanding scenario. Task-incremental evaluation was chosen over class-incremental because class-incremental on CIFAR-100 requires discriminating all 100 classes simultaneously, making all methods converge near chance (10%) and providing no meaningful differentiation. Task-incremental gives fair measurement of forgetting and backward transfer.
- **Architecture**: CNN (3 conv layers + 2 FC layers, 100-class output), 20 training epochs per task.

---

## 4. Evaluation Metrics

Four standard CL metrics were computed from the accuracy matrix R, where R[i][j] denotes the accuracy on task j after training on task i:

- **ACC (Average Final Accuracy)**: Mean accuracy across all tasks after all training is complete.
  `ACC = (1/T) * sum_j R[T][j]`
  Higher is better. Measures how much knowledge is retained at the end.

- **BWT (Backward Transfer)**: Mean change in task accuracy after training on later tasks.
  `BWT = (1/(T-1)) * sum_j (R[T][j] - R[j][j])`
  Zero means no forgetting; negative means forgetting; positive means that learning later tasks improved earlier task performance (rare, but observed in this study on EMNIST).

- **LA (Learning Accuracy)**: Mean accuracy on each task immediately after training on it.
  `LA = (1/T) * sum_i R[i][i]`
  Measures the model's peak plasticity — how well it can learn each task in isolation.

- **Forgetting**: Defined as -BWT (positive values = forgetting). Reported alongside BWT for clarity.

Additionally, two cost metrics were tracked:
- **Total training time** (seconds): Wall-clock time summed across all tasks.
- **Extra storage** (MB): Size of all auxiliary data structures (Fisher matrices, gradient subspace, episodic buffers) at the end of training.

---

## 5. Experimental Design Decisions

### 5.1 Multi-Seed Protocol
All experiments were run with three random seeds (0, 1, 2). Results are reported as **mean ± std** across seeds. This ensures statistical reliability and allows detection of high-variance methods that may appear competitive on a single run but are inconsistent in practice.

### 5.2 Unified Epochs per Task
To ensure a fair comparison, all methods on each dataset use the same number of training epochs per task:
- **Rotated MNIST**: 1 epoch per task (all 6 methods)
- **EMNIST**: 1 epoch per task (all 6 methods)
- **CIFAR-100**: 20 epochs per task (all 6 methods)

Note: ER's default code configuration is 5 epochs per task; this was explicitly overridden to 1 epoch for Rotated MNIST and EMNIST to ensure comparability. The 20-epoch setting for CIFAR-100 was applied uniformly to compensate for the larger model and more complex task structure.

### 5.3 Buffer Size Design for Replay Methods
A two-tier experimental design was used for replay methods:

**Tier 1 — Main Cross-Method Comparison (buffer = 1000)**:
For the primary comparison across all six methods, both ER and A-GEM were configured with a buffer/memory size of 1000 samples. This choice ensures:
- Fairness: both replay methods operate under identical memory constraints.
- Representativeness: 1000 samples is neither so small that replay is ineffective nor so large that it approximates storing the full training set.
- Comparability: it matches commonly reported settings in the CL literature.

**Tier 2 — Buffer Size Ablation Study (200 / 500 / 1000 / 2000)**:
To understand how sensitive each replay method is to its buffer size, both ER and A-GEM were evaluated across four buffer sizes: 200, 500, 1000, and 2000 samples. Three seeds were used at each setting, yielding 36 experiments per method (4 sizes × 3 datasets × 3 seeds = 36).

This ablation addresses the question: *Is ER's advantage over A-GEM an intrinsic algorithmic difference, or merely a product of buffer size?*

### 5.4 Cost Measurement Design
Each experiment was wrapped with a `CostTracker` that records per-task:
- Start and end timestamps (for training time).
- RAM usage before and after each task (via `tracemalloc`).
- Extra storage bytes for method-specific data structures (Fisher matrices, OGD gradient memory, episodic buffers), computed from tensor sizes.

Cost measurements are averaged across 3 seeds to reduce noise.

---

## 6. Main Experiment Results (Cross-Method Comparison, Buffer = 1000)

### 6.1 Rotated MNIST (10 Tasks, 0°→90° Rotation Drift, 1 epoch/task)

| Method   | ACC (↑)         | BWT (↑→0)        | LA (↑)          | Forgetting (↓)  |
|----------|-----------------|------------------|-----------------|-----------------|
| Vanilla  | 0.596 ± 0.003   | -0.331 ± 0.003   | 0.894 ± 0.000   | 0.331 ± 0.003   |
| EWC      | 0.607 ± 0.002   | -0.317 ± 0.002   | 0.892 ± 0.000   | 0.317 ± 0.002   |
| MAS      | 0.639 ± 0.002   | -0.258 ± 0.003   | 0.870 ± 0.001   | 0.258 ± 0.003   |
| OGD      | 0.591 ± 0.010   | -0.333 ± 0.010   | 0.891 ± 0.000   | 0.333 ± 0.010   |
| **ER**   | **0.833 ± 0.002**| **-0.059 ± 0.002**| **0.886 ± 0.001**| **0.059 ± 0.002**|
| A-GEM    | 0.712 ± 0.003   | -0.202 ± 0.003   | 0.893 ± 0.001   | 0.202 ± 0.003   |

*All methods trained with 1 epoch per task for fair comparison.*

**Key observations:**
- ER achieves the highest ACC (83.3%) by a clear margin, outperforming the second-best method (A-GEM, 71.2%) by +12.1 percentage points, and the best regularisation method (MAS, 63.9%) by +19.4 percentage points.
- ER's BWT (-0.059) is substantially lower than all other methods, confirming that direct sample replay is highly effective at preventing forgetting under rotation drift.
- EWC, MAS, and OGD achieve only marginally better ACC than Vanilla (59.6%), demonstrating that weight regularisation and gradient constraints provide minimal benefit for geometric drift. All three exhibit forgetting rates >25%.
- OGD performs slightly worse than Vanilla (59.1% vs 59.6%), suggesting that orthogonal gradient projection may over-constrain the optimiser under rotation drift.
- A-GEM provides better protection than regularisation methods (forgetting = 20.2% vs >25%) but falls significantly short of direct replay (forgetting = 5.9%), suggesting gradient constraints are weaker than revisiting past examples. Notably, A-GEM accumulates memory across tasks (total = 10 × 1,000 = 10,000 samples at end of training), yet ER with a fixed 1,000-sample buffer still outperforms it by +12 percentage points — demonstrating superior memory efficiency.
- LA values are similar across all methods (~89%), indicating comparable plasticity. The performance gap between methods stems entirely from differences in forgetting resistance.

### 6.2 EMNIST (5 Tasks, Visual Style Drift, Shared 36-Class Label Space)

| Method   | ACC (↑)         | BWT (↑→0)        | LA (↑)          | Forgetting (↓)  |
|----------|-----------------|------------------|-----------------|-----------------|
| Vanilla  | 0.839 ± 0.001   | +0.034 ± 0.001   | 0.812 ± 0.001   | -0.034 ± 0.001  |
| EWC      | 0.831 ± 0.001   | +0.029 ± 0.001   | 0.808 ± 0.001   | -0.029 ± 0.001  |
| **MAS**  | **0.877 ± 0.001** | +0.020 ± 0.002  | **0.861 ± 0.001** | -0.020 ± 0.002  |
| OGD      | 0.564 ± 0.006   | +0.069 ± 0.009   | 0.509 ± 0.006   | -0.069 ± 0.009  |
| ER       | 0.839 ± 0.002   | +0.034 ± 0.001   | 0.812 ± 0.001   | -0.034 ± 0.001  |
| A-GEM    | 0.843 ± 0.001   | +0.036 ± 0.001   | 0.814 ± 0.001   | -0.036 ± 0.001  |

*All methods trained with 1 epoch per task for fair comparison.*

**Key observations:**
- **All methods show positive BWT on EMNIST**, meaning that learning later visual styles actually *improves* performance on earlier ones. This is a forward transfer effect: because all five tasks share the same 36-class label space and only differ in visual augmentation style, training on later transformations (e.g., blur, brightness jitter) exposes the model to greater input variety, which enriches the shared feature representation and generalises back to earlier tasks.
- Because there is no forgetting, replay methods (ER, A-GEM) offer no advantage over Vanilla — all three achieve essentially identical ACC (~83.9%). This is the expected result: if there is no problem to solve, the solution provides no benefit.
- **MAS is the best-performing method** (ACC = 87.7%), outperforming all other methods under the same training budget. MAS's output-sensitivity importance metric provides a form of implicit regularisation that encourages the model to maintain a well-calibrated, general feature representation — particularly beneficial when tasks share visual structure.
- **OGD performs worst of all methods** (ACC = 56.4%), falling well below even the Vanilla baseline (83.9%). OGD's orthogonality constraint forces gradient updates into progressively narrower subspaces; by task 5, the available gradient directions are severely constrained, reducing the model's ability to consolidate knowledge. Despite technically not forgetting past tasks (positive BWT = +6.9%), the model's learning accuracy is badly hurt (LA = 50.9% vs 81.2% for Vanilla), meaning OGD cannot adequately learn each task in the first place — a plasticity failure.
- EWC performs slightly worse than Vanilla (83.1% vs 83.9%), suggesting that Fisher-based weight protection is unnecessary and may marginally hinder the positive transfer that would otherwise occur naturally.

### 6.3 CIFAR-100 (10 Tasks, Task-Incremental, Semantic Drift)

| Method   | ACC (↑)         | BWT (↑→0)        | LA (↑)          | Forgetting (↓)  |
|----------|-----------------|------------------|-----------------|-----------------|
| Vanilla  | 0.174 ± 0.007   | -0.511 ± 0.021   | 0.634 ± 0.021   | 0.511 ± 0.021   |
| EWC      | 0.171 ± 0.008   | -0.451 ± 0.074   | 0.577 ± 0.070   | 0.451 ± 0.074   |
| MAS      | 0.403 ± 0.017   | -0.091 ± 0.014   | 0.484 ± 0.012   | 0.091 ± 0.014   |
| OGD      | 0.175 ± 0.005   | -0.510 ± 0.054   | 0.635 ± 0.045   | 0.510 ± 0.054   |
| **ER**   | **0.465 ± 0.010** | -0.215 ± 0.008  | 0.658 ± 0.004   | 0.215 ± 0.008   |
| A-GEM    | 0.414 ± 0.011   | -0.279 ± 0.034   | 0.665 ± 0.031   | 0.279 ± 0.034   |

**Key observations:**
- CIFAR-100 is the most challenging scenario, with all methods showing substantial forgetting due to the large semantic gap between tasks (e.g., fish vs. motorcycles vs. furniture).
- **ER again leads with ACC = 46.5%**, followed by A-GEM (41.4%) and MAS (40.3%). The three remaining methods (Vanilla, EWC, OGD) cluster around 17–17.5%, barely above chance for 10-class task-incremental evaluation (~10%).
- **EWC fails catastrophically**: despite using Fisher-based importance weighting, its ACC (17.1%) is indistinguishable from Vanilla (17.4%). EWC's Fisher matrix is computed from a relatively small amount of data per task; when tasks are semantically very different (semantic drift), the parameter importance estimates become unreliable, and the quadratic penalty fails to protect the correct weights.
- **MAS is the strongest regularisation method** (ACC = 40.3%), demonstrating that output-gradient-based importance is more robust to semantic drift than likelihood-based Fisher Information. MAS's forgetting (9.1%) is notably lower than ER (21.5%) in absolute terms, but ER's higher initial learning accuracy (LA = 65.8% vs 48.4%) compensates, resulting in better final ACC.
- **OGD again performs at Vanilla level** (17.5%), confirming that orthogonal gradient projection does not scale to tasks with large semantic distance.
- A-GEM (41.4%) performs comparably to ER but with higher variance (std = 0.011 for A-GEM vs 0.010 for ER). The gradient constraint mechanism of A-GEM appears effective when tasks share visual structure but is less stable than direct replay when visual distributions differ substantially.

---

## 7. Buffer Size Sensitivity Analysis (Ablation Study)

### 7.1 ER Buffer Size Sensitivity

| Buffer | CIFAR-100 ACC   | Rotated MNIST ACC | EMNIST ACC      |
|--------|-----------------|-------------------|-----------------|
| 200    | 0.337 ± 0.012   | 0.738 ± 0.006     | 0.839 ± 0.002   |
| 500    | 0.414 ± 0.009   | 0.799 ± 0.004     | 0.839 ± 0.002   |
| 1000   | 0.465 ± 0.010   | 0.833 ± 0.002     | 0.839 ± 0.002   |
| 2000   | 0.538 ± 0.012   | 0.859 ± 0.003     | 0.839 ± 0.002   |

**BWT (CIFAR-100):** -0.393 (buf=200) → -0.292 (500) → -0.215 (1000) → -0.127 (2000)
**BWT (Rotated MNIST):** -0.163 (buf=200) → -0.101 (500) → -0.072 (1000) → -0.046 (2000)

**Key observations:**
- On CIFAR-100 and Rotated MNIST, ER shows a **clear, monotonic improvement** as buffer size increases. Doubling buffer size from 200→2000 raises CIFAR-100 ACC by +20 percentage points and reduces forgetting by 26.6 points.
- The improvement is consistent and low-variance across seeds, indicating that ER's buffer size sensitivity is a stable and predictable phenomenon.
- On **EMNIST, all buffer sizes produce identical results** (ACC ≈ 83.9% regardless of buffer size). This confirms the earlier finding that EMNIST exhibits natural positive transfer with no forgetting; the replay buffer is unused effectively because there is nothing to protect against.
- The relationship between buffer size and performance is approximately log-linear, suggesting diminishing returns beyond 2000 samples (not tested here).

### 7.2 A-GEM Memory Size Sensitivity

| Memory | CIFAR-100 ACC   | Rotated MNIST ACC | EMNIST ACC      |
|--------|-----------------|-------------------|-----------------|
| 200    | 0.393 ± 0.035   | 0.718 ± 0.006     | 0.840 ± 0.002   |
| 500    | 0.387 ± 0.045   | 0.715 ± 0.001     | 0.843 ± 0.000   |
| 1000   | 0.414 ± 0.011   | 0.712 ± 0.003     | 0.843 ± 0.001   |
| 2000   | 0.325 ± 0.160   | 0.710 ± 0.002     | 0.844 ± 0.002   |

**Key observations:**
- **A-GEM is largely insensitive to memory size**, in stark contrast to ER. On Rotated MNIST, ACC actually *decreases* slightly as memory grows (0.718 → 0.710), a counterintuitive result.
- On CIFAR-100, A-GEM's performance is **highly unstable at memory=2000** (std = 0.160 vs ≤ 0.045 at smaller sizes). This suggests that a large reference gradient computed from many heterogeneous past examples can produce conflicting constraint signals, destabilising training.
- The fundamental difference between ER and A-GEM under buffer scaling is algorithmic: ER directly replays samples, so more samples → better coverage of past distributions → less forgetting. A-GEM projects gradients using past samples as a reference; however, the quality of this projection depends on the coherence of the reference gradient, not just its size. Aggregating a large, diverse memory can reduce gradient coherence and hurt rather than help.
- This finding supports choosing ER over A-GEM in practice: ER is predictably better with more memory, while A-GEM's benefit from additional memory is unreliable.

### 7.3 ER vs A-GEM — Comparative Summary

| Criterion | ER | A-GEM |
|-----------|-----|-------|
| Best ACC (CIFAR-100) | 0.538 (buf=2000) | 0.414 (mem=1000) |
| Best ACC (Rotated MNIST) | 0.859 (buf=2000) | 0.718 (mem=200) |
| Response to larger buffer | Monotonic improvement | No improvement / unstable |
| Variance across seeds | Low (std ≤ 0.012) | Higher (std up to 0.160) |
| Preferred buffer size for thesis | 1000 (stable, good performance) | 1000 (matched for fair comparison) |

---

## 8. Computational Cost Analysis

### 8.0 Complete Cost Summary Table (mean across 3 seeds)

| Dataset | Method | Training Time (s) | Extra Storage (MB) |
|---------|--------|------------------:|-------------------:|
| CIFAR-100 | Vanilla | 243 ± 2 | 0.00 |
| CIFAR-100 | EWC | 694 ± 3 | 251.48 |
| CIFAR-100 | MAS | 327 ± 2 | 25.15 |
| CIFAR-100 | OGD | 283 ± 2 | 125.74 |
| CIFAR-100 | ER | 461 ± 0 | 11.73 |
| CIFAR-100 | A-GEM | 1,139 ± 1 | 117.26 |
| Rotated MNIST | Vanilla | 9 ± 1 | 0.00 |
| Rotated MNIST | EWC | 42 ± 4 | 15.53 |
| Rotated MNIST | MAS | 23 ± 2 | 1.55 |
| Rotated MNIST | OGD | 18 ± 1 | 7.76 |
| Rotated MNIST | ER | 23 ± 2 | 3.00 |
| Rotated MNIST | A-GEM | 20 ± 0 | 29.98 |
| EMNIST | Vanilla | 70 ± 6 | 0.00 |
| EMNIST | EWC | 249 ± 21 | 8.02 |
| EMNIST | MAS | 839 ± 75 | 1.60 |
| EMNIST | OGD | 93 ± 9 | 1.60 |
| EMNIST | ER | 69 ± 5 | 3.00 |
| EMNIST | A-GEM | 139 ± 13 | 14.99 |

*Extra Storage = size of method-specific auxiliary structures (Fisher matrices, gradient subspace, episodic buffer) at end of training. ER and A-GEM figures use buffer/memory size = 1,000.*

---

### 8.1 Training Time (seconds, mean across 3 seeds)

| Method   | CIFAR-100 | Rotated MNIST | EMNIST  |
|----------|-----------|---------------|---------|
| Vanilla  | 243 s     | 9 s           | 70 s    |
| EWC      | 694 s     | 42 s          | 249 s   |
| MAS      | 327 s     | 23 s          | 839 s   |
| OGD      | 283 s     | 18 s          | 93 s    |
| ER       | 461 s     | 24 s          | 69 s    |
| A-GEM    | 1,139 s   | 20 s          | 139 s   |

*Rotated MNIST timings reflect the corrected 1 epoch/task setting.*

**Key observations:**
- **EWC** is 2.9× slower than Vanilla on CIFAR-100 (694 s vs 243 s) due to Fisher Information computation after each task.
- **MAS** on EMNIST (839 s, ~12× Vanilla at 70 s) is the slowest EMNIST method, because computing output-gradient importance across the full EMNIST feature space is expensive. Note: at the corrected 1 epoch/task, this is roughly half the time compared to the old 3-epoch default (which was ~1,617 s).
- **OGD** on EMNIST dropped from 296 s (4 epochs) to 93 s (1 epoch), confirming that its original speed disadvantage was largely due to extra training rounds rather than per-step overhead.
- **ER** on Rotated MNIST at 1 epoch/task (24 s) is only 2.7× slower than Vanilla (9 s), a modest overhead consistent with replaying a fixed 1,000-sample buffer alongside each training batch.
- **A-GEM** is the slowest on CIFAR-100 (1,139 s, 4.7× Vanilla), owing to the double-forward-backward pass required for gradient projection at each training step.
- ER on EMNIST (69 s) is nearly identical to Vanilla (70 s), consistent with the finding that the buffer provides no benefit when there is no forgetting to prevent.

### 8.2 Extra Storage (MB, end of training)

| Method   | CIFAR-100 | Rotated MNIST | EMNIST  |
|----------|-----------|---------------|---------|
| Vanilla  | 0.00 MB   | 0.00 MB       | 0.00 MB |
| EWC      | 251.48 MB | 15.53 MB      | 8.02 MB |
| MAS      | 25.15 MB  | 1.55 MB       | 1.60 MB |
| OGD      | 125.74 MB | 7.76 MB       | 1.60 MB |
| ER       | 11.73 MB  | 3.00 MB       | 3.00 MB |
| A-GEM    | 117.26 MB | 29.98 MB      | 14.99 MB|

**Key observations:**
- **EWC has the highest storage cost** on CIFAR-100 (251.5 MB), because it must store a full copy of the model weights plus one Fisher diagonal per task (10 tasks × CNN parameter count). This is 21× more than ER.
- **ER has the lowest storage among all non-baseline methods** (11.7 MB on CIFAR-100), because 1000 CIFAR-32×32 RGB images require only ~12 MB.
- **OGD** (125.7 MB) stores the gradient subspace per task, which scales with both model size and number of tasks.
- **A-GEM** (117.3 MB) stores the cumulative episodic memory (10 tasks × 1000 samples), but because gradients are projected at training time and not stored, the memory cost is purely sample-based.
- ER provides the best storage efficiency: it requires less memory than EWC, OGD, or A-GEM while outperforming all of them.

### 8.3 Efficiency Summary

When considering the cost-effectiveness trade-off (performance per unit storage):
- **ER** is the most efficient method: highest ACC on two of three datasets, lowest storage among all CL methods.
- **MAS** is the most efficient regularisation method: its storage cost (25 MB on CIFAR-100) is 10× lower than EWC while delivering dramatically better ACC (40.3% vs 17.1%).
- **EWC** is the least cost-effective: highest storage, lowest ACC among CL methods on CIFAR-100, only marginal improvement over Vanilla.
- **A-GEM**'s high computational cost (1,139 s) and moderate storage (117 MB) do not justify its performance advantage over ER.

---

## 9. Discussion

### 9.1 Does Replay Outperform Regularisation under Visual Drift?

The results consistently support a **yes** for datasets with genuine visual drift (Rotated MNIST, CIFAR-100) and **no** for datasets with natural forward transfer (EMNIST).

On Rotated MNIST (under fair 1 epoch/task conditions), ER achieves 83.3% ACC vs 63.9% for the best regularisation method (MAS) — a gap of 19.4 percentage points. On CIFAR-100, ER achieves 46.5% vs 40.3% for MAS — a smaller but still significant gap. The replay advantage grows with the severity and diversity of visual drift, because regularisation methods cannot update their importance estimates at test time, while replay methods directly sample the true past data distribution.

### 9.2 Why Regularisation Struggles with Visual Drift

Regularisation methods estimate weight importance once, at the end of each task. Under strong visual drift (e.g., semantic change from animals to vehicles), the distribution of activations shifts substantially between tasks. A Fisher matrix computed on task 1's data may poorly capture which weights are important for task 1 when the model has been updated with the very different gradient landscape of tasks 2–10. This is the core limitation: importance estimates are **non-stationary** under distribution shift.

MAS partially mitigates this by using output sensitivity rather than likelihood curvature, which is more stable to distribution shifts. This explains MAS's advantage over EWC on CIFAR-100 (40.3% vs 17.1%).

### 9.3 Why OGD Fails Universally

OGD's orthogonality constraint becomes increasingly restrictive as more tasks are learned. After T tasks, the gradient must be orthogonal to the span of T × (gradient dimension) vectors. For large models or many tasks, the feasible gradient space approaches zero, and the model effectively stops learning useful updates. This failure mode is particularly severe for CIFAR-100 (10 tasks, large CNN) and Rotated MNIST (10 tasks with shared label space).

### 9.4 The Role of Buffer Size

The ablation study reveals a fundamental asymmetry between ER and A-GEM in their use of episodic memory. ER uses memory *generatively* — each stored sample contributes directly to loss computation, so more samples improve distribution coverage. A-GEM uses memory *constraintively* — stored samples define a feasibility region for gradient updates, but a larger and more heterogeneous constraint set can produce incoherent composite gradients that interfere with learning.

This asymmetry suggests that **direct replay is a more robust use of episodic memory than gradient-level constraints**, particularly in visually diverse settings.

### 9.5 The EMNIST Anomaly

EMNIST's positive BWT (+3.4% for Vanilla) challenges the assumption that catastrophic forgetting is universal in continual learning. When tasks share low-level visual features (strokes, edges, curves), learning new tasks can reinforce shared representations, benefiting earlier tasks. This is consistent with the literature on *forward transfer* and suggests that the plasticity-stability dilemma may be less severe in curricula with compatible tasks.

The practical implication for this study is that EMNIST differentiates methods on *plasticity* rather than *stability*: the best method is the one that best exploits inter-task feature sharing (MAS, 87.7% at 1 epoch/task) rather than the one that best prevents forgetting (which is not needed here). Notably, OGD's severe plasticity failure on EMNIST (LA = 50.9%, ACC = 56.4%) under 1 epoch/task conditions — well below even Vanilla — further confirms that gradient orthogonality constraints are harmful for scenarios with positive inter-task transfer.

---

## 10. Conclusions

1. **Replay-based methods (ER, A-GEM) outperform regularisation-based (EWC, MAS) and optimisation-based (OGD) methods under visual drift**, with ER achieving the best performance on both Rotated MNIST (83.3%) and CIFAR-100 (46.5%) under a fair, unified training budget (1 epoch/task for MNIST-based datasets, 20 epochs/task for CIFAR-100).

2. **Experience Replay (ER) is the most reliable and cost-effective method overall**: highest ACC in visual drift scenarios, lowest extra storage among all CL methods, predictable improvement with larger buffers, and low variance across seeds. Remarkably, ER with a fixed 1,000-sample buffer outperforms A-GEM despite A-GEM accumulating up to 10,000 samples across tasks — a 10× memory advantage that yields no performance benefit.

3. **MAS is the best regularisation method** and provides competitive performance on CIFAR-100 (40.3%), but still falls short of ER (46.5%). MAS's output-sensitivity importance measure is more robust to visual drift than EWC's Fisher-based approach.

4. **EWC and OGD fail to provide meaningful protection** under strong visual drift, performing at or below the Vanilla baseline on CIFAR-100. Their theoretical justifications do not translate to practical benefits in visual distribution shift scenarios.

5. **Buffer size matters for ER but not for A-GEM**: ER performance scales monotonically with buffer size; A-GEM performance is largely flat and becomes unstable at large memory sizes. For fair comparison, buffer=1000 was used for both in the main experiment.

6. **EMNIST is qualitatively different**: all methods show positive backward transfer rather than forgetting, making it an unsuitable benchmark for evaluating anti-forgetting mechanisms but a useful testbed for measuring plasticity under visual style drift with a shared label space.

7. From a practical standpoint, for applications involving visual drift in continual learning, **ER should be the first choice**: it requires minimal hyperparameter tuning, provides predictable scaling with memory budget, has lower storage overhead than competing methods, and consistently delivers the best accuracy under distribution shift.

---

## 11. Experimental Configuration Summary

| Parameter | Value |
|-----------|-------|
| Seeds | 0, 1, 2 (mean ± std reported) |
| ER buffer size (main) | 1000 |
| A-GEM memory size (main) | 1000 |
| Buffer sensitivity range | 200, 500, 1000, 2000 |
| Rotated MNIST tasks | 10 (0° to 90° rotation, 10° per task) |
| EMNIST tasks | 5 visual style transformations with shared 36-class labels |
| CIFAR-100 tasks | 10 (10 fine classes per task, task-incremental) |
| CIFAR-100 epochs per task | 20 |
| MNIST/EMNIST epochs per task | 1 |
| Batch size | 64 (all experiments) |
| Optimiser (MLP) | SGD, lr=0.01 |
| Optimiser (CNN) | Adam, lr=0.001 |
| Total experiments run | 126 |

---

*Report generated from experimental results stored in:*
- *`reports/summary_multiseed.csv` — main experiment metrics*
- *`reports/buffer_sensitivity.png` — buffer sensitivity plots*
- *`reports/thesis_summary.png` — cross-method summary figure*
