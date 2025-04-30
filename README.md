# Continual Learning Evaluation

This repository contains code for evaluating various continual learning methods (e.g., EWC, Replay, GEM, OGD) on datasets like Rotated MNIST and CIFAR-100, using Wide Residual Networks (WRN) and other models.

## Structure

- `data/` - (ignored) Contains raw datasets like CIFAR-100.
- `scripts/` - Training and evaluation scripts for different methods.
- `models/` - Model architectures (e.g., WRN).
- `results/` - Output logs and plots.

## Notes

- Datasets will be automatically downloaded using `torchvision.datasets`.
- Make sure to run the code on CUDA-enabled GPUs for CIFAR-100 experiments.

## How to run on Colab

```python
!git clone https://github.com/Augustin-Liu-666/continual-learning-eval.git
%cd continual-learning-eval
# Then run your training script, e.g.,
# !python train_wrn_cifar100.py