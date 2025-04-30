import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.cnn_cifar100_model import CIFAR100_CNN
from dataloader.semantic_cifar100_dataloader import train_loaders, test_loaders
from utils import evaluate, compute_forgetting_metrics

# EWC Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
lr = 0.001
ewc_lambda = 100

print("Training on Semantic CIFAR-100 (EWC)...")

model = CIFAR100_CNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

prev_params = []
fisher_matrices = []
accuracy_matrix = []

for task in range(len(train_loaders)):
    print(f"Training on Task {task+1}")
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loaders[task]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            # EWC penalty
            if task > 0:
                for t in range(task):
                    for i, param in enumerate(model.parameters()):
                        loss += (ewc_lambda / 2) * torch.sum(
                            fisher_matrices[t][i] * (param - prev_params[t][i]) ** 2
                        )
            loss.backward()
            optimizer.step()

    # Save current parameters
    current_params = [param.detach().clone() for param in model.parameters()]
    prev_params.append(current_params)

    # Compute Fisher Information Matrix
    fisher = []
    model.eval()
    for p in model.parameters():
        fisher.append(torch.zeros_like(p))
    model.train()

    for x, y in train_loaders[task]:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        for i, param in enumerate(model.parameters()):
            fisher[i] += param.grad.data ** 2
    fisher = [f / len(train_loaders[task]) for f in fisher]
    fisher_matrices.append(fisher)

    # Evaluate after current task
    task_accuracies = []
    for test_task in range(task + 1):
        acc = evaluate(model, test_loaders[test_task], device)
        task_accuracies.append(acc)
    print(f"Accuracy after Task {task+1}: {task_accuracies}")
    accuracy_matrix.append(task_accuracies)

# ----------- Plot Accuracy Matrix Like Vanilla Version ------------
plt.figure(figsize=(10, 6))
for task_id in range(len(accuracy_matrix[-1])):
    accs = [epoch_acc[task_id] if task_id < len(epoch_acc) else None for epoch_acc in accuracy_matrix]
    plt.plot(accs, label=f'Task {task_id + 1}')
plt.xlabel("Training Task")
plt.ylabel("Accuracy (%)")
plt.title("EWC Accuracy Curve - Semantic CIFAR-100")
plt.legend()
plt.grid(True)
plt.savefig("ewc_accuracy_curve_semantic_cifar100.png")
print("\n\u2705 Accuracy curve saved as 'ewc_accuracy_curve_semantic_cifar100.png'!")

# ----------------- Metrics ------------------
forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)
print("\n\U0001F4CA Evaluation Results (EWC on Semantic CIFAR-100):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")
