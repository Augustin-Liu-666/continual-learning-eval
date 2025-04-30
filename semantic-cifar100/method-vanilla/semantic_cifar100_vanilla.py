import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.wrn_cifar100_model import WRN
from dataloader.semantic_cifar100_dataloader import train_loaders, test_loaders
from utils import evaluate, compute_forgetting_metrics

# Basic settings
use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
print("\n✅ Semantic CIFAR-100 DataLoaders Ready!")
print(f"✅ Using device: {device}\n")

EPOCHS = 5  # 降低 epoch
lr = 0.001

print("Training on Semantic CIFAR-100 (Vanilla)...")

model = WRN(depth=28, widen_factor=10, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

accuracy_matrix = []

for task in range(len(train_loaders)):
    print(f"Training on Task {task+1}")
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loaders[task]):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    task_accuracies = []
    for test_task in range(task + 1):
        acc = evaluate(model, test_loaders[test_task], device)
        task_accuracies.append(acc)
    print(f"Accuracy after Task {task+1}: {task_accuracies}")
    accuracy_matrix.append(task_accuracies)

# Plot accuracy curve
plt.figure(figsize=(10, 6))
for task_id in range(len(accuracy_matrix[-1])):
    accs = [epoch_acc[task_id] if task_id < len(epoch_acc) else None for epoch_acc in accuracy_matrix]
    plt.plot(accs, label=f'Task {task_id + 1}')
plt.xlabel("Training Task")
plt.ylabel("Accuracy (%)")
plt.title("Semantic CIFAR-100 - Vanilla + WRN Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("vanilla_wrn_accuracy_curve_semantic_cifar100.png")
print("\n✅ Accuracy curve saved as 'vanilla_wrn_accuracy_curve_semantic_cifar100.png'!")

forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)

print("\n📊 Evaluation Results (Vanilla + WRN on Semantic CIFAR-100):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")