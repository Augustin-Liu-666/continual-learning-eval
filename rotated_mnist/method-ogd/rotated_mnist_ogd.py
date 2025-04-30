import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders
from rotated_mnist.model.simple_mlp import MLP
from utils import evaluate, compute_forgetting_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Rotated MNIST DataLoaders Ready!")

accuracy_matrix = []
gradient_memory = []

for task_id, train_loader in enumerate(train_loaders):
    print(f"Training on Task {task_id + 1}")
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)  # flatten image

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Project gradients
        if gradient_memory:
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            for past_grad in gradient_memory:
                dot = torch.dot(grads, past_grad)
                grads -= (dot / (past_grad.norm()**2 + 1e-8)) * past_grad
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    numel = p.grad.numel()
                    p.grad.copy_(grads[idx:idx+numel].view_as(p.grad))
                    idx += numel

        optimizer.step()

    # Save representative gradient from this task
    model.eval()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        gradient_memory.append(grad.detach())
        break

    # Evaluate
    acc_per_task = []
    for test_task in range(task_id + 1):
        acc = evaluate(model, test_loaders[test_task], device)
        acc_per_task.append(acc)
    accuracy_matrix.append(acc_per_task)
    print(f"Accuracy after Task {task_id + 1}: {acc_per_task}")

# Plot
accuracy_matrix_np = np.full((5, 5), np.nan)
for i, row in enumerate(accuracy_matrix):
    accuracy_matrix_np[i, :len(row)] = row
for t in range(5):
    plt.plot(range(1, 6), accuracy_matrix_np[:, t], label=f"Task {t+1}")
plt.xlabel("After Task")
plt.ylabel("Accuracy (%)")
plt.title("OGD Accuracy on Rotated MNIST")
plt.legend()
plt.savefig("ogd_accuracy_curve_rotated_mnist.png")
print("Accuracy curve saved as 'ogd_accuracy_curve_rotated_mnist.png'!")

# Metrics
forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)
print("\n📊 Evaluation Results (OGD on Rotated MNIST):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")