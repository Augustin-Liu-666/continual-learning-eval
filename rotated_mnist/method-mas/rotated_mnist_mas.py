# rotated_mnist_mas.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders

# Simple MLP Model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Evaluate model

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# Estimate parameter importance using MAS

def compute_importance(model, dataloader):
    model.eval()
    importance = {}
    for name, param in model.named_parameters():
        importance[name] = torch.zeros_like(param)

    for images, _ in dataloader:
        images = images.to(device)
        model.zero_grad()
        outputs = model(images)
        probs = torch.norm(torch.softmax(outputs, dim=1), p=2, dim=1).mean()
        probs.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += param.grad.data.clone().abs()

    for name in importance:
        importance[name] /= len(dataloader)
    return importance

# MAS penalty loss

def mas_loss(model, prev_params, importance, lam):
    loss = 0
    for name, param in model.named_parameters():
        if name in importance:
            loss += (importance[name] * (param - prev_params[name]).pow(2)).sum()
    return lam * loss

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

accuracies = []
pre_task_accs = []
importance = {}
prev_params = {}
lam = 100  # 10

# Training across tasks
for task_id, train_loader in enumerate(train_loaders):
    print(f"Training on Task {task_id + 1}")

    if task_id < len(train_loaders) - 1:
        fwt_acc = evaluate(model, test_loaders[task_id + 1])
        pre_task_accs.append(fwt_acc)

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if task_id > 0:
                loss += mas_loss(model, prev_params, importance, lam)
            loss.backward()
            optimizer.step()

    acc = []
    for test_loader in test_loaders[:task_id + 1]:
        acc.append(evaluate(model, test_loader))
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id + 1}: {acc}")

    prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    importance = compute_importance(model, train_loader)

# Accuracy curve
full_accuracies = []
for i in range(len(accuracies)):
    row = [None] * 5
    for j in range(i + 1):
        row[j] = accuracies[i][j]
    full_accuracies.append(row)
full_accuracies = np.array(full_accuracies)

plt.figure(figsize=(10, 6))
for task in range(5):
    plt.plot(range(1, 6), full_accuracies[:, task], marker='o', label=f"Task {task + 1}")
plt.xlabel("After Task")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Evolution Across Tasks (MAS Training on Rotated MNIST)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("mas_accuracy_curve_rotated_mnist.png")


print("\n✅ Accuracy curve saved as 'mas_accuracy_curve_rotated_mnist.png'!")

# Metrics
forgettings = []
for task in range(4):
    max_acc = max(full_accuracies[i, task] for i in range(task + 1, 5))
    last_acc = full_accuracies[4, task]
    forgettings.append(max_acc - last_acc)

average_forgetting = np.mean(forgettings)
bwt = -average_forgetting
fwt_values = pre_task_accs
average_fwt = np.mean(fwt_values) if fwt_values else float('nan')

print("\n📈 Evaluation Results (MAS Training on Rotated MNIST):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")