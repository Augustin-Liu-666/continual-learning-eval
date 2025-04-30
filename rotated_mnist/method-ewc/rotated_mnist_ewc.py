# rotated_mnist_ewc.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders

# 简单的MLP模型 / Simple MLP model
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

# 计算Fisher信息 / Compute Fisher Information Matrix
def compute_fisher(model, criterion, dataloader):
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.clone().pow(2)
    for n in fisher:
        fisher[n] /= len(dataloader)
    return fisher

# 评估函数 / Evaluation function
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

# EWC 正则项损失 / EWC penalty

def ewc_loss(model, prev_params, fisher, lam):
    loss = 0
    for n, p in model.named_parameters():
        if n in fisher:
            loss += (fisher[n] * (p - prev_params[n]).pow(2)).sum()
    return lam * loss

# 配置设备 / Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

accuracies = []
pre_task_accs = []
prev_params = {}
fisher_matrices = {}
lam = 15.0

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
                loss += ewc_loss(model, prev_params, fisher_matrices, lam)
            loss.backward()
            optimizer.step()

    acc = []
    for test_loader in test_loaders[:task_id + 1]:
        acc.append(evaluate(model, test_loader))
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id + 1}: {acc}")

    # 存储当前参数 / Save current params
    prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    fisher_matrices = compute_fisher(model, criterion, train_loader)

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
plt.title("Accuracy Evolution Across Tasks (EWC Training on Rotated MNIST)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("ewc_accuracy_curve_rotated_mnist.png")


print("\n✅ Accuracy curve saved as 'ewc_accuracy_curve_rotated_mnist.png'!")

# Evaluation
forgettings = []
for task in range(5 - 1):
    max_acc = max(full_accuracies[i, task] for i in range(task + 1, 5))
    last_acc = full_accuracies[4, task]
    forgettings.append(max_acc - last_acc)

average_forgetting = np.mean(forgettings)
bwt = -average_forgetting
fwt_values = pre_task_accs
average_fwt = np.mean(fwt_values) if fwt_values else float('nan')

print("\n📈 Evaluation Results (EWC Training on Rotated MNIST):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
