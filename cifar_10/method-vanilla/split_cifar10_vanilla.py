# split_cifar10_vanilla.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from cifar_10.dataloader.split_cifar10_dataloader import train_loaders, test_loaders
from cifar_10.models.cnn_cifar10_model import SimpleCNN

# 设置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义训练函数 / Define training function
def train(model, optimizer, criterion, loader):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 定义测试函数 / Define evaluation function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# 初始化模型与优化器 / Initialize model and optimizer
model = SimpleCNN(num_classes=2).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 保存每个任务结束时的准确率 / Accuracy tracking
accuracies = []

# Sequential Task Training
for task_id, train_loader in enumerate(train_loaders):
    print(f"Training on Task {task_id + 1}")
    # 每个新任务重新初始化最后一层 / Reset output layer for each task
    model.fc2 = nn.Linear(256, 2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):  # 训练10个epochs
        train(model, optimizer, criterion, train_loader)

    acc = []
    for test_loader in test_loaders[:task_id + 1]:
        a = evaluate(model, test_loader)
        acc.append(a)
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id + 1}: {acc}")

# 绘制准确率曲线 / Plot accuracy curve
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
plt.title("Accuracy Evolution Across Tasks (Vanilla Training on Split-CIFAR10)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("vanilla_accuracy_curve_cifar10.png")


print("\n✅ Accuracy curve saved as 'vanilla_accuracy_curve_cifar10.png'!")

# 计算 Forgetting / BWT / FWT
forgettings = []
for task in range(5 - 1):
    max_acc = max(full_accuracies[i, task] for i in range(task + 1, 5))
    last_acc = full_accuracies[4, task]
    forgettings.append(max_acc - last_acc)

average_forgetting = np.mean(forgettings)
bwt = -average_forgetting

fwt_values = [full_accuracies[i, i + 1] for i in range(4) if full_accuracies[i, i + 1] is not None]
average_fwt = np.mean(fwt_values) if fwt_values else float('nan')

print("\n📈 Evaluation Results (Vanilla Training on Split-CIFAR10):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
