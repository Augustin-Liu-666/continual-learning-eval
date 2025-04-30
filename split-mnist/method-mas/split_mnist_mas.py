# split_mnist_mas.py

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# 下载并准备Split-MNIST数据 / Download and prepare Split-MNIST
transform = transforms.Compose([transforms.ToTensor()])
full_mnist = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

# 划分Split-MNIST任务 / Split MNIST into 5 tasks
def create_split_mnist_tasks():
    task_data = []
    class_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    for (class1, class2) in class_pairs:
        idx = (full_mnist.targets == class1) | (full_mnist.targets == class2)
        data = full_mnist.data[idx]
        targets = full_mnist.targets[idx]
        targets = (targets == class2).long()
        task_dataset = torch.utils.data.TensorDataset(data.unsqueeze(1).float()/255.0, targets)
        task_data.append(task_dataset)
    return task_data

tasks = create_split_mnist_tasks()
task_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in tasks]

# 定义简单MLP模型 / Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 计算MAS重要性矩阵 / Compute MAS importance
def compute_mas_importance(model, loader):
    model.eval()
    importance = {}
    for n, p in model.named_parameters():
        importance[n] = torch.zeros_like(p)

    for data, target in loader:
        data = data.to(device)
        model.zero_grad()
        output = model(data)
        output_probs = torch.sigmoid(output)
        output_probs.mean().backward()
        for n, p in model.named_parameters():
            importance[n] += p.grad.abs()

    # 平均重要性 / Average importance
    for n in importance.keys():
        importance[n] /= len(loader)
    return importance

# MAS损失函数 / MAS loss function
def mas_loss(model, criterion, output, target, mas_lambda, old_params, importance_matrix):
    loss = criterion(output, target)
    if old_params is not None:
        for n, p in model.named_parameters():
            loss += (mas_lambda/2) * (importance_matrix[n] * (p - old_params[n]).pow(2)).sum()
    return loss

# 初始化 / Initialize
model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
mas_lambda = 100  # 正则化强度 / Regularization strength

accuracies = []  # 存储准确率 / Store accuracies
old_params = None
importance_matrix = None

# 训练和评估每个任务 / Train and evaluate on each task
for task_id, loader in enumerate(task_loaders):
    print(f"Training on Task {task_id+1}")
    model.train()
    for epoch in range(10):  # ✅ 10个epoch
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = mas_loss(model, criterion, output, target, mas_lambda, old_params, importance_matrix)
            loss.backward()
            optimizer.step()
    
    # 更新MAS重要性矩阵 / Update MAS importance matrix
    importance_matrix = compute_mas_importance(model, loader)
    old_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    # 评估 / Evaluate
    def evaluate(model, loaders):
        model.eval()
        accs = []
        with torch.no_grad():
            for loader in loaders:
                correct = 0
                total = 0
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    preds = output.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
                accs.append(100 * correct / total)
        return accs

    acc = evaluate(model, task_loaders[:task_id+1])
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id+1}: {acc}")

# ========== 绘制准确率曲线 / Plot Accuracy Curve ==========

full_accuracies = []
for i in range(len(accuracies)):
    row = [None]*5
    for j in range(i+1):
        row[j] = accuracies[i][j]
    full_accuracies.append(row)

full_accuracies = np.array(full_accuracies)

plt.figure(figsize=(10,6))
for task in range(5):
    plt.plot(range(1,6), full_accuracies[:, task], marker='o', label=f"Task {task+1}")
plt.xlabel('After Task')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Evolution Across Tasks (MAS Training)')
plt.xticks([1,2,3,4,5])
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.savefig('mas_accuracy_curve.png')


print("\n✅ Accuracy curve saved as 'mas_accuracy_curve.png'!")

# ========== 计算指标 / Calculate Metrics ==========

# Forgetting
forgettings = []
for task in range(5-1):
    accs = [full_accuracies[i, task] for i in range(task,5)]
    max_acc = max(accs)
    last_acc = accs[-1]
    forgetting = max_acc - last_acc
    forgettings.append(forgetting)

average_forgetting = np.mean(forgettings)

# BWT
bwt = np.mean(forgettings)

# FWT
fwt_values = []
for task_id in range(1,5):
    initial_acc = full_accuracies[task_id-1][task_id]
    if initial_acc is not None:
        fwt_values.append(initial_acc)
average_fwt = np.mean(fwt_values) if len(fwt_values)>0 else float('nan')

# 打印指标 / Print metrics
print("\n📈 Evaluation Results (MAS Training):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {-bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
