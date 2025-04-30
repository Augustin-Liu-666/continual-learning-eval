# split_mnist_si.py

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
alpha = 0.1  # SI importance scaling factor

# 下载Split-MNIST / Download Split-MNIST
data = datasets.MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())

def create_split_tasks():
    class_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    tasks = []
    for a, b in class_pairs:
        mask = (data.targets == a) | (data.targets == b)
        x = data.data[mask].unsqueeze(1).float() / 255.0
        y = data.targets[mask]
        y = (y == b).long()
        tasks.append(TensorDataset(x, y))
    return tasks

tasks = create_split_tasks()
task_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in tasks]

# 模型 / Simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# SI 辅助结构 / SI memory structures
def init_si(model):
    omega = {}
    prev_params = {}
    W = {}
    for name, param in model.named_parameters():
        omega[name] = torch.zeros_like(param)
        prev_params[name] = param.detach().clone()
        W[name] = torch.zeros_like(param)
    return omega, prev_params, W

def update_omega(model, omega, W, prev_params, epsilon=0.1):
    for name, param in model.named_parameters():
        delta = param.detach() - prev_params[name]
        omega[name] += W[name] / (delta ** 2 + epsilon)
        W[name].zero_()
        prev_params[name] = param.detach().clone()

def update_W(model, W):
    for name, param in model.named_parameters():
        if param.grad is not None:
            W[name] += -param.grad * param.detach()

def compute_si_loss(model, omega, prev_params):
    reg = 0.0
    for name, param in model.named_parameters():
        reg += torch.sum(omega[name] * (param - prev_params[name]) ** 2)
    return reg

# 初始化 / Setup
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
omega, prev_params, W = init_si(model)
accuracies = []

# 训练与评估 / Training & Evaluation
for t, loader in enumerate(task_loaders):
    print(f"Training on Task {t+1}")
    model.train()
    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            reg = compute_si_loss(model, omega, prev_params) if t > 0 else 0.0
            total_loss = loss + alpha * reg
            total_loss.backward()
            update_W(model, W)
            optimizer.step()

    update_omega(model, omega, W, prev_params)

    # 评估 / Evaluation
    model.eval()
    with torch.no_grad():
        task_acc = []
        for eval_loader in task_loaders[:t+1]:
            correct = 0
            total = 0
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc = 100 * correct / total
            task_acc.append(acc)
        accuracies.append(task_acc)
        print(f"Accuracy after Task {t+1}: {task_acc}")

# 绘图 / Plot
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
plt.xlabel("After Task")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Evolution Across Tasks (SI Training)")
plt.legend()
plt.grid(True)
plt.savefig("si_accuracy_curve.png")


# Forgetting/BWT/FWT
forgettings = []
for task in range(4):
    max_acc = max(full_accuracies[i][task] for i in range(task+1, 5))
    last_acc = full_accuracies[4][task]
    forgettings.append(max_acc - last_acc)

average_forgetting = np.mean(forgettings)
bwt = -average_forgetting
fwt_values = [full_accuracies[i][i+1] for i in range(4) if full_accuracies[i][i+1] is not None]
average_fwt = np.mean(fwt_values) if fwt_values else float('nan')

print("\n📈 Evaluation Results (SI Training):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
