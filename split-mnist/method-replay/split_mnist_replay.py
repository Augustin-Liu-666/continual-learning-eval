# split_mnist_replay.py

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
memory_buffer_size = 500  # Buffer容量 / Size of replay memory buffer

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

# 初始化 / Initialize
model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

accuracies = []  # 存储准确率 / Store accuracies
memory_buffer = []  # 经验回放的存储区 / Memory buffer

# 辅助函数：更新Memory Buffer / Helper function: Update memory buffer
def update_memory(memory_buffer, new_data, new_targets):
    for data, target in zip(new_data, new_targets):
        if len(memory_buffer) < memory_buffer_size:
            memory_buffer.append((data.cpu(), target.cpu()))
        else:
            idx = random.randint(0, memory_buffer_size - 1)
            memory_buffer[idx] = (data.cpu(), target.cpu())

# 训练和评估每个任务 / Train and evaluate on each task
for task_id, loader in enumerate(task_loaders):
    print(f"Training on Task {task_id+1}")
    model.train()
    for epoch in range(10):
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # 如果Memory Buffer里有数据，取一小部分出来混合
            if len(memory_buffer) > 0:
                mem_data, mem_target = zip(*random.sample(memory_buffer, min(batch_size, len(memory_buffer))))
                mem_data = torch.stack(mem_data).to(device)
                mem_target = torch.stack(mem_target).to(device)

                # 将新数据和记忆数据合并 / Combine current batch and memory batch
                data = torch.cat([data, mem_data])
                target = torch.cat([target, mem_target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 更新Memory Buffer / Update memory after training
            update_memory(memory_buffer, data, target)

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
plt.title('Accuracy Evolution Across Tasks (Replay Training)')
plt.xticks([1,2,3,4,5])
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.savefig('replay_accuracy_curve.png')


print("\n✅ Accuracy curve saved as 'replay_accuracy_curve.png'!")

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
print("\n📈 Evaluation Results (Replay Training):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {-bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
