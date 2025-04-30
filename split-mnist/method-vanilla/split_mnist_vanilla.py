# split_mnist_vanilla.py （更新版：每个任务训练10个epoch）

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])
full_mnist = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

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

model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

accuracies = []

for task_id, loader in enumerate(task_loaders):
    print(f"Training on Task {task_id+1}")
    model.train()
    for epoch in range(10):   # ✅ 统一改成10个epoch
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
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

# 绘制准确率曲线
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
plt.title('Accuracy Evolution Across Tasks (Vanilla Training)')
plt.xticks([1,2,3,4,5])
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.savefig('vanilla_accuracy_curve.png')


print("\n✅ Accuracy curve saved as 'vanilla_accuracy_curve.png'!")

# 计算指标
forgettings = []
for task in range(5-1):
    accs = [full_accuracies[i, task] for i in range(task,5)]
    max_acc = max(accs)
    last_acc = accs[-1]
    forgetting = max_acc - last_acc
    forgettings.append(forgetting)

average_forgetting = np.mean(forgettings)
bwt = np.mean(forgettings)

fwt_values = []
for task_id in range(1,5):
    initial_acc = full_accuracies[task_id-1][task_id]
    if initial_acc is not None:
        fwt_values.append(initial_acc)
average_fwt = np.mean(fwt_values) if len(fwt_values) > 0 else float('nan')

print("\n📈 Evaluation Results (Vanilla Training):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {-bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")