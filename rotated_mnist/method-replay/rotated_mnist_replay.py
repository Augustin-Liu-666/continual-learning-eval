# rotated_mnist_replay.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders

# Model Definition
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

# Evaluation Function
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory
buffer = defaultdict(list)
buffer_size = 200

# Training Setup
model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

accuracies = []
pre_task_accs = []

# Training Loop
for task_id, train_loader in enumerate(train_loaders):
    print(f"Training on Task {task_id + 1}")

    if task_id < len(train_loaders) - 1:
        fwt_acc = evaluate(model, test_loaders[task_id + 1])
        pre_task_accs.append(fwt_acc)

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Add replay samples
            if buffer:
                replay_x, replay_y = zip(*random.sample(list(buffer.values()), min(len(buffer), buffer_size)))
                replay_x = torch.cat(replay_x).to(device)
                replay_y = torch.cat(replay_y).to(device)

                images = torch.cat([images, replay_x])
                labels = torch.cat([labels, replay_y])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Update memory buffer
    for images, labels in train_loader:
        for x, y in zip(images, labels):
            if len(buffer) >= buffer_size:
                buffer.pop(next(iter(buffer)))
            buffer[len(buffer)] = (x.unsqueeze(0), torch.tensor([y]))

    acc = []
    for test_loader in test_loaders[:task_id + 1]:
        acc.append(evaluate(model, test_loader))
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id + 1}: {acc}")

# Accuracy Curve
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
plt.title("Accuracy Evolution Across Tasks (Replay Training on Rotated MNIST)")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("replay_accuracy_curve_rotated_mnist.png")

print("\n✅ Accuracy curve saved as 'replay_accuracy_curve_rotated_mnist.png'!")

# Evaluation Metrics
forgettings = []
for task in range(4):
    max_acc = max(full_accuracies[i, task] for i in range(task + 1, 5))
    last_acc = full_accuracies[4, task]
    forgettings.append(max_acc - last_acc)

average_forgetting = np.mean(forgettings)
bwt = -average_forgetting
fwt_values = pre_task_accs
average_fwt = np.mean(fwt_values) if fwt_values else float('nan')

print("\n📈 Evaluation Results (Replay Training on Rotated MNIST):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
