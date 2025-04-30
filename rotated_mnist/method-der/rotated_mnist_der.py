# rotated_mnist_der.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import matplotlib.pyplot as plt
import numpy as np # type: ignore
from collections import deque
import random

from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders

# Define model
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

# Evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100. * correct / total

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Buffer structure: (x, y, logits)
buffer_size = 1000
buffer = deque(maxlen=buffer_size)

accuracies = []
pre_task_accs = []

# Training loop
for task_id, train_loader in enumerate(train_loaders):
    print(f"Training on Task {task_id + 1}")

    if task_id < len(train_loaders) - 1:
        acc_fwt = evaluate(model, test_loaders[task_id + 1])
        pre_task_accs.append(acc_fwt)

    for epoch in range(20):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.train()
            output = model(x)
            loss = criterion(output, y)

            # DER: replay samples + KL distillation on logits
            if buffer:
                b_x, b_y, b_logits = zip(*random.sample(buffer, min(len(buffer), 128)))
                b_x = torch.cat(b_x).to(device)
                b_y = torch.cat(b_y).to(device)
                b_logits = torch.cat(b_logits).to(device)
                logits_current = model(b_x)

                ce_loss = criterion(logits_current, b_y)
                kd_loss = kl_loss(torch.log_softmax(logits_current, dim=1),
                                  torch.softmax(b_logits, dim=1))
                loss += ce_loss + kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update buffer
            for i in range(x.size(0)):
                with torch.no_grad():
                    buffer.append((x[i:i+1].cpu(),
                                   y[i:i+1].cpu(),
                                   model(x[i:i+1]).detach().cpu()))

    acc = [evaluate(model, test_loader) for test_loader in test_loaders[:task_id + 1]]
    accuracies.append(acc)
    print(f"Accuracy after Task {task_id + 1}: {acc}")

# Plot
acc_arr = np.zeros((5, 5))
for i, row in enumerate(accuracies):
    for j, val in enumerate(row):
        acc_arr[i, j] = val

plt.figure(figsize=(10, 6))
for task in range(5):
    x_vals = list(range(task + 1, 6))
    y_vals = acc_arr[task:, task]
    plt.plot(x_vals, y_vals, marker='o', label=f"Task {task+1}")
plt.xlabel("After Task")
plt.ylabel("Accuracy (%)")
plt.title("DER Accuracy on Rotated MNIST")
plt.grid(True)
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 100)
plt.legend()
plt.savefig("der_accuracy_curve_rotated_mnist.png")


# Metrics
forgettings = [max(acc_arr[t+1:, t]) - acc_arr[4, t] for t in range(4)]
print("Evaluation Results (DER on Rotated MNIST):")
print(f"Average Forgetting: {np.mean(forgettings):.2f}%")
print(f"Average BWT: {-np.mean(forgettings):.2f}%")
print(f"Average FWT: {np.mean(pre_task_accs):.2f}%")