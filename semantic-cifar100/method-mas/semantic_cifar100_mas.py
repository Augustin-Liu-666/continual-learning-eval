
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from cnn_cifar100_model import CIFAR100_CNN
# Add parent paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(parent_dir, 'models'))
sys.path.append(os.path.join(parent_dir, 'semantic-cifar100', 'dataloader'))

from cnn_cifar100_model import CIFAR100_CNN
from semantic_cifar100_dataloader import train_loaders, test_loaders
from utils import evaluate, compute_forgetting_metrics

# MAS-specific components
class MAS:
    def __init__(self, model, lambda_):
        self.model = model
        self.lambda_ = lambda_
        self.importance = {}
        self.prev_params = {}

    def compute_importance(self, data_loader, device):
        self.model.eval()
        importance = {}
        for name, param in self.model.named_parameters():
            importance[name] = torch.zeros_like(param)

        for x, _ in data_loader:
            x = x.to(device)
            self.model.zero_grad()
            output = self.model(x)
            loss = torch.sum(torch.norm(output, dim=1))
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.abs().detach()

        for name in importance:
            importance[name] /= len(data_loader)

        self.importance = importance
        self.prev_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def penalty(self):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.importance:
                loss += torch.sum(self.importance[name] * (param - self.prev_params[name])**2)
        return self.lambda_ * loss


# Training settings
EPOCHS = 10
LR = 0.001
LAMBDA = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Semantic CIFAR-100 DataLoaders Ready!")
print("Training on Semantic CIFAR-100 (MAS)...")

model = CIFAR100_CNN(num_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
mas = MAS(model, lambda_=LAMBDA)

accuracy_matrix = []

for task in range(len(train_loaders)):
    print(f"Training on Task {task+1}")
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loaders[task]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            if task > 0:
                loss += mas.penalty()
            loss.backward()
            optimizer.step()

    mas.compute_importance(train_loaders[task], device)

    task_accuracies = []
    for test_task in range(task + 1):
        acc = evaluate(model, test_loaders[test_task], device)
        task_accuracies.append(acc)
    print(f"Accuracy after Task {task+1}: {task_accuracies}")
    accuracy_matrix.append(task_accuracies)

# Plot accuracy curve
plt.figure(figsize=(10, 6))
for task_id in range(len(accuracy_matrix[-1])):
    accs = [epoch_acc[task_id] if task_id < len(epoch_acc) else None for epoch_acc in accuracy_matrix]
    plt.plot(accs, label=f'Task {task_id + 1}')
plt.xlabel("Training Task")
plt.ylabel("Accuracy (%)")
plt.title("Semantic CIFAR-100 - MAS Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("mas_accuracy_curve_semantic_cifar100.png")
print("\n✅ Accuracy curve saved as 'mas_accuracy_curve_semantic_cifar100.png'!")

forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)
print("\n\U0001F4CA Evaluation Results (MAS on Semantic CIFAR-100):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")