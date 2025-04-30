import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.cnn_cifar100_model import CIFAR100_CNN
from dataloader.semantic_cifar100_dataloader import train_loaders, test_loaders
from utils import evaluate, compute_forgetting_metrics

# Basic settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
lr = 0.001
MEMORY_PER_CLASS = 20

print("Semantic CIFAR-100 DataLoaders Ready!")
print("Training on Semantic CIFAR-100 (Replay)...")

model = CIFAR100_CNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

replay_memory = []
accuracy_matrix = []

for task in range(len(train_loaders)):
    print(f"Training on Task {task+1}")
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loaders[task]:
            x, y = x.to(device), y.to(device)

            # Add replay samples
            if replay_memory:
                mem_x, mem_y = zip(*random.sample(replay_memory, min(len(replay_memory), len(x))))
                mem_x = torch.stack(mem_x).to(device)
                mem_y = torch.tensor(mem_y).to(device)
                x = torch.cat([x, mem_x])
                y = torch.cat([y, mem_y])

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    # Update replay memory
    model.eval()
    with torch.no_grad():
        for x, y in train_loaders[task]:
            for i in range(len(x)):
                if len(replay_memory) < MEMORY_PER_CLASS * 5 * (task + 1):
                    replay_memory.append((x[i].cpu(), y[i].item()))

    # Evaluation
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
plt.title("Semantic CIFAR-100 - Replay Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("replay_accuracy_curve_semantic_cifar100.png")
print("\n✅ Accuracy curve saved as 'replay_accuracy_curve_semantic_cifar100.png'!")

forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)

print("\n\U0001F4CA Evaluation Results (Replay on Semantic CIFAR-100):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")
