import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from rotated_mnist.dataloader.rotated_mnist_dataloader import train_loaders, test_loaders
from rotated_mnist.model.simple_mlp import MLP  # type: ignore
from utils import evaluate, compute_forgetting_metrics  # type: ignore
from collections import deque

# 配置
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 简化配置
num_tasks = 5
epochs = 1
lr = 0.01
buffer_size = 5  # 减少为 5

model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 缓存之后用 deque 实现 FIFO 
past_data = deque(maxlen=buffer_size * num_tasks)

accuracy_matrix = []

print("\n✅ Rotated MNIST DataLoaders Ready!")

for task in range(num_tasks):
    print(f"Training on Task {task+1}")
    model.train()
    for epoch in range(epochs):
        for (x, y) in train_loaders[task]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 当前任务损失
            output = model(x)
            loss = criterion(output, y)

            # 重操经验 buffer 损失
            if len(past_data) > 0:
                replay_samples = random.sample(past_data, min(5, len(past_data)))
                replay_x = torch.cat([sample[0] for sample in replay_samples]).to(device)
                replay_y = torch.cat([sample[1] for sample in replay_samples]).to(device)
                replay_output = model(replay_x)
                loss += criterion(replay_output, replay_y)

            loss.backward()
            optimizer.step()

            # 缓存当前样本
            past_data.append((x.detach(), y.detach()))

    # 每个任务后进行测试
    model.eval()
    acc_list = []
    with torch.no_grad():
        for test_task in range(task + 1):
            acc = evaluate(model, test_loaders[test_task], device)
            acc_list.append(acc)
        accuracy_matrix.append(acc_list)
    print(f"Accuracy after Task {task+1}: {acc_list}")

# 结果评价
import matplotlib.pyplot as plt
acc_matrix = np.zeros((num_tasks, num_tasks))
for i, row in enumerate(accuracy_matrix):
    acc_matrix[i, :len(row)] = row

# 折线图
for t in range(num_tasks):
    plt.plot(range(t+1, num_tasks+1), acc_matrix[t:, t], label=f"Task {t+1}")
plt.xlabel("After Task")
plt.ylabel("Accuracy (%)")
plt.title("GEM Accuracy on Rotated MNIST (Memory Optimized)")
plt.legend()
plt.tight_layout()
plt.savefig("gem_accuracy_curve_rotated_mnist.png")
print("Accuracy curve saved as 'gem_accuracy_curve_rotated_mnist.png'!")

# 按照先前缓存的原则评估 forgetting
forget, bwt, fwt = compute_forgetting_metrics(accuracy_matrix)
print("\n📊 Evaluation Results (GEM Optimized on Rotated MNIST):")
print(f"Average Forgetting: {forget:.2f}%")
print(f"Average BWT: {bwt:.2f}%")
print(f"Average FWT: {fwt:.2f}%")