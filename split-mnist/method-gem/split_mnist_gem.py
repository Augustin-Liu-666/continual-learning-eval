# split_mnist_gem.py

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import quadprog

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
memory_buffer_size = 500
memory_per_task = memory_buffer_size // 5

# 下载并准备Split-MNIST数据 / Download and prepare Split-MNIST
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
        task_dataset = TensorDataset(data.unsqueeze(1).float()/255.0, targets)
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
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
accuracies = []

memory_data = {}
memory_target = {}

def get_flat_grad_vector(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)

def overwrite_grad(model, new_grad_vector):
    index = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_shape = p.grad.shape
            num_param = p.grad.numel()
            p.grad.data = new_grad_vector[index:index+num_param].view(grad_shape).data
            index += num_param

def project_gradients(g_current, g_past_tasks):
    gradients = torch.stack(g_past_tasks)
    t = gradients.size(0)
    P = torch.matmul(gradients, gradients.t()).double().cpu().numpy()
    P = 0.5 * (P + P.T) + np.eye(t) * 1e-3
    q = (-1.0 * torch.matmul(gradients, g_current)).double().cpu().numpy()
    G = np.eye(t)
    h = np.zeros(t)

    v = quadprog.solve_qp(P, q, G, h)[0]
    v_tensor = torch.tensor(v, dtype=gradients.dtype, device=gradients.device)
    new_grad = torch.matmul(v_tensor, gradients).float()
    g_proj = g_current + new_grad
    return g_proj

for task_id, loader in enumerate(task_loaders):
    print(f"Training on Task {task_id+1}")
    model.train()

    memory_data[task_id] = []
    memory_target[task_id] = []

    for epoch in range(10):
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            g_current = get_flat_grad_vector(model)

            if task_id > 0:
                past_grads = []
                for past_task in range(task_id):
                    mem_data = torch.stack(memory_data[past_task]).to(device)
                    mem_target = torch.stack(memory_target[past_task]).to(device)

                    mem_output = model(mem_data)
                    mem_loss = criterion(mem_output, mem_target)

                    optimizer.zero_grad()
                    mem_loss.backward()
                    past_grad = get_flat_grad_vector(model)
                    past_grads.append(past_grad)

                g_proj = project_gradients(g_current, past_grads)
                overwrite_grad(model, g_proj)

            optimizer.step()

            for d, t in zip(data.cpu(), target.cpu()):
                if len(memory_data[task_id]) < memory_per_task:
                    memory_data[task_id].append(d)
                    memory_target[task_id].append(t)
                else:
                    idx = random.randint(0, memory_per_task-1)
                    memory_data[task_id][idx] = d
                    memory_target[task_id][idx] = t

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
plt.title('Accuracy Evolution Across Tasks (GEM Training)')
plt.xticks([1,2,3,4,5])
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.savefig('gem_accuracy_curve.png')


print("\n✅ Accuracy curve saved as 'gem_accuracy_curve.png'!")

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
average_fwt = np.mean(fwt_values) if len(fwt_values)>0 else float('nan')

print("\n📈 Evaluation Results (GEM Training):")
print(f"Average Forgetting: {average_forgetting:.2f}%")
print(f"Average BWT: {-bwt:.2f}%")
print(f"Average FWT: {average_fwt:.2f}%")
