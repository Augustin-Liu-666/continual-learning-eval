import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from iam.dataloader.iam_dataloader import get_iam_tasks_by_class_split
from iam.model.cnn_iam import CNN_IAM
from calc_metrics import compute_cl_metrics
from cost_tracker import CostTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_tasks = 8
epochs = 10
lr = 0.001
memory_size = 200
report_dir = "iam/reports"
os.makedirs(report_dir, exist_ok=True)

train_loaders, test_loaders = get_iam_tasks_by_class_split(n_tasks=num_tasks, batch_size=batch_size, root_dir="data/iam")


class AGEM:
    def __init__(self, model, device, memory_size=200):
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.mem_x = None
        self.mem_y = None

    def update_memory(self, loader):
        collected_x, collected_y = [], []
        for x, y in loader:
            for xi, yi in zip(x, y):
                collected_x.append(xi)
                collected_y.append(yi.item())
                if len(collected_x) >= self.memory_size:
                    break
            if len(collected_x) >= self.memory_size:
                break
        new_x = torch.stack(collected_x)
        new_y = torch.tensor(collected_y)
        if self.mem_x is None:
            self.mem_x, self.mem_y = new_x, new_y
        else:
            self.mem_x = torch.cat([self.mem_x, new_x], dim=0)
            self.mem_y = torch.cat([self.mem_y, new_y], dim=0)

    def project_gradient(self, criterion):
        if self.mem_x is None:
            return
        current_grad = torch.cat([
            p.grad.data.view(-1).clone() if p.grad is not None else torch.zeros(p.numel(), device=self.device)
            for p in self.model.parameters()
        ])
        idx = torch.randperm(len(self.mem_x))[:min(256, len(self.mem_x))]
        self.model.zero_grad()
        criterion(self.model(self.mem_x[idx].to(self.device)), self.mem_y[idx].to(self.device)).backward()
        ref_grad = torch.cat([
            p.grad.data.view(-1).clone() if p.grad is not None else torch.zeros(p.numel(), device=self.device)
            for p in self.model.parameters()
        ])
        dot = torch.dot(current_grad, ref_grad)
        if dot < 0:
            current_grad = current_grad - (dot / (torch.dot(ref_grad, ref_grad) + 1e-8)) * ref_grad
        idx_start = 0
        for p in self.model.parameters():
            numel = p.numel()
            grad_slice = current_grad[idx_start:idx_start + numel].view(p.shape)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.data.copy_(grad_slice)
            idx_start += numel

    def memory_bytes(self):
        if self.mem_x is None:
            return 0
        return self.mem_x.element_size() * self.mem_x.numel() + self.mem_y.element_size() * self.mem_y.numel()


model = CNN_IAM(num_classes=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
agem = AGEM(model, device, memory_size=memory_size)
tracker = CostTracker(method_name="AGEM")
acc_log = pd.DataFrame(index=[f"T{i}" for i in range(num_tasks)])

for t in range(num_tasks):
    print(f"\n▶️ Training on Task T{t}")
    tracker.start_task(t)
    model.train()
    for _ in range(epochs):
        for x, y in train_loaders[t]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            if t > 0:
                agem.project_gradient(criterion)
            optimizer.step()

    agem.update_memory(train_loaders[t])
    n_mem = len(agem.mem_x) if agem.mem_x is not None else 0
    tracker.end_task(t, extra_storage_bytes=agem.memory_bytes(),
                     extra_storage_label=f"episodic_memory ({n_mem} samples)")

    model.eval()
    for k in range(t + 1):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[k]:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc_log.loc[f"T{k}", f"T{t}"] = correct / total

acc_csv_path = os.path.join(report_dir, "acc_log_agem.csv")
acc_log.to_csv(acc_csv_path)
tracker.print_summary()
tracker.save_report(os.path.join(report_dir, "cost_agem.csv"))

plt.figure(figsize=(8, 6))
for col in acc_log.columns:
    plt.plot(acc_log.index, acc_log[col], marker='o', label=col)
plt.title("Accuracy vs Task (IAM - A-GEM)")
plt.xlabel("Training Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "acc_plot_agem.png"))
print(f"✅ Saved: {acc_csv_path}")
compute_cl_metrics(acc_csv_path)
