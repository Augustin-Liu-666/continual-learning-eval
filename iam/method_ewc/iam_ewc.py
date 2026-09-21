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
lambda_ewc = 10.0
report_dir = "iam/reports"
os.makedirs(report_dir, exist_ok=True)

train_loaders, test_loaders = get_iam_tasks_by_class_split(n_tasks=num_tasks, batch_size=batch_size, root_dir="data/iam")


class EWC:
    def __init__(self, model, dataloader, device, lambda_ewc):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            criterion(self.model(x), y).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ewc * loss


model = CNN_IAM(num_classes=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
prev_ewcs = []
tracker = CostTracker(method_name="EWC")
params_bytes = CostTracker.params_bytes(model)
acc_log = pd.DataFrame(index=[f"T{i}" for i in range(num_tasks)])

for t in range(num_tasks):
    print(f"\n▶️ Training on Task T{t}")
    tracker.start_task(t)
    model.train()
    for _ in range(epochs):
        for x, y in train_loaders[t]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            for ewc in prev_ewcs:
                loss += ewc.penalty(model)
            loss.backward()
            optimizer.step()

    prev_ewcs.append(EWC(model, train_loaders[t], device, lambda_ewc))
    extra_bytes = len(prev_ewcs) * 2 * params_bytes
    tracker.end_task(t, extra_storage_bytes=extra_bytes,
                     extra_storage_label=f"fisher+params ({len(prev_ewcs)} tasks)")

    model.eval()
    for k in range(t + 1):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[k]:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc_log.loc[f"T{k}", f"T{t}"] = correct / total

acc_csv_path = os.path.join(report_dir, "acc_log_ewc.csv")
acc_log.to_csv(acc_csv_path)
tracker.print_summary()
tracker.save_report(os.path.join(report_dir, "cost_ewc.csv"))

plt.figure(figsize=(8, 6))
for col in acc_log.columns:
    plt.plot(acc_log.index, acc_log[col], marker='o', label=col)
plt.title("Accuracy vs Task (IAM - EWC)")
plt.xlabel("Training Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "acc_plot_ewc.png"))
print(f"✅ Saved: {acc_csv_path}")
compute_cl_metrics(acc_csv_path)
