import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from iam.dataloader.iam_dataloader import get_iam_tasks_by_class_split
from iam.model.cnn_iam import CNN_IAM
from iam.method_ogd.ogd import OGD
from calc_metrics import compute_cl_metrics
from cost_tracker import CostTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_tasks = 8
epochs = 10
lr = 0.001
report_dir = "iam/reports"
os.makedirs(report_dir, exist_ok=True)

train_loaders, test_loaders = get_iam_tasks_by_class_split(n_tasks=num_tasks, batch_size=batch_size, root_dir="data/iam")

model = CNN_IAM(num_classes=200).to(device)
ogd = OGD(model, device, lr=lr)
tracker = CostTracker(method_name="OGD")
params_bytes = CostTracker.params_bytes(model)
acc_log = pd.DataFrame(index=[f"T{i}" for i in range(num_tasks)])

for t in range(num_tasks):
    print(f"\n▶️ Training on Task T{t}")
    tracker.start_task(t)
    ogd.train_task(train_loaders[t], epochs=epochs)

    n_grads = len(ogd.task_grads)
    tracker.end_task(t, extra_storage_bytes=n_grads * params_bytes,
                     extra_storage_label=f"task_grads ({n_grads} vectors)")

    model.eval()
    for k in range(t + 1):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loaders[k]:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc_log.loc[f"T{k}", f"T{t}"] = correct / total

acc_csv_path = os.path.join(report_dir, "acc_log_ogd.csv")
acc_log.to_csv(acc_csv_path)
tracker.print_summary()
tracker.save_report(os.path.join(report_dir, "cost_ogd.csv"))

plt.figure(figsize=(8, 6))
for col in acc_log.columns:
    plt.plot(acc_log.index, acc_log[col], marker='o', label=col)
plt.title("Accuracy vs Task (IAM - OGD)")
plt.xlabel("Training Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "acc_plot_ogd.png"))
print(f"✅ Saved: {acc_csv_path}")
compute_cl_metrics(acc_csv_path)
