import argparse, os, random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from emnist.dataloader.emnist_dataloader import get_emnist_tasks
from emnist.model.simple_mlp import MLP
from emnist.method_ogd.ogd import OGD
from emnist.cost_tracker import CostTracker
from calc_metrics import compute_cl_metrics


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks',  type=int,   default=5)
    parser.add_argument('--epochs',     type=int,   default=4)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=0.0003)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    report_dir = "emnist/reports"
    os.makedirs(report_dir, exist_ok=True)
    acc_log_path  = os.path.join(report_dir, f"ogd_acc_log_seed{args.seed}.csv")
    cost_log_path = os.path.join(report_dir, f"ogd_cost_log_seed{args.seed}.csv")

    train_loaders, test_loaders = get_emnist_tasks(n_tasks=args.num_tasks, batch_size=args.batch_size)
    print(f"\n✅ EMNIST Tasks Ready: {args.num_tasks}")

    model = MLP().to(device)
    ogd = OGD(model=model, device=device, lr=args.lr)
    acc_log = pd.DataFrame(index=[f"T{i}" for i in range(args.num_tasks)])
    tracker = CostTracker(method_name="OGD")

    for t in range(args.num_tasks):
        print(f"\n▶️ Training on Task {t}")
        tracker.start_task(task_id=t)
        ogd.train_task(train_loaders[t], epochs=args.epochs)
        ogd.end_task(train_loaders[t])

        ogd_extra_bytes = CostTracker.params_bytes(model) * 2
        tracker.end_task(task_id=t,
                         extra_storage_bytes=ogd_extra_bytes,
                         extra_storage_label="prev_grads + prev_params (1 task)")

        model.eval()
        for k in range(t + 1):
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loaders[k]:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            acc_log.loc[f"T{k}", f"T{t}"] = correct / total

    acc_log.to_csv(acc_log_path)
    print(f"\n✅ Accuracy log saved to: {acc_log_path}")
    tracker.print_summary()
    tracker.save_report(cost_log_path)

    compute_cl_metrics(acc_log_path)
