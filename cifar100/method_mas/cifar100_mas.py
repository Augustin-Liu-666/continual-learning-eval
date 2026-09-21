import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, matplotlib.pyplot as plt

from cifar100.dataloader.cifar100_dataloader import get_cifar100_tasks, TASK_FINE_CLASSES
from cifar100.model.cnn_cifar100 import CNN_CIFAR100
from calc_metrics import compute_cl_metrics
from cost_tracker import CostTracker


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


class MAS:
    def __init__(self, model, device, lambda_reg=1.0):
        self.model = model; self.device = device; self.lambda_reg = lambda_reg
        self.importance = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, dataloader):
        self.model.eval()
        new_imp = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        for x, _ in dataloader:
            x = x.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            torch.norm(out, p=2, dim=1).mean().backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_imp[n] += p.grad.abs().detach()
        for n in new_imp:
            new_imp[n] /= len(dataloader)
        for n in self.importance:
            self.importance[n] = new_imp[n]
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.clone().detach()

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss += (self.importance[n] * (p - self.prev_params[n]).pow(2)).sum()
        return self.lambda_reg * loss


def evaluate(model, loaders, device, task_classes=None):
    """Task-incremental eval: mask logits to current task's classes if task_classes given."""
    model.eval()
    out = []
    inf_mask = torch.full((100,), float('-inf'))
    with torch.no_grad():
        for j, loader in enumerate(loaders):
            c = t = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                if task_classes is not None:
                    mask = inf_mask.clone().to(device)
                    mask[list(task_classes[j])] = 0.0
                    logits = logits + mask
                c += (logits.argmax(1) == y).sum().item(); t += y.size(0)
            out.append(c / t)
    return out


def train_mas(model, train_loaders, test_loaders, device, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    mas = MAS(model, device, lambda_reg=args.lambda_reg)
    tracker = CostTracker(method_name="MAS")
    params_bytes = CostTracker.params_bytes(model)
    acc_matrix = []

    for task_id, loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()
        for _ in range(args.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if task_id > 0:
                    loss += mas.penalty()
                loss.backward()
                optimizer.step()

        mas.update(loader)
        tracker.end_task(task_id,
                         extra_storage_bytes=2 * params_bytes,
                         extra_storage_label="importance + prev_params")
        acc = evaluate(model, test_loaders[:task_id+1], device, task_classes=TASK_FINE_CLASSES)
        acc_matrix.append(acc)
        print(f"After Task {task_id+1}: {[f'{a:.3f}' for a in acc]}")
    return acc_matrix, tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks',    type=int,   default=10)
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loaders, test_loaders = get_cifar100_tasks(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = CNN_CIFAR100(num_classes=100).to(device)

    acc_matrix, tracker = train_mas(model, train_loaders, test_loaders, device, args)

    os.makedirs("cifar100/reports", exist_ok=True)
    df = pd.DataFrame(acc_matrix)
    df.index   = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}"       for i in range(args.n_tasks)]
    csv_path = f"cifar100/reports/acc_matrix_MAS_{args.n_tasks}tasks_seed{args.seed}.csv"
    df.to_csv(csv_path)
    tracker.print_summary()
    tracker.save_report(f"cifar100/reports/cost_MAS_{args.n_tasks}tasks_seed{args.seed}.csv")

    plt.figure(figsize=(9,5))
    for task in range(args.n_tasks):
        vals = [acc_matrix[t][task] if len(acc_matrix[t]) > task else None for t in range(len(acc_matrix))]
        plt.plot(range(1, len(vals)+1), vals, marker='o', label=f"Task {task+1}")
    plt.title("Accuracy vs Task (CIFAR-100 – MAS)")
    plt.xlabel("Training Task"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(f"cifar100/reports/acc_plot_MAS_{args.n_tasks}tasks.png")
    plt.close()
    compute_cl_metrics(csv_path)
