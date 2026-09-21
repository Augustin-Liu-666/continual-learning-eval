import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, matplotlib.pyplot as plt

from cifar100.dataloader.cifar100_dataloader import get_cifar100_tasks, TASK_FINE_CLASSES
from cifar100.model.cnn_cifar100 import CNN_CIFAR100
from calc_metrics import compute_cl_metrics
from cost_tracker import CostTracker


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


class AGEM:
    def __init__(self, model, device, memory_size=200):
        self.model = model; self.device = device
        self.memory_size = memory_size
        self.mem_x = self.mem_y = None

    def update_memory(self, loader):
        xs, ys = [], []
        for x, y in loader:
            for xi, yi in zip(x, y):
                xs.append(xi); ys.append(yi.item())
                if len(xs) >= self.memory_size:
                    break
            if len(xs) >= self.memory_size:
                break
        new_x = torch.stack(xs)
        new_y = torch.tensor(ys)
        if self.mem_x is None:
            self.mem_x, self.mem_y = new_x, new_y
        else:
            self.mem_x = torch.cat([self.mem_x, new_x])
            self.mem_y = torch.cat([self.mem_y, new_y])

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
        ptr = 0
        for p in self.model.parameters():
            n = p.numel()
            g = current_grad[ptr:ptr+n].view(p.shape)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.data.copy_(g)
            ptr += n

    def memory_bytes(self):
        if self.mem_x is None: return 0
        return self.mem_x.element_size()*self.mem_x.numel() + self.mem_y.element_size()*self.mem_y.numel()


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


def train_agem(model, train_loaders, test_loaders, device, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    agem = AGEM(model, device, memory_size=args.memory_size)
    tracker = CostTracker(method_name="A-GEM")
    acc_matrix = []

    for task_id, loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()
        for _ in range(args.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                if task_id > 0:
                    agem.project_gradient(criterion)
                optimizer.step()

        agem.update_memory(loader)
        n_mem = len(agem.mem_x) if agem.mem_x is not None else 0
        tracker.end_task(task_id,
                         extra_storage_bytes=agem.memory_bytes(),
                         extra_storage_label=f"episodic_memory ({n_mem} samples)")
        acc = evaluate(model, test_loaders[:task_id+1], device, task_classes=TASK_FINE_CLASSES)
        acc_matrix.append(acc)
        print(f"After Task {task_id+1}: {[f'{a:.3f}' for a in acc]}")
    return acc_matrix, tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks',     type=int,   default=10)
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--memory_size', type=int,   default=200)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--seed',        type=int,   default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loaders, test_loaders = get_cifar100_tasks(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = CNN_CIFAR100(num_classes=100).to(device)

    acc_matrix, tracker = train_agem(model, train_loaders, test_loaders, device, args)

    os.makedirs("cifar100/reports", exist_ok=True)
    df = pd.DataFrame(acc_matrix)
    df.index   = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}"       for i in range(args.n_tasks)]
    csv_path = f"cifar100/reports/acc_matrix_AGEM_{args.n_tasks}tasks_mem{args.memory_size}_seed{args.seed}.csv"
    df.to_csv(csv_path)
    tracker.print_summary()
    tracker.save_report(f"cifar100/reports/cost_AGEM_{args.n_tasks}tasks_mem{args.memory_size}_seed{args.seed}.csv")

    plt.figure(figsize=(9,5))
    for task in range(args.n_tasks):
        vals = [acc_matrix[t][task] if len(acc_matrix[t]) > task else None for t in range(len(acc_matrix))]
        plt.plot(range(1, len(vals)+1), vals, marker='o', label=f"Task {task+1}")
    plt.title("Accuracy vs Task (CIFAR-100 – A-GEM)")
    plt.xlabel("Training Task"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(f"cifar100/reports/acc_plot_AGEM_{args.n_tasks}tasks.png")
    plt.close()
    compute_cl_metrics(csv_path)
