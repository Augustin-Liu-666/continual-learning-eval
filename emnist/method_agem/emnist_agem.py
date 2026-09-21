import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

from emnist.dataloader.emnist_dataloader import get_emnist_tasks
from emnist.model.simple_mlp import MLP
from cost_tracker import CostTracker


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, test_loaders, device):
    model.eval()
    acc_list = []
    with torch.no_grad():
        for test_loader in test_loaders:
            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            acc_list.append(correct / total)
    return acc_list


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


def train_agem(model, train_loaders, test_loaders, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    agem = AGEM(model, device, memory_size=args.memory_size)
    acc_matrix = []
    tracker = CostTracker(method_name="AGEM")

    for task_id, train_loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()
        for _ in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                if task_id > 0:
                    agem.project_gradient(criterion)
                optimizer.step()

        agem.update_memory(train_loader)
        n_mem = len(agem.mem_x) if agem.mem_x is not None else 0
        tracker.end_task(task_id, extra_storage_bytes=agem.memory_bytes(),
                         extra_storage_label=f"episodic_memory ({n_mem} samples)")

        acc = evaluate(model, test_loaders[:task_id + 1], device)
        acc_matrix.append(acc)
        print(f"After Task {task_id+1}, ACC: {acc}")

    return acc_matrix, tracker


def summarize(acc_matrix):
    n_tasks = len(acc_matrix)
    final_acc = acc_matrix[-1]
    avg_final_acc = sum(final_acc) / n_tasks
    forgetting = 0.0
    for task in range(n_tasks):
        valid_accs = [accs[task] for accs in acc_matrix if len(accs) > task]
        if valid_accs:
            forgetting += max(valid_accs) - final_acc[task]
    forgetting /= n_tasks
    return avg_final_acc, forgetting


def save_acc_matrix(acc_matrix, method_name, n_tasks, seed=0, memory_size=None):
    df = pd.DataFrame(acc_matrix)
    df.index = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}" for i in range(n_tasks)]
    os.makedirs("emnist/reports", exist_ok=True)
    mem_tag = f"_mem{memory_size}" if memory_size is not None else ""
    path = f"emnist/reports/acc_matrix_{method_name}_{n_tasks}tasks{mem_tag}_seed{seed}.csv"
    df.to_csv(path)
    print(f"Accuracy matrix saved to {path}")


def plot_acc_lines(acc_matrix, method_name, n_tasks):
    for task in range(n_tasks):
        acc_over_time = [acc[task] if len(acc) > task else None for acc in acc_matrix]
        plt.plot(range(1, len(acc_over_time) + 1), acc_over_time, label=f"Task {task+1}")
    plt.xlabel("Training Task Number")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Task (Method: {method_name})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("emnist/reports", exist_ok=True)
    path = f"emnist/reports/acc_plot_{method_name}_{n_tasks}tasks.png"
    plt.savefig(path)
    print(f"Accuracy plot saved to {path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--memory_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loaders, test_loaders = get_emnist_tasks(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = MLP(input_size=784, hidden_size=256, output_size=36).to(device)

    acc_matrix, tracker = train_agem(model, train_loaders, test_loaders, device, args)
    avg_acc, forgetting = summarize(acc_matrix)

    print(f"\nAvg Final ACC: {avg_acc*100:.2f}%")
    print(f"Avg Forgetting: {forgetting*100:.2f}%")

    save_acc_matrix(acc_matrix, method_name="EMNIST_AGEM", n_tasks=args.n_tasks, seed=args.seed, memory_size=args.memory_size)
    plot_acc_lines(acc_matrix, method_name="EMNIST_AGEM", n_tasks=args.n_tasks)
    tracker.print_summary()
    tracker.save_report(f"emnist/reports/cost_AGEM_{args.n_tasks}tasks_mem{args.memory_size}_seed{args.seed}.csv")
