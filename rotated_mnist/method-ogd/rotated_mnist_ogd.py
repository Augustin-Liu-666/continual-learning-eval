import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

from rotated_mnist.dataloader.rotated_mnist_dataloader import get_rotated_mnist_dataloaders
from rotated_mnist.model.simple_mlp import MLP
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


class OGD:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.task_gradients = []

    def store_task_gradient(self, loader):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            grad = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
            self.task_gradients.append(grad.detach())
            break

    def project_gradient(self):
        if not self.task_gradients:
            return
        current_grad = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        for past_grad in self.task_gradients:
            proj = torch.dot(current_grad, past_grad) / (torch.norm(past_grad) ** 2 + 1e-8)
            current_grad = current_grad - proj * past_grad
        idx = 0
        for p in self.model.parameters():
            if p.grad is not None:
                numel = p.grad.numel()
                p.grad.data = current_grad[idx:idx + numel].view_as(p.grad).data.clone()
                idx += numel


def train_ogd(model, train_loaders, test_loaders, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    ogd = OGD(model, device)
    acc_matrix = []
    tracker = CostTracker(method_name="OGD")
    params_bytes = CostTracker.params_bytes(model)

    for task_id, train_loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            ogd.project_gradient()
            optimizer.step()

        ogd.store_task_gradient(train_loader)
        # OGD stores one gradient vector per past task
        extra_bytes = len(ogd.task_gradients) * params_bytes
        tracker.end_task(task_id, extra_storage_bytes=extra_bytes,
                         extra_storage_label=f"gradient_vecs ({len(ogd.task_gradients)} tasks)")

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


def save_acc_matrix(acc_matrix, method_name, n_tasks, seed=0):
    df = pd.DataFrame(acc_matrix)
    df.index = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}" for i in range(n_tasks)]
    os.makedirs("reports", exist_ok=True)
    path = f"reports/acc_matrix_{method_name}_{n_tasks}tasks_seed{seed}.csv"
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
    os.makedirs("reports", exist_ok=True)
    path = f"reports/acc_plot_{method_name}_{n_tasks}tasks.png"
    plt.savefig(path)
    print(f"Accuracy plot saved to {path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loaders, test_loaders = get_rotated_mnist_dataloaders(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = MLP(input_size=784, hidden_size=256, output_size=10).to(device)

    acc_matrix, tracker = train_ogd(model, train_loaders, test_loaders, device, args)
    avg_acc, forgetting = summarize(acc_matrix)

    print(f"\nAvg Final ACC: {avg_acc*100:.2f}%")
    print(f"Avg Forgetting: {forgetting*100:.2f}%")

    save_acc_matrix(acc_matrix, method_name="OGD", n_tasks=args.n_tasks, seed=args.seed)
    plot_acc_lines(acc_matrix, method_name="OGD", n_tasks=args.n_tasks)
    tracker.print_summary()
    tracker.save_report(f"reports/cost_OGD_{args.n_tasks}tasks_seed{args.seed}.csv")
