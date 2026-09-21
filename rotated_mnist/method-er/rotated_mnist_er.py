import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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


def train_er(model, train_loaders, test_loaders, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    memory_data = []
    memory_label = []
    acc_matrix = []
    tracker = CostTracker(method_name="ER")

    for task_id, train_loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()

        if memory_data:
            memory_x = torch.cat(memory_data)
            memory_y = torch.cat(memory_label)
        else:
            memory_x = memory_y = None

        for _ in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if memory_x is not None:
                    mem_idx = torch.randperm(memory_x.size(0))[:args.batch_size]
                    loss += criterion(model(memory_x[mem_idx].to(device)), memory_y[mem_idx].to(device))
                loss.backward()
                optimizer.step()

        # Update replay buffer (fixed per-task allocation)
        buf_per_task = args.buffer_size // args.n_tasks
        mem_x_list, mem_y_list = [], []
        for x, y in train_loader:
            mem_x_list.append(x)
            mem_y_list.append(y)
            if sum(t.size(0) for t in mem_x_list) >= buf_per_task:
                break
        memory_data.append(torch.cat(mem_x_list)[:buf_per_task])
        memory_label.append(torch.cat(mem_y_list)[:buf_per_task])

        extra_bytes = sum(t.element_size() * t.numel() for t in memory_data + memory_label)
        tracker.end_task(task_id, extra_storage_bytes=extra_bytes,
                         extra_storage_label=f"replay_buffer ({sum(t.size(0) for t in memory_data)} samples)")

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


def save_acc_matrix(acc_matrix, method_name, n_tasks, seed=0, buffer_size=None):
    df = pd.DataFrame(acc_matrix)
    df.index = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}" for i in range(n_tasks)]
    os.makedirs("reports", exist_ok=True)
    buf_tag = f"_buf{buffer_size}" if buffer_size is not None else ""
    path = f"reports/acc_matrix_{method_name}_{n_tasks}tasks{buf_tag}_seed{seed}.csv"
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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loaders, test_loaders = get_rotated_mnist_dataloaders(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = MLP(input_size=784, hidden_size=256, output_size=10).to(device)

    acc_matrix, tracker = train_er(model, train_loaders, test_loaders, device, args)
    avg_acc, forgetting = summarize(acc_matrix)

    print(f"\nAvg Final ACC: {avg_acc*100:.2f}%")
    print(f"Avg Forgetting: {forgetting*100:.2f}%")

    save_acc_matrix(acc_matrix, method_name="ER", n_tasks=args.n_tasks, seed=args.seed, buffer_size=args.buffer_size)
    plot_acc_lines(acc_matrix, method_name="ER", n_tasks=args.n_tasks)
    tracker.print_summary()
    tracker.save_report(f"reports/cost_ER_{args.n_tasks}tasks_buf{args.buffer_size}_seed{args.seed}.csv")
