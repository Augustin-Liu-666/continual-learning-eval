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
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
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
            memory_dataset = torch.utils.data.TensorDataset(
                torch.cat(memory_data), torch.cat(memory_label)
            )
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset, batch_size=args.batch_size, shuffle=True
            )
            combined_loader = list(train_loader) + list(memory_loader)
            random.shuffle(combined_loader)
        else:
            combined_loader = train_loader

        for x, y in combined_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        buffer_size = args.buffer_size // args.n_tasks
        mem_x, mem_y = [], []
        for i, (x, y) in enumerate(train_loader):
            mem_x.append(x)
            mem_y.append(y)
            if len(mem_x) * x.size(0) >= buffer_size:
                break
        memory_data.append(torch.cat(mem_x)[:buffer_size])
        memory_label.append(torch.cat(mem_y)[:buffer_size])

        mem_bytes = sum(t.element_size() * t.numel() for t in memory_data + memory_label)
        n_mem = sum(t.size(0) for t in memory_data)
        tracker.end_task(task_id, extra_storage_bytes=mem_bytes,
                         extra_storage_label=f"replay_buffer ({n_mem} samples)")

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
        if not valid_accs:
            continue
        forgetting += max(valid_accs) - final_acc[task]
    forgetting /= n_tasks
    return avg_final_acc, forgetting


def save_acc_matrix(acc_matrix, method_name, n_tasks, seed=0, buffer_size=None):
    df = pd.DataFrame(acc_matrix)
    df.index = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}" for i in range(n_tasks)]
    os.makedirs("emnist/reports", exist_ok=True)
    buf_tag = f"_buf{buffer_size}" if buffer_size is not None else ""
    path = f"emnist/reports/acc_matrix_{method_name}_{n_tasks}tasks{buf_tag}_seed{seed}.csv"
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
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_loaders, test_loaders = get_emnist_tasks(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = MLP(input_size=784, hidden_size=256, output_size=36).to(device)

    acc_matrix, tracker = train_er(model, train_loaders, test_loaders, device, args)
    avg_acc, forgetting = summarize(acc_matrix)

    print(f"\nAvg Final ACC: {avg_acc*100:.2f}%")
    print(f"Avg Forgetting: {forgetting*100:.2f}%")

    save_acc_matrix(acc_matrix, method_name="EMNIST_ER", n_tasks=args.n_tasks, seed=args.seed, buffer_size=args.buffer_size)
    plot_acc_lines(acc_matrix, method_name="EMNIST_ER", n_tasks=args.n_tasks)
    tracker.print_summary()
    tracker.save_report(f"emnist/reports/cost_ER_{args.n_tasks}tasks_buf{args.buffer_size}_seed{args.seed}.csv")
