import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, matplotlib.pyplot as plt

from cifar100.dataloader.cifar100_dataloader import get_cifar100_tasks, TASK_FINE_CLASSES
from cifar100.model.cnn_cifar100 import CNN_CIFAR100
from calc_metrics import compute_cl_metrics
from cost_tracker import CostTracker


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


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


def train_er(model, train_loaders, test_loaders, device, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    memory_x, memory_y = None, None
    tracker = CostTracker(method_name="ER")
    acc_matrix = []
    buf_per_task = args.buffer_size // args.n_tasks

    for task_id, loader in enumerate(train_loaders):
        tracker.start_task(task_id)
        model.train()
        for _ in range(args.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if memory_x is not None:
                    idx = torch.randperm(memory_x.size(0))[:args.batch_size]
                    loss += criterion(model(memory_x[idx].to(device)), memory_y[idx].to(device))
                loss.backward()
                optimizer.step()

        # Update buffer
        mem_x_list, mem_y_list = [], []
        for x, y in loader:
            mem_x_list.append(x); mem_y_list.append(y)
            if sum(t.size(0) for t in mem_x_list) >= buf_per_task:
                break
        new_x = torch.cat(mem_x_list)[:buf_per_task]
        new_y = torch.cat(mem_y_list)[:buf_per_task]
        memory_x = new_x if memory_x is None else torch.cat([memory_x, new_x])
        memory_y = new_y if memory_y is None else torch.cat([memory_y, new_y])

        mem_bytes = memory_x.element_size()*memory_x.numel() + memory_y.element_size()*memory_y.numel()
        tracker.end_task(task_id,
                         extra_storage_bytes=mem_bytes,
                         extra_storage_label=f"replay_buffer ({len(memory_x)} samples)")
        acc = evaluate(model, test_loaders[:task_id+1], device, task_classes=TASK_FINE_CLASSES)
        acc_matrix.append(acc)
        print(f"After Task {task_id+1}: {[f'{a:.3f}' for a in acc]}")
    return acc_matrix, tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks',     type=int,   default=10)
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--buffer_size', type=int,   default=1000)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--seed',        type=int,   default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loaders, test_loaders = get_cifar100_tasks(n_tasks=args.n_tasks, batch_size=args.batch_size)
    model = CNN_CIFAR100(num_classes=100).to(device)

    acc_matrix, tracker = train_er(model, train_loaders, test_loaders, device, args)

    os.makedirs("cifar100/reports", exist_ok=True)
    df = pd.DataFrame(acc_matrix)
    df.index   = [f"After_Task_{i+1}" for i in range(len(acc_matrix))]
    df.columns = [f"Task_{i+1}"       for i in range(args.n_tasks)]
    csv_path = f"cifar100/reports/acc_matrix_ER_{args.n_tasks}tasks_buf{args.buffer_size}_seed{args.seed}.csv"
    df.to_csv(csv_path)
    tracker.print_summary()
    tracker.save_report(f"cifar100/reports/cost_ER_{args.n_tasks}tasks_buf{args.buffer_size}_seed{args.seed}.csv")

    plt.figure(figsize=(9,5))
    for task in range(args.n_tasks):
        vals = [acc_matrix[t][task] if len(acc_matrix[t]) > task else None for t in range(len(acc_matrix))]
        plt.plot(range(1, len(vals)+1), vals, marker='o', label=f"Task {task+1}")
    plt.title("Accuracy vs Task (CIFAR-100 – ER)")
    plt.xlabel("Training Task"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(f"cifar100/reports/acc_plot_ER_{args.n_tasks}tasks.png")
    plt.close()
    compute_cl_metrics(csv_path)
