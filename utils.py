import torch
import numpy as np

# Evaluate accuracy on a single test loader
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)

            # 保持原始图像形状用于 CNN，不要展平
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total

# Calculate Forgetting, BWT, FWT metrics
def compute_forgetting_metrics(acc_matrix):
    """
    acc_matrix: list of list
        acc_matrix[i][j] 表示第 i 次训练后对第 j 个任务的准确率
    """
    num_tasks = len(acc_matrix)
    max_len = max(len(row) for row in acc_matrix)

    # 将不规则 acc_matrix 转换为等长的矩阵，缺失的补 nan
    padded = np.full((num_tasks, max_len), np.nan)
    for i, row in enumerate(acc_matrix):
        padded[i, :len(row)] = row

    # Forgetting
    forgettings = []
    for t in range(max_len - 1):
        if np.isnan(padded[-1, t]):
            continue
        max_acc = np.nanmax(padded[t+1:, t])
        forgetting = max_acc - padded[-1, t]
        forgettings.append(forgetting)
    avg_forgetting = np.nanmean(forgettings)

    # BWT
    bwt = []
    for t in range(max_len - 1):
        if t + 1 < num_tasks and not np.isnan(padded[-1, t]) and not np.isnan(padded[t, t]):
            bwt.append(padded[-1, t] - padded[t, t])
    avg_bwt = np.nanmean(bwt)

    # FWT
    fwt = []
    for i in range(1, num_tasks):
        for j in range(i):
            if not np.isnan(padded[i, j]):
                fwt.append(padded[i, j])
    avg_fwt = np.nanmean(fwt) if fwt else float("nan")

    return avg_forgetting, avg_bwt, avg_fwt