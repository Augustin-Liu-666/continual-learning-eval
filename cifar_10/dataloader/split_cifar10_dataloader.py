# split_cifar10_dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 类别索引 / Class indexes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Split CIFAR-10的任务划分 / Task splits
split_tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# CIFAR-10 标准数据增强 / Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# 下载CIFAR-10 / Download CIFAR-10
total_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
total_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# 划分任务 / Create task-specific datasets
def create_split_cifar10_tasks(dataset):
    task_datasets = []
    targets = torch.tensor(dataset.targets)
    for class1, class2 in split_tasks:
        mask = (targets == class1) | (targets == class2)
        data = dataset.data[mask.numpy()]
        labels = targets[mask]
        labels = (labels == class2).long()  # class2作为正类1, class1为0

        # 将numpy数据转成tensor
        data = torch.tensor(data).permute(0, 3, 1, 2).float() / 255.0
        # 注意: 需要按照训练时的标准化处理
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(3,1,1)
        data = (data - mean) / std

        task_dataset = TensorDataset(data, labels)
        task_datasets.append(task_dataset)
    return task_datasets

# 创建任务数据 / Create task datasets
train_tasks = create_split_cifar10_tasks(total_train)
test_tasks = create_split_cifar10_tasks(total_test)

# 每个任务的DataLoader / Per-task Dataloaders
batch_size = 64
train_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in train_tasks]
test_loaders = [DataLoader(task, batch_size=batch_size, shuffle=False) for task in test_tasks]

print("✅ Split-CIFAR10 DataLoaders Ready!")
