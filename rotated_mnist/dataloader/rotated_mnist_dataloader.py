# rotated_mnist_dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 配置设备 / Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置旋转角度 / Rotation angles per task
rotation_angles = [0, 15, 30, 45, 60]

# 下载原始MNIST数据 / Download MNIST
base_transform = transforms.ToTensor()
train_base = datasets.MNIST(root='../data', train=True, download=True, transform=base_transform)
test_base = datasets.MNIST(root='../data', train=False, download=True, transform=base_transform)

# 构建旋转任务 / Create rotated tasks
def create_rotated_tasks(dataset, angles):
    task_datasets = []
    for angle in angles:
        rotated_transform = transforms.Compose([
            transforms.RandomRotation((angle, angle)),  # 固定旋转角度
            transforms.ToTensor()
        ])
        rotated_data = []
        rotated_labels = []
        for img, label in dataset:
            img = transforms.functional.rotate(img, angle)
            rotated_data.append(img)
            rotated_labels.append(label)
        data_tensor = torch.stack(rotated_data)
        labels_tensor = torch.tensor(rotated_labels)
        task_dataset = TensorDataset(data_tensor, labels_tensor)
        task_datasets.append(task_dataset)
    return task_datasets

# 创建训练和测试任务 / Create train and test tasks
train_tasks = create_rotated_tasks(train_base, rotation_angles)
test_tasks = create_rotated_tasks(test_base, rotation_angles)

# 每个任务的DataLoader / Per-task DataLoaders
batch_size = 64
train_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in train_tasks]
test_loaders = [DataLoader(task, batch_size=batch_size, shuffle=False) for task in test_tasks]

print("✅ Rotated MNIST DataLoaders Ready!")
