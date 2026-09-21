import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# Select the available compute device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download and load the original MNIST dataset.
base_transform = transforms.ToTensor()
train_base = datasets.MNIST(root="data", train=True, download=True, transform=base_transform)
test_base = datasets.MNIST(root="data", train=False, download=True, transform=base_transform)

# Generate evenly spaced rotation angles.
def get_rotation_angles(n_tasks):
    max_rotation = 90  # Set to 180 for a wider drift range.
    return [i * max_rotation / (n_tasks - 1) for i in range(n_tasks)]

# Build one task dataset per rotation angle.
def create_rotated_tasks(dataset, angles):
    task_datasets = []
    for angle in angles:
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

# Return aligned lists of training and test loaders.
def get_rotated_mnist_dataloaders(n_tasks=5, batch_size=64):
    rotation_angles = get_rotation_angles(n_tasks)
    train_tasks = create_rotated_tasks(train_base, rotation_angles)
    test_tasks = create_rotated_tasks(test_base, rotation_angles)
    train_loaders = [DataLoader(task, batch_size=batch_size, shuffle=True) for task in train_tasks]
    test_loaders = [DataLoader(task, batch_size=batch_size, shuffle=False) for task in test_tasks]
    print(f"✅ Rotated MNIST DataLoaders Ready! Tasks: {n_tasks}, Angles: {rotation_angles}")
    return train_loaders, test_loaders
