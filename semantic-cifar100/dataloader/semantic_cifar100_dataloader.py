import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset

# Config
BATCH_SIZE = 64
NUM_TASKS = 10  # Split CIFAR-100 into 10 tasks (each 10 classes)

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load CIFAR-100
train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

# Custom dataset for remapped labels
class MappedSubset(Dataset):
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, self.label_mapping[y]

    def __len__(self):
        return len(self.subset)

# Split data into tasks
train_loaders = []
test_loaders = []

classes_per_task = 100 // NUM_TASKS
for task_id in range(NUM_TASKS):
    class_range = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
    label_mapping = {original: idx for idx, original in enumerate(class_range)}

    train_indices = [i for i, label in enumerate(train_dataset.targets) if label in class_range]
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label in class_range]

    mapped_train = MappedSubset(Subset(train_dataset, train_indices), label_mapping)
    mapped_test = MappedSubset(Subset(test_dataset, test_indices), label_mapping)

    train_loader = DataLoader(mapped_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mapped_test, batch_size=BATCH_SIZE, shuffle=False)

    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

print("\n✅ Semantic CIFAR-100 DataLoaders Ready!")
