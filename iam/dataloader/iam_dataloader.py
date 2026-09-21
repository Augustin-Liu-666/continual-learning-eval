import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class IAMDataset(Dataset):
    def __init__(self, root_dir, class_ids=None, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_file = os.path.join(root_dir, "meta", "labels.txt")
        self.transform = transform
        self.samples = []

        with open(self.label_file, "r") as f:
            for line in f:
                image_name, label = line.strip().split(",")
                label = int(label)
                if class_ids is not None and label not in class_ids:
                    continue
                image_path = os.path.join(self.image_dir, image_name)
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_iam_task_transforms():
    base_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return [base_transform] * 8


def get_iam_tasks_by_class_split(n_tasks=8, batch_size=64, root_dir="data/iam", train_ratio=0.8):
    assert 200 % n_tasks == 0, "The 200 classes must divide evenly across tasks."
    class_per_task = 200 // n_tasks
    transforms_list = get_iam_task_transforms()
    train_loaders, test_loaders = [], []

    for i in range(n_tasks):
        class_ids = list(range(i * class_per_task, (i + 1) * class_per_task))
        dataset = IAMDataset(root_dir=root_dir, class_ids=class_ids, transform=transforms_list[i])

        n_train = int(len(dataset) * train_ratio)
        n_test = len(dataset) - n_train
        if n_test == 0:
            n_train = max(0, len(dataset) - 1)
            n_test = len(dataset) - n_train

        train_set, test_set = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        train_loaders.append(DataLoader(train_set, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_set, batch_size=batch_size, shuffle=False))

    print(f"\n✅ IAM Tasks Loaded: {n_tasks} tasks × {class_per_task} classes each (train/test split {train_ratio:.0%}/{1-train_ratio:.0%})")
    return train_loaders, test_loaders
