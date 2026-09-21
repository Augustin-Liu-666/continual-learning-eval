import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import random

N_CLASSES = 36  # EMNIST: A-Z and 0-9, for 36 classes in total.

from PIL import ImageFilter

def get_emnist_tasks(n_tasks=5, batch_size=64):
    assert 1 <= n_tasks <= 5, "The number of tasks must be between 1 and 5."
    raw_train = datasets.EMNIST(root="data", split="byclass", train=True, download=True)
    raw_test = datasets.EMNIST(root="data", split="byclass", train=False, download=True)

    def filter_36(data):
        return [(img, label) for img, label in data if label < 36]

    train_data = filter_36(raw_train)
    test_data = filter_36(raw_test)

    task_transforms = [
        transforms.Lambda(lambda x: x),  # Original image
        transforms.Lambda(lambda x: transforms.functional.rotate(x, angle=random.uniform(-30, 30))),  # Rotation
        transforms.Lambda(lambda x: x.filter(ImageFilter.GaussianBlur(radius=1.0))),  # Blur
        transforms.ColorJitter(brightness=0.5),  # Brightness
        transforms.Compose([  # Combined transformation
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomApply([transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=random.uniform(-20, 20)))], p=0.8)
        ])
    ]

    train_loaders, test_loaders = [], []

    for i in range(n_tasks):
        tform = task_transforms[i]

        train_imgs, train_labels = [], []
        for img, label in train_data:
            img = tform(img)
            img = transforms.ToTensor()(img)
            train_imgs.append(img)
            train_labels.append(label)
        train_tensor = TensorDataset(torch.stack(train_imgs), torch.tensor(train_labels))
        train_loaders.append(DataLoader(train_tensor, batch_size=batch_size, shuffle=True))

        test_imgs, test_labels = [], []
        for img, label in test_data:
            img = tform(img)
            img = transforms.ToTensor()(img)
            test_imgs.append(img)
            test_labels.append(label)
        test_tensor = TensorDataset(torch.stack(test_imgs), torch.tensor(test_labels))
        test_loaders.append(DataLoader(test_tensor, batch_size=batch_size, shuffle=False))

    print(f"✅ EMNIST Tasks Ready: {n_tasks}")
    return train_loaders, test_loaders
