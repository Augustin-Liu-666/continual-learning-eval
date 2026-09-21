import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# superclass i → 5 fine-grained class IDs (fixed CIFAR-100 mapping)
SUPERCLASS_TO_FINE = {
    0:  [4, 30, 55, 72, 95],
    1:  [1, 32, 67, 73, 91],
    2:  [54, 62, 70, 82, 92],
    3:  [9, 10, 16, 28, 61],
    4:  [0, 51, 53, 57, 83],
    5:  [22, 39, 40, 86, 87],
    6:  [5, 20, 25, 84, 94],
    7:  [6,  7, 14, 18, 24],
    8:  [3, 42, 43, 88, 97],
    9:  [12, 17, 37, 68, 76],
    10: [23, 33, 49, 60, 71],
    11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75],
    13: [26, 45, 77, 79, 99],
    14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93],
    16: [36, 50, 65, 74, 80],
    17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],
    19: [41, 69, 81, 85, 89],
}

# Pair adjacent superclasses → 10 tasks × 10 fine classes each
TASK_FINE_CLASSES = [
    sorted(SUPERCLASS_TO_FINE[2*i] + SUPERCLASS_TO_FINE[2*i+1])
    for i in range(10)
]


def _load_raw(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b'fine_labels'])
    return imgs, labels


def _normalise(imgs):
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(1,3,1,1)
    std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(1,3,1,1)
    return (imgs - mean) / std


def get_cifar100_tasks(n_tasks=10, batch_size=64, root='data'):
    assert n_tasks == 10, "CIFAR-100 split is fixed at 10 tasks"
    train_imgs, train_labels = _load_raw(f'{root}/cifar-100-python/train')
    test_imgs,  test_labels  = _load_raw(f'{root}/cifar-100-python/test')

    train_imgs = _normalise(train_imgs)
    test_imgs  = _normalise(test_imgs)

    train_loaders, test_loaders = [], []
    for task_id, fine_classes in enumerate(TASK_FINE_CLASSES):
        for split, imgs, labels, shuffle in [
            ('train', train_imgs, train_labels, True),
            ('test',  test_imgs,  test_labels,  False),
        ]:
            mask = np.isin(labels, fine_classes)
            x = torch.tensor(imgs[mask])
            y = torch.tensor(labels[mask], dtype=torch.long)  # global labels 0-99
            ds = TensorDataset(x, y)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            if split == 'train':
                train_loaders.append(loader)
            else:
                test_loaders.append(loader)

    print(f"✅ CIFAR-100 Tasks Ready: {n_tasks} tasks × 10 classes each "
          f"(~{len(train_loaders[0].dataset)} train, ~{len(test_loaders[0].dataset)} test per task)")
    return train_loaders, test_loaders
