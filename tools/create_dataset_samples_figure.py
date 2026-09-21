import gzip
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data"
OUT = PROJECT / "reports" / "dataset_samples.png"
TMP = Path("/tmp/cl_dataset_samples")


def read_idx_images(path: Path, n=10):
    with open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file")
    count = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    return arr.reshape(count, rows, cols)[:n]


def ensure_cifar100():
    TMP.mkdir(parents=True, exist_ok=True)
    archive = TMP / "cifar-100-python.tar.gz"
    extracted = TMP / "cifar-100-python"
    if not extracted.exists():
        if not archive.exists():
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(TMP)
    return extracted


def load_cifar100(n=10):
    extracted = ensure_cifar100()
    with open(extracted / "train", "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"][:n]
    imgs = data.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
    return imgs


def make_grid():
    mnist_path = DATA_ROOT / "MNIST" / "raw" / "train-images-idx3-ubyte"
    emnist_path = DATA_ROOT / "EMNIST" / "raw" / "emnist-byclass-train-images-idx3-ubyte"

    mnist = read_idx_images(mnist_path, n=25)
    emnist = read_idx_images(emnist_path, n=10)
    cifar = load_cifar100(n=10)

    angles = list(range(0, 100, 10))
    rotated = []
    base_digit = mnist[0]
    for angle in angles:
        pil = Image.fromarray(base_digit)
        rotated.append(np.asarray(pil.rotate(angle, fillcolor=0)))

    # EMNIST IDX images are stored transposed relative to ordinary display.
    emnist_display = [np.flipud(np.rot90(img, k=3)) for img in emnist]

    fig, axes = plt.subplots(3, 10, figsize=(13.5, 4.5))
    rows = [
        ("Rotated MNIST", rotated, "gray"),
        ("EMNIST", emnist_display, "gray"),
        ("CIFAR-100", cifar, None),
    ]

    for r, (label, images, cmap) in enumerate(rows):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(images[c], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=48, va="center")
            if r == 0:
                ax.set_title(f"{angles[c]}°", fontsize=9)

    fig.suptitle("Representative Dataset Samples", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.94], h_pad=0.7, w_pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    make_grid()
