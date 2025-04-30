# cnn_cifar10_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 第一层卷积 / First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积 / Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        # 第三层卷积 / Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        # 池化层 / Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层 / Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32x32 -> 32x32
        x = self.pool(x)            # 32x32 -> 16x16
        x = F.relu(self.conv2(x))   # 16x16 -> 16x16
        x = self.pool(x)            # 16x16 -> 8x8
        x = F.relu(self.conv3(x))   # 8x8 -> 8x8
        x = self.pool(x)            # 8x8 -> 4x4
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN(num_classes=2)
    print(model)
    # 测试前向传播 / Test a dummy forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    out = model(dummy_input)
    print(out.shape)  # Should be [1, 2]
