import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_IAM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_IAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # Output: 32 x 32 x 128
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                             # Output: 32 x 16 x 64

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Output: 64 x 16 x 64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                             # Output: 64 x 8 x 32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128 x 8 x 32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)                             # Output: 128 x 4 x 16

        self.fc1 = nn.Linear(128 * 4 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten the convolutional features.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
