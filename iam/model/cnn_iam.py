import torch
import torch.nn as nn

class CNN_IAM(nn.Module):
    def __init__(self, num_classes=200):
        super(CNN_IAM, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B, 32, 32, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 32, 16, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 16, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 64, 8, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 8, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 128, 4, 16]

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # [B, 256, 4, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                 # [B, 256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                    # [B, 256]
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x