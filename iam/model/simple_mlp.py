# iam/model/simple_mlp.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=32 * 128, hidden_size=512, output_size=12214):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # [B, 1, 32, 128] → [B, 4096]
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)