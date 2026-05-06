import torch.nn as nn


class StudentAdapter(nn.Module):
    def __init__(self, in_dim=128, out_dim=512, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)