import torch
from torch import nn


class VisionQNetwork(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        action_dim: int = 6,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_outputs: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_fc = nn.Linear(64, feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(image)
        feats = feats.view(feats.size(0), -1)
        feats = torch.relu(self.image_fc(feats))
        x = torch.cat([feats, action], dim=1)
        return self.head(x)
