"""AutoPhoto aesthetic reward — the pretrained scorer reused as the RL reward.

AutoPhoto's reward is a low-pass ResNet18 (Adobe `models_lpf`) with a final linear
head to a scalar aesthetic score; the released `resnet-model42.pt` is loaded and the
head sign-flipped so higher = better (their convention). This is the one piece of
AutoPhoto we reuse directly (pure PyTorch); the RL *policy* is retrained in our
Blender env. See REFERENCES.md.

`score(image) -> float` is the per-view reward; `score_and_features(image)` also
returns the 512-d penultimate features (the observation AutoPhoto feeds its policy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AestheticReward:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cuda", filter_size: int = 3) -> None:
        from torchvision import transforms

        from src.policy.autophoto.vendor.models_lpf import resnet18

        model = resnet18(filter_size=filter_size)
        model.fc = nn.Linear(512, 1)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state)
        # AutoPhoto sign-flips the head so a higher score means a better photo.
        model.fc.weight = nn.Parameter(-model.fc.weight)
        model.fc.bias = nn.Parameter(-model.fc.bias)
        model.eval().to(device)
        self.model = model
        self.device = device

        # capture the 512-d input to fc (the policy observation) on each forward
        self._feat: torch.Tensor | None = None
        self.model.fc.register_forward_pre_hook(
            lambda _m, inp: setattr(self, "_feat", inp[0].detach()))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ])

    @torch.no_grad()
    def score(self, image) -> float:
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return float(self.model(t).item())

    @torch.no_grad()
    def score_and_features(self, image) -> tuple[float, np.ndarray]:
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        s = float(self.model(t).item())
        feat = self._feat.squeeze(0).float().cpu().numpy()   # (512,)
        return s, feat

    @property
    def feature_dim(self) -> int:
        return 512


__all__ = ["AestheticReward", "IMAGENET_MEAN", "IMAGENET_STD"]
