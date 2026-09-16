"""VENDORED from jungwoo-ahn/DronePhotographer origin/main @ 30cdb168.

`src/policy/autophoto/{reward.py,vendor/models_lpf/*}` — AutoPhoto's pretrained aesthetic
ResNet18, which IS the policy's observation: `PhotoEnv` feeds the RL agent this network's
512-d features, never the raw frame. Reproducing it exactly is therefore not optional.

Flattened (`models_lpf.resnet` -> `_resnet`) and import paths rewritten; nothing else changed.
Weights: `baseline_ckpts/weights/autophoto/resnet-model42.pt`.
"""
from src.baselines.autophoto_vendor._reward import AestheticReward  # noqa: F401

__all__ = ["AestheticReward"]
