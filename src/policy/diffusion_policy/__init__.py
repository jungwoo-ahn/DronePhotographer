"""Diffusion Policy baseline (issue #22) — frozen DINOv2 + conditional 1D U-Net.

"Ours without the world model": a modern frozen vision backbone + a DDPM action
head, contrasted against the Cosmos video world-action policy to isolate the
previsualization contribution. See REFERENCES.md.
"""

from src.policy.diffusion_policy.denoiser import ConditionalUnet1D
from src.policy.diffusion_policy.model import DiffusionPolicy, DPLossOutputs, DPOutputs

__all__ = ["ConditionalUnet1D", "DiffusionPolicy", "DPLossOutputs", "DPOutputs"]
