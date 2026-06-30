# `src/policy/diffusion_policy/`

Diffusion Policy baseline (issue #22) — "ours without the world model".

A frozen DINOv2-large encoder turns the current frame into a global observation
embedding; the normalized goal vector is embedded and concatenated; a conditional
1D U-Net (Chi et al.) denoises the 5D action chunk via DDPM (epsilon-prediction),
sampled at inference with DDIM. No future-frame prediction (the world model we
ablate), no value head.

- `denoiser.py` — `ConditionalUnet1D` (1D U-Net over the chunk axis, FiLM conditioning).
- `model.py` — `DiffusionPolicy` (frozen DINOv2 obs encoder + DDPM/DDIM action head).
- `dataset.py` — `DiffusionPolicyDataset` + `DPCollate` (DINOv2 processor runs in dataloader workers).
- `trainer.py` — `DPTrainer` (DDPM noise-prediction loss, held-out validation, DDP, EMA best/last ckpt).
- `REFERENCES.md` — Diffusion Policy / DINOv2 / diffusers attribution.

Run `scripts/train_diffusion_policy.py --config configs/policy/diffusion_policy_dinov2.yaml`;
eval with `scripts/eval_diffusion_policy.py`. Same v7 data / HER goal sampling /
held-out `val_scenes.txt` / 5D action / pose-distance eval as the Cosmos and VLA
baselines, so they are directly comparable. Reuses `src/policy/common/` (goal
space, 5D action, dataset base) rather than duplicating it. Being frozen-backbone
the head trains fast at a large batch (no bs=1 dispatch stall).
