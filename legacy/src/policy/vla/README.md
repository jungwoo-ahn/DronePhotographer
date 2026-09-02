# `src/policy/vla/`

π0-style VLA baseline (issue #22) — "ours without the world model".

A VLM (Qwen3-VL-2B) encodes the current frame; the normalized goal vector is
projected to soft tokens appended to the VLM hidden states; a flow-matching
action expert cross-attends to that context and denoises the 5D action chunk. No
future-frame prediction (the world model we ablate), no value head.

- `model.py` — `VLAActionPolicy` (backbone + goal soft-tokens + flow action expert).
- `action_expert.py` — cross-attention flow-matching action expert.
- `dataset.py` — `VLADroneDataset` + `VLACollate` (Qwen processor runs in dataloader workers).
- `trainer.py` — `VLATrainer` (flow loss, held-out validation, DDP, EMA best/last ckpt).
- `REFERENCES.md` — π0 / Qwen3-VL attribution and the faithful-simplification note.

Run `scripts/train_vla_policy.py --config configs/policy/vla_qwen3_2b.yaml`; eval
with `scripts/eval_vla_policy.py`. Same v7 data / HER goal sampling / held-out
`val_scenes.txt` / 5D action / pose-distance eval as the Cosmos and
diffusion-policy baselines, so they are directly comparable. Reuses
`src/policy/common/` (goal space, 5D action, dataset base, flow) rather than
duplicating it.

This directory is the home for VLA-family baselines; additional ones (OpenVLA,
RT-2, ...) can live alongside, mirroring this layout.
