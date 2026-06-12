# Cosmos integration — upstream references

This directory ports patterns from NVIDIA's open-source Cosmos stack. We do **not** vendor `cosmos-policy` as a dependency; we re-implement the small pieces we need against `diffusers` directly. The references below are for attribution and for keeping our code aligned with upstream design intent.

## Upstream packages

| Upstream | Version pinned | License | What we use |
|---|---|---|---|
| `nvidia/Cosmos-Predict2.5-2B` (HuggingFace) | latest | NVIDIA Open Model License | The model weights themselves (video diffusion transformer + VAE + Cosmos-Reason1 text encoder). Loaded via `diffusers.DiffusionPipeline.from_pretrained`. |
| `nvlabs/cosmos-policy` | commit `18a2accadf4e7a3531e56754102af5a24d2316da` (fetched 2026-05-25 to /tmp) | Apache-2.0 | Architectural reference only. |
| `diffusers` | ≥0.35.2 | Apache-2.0 | Backbone loader, scheduler, VAE. Real dependency. |

## Files referenced from `cosmos-policy@18a2acc`

These are *not* copied; we read them and re-implemented the equivalent functionality minimally for our use case (shot-profile vector conditioning, 5D action head, value head).

| Upstream file | Our adaptation | What we kept | What we changed |
|---|---|---|---|
| `cosmos_policy/conditioner.py` | `src/policy/cosmos/conditioner.py` | Mutable `BaseCondition` dataclass shape; idea of an embedder that maps raw batch fields to cross-attention tensors. | Replaced `Text2WorldCondition` (T5 embeddings via cross-attention) with `ShotProfileVectorConditioner` (8-dim goal vector → cross-attention tokens via a learned `Linear`). No `_src.predict2` import — standalone PyTorch. |
| `cosmos_policy/models/policy_video2world_model.py` | `src/policy/cosmos/model.py` | Pattern of a wrapper around a video diffusion backbone that adds extra heads. The "action-as-latent-frame" idea is replaced by an explicit action MLP head (simpler for 5D continuous actions). | Drops dependence on `ImaginaireModel`/`_src` internals. Loads backbone via `DiffusionPipeline.from_pretrained` instead. Adds an explicit 5D action head + scalar value head; both predict from the transformer's pooled hidden state. |
| `cosmos_policy/datasets/aloha_dataset.py` | `src/policy/cosmos/dataset.py` | The split into per-platform dataset classes (one per robot family). | Replaced with `CosmosDroneDataset` using our v6 placement-keyed annotation schema. Adds 4-frame VAE padding for length-1 trajectory prototype (repeats the rendered image 4× along time). |
| `cosmos_policy/trainer.py` | `src/policy/cosmos/trainer.py` | Outer epoch/iter loop structure. Checkpoint-on-save_iter pattern. | Drops `ImaginaireTrainer` superclass (depends on `_src` framework). Plain PyTorch training loop with `torch.optim.AdamW` + `torch.utils.tensorboard.SummaryWriter`. Joint loss: `λ_flow * flow_matching + λ_action * mse(action) + λ_value * mse(value)`. |

## License notes

- **Apache-2.0** code from `cosmos-policy` is permissively reusable; this file fulfils the attribution clause.
- The Cosmos-Predict2.5-2B **weights** are governed by the [NVIDIA Open Model License](https://github.com/nvidia-cosmos/cosmos-predict2.5#license) — commercial use is allowed; derivative models are allowed; attribution and the license terms must be preserved in any redistribution.
- Nothing from `cosmos_policy._src/` is reproduced here; that subdirectory depends on the unreleased `imaginaire` framework.

## How to refresh

If a newer cosmos-policy commit changes a pattern materially (e.g. new conditioner abstraction), re-clone to `/tmp/cosmos-policy` and diff against this directory. Update the pinned commit hash above and note material changes in commit messages.
