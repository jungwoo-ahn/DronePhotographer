# Cosmos-Predict2.5 — input/output formats and our adaptations

This document is the **contract** for what shapes/dtypes flow between modules in `src/policy/cosmos/`. It maps directly to GH issue **#18 (005-cosmos-policy-integration)**, which defines what we take from upstream cosmos-policy and what we deliberately change.

Source-of-truth files for the upstream specs (verified at commit `18a2acc` of `nvlabs/cosmos-policy`):

- VAE: `cosmos_policy/_src/predict2/tokenizers/base_vae.py`
- Transformer: `cosmos_policy/_src/predict2/networks/minimal_v1_lvg_dit.py`
- Conditioner: `cosmos_policy/conditioner.py` + `cosmos_policy/_src/predict2/conditioner.py`
- Dataset (reference): `cosmos_policy/datasets/aloha_dataset.py`
- Training loss: `cosmos_policy/_src/predict2/models/text2world_model.py:339–397`

---

## Tensor shape glossary

| Symbol | Value | Where |
|---|---|---|
| `B` | batch size (per-GPU) | everywhere |
| `C` | image channels = **3** | image / video |
| `T` | temporal frames in clip = **4** (prototype) | image / video |
| `H, W` | image height, width = **480, 720** (Cosmos 720p) | image / video |
| `C_lat` | latent channels = **16** | VAE latent (Cosmos predict2) |
| `T_lat, H_lat, W_lat` | latent temporal/spatial dims (depend on VAE tokenizer) | VAE latent |
| `N_tok` | cross-attention tokens — **512** matching T5; 4 of those positions carry our goal | conditioner |
| `D_xattn` | cross-attention emb dim = **1024** (T5-11B hidden size) | conditioner |
| `D_goal` | goal-vector dim = **8** (V5_SCORE_KEYS); the v6 prototype uses 2 (az/el) | conditioner input |
| `D_act` | action dim = **5** `(Δx, Δy, Δz, Δyaw, Δpitch)` in camera-local frame | action head |
| `D_val` | per-step value: **chunk_size** (`cost_to_go`) or **chunk_size × goal_dim** (`achieved_profile` / `profile_delta`) — selectable via `value_target_mode` | value latent |
| `chunk` | future action steps predicted per diffusion sample = **1** (v6) → **4/8** (post-#17) | model.chunk_size |
| `T_total` | full latent sequence length = `T_img + 1 (+1 if value_latent)` | model |

---

## Module-by-module contract

### 1. `vae.py` — `CosmosVAEWrapper`

**Input** (image in, normalized): `(B, C, T, H, W)` float32 in `[-1, 1]`.

**Output** (latent): `(B, C_lat=16, T_lat, H_lat, W_lat)` float32 (already rescaled by `(latent_mean, latent_std)` for native Cosmos VAEs, or by `config.scaling_factor` for diffusers-wrapped VAEs).

**Axis convention**: `(B, C, T, H, W)` everywhere — same on input and output. No `(B, T, C, H, W)` permutation.

**Helpers**:
- `assemble_clip(state_image, next_state_image, T=4) -> (B, C, T, H, W)` — frame 0 is state, frames 1..T-1 repeat next_state. Filler frames use `next_state` (not zeros) to avoid shifting VAE statistics.
- `encode_pair(state_image, next_state_image, T=4) -> latent` — convenience: assemble then encode.

**Why this differs from upstream**: cosmos-policy's tokenizer has no fixed 4-frame requirement (it's an arch property of Wan2pt1/Wan2pt2), but for our **length-1 trajectory prototype** we still need T>1 so the temporal axis is non-trivial. We choose T=4 to match the cookbook examples. Once issue #17 (camera-trajectory-sampling) provides real trajectories, this becomes a true 4-frame clip and the filler-repeat goes away.

---

### 2. `conditioner.py` — `ShotProfileVectorConditioner` (zero-init prefix on a T5 anchor)

**Input** (batch dict): `{"goal_vec": (B, D_goal=8) float}` — already normalized to `[-1, 1]` via `src/policy/common/goal_space.normalize_goal`.

**Output** (`ShotProfileCondition` dataclass):

| Field | Shape | Notes |
|---|---|---|
| `crossattn_emb` | `(B, real_len + K, D)` | Real Qwen text tokens (~6) + K=4 goal tokens. D = Qwen per-layer concat (3584×28 = 100352). No padding tokens emitted. |
| `padding_mask` | `(B, real_len + K) bool` | All True — every emitted token is valid (no padding to mask). |
| `raw_goal` | `(B, D_goal=8)` | Kept for debugging / value-head input if we want it |

**Architecture (zero-init goal tokens, ControlNet-style — no padding emitted)**:

```
anchor_text: (real_len, D)  ← frozen buffer (real text only), from assets/text_anchor.pt
gate:        ()             ← nn.Parameter, initialized to 0
goal_proj:   Linear(8, K·D)

# At forward:
text = anchor_text.expand(B, -1, -1)                              # (B, real_len, D)
goal_tokens = (gate * keep) * goal_proj(goal_vec).view(B, K, D)   # 0 at init
emb = cat([text, goal_tokens], dim=1)                            # (B, real_len + K, D), all valid
```

At init `gate=0` → the goal tokens are exactly zero, so the context is the anchor's
real text + K zero tokens (the pretrained Qwen-conditioned path). Gradient descent
ramps the gate up as the action/value/flow losses demand goal-specific signal.

> We deliberately never emit the anchor's padding region. The earlier design kept
> all 512 anchor positions (real text + ~500 *non-zero*, per-position-normalized
> padding embeddings) behind a `padding_mask` that the diffusers backbone adapter
> never applied (it sets the transformer's spatial `padding_mask`, not the
> cross-attention `attention_mask`) — so the K goal tokens were drowned by ~500
> constant padding tokens. Emitting only valid tokens fixes that at the source.

**Building the anchor** (one-time, requires the Qwen2.5-VL text encoder):

```bash
python scripts/build_text_anchor.py --prompt "A drone cinematography" --output assets/text_anchor.pt
```

This is the **only** time the Qwen2.5-VL text encoder is loaded; the saved tensor
(real text tokens only) is loaded by the conditioner at every training/inference run.

**Adaptation from upstream** (issue #18: "Goal condition: We need to embed shot profiles somehow"):

| Upstream Text2WorldCondition | Our ShotProfileCondition |
|---|---|
| Per-sample T5-11B encode of a natural-language command, `(B, 512, 1024) bf16` | Per-sample = fixed anchor `(512, 1024)` + zero-init goal injection at K positions |
| T5-11B loaded throughout training (or precomputed-cached for unique commands) | T5-11B loaded once, encoder discarded; only the ~1 MB anchor stays |
| Encoder is ~11 B params | Conditioner is ~33K params (Linear + gate scalar) |

**Why zero-init prefix instead of replacing T5 with a Linear**:

- A randomly-initialized Linear puts the conditioner output **off the T5 manifold** — the frozen Cosmos backbone has never seen those activations.
- The anchor keeps **all 512 positions in-distribution** by design.
- The zero-init gate gives the same "training starts as the pretrained model" guarantee that ControlNet uses for spatial conditioning.

**Escalation path**: if the gate plateaus at a small value while loss stalls, the goal signal isn't fitting through the prefix-injection bottleneck. The next step is **decoupled cross-attention (IP-Adapter style)**: add a parallel `K, V` projection in each cross-attention layer that attends to a separate stream of goal tokens, with the output projection zero-initialized. This requires touching the backbone's `Attention` blocks but gives the model far more capacity to integrate the goal at every layer.

---

### 3. `action_latent.py` + `model.py` — `CosmosWorldActionPolicy` (latent-frame action/value)

**Latent sequence layout** (training; analogous at inference):
```
pos 0 .. T_img-1   : image latents (from VAE)
pos T_img          : action latent  ← filled by tiling action_chunk to fit C·H·W
pos T_img + 1      : value latent   ← filled by tiling the (chunk_size,) per-step value (optional)
```
Total length `T_total = T_img + num_extra_frames` where `num_extra_frames ∈ {1, 2}`.

**Action injection/extraction is parameter-free** (ported verbatim from `cosmos_policy/models/policy_text2world_model.py::replace_latent_with_action_chunk` and `cosmos_utils.py::extract_action_chunk_from_latent_sequence`):

- **Inject**: flatten `(chunk_size, action_dim)` → repeat to fill the latent frame's `C·H·W` volume → reshape back. No learned encoder.
- **Extract**: flatten the predicted latent frame → split into `K = ⌊C·H·W / (chunk·act_dim)⌋` whole tiles → **mean across tiles**. The averaging is what makes the decoding robust to per-element diffusion-sampler noise.
- **Value**: the `(chunk_size,)` per-step cost-to-go is tiled/averaged exactly like the action chunk (per-step dim = 1) via `inject_value_seq` / `extract_value_seq`. (`inject_value` / `extract_value` keep the legacy scalar broadcast for other callers.)

For 480×720 inputs with the Cosmos VAE's 8× spatial compression, a latent frame is `(16, 1, 60, 90) = 86,400` elements. A `(chunk_size=4, action_dim=5) = 20`-float chunk fits **4,320 times** — that's the averaging margin at decode.

**`CosmosWorldActionPolicy` API**:

```python
# Training
loss = policy.compute_loss(
    image_latent: (B, C, T_img, H, W),
    action_chunk: (B, chunk_size, ACTION_DIM),
    goal_vec:     (B, D_goal),
    value_target: (B, chunk_size)  | None,   # per-step cost-to-go
) -> scalar              # single flow-matching MSE over the full sequence

# Inference
out = policy.sample(
    image_latent: (B, C, T_img, H, W),  # conditioning frames; pinned at every step
    goal_vec:     (B, D_goal),
    n_steps:      int = 32,             # rectified-flow Euler steps
) -> PolicyOutputs
# out.pred_action_chunk: (B, chunk_size, ACTION_DIM)
# out.pred_value:        (B, chunk_size)   # per-step cost-to-go
# out.pred_latents:      (B, C, T_total, H, W)
```

**Backbone call convention** (handled by `BackboneAdapter`):

- **`DiffusersStyleAdapter`** (default): `transformer(hidden_states=x, timestep=t, encoder_hidden_states=cross, encoder_attention_mask=mask)` returning `out.sample`. Use when loading via `diffusers.DiffusionPipeline.from_pretrained`.
- **`CosmosNativeAdapter`**: `transformer(x_B_C_T_H_W=x, timesteps_B_T=t, crossattn_emb=cross, padding_mask=mask)` returning a tensor directly. Use when calling cosmos-policy's raw `MinimalV1LVGDiT`.

Pick at construction time. Write a third adapter only if a new backbone API emerges.

**Chunk size**:

- **v6 single-shot data**: `chunk_size=1` — each placement is one camera pose, only one (prev → next) action available per sample. The dataset auto-tiles the single action across the chunk axis for backwards-compatibility with larger chunks.
- **Trajectory data (issue #17)**: bump to `chunk_size=4` or `8` — the model architecture is unchanged, just the dataset emits real K-step sequences.

**Adaptation from upstream** (fully aligned now):

| Upstream `Video2WorldModel` | Our `CosmosWorldActionPolicy` |
|---|---|
| Action = injected latent frame; tile-repeated `(chunk, act_dim)` flattened | Same — `action_latent.inject_action_chunk` is a verbatim port |
| Value = injected latent frame; scalar broadcast | Same — `action_latent.inject_value` |
| Extraction = split flattened latent into tiles, average | Same — `action_latent.extract_action_chunk` |
| Loss = pure flow matching over the whole `(image + action + value)` sequence | Same |
| Inference = sample from the diffusion process, then extract | Same — `policy.sample()` |
| Backbone trains end-to-end | Backbone **frozen** for our prototype (only conditioner trains; ~33K params) |

**Why this is now multimodal-aware**: with the action expressed as a noise-injected latent the model predicts, the diffusion sampler naturally returns *one realization* per call from the model's learned action distribution. For drone cinematography where "orbit left" and "orbit right" are both valid responses to the same `(state, goal)`, this avoids the mode-averaging that a deterministic MLP head would suffer from.

---

### 4. `dataset.py` — `CosmosDroneDataset.__getitem__`

Operates exclusively on the **v7 schema** (`docs/v7_handoff_jooyeol.md` on branch `v7_data_for_cosmos_policy`). One `data.json` per placement; one sample per K-step window inside each `accepted_pairs[i].trajectory_32f`.

| Field | Source |
|---|---|
| `state_image: (3, H, W) [-1, 1]` | Window's start frame JPEG, resolved via `<placement_dir>/<render_records[i][j].path_rel>` |
| `next_state_image: (3, H, W) [-1, 1]` | Window's end frame — ALOHA-style T_img=2 supervision |
| `goal_vec: (D_goal,)` | `render_records[i][end].scores` (the 8 V5 keys), normalized to `[-1, 1]` |
| `action_chunk: (chunk_size, 5)` | K consecutive 5D actions computed from `accepted_pairs[i].trajectory_32f[j].{pos,forward,up}` deltas |
| `value_target: (chunk_size,)` | Per-step cost-to-go: `value[k] = −pose_distance(keyframe_k, goal)` for the state before each action (value[0] = start→goal). Normalized by `VALUE_SCALE`. |
| `meta` | `{annotation_path, pair_idx, start_frame_idx, end_frame_idx, chunk_size, scene, object}` |

With `chunk_size=8, stride=1` over each 32-frame trajectory: **24 windows × K_accepted pairs per placement**. Across 7,885 placements × ~8 K_accepted × 24 windows ≈ **1.5M training samples**.

**Sampling scheme (`sampling_scheme`, default now `multiscale_bidir`)**: instead of the sliding window above, emit — per start frame — one window per signed offset `±o ∈ ±{8,16,24}` whose endpoint exists. The goal is that endpoint, so the SAME start with DIFFERENT endpoints has DIFFERENT action targets → the action must depend on the goal (fixing the collapse to `f(state)`). Offset 16/24 "merge" 2/3 real steps into each of the 8 actions by re-encoding between STRIDED keyframes (camera-local deltas don't sum). ~96 windows/pair; negative offsets subsume `augment_reverse`. `next_state_image`/`goal_vec` come from the endpoint (= goal frame). See `annotations.iter_multiscale_windows`. `sampling_scheme="sliding_window"` restores the legacy HER-window behavior.


**Output dict per sample**:

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `state_image` | `(C=3, H, W)` | float32 | In `[-1, 1]`. The starting view. |
| `next_state_image` | `(C=3, H, W)` | float32 | In `[-1, 1]`. The target view (whose profile is the goal). |
| `goal_vec` | `(D_goal,)` | float32 | Normalized via `normalize_goal`. v6 uses `D_goal=2`. |
| `action_chunk` | `(chunk_size, ACTION_DIM=5)` | float32 | K future (Δx, Δy, Δz, Δyaw, Δpitch) actions in prev-frame camera-local basis. v6 single-shot: auto-tiles one action across the chunk axis. |
| `value_target` | `(chunk_size,)` or `(chunk_size, goal_dim)` | float32 | Per-step value at the state BEFORE each action, selected by `value_target_mode`: **`cost_to_go`** `(chunk_size,)` = `−pose_distance(keyframe_k, goal)` (pose-based, clamp-immune, ÷`VALUE_SCALE`); **`achieved_profile`** `(chunk_size, goal_dim)` = `normalize_goal(profile(keyframe_k))`; **`profile_delta`** `(chunk_size, goal_dim)` = `normalize_goal(goal) − achieved`. Profile modes are score-derived (inherit the off-screen clamp). `resolve_value_spec(mode, goal_dim)` → `(value_dim, value_scale)`. See `src/policy/common/reward.py`, `dataset_base.py`. |
| `meta` | `dict` | — | `annotation_path`, `placement_idx`, `prev_view_idx`, `next_view_idx`, `scene`, `object`. |

**Image normalization**: PNG → `np.asarray / 127.5 - 1.0` → `(C, H, W)` torch tensor in `[-1, 1]`. Resized to `target_resolution=(H, W)` (default `(480, 720)` for Cosmos 720p).

**Adaptation from upstream** (`aloha_dataset.py`):

| Upstream ALOHA dict | Our drone dict |
|---|---|
| `video: (3, T_expanded, 224, 224) uint8` | `state_image + next_state_image: (3, H, W) float32 [-1, 1]` |
| `t5_text_embeddings: (512, 1024)` | Dropped — replaced by `goal_vec: (D_goal,)` |
| `actions: (25, 14) float32` | `action_chunk: (chunk_size, 5) float32` |
| `proprio, future_proprio, value_function_return` | Dropped (no proprioception in drone setting); value is computed downstream by `policy.compute_loss` |
| `action_latent_idx, value_latent_idx` | Computed by `policy.action_latent_idx(t_img)` / `policy.value_latent_idx(t_img)` — not part of the sample dict |

---

### 5. `trainer.py` — `CosmosPolicyTrainer.fit()`

**Per-iteration loss assembly**:
```
clip          = vae.assemble_clip(state_image, next_image)   # (B, C, T=4, H, W)
image_latent  = vae.encode(clip)                              # (B, 16, T_lat, H_lat, W_lat)
loss_out      = policy.compute_loss(image_latent, action_chunk, goal_vec, value_target)
loss          = loss_out.total
```

Inside `compute_loss`:
1. `build_training_latents` assembles `x0` = `[image_latents, action_latent, value_latent]` with the action chunk + value tiled in.
2. Sample `t ~ U(0,1)` and `noise ~ N(0,I)`, form `x_t = (1-t)·noise + t·x0`, target velocity `v = x0 - noise` (rectified flow).
3. Run backbone → `v_pred`, compute squared error tensor `sq = (v_pred - v)²` over the full sequence.
4. **Split `sq` by latent-sequence position** and return per-component means:
   - `loss_world  = sq[:, :, :T_img].mean()`         — averaged over the image latents
   - `loss_action = sq[:, :, T_img].mean()`           — averaged over the action latent
   - `loss_value  = sq[:, :, T_img + 1].mean()`       — averaged over the value latent
5. Combine: `total = λ_world · world + λ_action · action + λ_value · value`.

**Joint training, properly balanced**: world + action + value are trained together under one flow-matching objective. Per-component means (normalized by each component's element count) put the three on a comparable scale regardless of T_img — important because uniform averaging would let the image-latent volume dominate by a factor of T_img. The lambdas (default 1.0 each in `configs/policy/cosmos_2b_v6_proto.yaml::loss`) are the knobs to bias which component fits faster.

**Adaptation from upstream** (`text2world_model.py:training_step`):

| Upstream | Ours |
|---|---|
| EDM σ + `BALANCED_TWO_HEADS_V1` + per-σ weighting `(σ² + σ_data²) / (σ · σ_data)²` | **Same** (see "EDM σ sampling + BALANCED_TWO_HEADS_V1" section below) |
| Single joint loss over (image + action + value) latent frames | Same, but decomposed by latent position into `world` / `action` / `value` so we can log + reweight each |
| `imaginaire.trainer.ImaginaireTrainer` with DDP/FSDP, callbacks, profiling | Plain PyTorch loop with AdamW + linear LR warmup |
| Checkpointer with safetensors + sharding | `torch.save` of full state dict |

---

## Summary — issue #18 mapping

| Issue #18 requirement | Our implementation file | Notes |
|---|---|---|
| "Use Cosmos 2B" video model | `model.py` (backbone via diffusers) | Frozen for the prototype |
| "Action head: similar, 5D or 6D" | `action_latent.py` + `model.py` | 5D actions tile-injected into a latent frame; parameter-free encode/decode |
| "Cosmos VAE 4 duplicate images" | `vae.py::assemble_clip` | T=4 with state + 3× next_state |
| "Goal condition: embed shot profiles" | `conditioner.py` (zero-init prefix on T5 anchor) | 8-dim goal → 4 injected tokens of dim 1024 |
| "Value function: score distance reward" | `action_latent.py::inject_value` + `extract_value` + `src/policy/common/reward.py::score_distance_reward` | Scalar tile-injected; mean-pooled at decode |

---

## EDM σ sampling + BALANCED_TWO_HEADS_V1

We use cosmos-policy's training recipe end-to-end. Source: `_src/predict2/models/video2world_model.py:118` (the σ-sampling switch), `_src/predict2/models/text2world_model.py:919` (the loss), and `_src/imaginaire/modules/edm_sde.py` (the base log-normal distribution).

**σ sampling** (`src/policy/cosmos/edm.py::sample_sigma`):

1. **Base distribution**: log-normal `log(σ) ~ N(p_mean=-1.2, p_std=1.2)` — Karras 2022 default; matches upstream `HybridEDMSDE`.
2. **`BALANCED_TWO_HEADS_V1`** (enabled by default): independently of the base draw,
   - with probability `high_sigma_ratio` (default 0.25): replace σ with log-uniform on `[200, 100000]`
   - with probability `low_sigma_ratio` (default 0.25): replace σ with uniform on `[1e-5, 2.0]`

The "two heads" are the EDM denoiser's two preconditioned outputs — noise prediction (`eps`) dominates at large σ, clean-sample prediction (`x0`) dominates at small σ. The base log-normal undersamples both tails, so both heads end up poorly supervised at the extremes. The replacement step forces a quarter of every batch into each tail.

**Loss** (`model.py::compute_loss`):

```
σ = sample_sigma(...)                          # log-normal + BALANCED_TWO_HEADS_V1
x_t = x0 + σ · ε                                # EDM forward process
c_skip, c_out, c_in, c_noise = edm_scaling(σ)  # Karras 2022 eq. 7
net_out = transformer(c_in · x_t, c_noise, cond)
x0_pred = c_skip · x_t + c_out · net_out       # EDM x0 prediction
w(σ) = 1/σ_data² + 1/σ²                         # per-σ weighting (Karras Table 1)
sq = (x0 - x0_pred)² · w(σ)
total = λ_world · world(sq) + λ_action · action(sq) + λ_value · value(sq)
```

Identical preconditioning to upstream; the only deviation is we keep our per-component (`world`/`action`/`value`) loss decomposition rather than a single uniform mean.

**Inference** (`model.py::sample`): Karras σ schedule (Algorithm 1, Karras 2022) plus a plain Euler step at each σ:
```
σ_i = (σ_max^(1/ρ) + i/(N-1)·(σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ,  ρ = 7
d = (x - x0_pred) / σ
x ← x + (σ_next - σ) · d
```
Image-conditioning frames are re-pinned at every step so the diffusion process only denoises the action + value latent positions.

**Configuration** (`configs/policy/cosmos_2b_v6_proto.yaml::edm`):
```yaml
edm:
  sigma_data: 1.0
  sigma_min: 0.002
  sigma_max: 80.0
  p_mean: -1.2
  p_std: 1.2
  use_balanced_two_heads: true
  high_sigma_ratio: 0.25
  low_sigma_ratio: 0.25
  rho: 7.0
```

Disable `use_balanced_two_heads` for ablations vs. the pure log-normal schedule.

## FAQ: should we precompute and save the conditioning as a separate embedding file?

**Upstream cosmos-policy does this** for text. See `cosmos_policy/datasets/save_aloha_t5_text_embeddings.py` (and the `_libero_` / `_robocasa_` variants): it loads T5-11B once, encodes the dataset's `unique_commands` strings, saves a `{command: (1, 512, 1024) bfloat16}` dict to a pickle file. The dataset then reads embeddings from disk at training time. **Why they do this**:

  1. T5-11B is ~22 GB in bfloat16 — having it resident on the training GPU alongside the 2B video model would waste VRAM that could go to bigger batches.
  2. The same command string (`"fold shirt"`) appears across many samples, so reusing one cached encoding amortizes the cost.
  3. T5-11B forward is non-trivial (~hundreds of ms per command on a single H100); training would be I/O-bound on text encoding without the cache.

**We don't need to do this** for the shot-profile goal vector. Our "encoder" is a single `Linear(8, K·D)` + zero-init gate that operates on a frozen **Qwen2.5-VL** text anchor (`assets/text_anchor.pt`). The anchor is precomputed once via `scripts/build_text_anchor.py`; the goal-projection weights themselves are trainable and change every step, so they cannot be cached. (Cosmos-Predict2.5's text encoder is Qwen2.5-VL, not T5 — the upstream cosmos-policy reference above is the older ALOHA codebase, which used T5.)

**What we should precompute instead**: the **VAE latents** for the rendered drone images. The Cosmos VAE is frozen for the prototype and its output is sample-dependent (not goal-dependent), so a `{(annotation_path, placement_idx, view_idx): latent}` cache buys a lot. That's what `scripts/encode_vae_latents.py` is for — it saves `(C_lat=16, T_lat, H_lat, W_lat) bfloat16` tensors keyed by view identity. Each latent is ~1–10 MB depending on resolution; the full v6 set of ~20K views fits in well under 200 GB on disk.

Summary:

| Modality | Encoder | Output shape | Encoder size | Should we cache? |
|---|---|---|---|---|
| Upstream Cosmos text | T5-11B (frozen, per-sample) | `(B, 512, 1024) bfloat16` | ~11 B params, ~22 GB | **Yes** — cosmos-policy ships a script for this |
| Our text anchor (one fixed prompt) | Qwen2.5-VL (one-shot) | `(real_len, D)` bfloat16, D=100352 | One-time encode | **Yes** — `scripts/build_text_anchor.py` saves `assets/text_anchor.pt` (real tokens only) |
| Our goal projection (in the model) | `Linear(8, K·D)` + zero-init gate (trainable) | K goal tokens concatenated after the text | trainable | **No** — encoder weights change every step |
| Image / video | Cosmos VAE (frozen) | `(B, 16, T_lat, H_lat, W_lat) bfloat16` | ~0.4 B params | **Yes** — see `scripts/encode_vae_latents.py` |

## Hot-path overhead estimate (per training iteration)

For prototype settings (B=1, T=4, H=480, W=720, frozen backbone):

| Step | Approx FLOPs | Notes |
|---|---|---|
| VAE encode (×1 clip per pair) | dominated by VAE Wan2 architecture | Frozen — could precompute via `scripts/encode_vae_latents.py` |
| Backbone forward (1 per iter at training; `n_steps` per inference) | dominated by 2B-param transformer | Frozen weights but still needs forward pass through backbone for the conditioner gradient to land |
| Conditioner | negligible (~33K params + 1-scalar gate) | The only trainable component when `freeze_backbone=True` |

Training wall-clock: ~1–2 s/iter on a single H100 in `bfloat16` with the backbone frozen. The 5000-iter prototype run fits in ~2–3 hours.

Inference wall-clock: `n_steps × backbone_forward_cost`. At `n_steps=32` and ~50 ms per backbone forward → ~1.5 s per action chunk. Reduce `n_steps` (e.g. 8) for real-time-ish use if quality holds.
