# The cosmos3 camera-policy checkpoint

`camera_policy_nano_v6/iter_000045000` is the trained camera policy, fetched from the
training cluster. This note is what is actually inside it, because the answer changed
several assumptions: it is not a small model, half of it is redundant for inference,
and the code that produced it is not in this repo.

## What it is

A cosmos3 **action-SFT** model: a Qwen-style language backbone (vocab 151,936, hidden
4096) with a second `mlp_moe_gen` MLP per layer, a diffusion time embedder, VAE
adapters (`vae2llm`, `llm2vae`, 192-dim latents) and the action adapters
(`action2llm`, `llm2action`, `action_modality_embed`). `max_action_dim: 64`,
`chunk_length: 8`, `fps: 30`. It was fine-tuned from `Cosmos3-Nano` across 8 ranks
with FSDP (`data_parallel_shard_degree=8`).

"Nano" is the base model's name, not its size. The net is **36 layers, 15,173,136,576
parameters** — the doubled MLP per layer is most of why a 4096-wide backbone gets that
large.

Two numbers the config does not tell you, read off the weights themselves: the backbone
is 36 layers, and although `max_action_dim` is 64, only columns **0-9** of
`llm2action.bias.weight` are non-zero. The action space the policy was actually trained
on is 10-dimensional; the other 54 are padding.

## What the 91 GB is

`model/` is 8 × 11.38 GB `.distcp` shards plus a 2.75 MB `.metadata` index, holding
1,618 entries — the same 809 tensors twice:

| key | dtype | size | what it is |
| --- | --- | --- | --- |
| `net` | bf16 | 28.26 GiB | the live weights |
| `net_ema` | fp32 | 56.52 GiB | the EMA copy, rate 0.1 |

**Two thirds of the file is the EMA copy's extra precision.** EMA is the copy to
serve: `ema.enabled: true` in the config, and the run's reported losses (this
iteration's held-out action loss of 0.00109, per its `KEEP` file) are EMA numbers. Cast
it to bf16 and the servable checkpoint is 28.26 GiB, not 91 GB.

`optim/` (55.9 GB) was deliberately not fetched — weights only, since nothing here
resumes training. `scheduler/` and `trainer/` came along and are small.

## Consolidating it

```bash
python scripts/consolidate_cosmos_ckpt.py \
    --src checkpoints/camera_policy_nano_v6/iter_000045000 \
    --weights ema --dtype bf16

# and to check the result against the shards it came from
python scripts/consolidate_cosmos_ckpt.py --src <same> \
    --verify checkpoints/.../camera_policy_nano_v6-iter_000045000-ema-bf16.safetensors
```

It writes one `.safetensors` of plain tensors with the `net_ema.` prefix stripped, plus
a JSON sidecar of provenance (the same fields are embedded in the file's own metadata
header). Three things about how it works are deliberate:

- **It streams.** Keys are loaded in groups of at most `--chunk-gib` source bytes, so
  only the bf16 result stays resident rather than 57 GiB of fp32 alongside it.
- **It refuses a partial answer.** If the loader returns fewer keys than were asked
  for, or any tensor holds a nan or an inf, it exits without writing — a checkpoint
  that is quietly missing a layer is worse than no checkpoint.
- **`--verify` compares bitwise.** The only transformation is a dtype cast, which is
  deterministic, so a sampled tensor must be *exactly* equal to its cast shard value.
  Anything less means it was mis-assembled, not merely rounded. The sample always
  includes the largest tensor and every action adapter.

safetensors rather than `torch.save`: it mmaps, so loading does not cost 28 GiB of
page cache up front, and it needs no `weights_only=False`.

### One trap: the index was pickled under Python 3.13

`.metadata` references `pathlib._local.PosixPath`, a module that only exists from
Python 3.13. Read it under 3.12 and the unpickle dies with
`ModuleNotFoundError: No module named 'pathlib._local'` before a single tensor is
touched. The script aliases the module to `pathlib` up front; nothing else is needed,
because the classes are the same ones.

## What still blocks serving it

Consolidation is **not** what stood between this checkpoint and
`scripts/serve_policy.py`. Three other things do:

1. **The model code.** The net is defined in `cosmos_framework`, which is not vendored
   here and is not on this machine; the tree it trained from
   (`.../DronePhotographerV12`) no longer exists. Weights without the module they load
   into are inert.
2. **The VAE.** The action adapters were trained against specific 192-dim latents, so
   the matching encoder has to come with it.
3. **Nothing else.** 28.26 GiB of bf16 fits on one A6000 (48 GB) with room for
   activations — but not on a 24 GB card, which rules out the 3090 boxes.

`CosmosPolicy.__init__` raises with this list rather than a stale complaint about
shards, so the next person to try reads the real reason.
