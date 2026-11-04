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

## Getting it

**Checkpoints are never committed.** `checkpoints/` is in `.gitignore`, one iteration is
~91 GB, and the consolidated file is another 28 GiB. What the repo carries is the
recipe:

```bash
pip install paramiko
export CKPT_SSH_PASSWORD=...          # or --key for a private key
python scripts/fetch_cosmos_ckpt.py \
    --host <cluster> --port <port> --user <you> \
    --run  /home/.../action_sft/camera_policy_nano_v6 \
    --iter iter_000045000 \
    --dest checkpoints/camera_policy_nano_v6/iter_000045000

# what is already on disk, checked against the cluster file by file
python scripts/fetch_cosmos_ckpt.py ... --verify
```

It is resumable (a partial shard is appended to, not restarted), fetches the big shards
`--jobs` at a time, and skips `optim/` -- 56 GB of Adam state that only a resumed
training run wants, not an inference one. `--dry-run` prints the manifest first.

Two things in it are scar tissue from doing this by hand, and are worth not
re-learning:

- **A zero-byte remote file is not an already-fetched file.** The natural way to test
  "do I have this?" is `getsize(local) == size` with a missing file counting as 0 --
  which for an empty remote file is `0 == 0`, true before the file exists. That silently
  skipped seven empty `scheduler/__N_0.distcp` files here. The check tests existence
  first.
- **Summing bytes is not a completion test.** The byte total still matched after those
  seven were skipped, because the total only summed files that existed, so the fetch
  reported success. Completion is now "every file present at exactly the remote size",
  reported per file.

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

## Serving it

`scripts/serve_cosmos_policy.sh` serves it from this repo. The code here is
`src/policy/cosmos3/`; the checkpoint and VAE are read from where training left them, in
`DronePhotographerV12` on the cluster, and nothing writes to them.

It needs `repos/cosmos-framework` (gitignored) at the commit the checkpoint trained on:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git repos/cosmos-framework
git -C repos/cosmos-framework checkout 2190a72a5799ab1bc8e4ddc19ae09cd1f14d7cbc
cd repos/cosmos-framework && uv sync --frozen --all-extras --group=cu128-train
```

The framework stays unpatched: `src/policy/cosmos3/recipe.py` registers the 10D action type
and the experiment from outside it.

Measured on one B200 with `iter_000045000`, 4 draws of 30 sampler steps per reply (2026-11-03):

- Loading the weights takes 91 s. The first reply takes 35 s, and later ones about 3.1 s
  (p90 3.4 s). One draw is about 0.7 s of that.
- Over 32 of the eval episodes, on their rendered start frames, the first chunk's motion
  points toward the requested goal: cosine with the start-to-goal direction has median
  +0.94 and is positive in 31 of 32. With goals shuffled between episodes the median is
  +0.77 and 75% are positive, so the goal does change the answer, but many episodes in this
  set share a direction.

None of this has seen a real camera image.
