# Running PhotoFlow (gpt-5) on bench_v2 off the cluster

Written 2026-11-12 by jooyeol (with Claude). This is V12 code: `astra-baseline` plus one commit.
jooyeol has no push access to the V12 repo, so the branch lives in the other one, as
**`v12-photoflow-unic-bench-v2` in `jungwoo-ahn/DronePhotographer`** (it shares no history with
that repo's `main`; jungwoo can pull it into V12 with
`git fetch https://github.com/jungwoo-ahn/DronePhotographer.git v12-photoflow-unic-bench-v2`).
It adds two baselines to `scripts/closed_loop_eval.py`:

* `--policy unic` — already run on the cluster, 240/240 (`runs/bench_v2/rollout_unic.json`).
* `--policy photoflow` — a PhotoFlow-style search driven by gpt-5 over the LetSur gateway.
  **Not run yet.** This note is how to run it on another machine in under a day.

## What it costs

Measured on one episode on the cluster: 31 gpt-5 calls, **$1.24**, **23.6 min**, of which the
28 preview renders are about 3 min. The rest is waiting on the API, so the run parallelises
almost perfectly and one GPU can serve many shards.

| episodes at once, `N` | wall-clock for 240 episodes |
|---|---|
| 1 | ~94 h |
| 5 | ~19 h |
| 8 | ~12 h |
| **10 (default)** | **~9.5 h** |
| 16 | ~6 h |

Total API cost is about **$300** whatever `N` is (≈7,400 calls). These are extrapolated from
that one episode; expect ±30%. Nobody has tried more than one stream against the gateway, so if
the logs fill with HTTP 429 retries, stop and relaunch with a smaller `N`.

## What the machine needs

* Linux x86_64 with NVIDIA GPUs and driver (Blender 4.5 Cycles renders on them). **Each
  worker's Blender held ~8.7 GB of GPU memory** while rendering (measured on Chill-Camping; other
  scenes may differ). So `N` is limited by total GPU memory: about 2 workers per 24 GB card,
  5 per 48 GB, 8 per 80 GB. N=10 wants ~90 GB across the cards in `GPUS`. The GPUs are
  otherwise nearly idle, so sharing them with something else is fine if the memory is there.
* Python ≥ 3.10 with `numpy`, `pillow`, `torch` (CPU torch is enough: PhotoFlow runs no local
  model, the harness only imports it).
* ~15 GB disk, and outbound HTTPS to `gw.letsur.ai`.
* No Slurm, no Blender install, no cosmos-framework.

## Setup (once)

```bash
# 1. code
git clone -b v12-photoflow-unic-bench-v2 --single-branch \
    https://github.com/jungwoo-ahn/DronePhotographer.git DronePhotographerV12
cd DronePhotographerV12

# 2. data bundle (5.4 GB): Blender, the 6 scenes and 89 subjects bench_v2 uses, per-episode data
#    on the cluster: /home/nas_main/jooyeolyun/transfer/photoflow_bench_v2_bundle.tar
scp <cluster>:/home/nas_main/jooyeolyun/transfer/photoflow_bench_v2_bundle.tar ..
mkdir ../pf_bundle && tar xf ../photoflow_bench_v2_bundle.tar -C ../pf_bundle
cp -r ../pf_bundle/v12/. .                 # data/ and runs/bench_v2/_geom, both git-ignored

# 3. where things are, and the key (never commit it)
export DP_V12_ROOT=$PWD
export DP_SHARED_ROOT=$(realpath ../pf_bundle/shared)
export LETSUR_API_KEY=...                  # or: mkdir -p ~/.config/letsur && put it in ~/.config/letsur/key (chmod 600)
```

The bundle is rebuilt on the cluster with `python scripts/photoflow_make_bundle.py --out <tar>`.

## Smoke test (≈5 min, ≈$0.30)

One episode with a cut-down search, to see Blender and the gateway both work:

```bash
N=1 SHARDS=0 EXTRA_ARGS="--render-budget 4" bash scripts/run_photoflow_parallel.sh
#   in another terminal:
tail -f runs/bench_v2/photoflow_shards/shard_00.log  # want: "[eval] 1/10 ... d 0.5790 -> ..."
#   Ctrl-C the launcher once that first line appears, then:
rm -rf runs/bench_v2/photoflow_shards                # REQUIRED before the real run (the smoke
                                                     # episode used a cut-down search)
```

## The real run

```bash
tmux new -s photoflow                       # it must survive your SSH session
N=10 GPUS=0 bash scripts/run_photoflow_parallel.sh
```

The 240 episodes are always cut into the same 24 shards of 10 (strided, so each shard covers
every scene). `N` workers take shards off that list one after another.

* `N` — how many episodes run at once (default 10).
* `GPUS` — comma-separated ids; workers go round-robin, e.g. `GPUS=0,1,2,3`.
* `PY` — the python to use, if not `python`.
* `SHARDS` — which of the 24 shard indices this machine runs (default all). Two machines can
  split a run: `SHARDS="$(seq 0 11)"` on one, `SHARDS="$(seq 12 23)"` on the other, then copy
  one machine's `runs/bench_v2/photoflow_shards/` into the other's before merging.

Watch it:

```bash
tail -n 2 runs/bench_v2/photoflow_shards/shard_*.log
python scripts/merge_photoflow_shards.py    # episodes done so far and dollars spent; writes nothing until 240/240
```

**Resume.** Each shard saves after every episode, and a finished shard is skipped. After a
crash, reboot or Ctrl-C, run the launcher again; finished episodes are not billed again. An
episode that was mid-search is redone from its start (≤ $1.24 lost per worker). `N` and `GPUS`
can be changed on a relaunch, e.g. to back off after rate-limit errors.

## Bringing the result back

When all shards finish, the launcher writes `runs/bench_v2/rollout_photoflow.json`. The scorers
need the final frames too, so copy the whole shard folder to the cluster and merge there, which
also rewrites the other machine's paths:

```bash
# on the other machine
tar cf photoflow_result.tar -C runs/bench_v2 photoflow_shards
# on the cluster
cd /home/nas_main/jungwooahn/projects/DronePhotographerV12
tar xf photoflow_result.tar -C runs/bench_v2
python <this branch>/scripts/merge_photoflow_shards.py --rebase-from /abs/path/on/other/machine/DronePhotographerV12
```

Then score it like every other run (`score_benchmark.py detect` on `rollout_photoflow.json`,
`perceptual_eval.py`, `score_benchmark.py table --tau 0.20 0.10`, and jooyeol's
`scripts/clip_text_similarity.py` for the text-similarity row).

## What was changed in the harness

* `src/baselines/adapters.py`: `UNICAdapter`, `PhotoFlowAdapter`; `src/baselines/photoflow/`,
  `src/baselines/unic_model.py`, `src/baselines/unic_vendor/`.
* `src/baselines/astra.py`: gpt-5 price; key also read from `~/.config/letsur/key`.
* `scripts/closed_loop_eval.py`: the two policies, `--render-budget`, `--scene-scale-dir`, and
  the roots now come from `DP_V12_ROOT` / `DP_SHARED_ROOT` when set (unset = the cluster paths,
  so nothing changes for existing runs).
* `scripts/blender_render_v7.py`: honours `DP_SHARED_ROOT`; writes a scene summary for PhotoFlow.
* `scripts/sbatch_closed_loop_eval.sh`: `CLEVAL_REQUEUE_FLAGS` chooses where a preempted run requeues.
