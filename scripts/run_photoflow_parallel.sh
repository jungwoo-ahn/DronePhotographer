#!/bin/bash
# PhotoFlow (gpt-5) on bench_v2, N episodes at a time. Plain bash: no Slurm needed.
# Full instructions: docs/photoflow_bench_v2_runbook.md
#
#   export DP_V12_ROOT=/path/to/this/checkout  DP_SHARED_ROOT=/path/to/bundle/shared
#   export LETSUR_API_KEY=...            # or put the key in ~/.config/letsur/key
#   N=10 GPUS=0,1 bash scripts/run_photoflow_parallel.sh
#
# The 240 episodes are always cut into the same 24 shards of 10 (strided, so every shard sees
# every scene). N workers each take shards off that list one after another, so:
#
# N       how many run at once (default 10). One episode is ~24 min, almost all of it waiting
#         on gpt-5, so wall-clock is about 94 h / N. N can be changed between launches.
# GPUS    comma-separated GPU ids; workers are dealt round-robin (default: 0). Each worker
#         keeps one Blender (Cycles, GPU) server alive.
# PY      python with numpy, pillow, torch (default: python).
# SHARDS  which of the 24 shard indices THIS machine runs, e.g. SHARDS="$(seq 0 11)"
#         (default: all). Two machines can split a run with disjoint SHARDS.
#
# Resumable: a shard saves after every episode and continues from its <out>.partial.json, and
# a finished shard is skipped. After a crash, reboot or Ctrl-C just run the command again.
# Finished episodes are never re-billed.
set -uo pipefail
: "${DP_V12_ROOT:?set DP_V12_ROOT to this checkout}"
: "${DP_SHARED_ROOT:?set DP_SHARED_ROOT to the shared/ folder of the data bundle}"
K=24; N=${N:-10}; GPUS=${GPUS:-0}; PY=${PY:-python}
cd "$DP_V12_ROOT"
[ -x "$DP_SHARED_ROOT/blender/blender" ] || { echo "no blender at $DP_SHARED_ROOT/blender/blender"; exit 1; }
[ -d "$DP_SHARED_ROOT/blender/syslibs/lib" ] && export LD_LIBRARY_PATH="$DP_SHARED_ROOT/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DP_V12_ROOT:${PYTHONPATH:-}"
OUT=runs/bench_v2/photoflow_shards
mkdir -p "$OUT/manifests"
"$PY" - "$K" "$OUT/manifests" <<'PYEOF' || exit 1
import json, sys
k_total, out = int(sys.argv[1]), sys.argv[2]
doc = json.load(open("configs/eval_episodes_v2.json"))
for k in range(k_total):
    d = dict(doc, episodes=doc["episodes"][k::k_total]); d["n_episodes"] = len(d["episodes"])
    json.dump(d, open(f"{out}/s{k:02d}of{k_total}.json", "w"))
PYEOF
IFS=, read -ra G <<< "$GPUS"
read -ra TODO <<< "$(echo ${SHARDS:-$(seq 0 $((K - 1)))})"

worker() {          # worker w runs TODO[w], TODO[w+N], ... one after another on its GPU
    local w=$1 rc=0 i k s
    for ((i = w; i < ${#TODO[@]}; i += N)); do
        k=${TODO[$i]}; s=$(printf '%02d' "$k")
        [ -f "$OUT/rollout_photoflow_s${s}.json" ] && continue
        CUDA_VISIBLE_DEVICES=${G[$((w % ${#G[@]}))]} "$PY" scripts/closed_loop_eval.py \
            --policy photoflow --checkpoint gpt-5 \
            --episodes-from "$OUT/manifests/s${s}of${K}.json" \
            --render-res 1024 768 --render-profile v7 --stop-on-shoot 1 --shoot-threshold 0.5 \
            --samples 1 --seed 0 --out "$OUT/rollout_photoflow_s${s}.json" ${EXTRA_ARGS:-} \
            >> "$OUT/shard_${s}.log" 2>&1 || rc=1
    done
    return $rc
}

pids=()
trap 'pkill -P $$ 2>/dev/null; kill "${pids[@]}" 2>/dev/null; pkill -f "$DP_SHARED_ROOT/blender/blender" 2>/dev/null' INT TERM
for ((w = 0; w < N && w < ${#TODO[@]}; w++)); do
    worker "$w" &
    pids+=($!)
    sleep "${STAGGER:-20}"               # stagger the Blender start-ups
done
echo "${#pids[@]} workers over ${#TODO[@]} shards, GPUs $GPUS; logs in $DP_V12_ROOT/$OUT/shard_*.log"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
"$PY" scripts/merge_photoflow_shards.py
exit $rc
