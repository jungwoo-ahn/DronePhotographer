#!/bin/bash
#SBATCH --job-name=camtrain
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=768G
#SBATCH --time=3-00:00:00
# ABSOLUTE. Slurm resolves a relative --output against the SUBMIT directory, and this run is
# submitted from a different account's shell (jooyeolyun, core-own = 8 GPU) whose home has no
# runs/ -- a relative path there means Slurm cannot open the log and the job fails at launch,
# before anything in this script runs or can say why.
#SBATCH --output=/home/nas_main/jungwooahn/projects/DronePhotographerV12/runs/camtrain_%j.out
#SBATCH --signal=B:SIGTERM@300
# Worker pods are driver-only with NO C compiler, so torch's inductor backend dies with
# "InvalidCxxCompiler: No working C++ compiler found". Cluster policy's answer is to bring
# an image that ships one; the venv still supplies python/torch, this is just the toolchain
# carrier (ubuntu22.04 base => glibc 2.35, which slurmd requires).
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
# Cosmos3-Nano camera-policy SFT.
#
# Single-GPU does NOT fit: with data_parallel_shard_degree=1 FSDP shards nothing, so the
# full parameters plus FusedAdam's fp32 master weights and moments sit on one device and
# the optimizer step OOMs a 178 GiB B200 (175 GiB already allocated). Shard across the
# GPUs instead — the reference LIBERO recipe runs HSDP 2x8 for the same reason.
#
#   sbatch --qos=own scripts/sbatch_train_camera.sh \
#       --sft-toml configs/policy/camera_policy_nano_100k.toml            # the real run, 8 GPUs
#   sbatch --qos=own --gres=gpu:2 --mem=192G scripts/sbatch_train_camera.sh   # 2-GPU smoke
#
# The defaults above are the 8-GPU shape. --mem and --gres must move TOGETHER: the admission
# webhook caps a job at 244000 MiB x num_gpus, so 768G with --gres=gpu:2 is rejected outright.
#
# 100k is ~7 days at the measured 585 iter/h, which no single job survives:
# --time caps a link at 24 h and preemptible QOS ends one whenever an `own` job wants the GPU.
# So the chain is made of jobs that each queue their OWN SUCCESSOR, and the trainer resumes
# from `latest_checkpoint.txt`.
#
# THE SUCCESSOR IS SUBMITTED AT START, NOT AT DEATH. The first version resubmitted from the
# SIGTERM trap, and it lost the chain on its first preemption: the trap dropped the sentinel,
# waited for the trainer to write an emergency checkpoint, and was SIGKILLed before it reached
# the `sbatch` line. Nothing ran for four days. A dying process is the worst possible place to
# put the step that keeps the run alive, so the successor is queued with
# `--dependency=afterany` while this job is healthy, and it starts whenever this one ends —
# preempted, timed out, or killed outright.
#
# What the trap still does is quick and best-effort: drop the sentinel so the framework can
# save (it polls for that file in on_training_step_end), and, if a PERSON stopped the job,
# write STOP_FILE and cancel the successor so a `scancel` means what it looks like.
#
# Guards, because a self-perpetuating chain needs them:
#   * STOP_FILE     — a deliberate stop; every later link exits immediately.
#   * banked >= max — the run finished; no successor.
#   * stalled links — if the banked checkpoint has not advanced for MAX_STALLED_LINKS links,
#                     the queue is ending links faster than save_iter can write and the chain
#                     is burning slots without training.
#   * MIN_USEFUL_S  — a link that CRASHES (not preempted) that fast is a loop, not progress.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"

MIN_USEFUL_S=${MIN_USEFUL_S:-900}      # only for a CRASHING link, never a preempted one
MAX_STALLED_LINKS=${MAX_STALLED_LINKS:-6}
SAVE_GRACE_S=${SAVE_GRACE_S:-240}      # of the ~5 min Slurm gives between SIGTERM and SIGKILL
_START=$(date +%s)
_ARGS=("$@")            # verbatim, so a resubmitted link is the same job

# `--sft-toml` is consumed HERE and handed on through the environment, because
# scripts/train_camera_policy.sh already passes its own `--sft-toml "$SFT_TOML"` to the
# trainer; forwarding a second one would leave two on the command line and make which
# config ran depend on argparse's tie-breaking.
_PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --sft-toml) export SFT_TOML="$2"; shift 2 ;;
        --sft-toml=*) export SFT_TOML="${1#*=}"; shift ;;
        *) _PASS+=("$1"); shift ;;
    esac
done
if [ -n "${SFT_TOML:-}" ]; then
    case "$SFT_TOML" in /*) ;; *) SFT_TOML="$V12/$SFT_TOML" ;; esac   # never relative: train_camera_policy.sh cds
    export SFT_TOML
    [ -f "$SFT_TOML" ] || { echo "no such config: $SFT_TOML" >&2; exit 2; }
fi

# Where the framework's TerminationSignalCheckpoint looks for its sentinel. Must be a path
# the container also sees, so it lives with the run rather than in /tmp.
export SLURM_LOG_DIR="$V12/runs/train/_sigterm/${SLURM_JOB_ID:-manual}"
mkdir -p "$SLURM_LOG_DIR"
# Derived from the config, never hardcoded. The previous version named camera_policy_nano_v5
# outright, so pointing this script at any other config left the chain reading a directory the
# run does not write: the banked iteration would never advance, the stall counter would climb
# on every link, and MAX_STALLED_LINKS would kill a perfectly healthy chain after six of them.
_toml() {   # _toml <key> -- first bare `key = "value"` or `key = value` in SFT_TOML
    sed -n "s/^ *$1 *= *\"\{0,1\}\([^\"#]*[^\" #]\)\"\{0,1\} *\(#.*\)\{0,1\}$/\1/p" \
        "${SFT_TOML:-/dev/null}" | head -1
}
_override() {   # _override <hydra.path> -- value from the trailing key=value args, if any
    local a
    for a in ${_PASS[@]+"${_PASS[@]}"}; do
        case "$a" in "$1"=*) printf '%s' "${a#*=}"; return ;; esac
    done
}
# A TRAILING OVERRIDE WINS OVER THE TOML, so the chain has to read it the same way the trainer
# does -- "Trailing key.path=value positionals are applied last (so they win over TOML)"
# (cosmos_framework/scripts/train.py). Reading only the TOML sent a smoke launched with
# `job.name=..._smoke trainer.max_iter=100` into the worst version of the bug this derivation
# was added to fix: the trainer wrote to the smoke directory while the chain watched the real
# run's, saw banked 0 against max_iter 100000, and queued itself a successor forever.
_setting() {    # _setting <hydra.path> <toml key>
    local v; v=$(_override "$1"); [ -n "$v" ] && { printf '%s' "$v"; return; }
    _toml "$2"
}
_RUN_NAME=$(_setting job.name name)
_RUN_PROJECT=$(_setting job.project project)
_RUN_GROUP=$(_setting job.group group)
: "${_RUN_NAME:=camera_policy_nano_v5}" "${_RUN_PROJECT:=cosmos3_camera_policy}" "${_RUN_GROUP:=action_sft}"
_CKPT_DIR="$V12/runs/train/$_RUN_PROJECT/$_RUN_GROUP/$_RUN_NAME/checkpoints"
# Per-run, so a deliberate stop of one run cannot silently kill another. A shared
# .camtrain_stop left over from v5 would have ended v6 on its first link with no visible cause.
_CHAIN_STATE="$V12/runs/train/.camtrain_chain_$_RUN_NAME"
_STOP_FILE="$V12/runs/train/.camtrain_stop_$_RUN_NAME"
_SUCCESSOR=""

# New directories must stay writable by the rest of the lab group: this run is submitted from
# one account (jooyeolyun, core-own = 8 GPU) into another's tree (jungwooahn), and a default
# 022 umask would leave checkpoints that the tree's owner cannot prune or pin.
umask 002

_ckpt_iter() {   # the iteration the chain has actually banked, 0 if none
    local p="$_CKPT_DIR/latest_checkpoint.txt"
    [ -f "$p" ] && sed -n 's/^iter_0*\([0-9][0-9]*\).*/\1/p' "$p" | head -1 || echo 0
}

_max_iter() {    # what this run trains to, so the chain knows when it is done
    _setting trainer.max_iter max_iter
}

_qos_args() {
    # The name a person can SUBMIT, not the one the scheduler reports: `--qos=own` comes back
    # as `free-own` and `--qos=extra` as `free-extra`, and echoing either into sbatch is a
    # requeue that fails on its first preemption.
    #
    # `own` is now reachable. It was excluded while this run lived on preemptible capacity,
    # and the measurement is why that changed: on `extra` the chain banked 1500 iterations in
    # 19 hours across nine preempted links — 79 iter/h against a nominal 593, because each
    # link reloads the model and discards up to save_iter of unbanked work. 13% efficiency
    # turns 25 days into 190.
    case "${SLURM_JOB_QOS:-}" in
        *share*) echo "-A share --qos=share" ;;
        *own*)   echo "--qos=own" ;;
        *)       echo "--qos=extra" ;;
    esac
}

_queue_successor() {
    # NO_CHAIN=1 for anything that is not the real run. A smoke queues its successor before it
    # has trained a step, so `banked >= max_iter` cannot stop it at submission time; the
    # successor then starts, finds the smoke already finished, and exits — a wasted slot each
    # time, and one preemptible link's worth of confusion in the queue.
    [ -n "${NO_CHAIN:-}" ] && { echo "[chain] NO_CHAIN set — single job, no successor"; return; }
    [ -f "$_STOP_FILE" ] && { echo "[chain] $_STOP_FILE exists — not queueing a successor"; return; }
    local banked target
    banked=$(_ckpt_iter); target=$(_max_iter)
    if [ -n "$target" ] && [ "$banked" -ge "$target" ] 2>/dev/null; then
        echo "[chain] banked iteration $banked has reached max_iter $target — chain complete"
        touch "$_STOP_FILE"; return
    fi
    # stall accounting, measured on what the chain has actually banked
    local prev stalled
    prev=$(cut -d' ' -f1 "$_CHAIN_STATE" 2>/dev/null || echo -1)
    stalled=$(cut -d' ' -f2 "$_CHAIN_STATE" 2>/dev/null || echo 0)
    if [ "$banked" -gt "$prev" ] 2>/dev/null; then stalled=0; else stalled=$((stalled + 1)); fi
    echo "$banked $stalled" > "$_CHAIN_STATE"
    if [ "$stalled" -ge "$MAX_STALLED_LINKS" ]; then
        echo "[chain] $stalled consecutive links banked no new checkpoint (still at $banked)." \
             "The queue is ending links faster than save_iter can write, so the chain would" \
             "burn slots without training. Lower checkpoint.save_iter or run somewhere less" \
             "contended, then delete $_CHAIN_STATE and resubmit." >&2
        return
    fi
    # THE SUCCESSOR MUST INHERIT THIS JOB'S ALLOCATION. Without these flags sbatch re-reads the
    # #SBATCH defaults, so a run launched with `--gres=gpu:2 --mem=192G` came back as 8 GPUs on
    # its second link (and vice versa) -- a different global batch halfway through training,
    # visible nowhere except the first line of the next log.
    local out alloc
    alloc="--gres=gpu:${NGPU} --mem=${SLURM_MEM_PER_NODE:-768000}M"
    [ -n "${SLURM_CPUS_PER_GPU:-}" ] && alloc="$alloc --cpus-per-gpu=${SLURM_CPUS_PER_GPU}"
    local tl; tl=$(squeue -h -j "$SLURM_JOB_ID" -o "%l" 2>/dev/null | tr -d ' ')
    case "$tl" in ""|UNLIMITED|NOT_SET) ;; *) alloc="$alloc --time=$tl" ;; esac
    out=$(sbatch --dependency=afterany:"$SLURM_JOB_ID" $(_qos_args) $alloc "$0" "${_ARGS[@]}" 2>&1) || {
        echo "[chain] could not queue a successor: $out" >&2; return; }
    _SUCCESSOR=$(echo "$out" | grep -oE '[0-9]+$' | head -1)
    echo "[chain] successor ${_SUCCESSOR} queued (afterany:${SLURM_JOB_ID}); banked $banked," \
         "stalled $stalled"
}

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
export NPROC_PER_NODE="$NGPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True   # optimizer step is the fragmentation peak

_handle_term() {
    # Fast and best-effort. Anything slow here is work the SIGKILL will interrupt, and the
    # successor is already queued, so nothing the chain depends on happens in this function.
    touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED" 2>/dev/null || true
    # NEVER cancel the successor from here. This used to ask sacct whether the scheduler or a
    # person ended the job, and stop the chain if it could not tell. It cannot tell: at the
    # moment SIGTERM arrives sacct still reports RUNNING for a timeout exactly as it does for
    # a `scancel`, so the "a person stopped this" branch fired on the ordinary 3-day boundary
    # -- wrote the stop file, cancelled the successor, and ended the run at iteration 42000 of
    # 100000 with nothing in the log but a line claiming a person had done it.
    #
    # The asymmetry decides the default: guessing "scheduler" when it was a person costs ONE
    # job that starts and exits immediately, while guessing "person" when it was the scheduler
    # costs the whole run. So the chain always continues, and stopping it is an explicit act:
    # `touch $_STOP_FILE` (every later link checks it at startup), then scancel.
    echo "[chain] SIGTERM at banked $(_ckpt_iter); successor ${_SUCCESSOR:-?} takes over." \
         "To stop the chain: touch $_STOP_FILE" >&2
    exit 143
}

trap _handle_term SIGTERM

if [ -f "$_STOP_FILE" ]; then
    echo "[chain] $_STOP_FILE exists — this chain was stopped deliberately. Delete it to resume."
    exit 0
fi
echo "GPUs: $NGPU  (sharding optimizer state across all of them)"
echo "config: ${SFT_TOML:-<default>}"
echo "run:    $_RUN_NAME  ($_CKPT_DIR)"
echo "banked: $(_ckpt_iter) / $(_max_iter)"
_queue_successor
# NOTE the path: the TOML's [model.parallelism] table lands at model.CONFIG.parallelism.*,
# so a hydra tail override must use that longer path — `model.parallelism.…` raises
# "Key 'parallelism' is not in struct".
bash scripts/train_camera_policy.sh \
    model.config.parallelism.data_parallel_shard_degree="$NGPU" \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    ${_PASS[@]+"${_PASS[@]}"} &
_child=$!
set +e
wait "$_child"
_rc=$?
set -e

if [ "$_rc" -eq 0 ]; then
    echo "[train] finished cleanly (max_iter reached) — stopping the chain"
    touch "$_STOP_FILE"
    [ -n "$_SUCCESSOR" ] && scancel "$_SUCCESSOR" 2>/dev/null || true
    exit 0
fi
# A real crash, not a signal we trapped. The successor is already queued and will resume from
# the last checkpoint, which is right for a transient fault and wrong for a broken config —
# so a crash this early stops the chain instead of letting it loop.
_elapsed=$(( $(date +%s) - _START ))
echo "[train] launcher exited $_rc after ${_elapsed}s" >&2
if [ "$_elapsed" -lt "$MIN_USEFUL_S" ]; then
    echo "[train] crashed inside ${MIN_USEFUL_S}s — stopping the chain rather than looping." \
         "Fix the cause, delete $_STOP_FILE, and resubmit." >&2
    touch "$_STOP_FILE"
    [ -n "$_SUCCESSOR" ] && scancel "$_SUCCESSOR" 2>/dev/null || true
fi
exit "$_rc"
