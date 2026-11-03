"""Consolidate a cosmos3 FSDP checkpoint into one framework-independent state dict.

    python scripts/consolidate_cosmos_ckpt.py \
        --src checkpoints/camera_policy_nano_v6/iter_000045000 \
        --weights ema --dtype bf16

The training run sharded the model across 8 ranks with FSDP, so `model/` is eight
`.distcp` files plus a `.metadata` index and nothing can read a single weight out of it
without torch's distributed-checkpoint machinery. This script turns that into one
`.safetensors` file of plain tensors, which loads with mmap on a box that has never
heard of `cosmos_framework` -- the training code is not in this repo and the tree it
ran from no longer exists, so a state dict that needs it is a state dict we cannot use.

Two things worth knowing before you pick the flags:

**Half the checkpoint is the EMA copy, in fp32.** `net` is the live bf16 weights and
`net_ema` is the exponential moving average kept at rate 0.1, stored as fp32 -- the
same ~15.2 B parameters twice, 28 GiB and 57 GiB. EMA is the copy to serve (it is what
the run's reported losses were measured on), so `--weights ema --dtype bf16` is the
default and gives a 28 GiB file rather than a 91 GB tree.

**Nothing here is loaded onto a GPU.** Consolidation is pure I/O and a dtype cast, so
it runs on a login node without touching anyone's device.
"""

import argparse
import json
import pathlib
import sys
import time
import types

import torch

# The .metadata index was pickled under Python 3.13, where PosixPath moved to
# `pathlib._local`. On 3.12 the unpickle dies before it reads a single tensor, so alias
# the module to `pathlib`, which still holds the same classes. Must happen before the
# torch.distributed.checkpoint import pulls the unpickler in.
if sys.version_info < (3, 13) and "pathlib._local" not in sys.modules:
    _m = types.ModuleType("pathlib._local")
    for _n in ("Path", "PurePath", "PosixPath", "WindowsPath",
               "PurePosixPath", "PureWindowsPath"):
        setattr(_m, _n, getattr(pathlib, _n))
    sys.modules["pathlib._local"] = _m

from torch.distributed.checkpoint import FileSystemReader                  # noqa: E402
from torch.distributed.checkpoint.default_planner import (                 # noqa: E402
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.format_utils import _load_state_dict     # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32, "keep": None}
PREFIX = {"ema": "net_ema.", "live": "net."}
GIB = 2 ** 30


def plan(md, prefix):
    """Every tensor under `prefix`, as (key, shape, dtype, bytes), in index order."""
    out = []
    for k, v in md.state_dict_metadata.items():
        if not k.startswith(prefix):
            continue
        size, props = getattr(v, "size", None), getattr(v, "properties", None)
        if size is None or props is None:
            # A non-tensor entry (an int, a dict) under the same prefix. There are none
            # in this checkpoint, but refusing quietly to carry one would be a silent
            # drop, so say so and keep going.
            print(f"  ! skipping non-tensor entry {k} ({type(v).__name__})")
            continue
        n = 1
        for s in size:
            n *= s
        out.append((k, tuple(size), props.dtype, n * props.dtype.itemsize))
    return out


def chunks(entries, budget):
    """Group keys so one load holds at most `budget` bytes of source tensors.

    Streaming in groups is what keeps this honest on a shared box: the whole EMA copy
    is 57 GiB in fp32 and we only ever want the bf16 result resident.
    """
    cur, cur_b = [], 0
    for e in entries:
        if cur and cur_b + e[3] > budget:
            yield cur
            cur, cur_b = [], 0
        cur.append(e)
        cur_b += e[3]
    if cur:
        yield cur


def consolidate(src, out, which, dtype, fmt, budget, strip, check):
    reader = FileSystemReader(src)
    md = reader.read_metadata()
    prefix = PREFIX[which]
    entries = plan(md, prefix)
    if not entries:
        raise SystemExit(f"no tensors under {prefix!r} in {src}")

    src_bytes = sum(e[3] for e in entries)
    target = dtype or None
    print(f"[plan] {len(entries)} tensors under {prefix!r}, {src_bytes / GIB:.2f} GiB on disk")
    print(f"[plan] -> {out}  ({fmt}, dtype={'as stored' if target is None else target})")

    tensors, n_params, t0, done_b = {}, 0, time.time(), 0
    bad = []
    for i, group in enumerate(chunks(entries, budget), 1):
        sd = {}
        _load_state_dict(sd, storage_reader=reader, no_dist=True,
                         planner=_EmptyStateDictLoadPlanner(keys=[e[0] for e in group]))
        missing = [e[0] for e in group if e[0] not in sd]
        if missing:
            raise SystemExit(f"loader returned {len(missing)} fewer keys than asked, "
                             f"first {missing[0]!r} -- refusing a partial state dict")
        for key, shape, _, nbytes in group:
            t = sd.pop(key)
            if check and not torch.isfinite(t).all():
                bad.append(key)
            if target is not None and t.dtype != target:
                t = t.to(target)
            name = key[len(prefix):] if strip else key
            tensors[name] = t.contiguous()
            n_params += t.numel()
            done_b += nbytes
        el = time.time() - t0
        pct = 100 * done_b / src_bytes
        rate = done_b / GIB / max(el, 1e-6)
        eta = (src_bytes - done_b) / GIB / max(rate, 1e-6)
        print(f"[read] group {i:>3}  {pct:5.1f}%  {done_b / GIB:7.2f}/{src_bytes / GIB:.2f} GiB  "
              f"{rate:5.2f} GiB/s  eta {eta / 60:5.1f} min", flush=True)

    if bad:
        raise SystemExit(f"{len(bad)} tensors hold a nan or an inf, first {bad[0]!r} -- "
                         "not writing a checkpoint that cannot be served")

    out_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"[read] done: {n_params / 1e9:.3f} B parameters, {out_bytes / GIB:.2f} GiB in memory"
          + (", all finite" if check else ""))

    prov = {
        "source": str(src),
        "source_checkpoint_id": str(getattr(getattr(md, "storage_meta", None),
                                            "checkpoint_id", "")),
        "weights": which, "prefix_stripped": prefix if strip else "",
        "dtype": "as stored" if target is None else str(target),
        "n_tensors": str(len(tensors)), "n_params": str(n_params),
        "all_finite": str(bool(check and not bad)),
        "torch": torch.__version__,
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {out} ...", flush=True)
    t1 = time.time()
    if fmt == "safetensors":
        from safetensors.torch import save_file
        save_file(tensors, str(out), metadata=prov)
    else:
        torch.save(tensors, out)
    wrote = out.stat().st_size
    print(f"[write] {wrote / GIB:.2f} GiB in {time.time() - t1:.0f}s")

    side = out.with_suffix(out.suffix + ".json")
    side.write_text(json.dumps({**prov, "file": out.name, "bytes": wrote}, indent=2) + "\n")
    print(f"[write] {side}")
    return out


def verify(src, path, which, strip, n_samples):
    """Re-read a sample of tensors from the shards and compare them to the file.

    The only transformation consolidation applies is a dtype cast, which is
    deterministic, so the comparison is bitwise: anything short of exact equality means
    a tensor was mis-assembled from the shards, not merely rounded. The sample always
    includes the biggest tensor and every action adapter, because those are the ones a
    silent truncation would hurt most and the ones no other checkpoint can supply.
    """
    import random
    from safetensors import safe_open

    prefix = PREFIX[which]
    md = FileSystemReader(src).read_metadata()
    entries = {k: e for k, *e in ((k, sh, dt, nb) for k, sh, dt, nb in plan(md, prefix))}

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        names = list(f.keys())
        print(f"[verify] {path}")
        print(f"[verify] {len(names)} tensors, written {meta.get('written', '?')}, "
              f"dtype {meta.get('dtype', '?')}, weights {meta.get('weights', '?')}")

        want = {(prefix + n if strip else n) for n in names}
        if want != set(entries):
            missing, extra = sorted(set(entries) - want), sorted(want - set(entries))
            raise SystemExit(f"key sets differ: {len(missing)} missing "
                             f"(first {missing[:1]}), {len(extra)} extra (first {extra[:1]})")
        print(f"[verify] key set matches the shards exactly ({len(want)} keys)")

        pool = sorted(names)
        must = [n for n in pool if "action" in n or "llm2vae" in n or "vae2llm" in n]
        big = max(pool, key=lambda n: entries[(prefix + n) if strip else n][2])
        rest = [n for n in pool if n not in must and n != big]
        random.Random(0).shuffle(rest)
        sample = list(dict.fromkeys([big] + must + rest))[:max(n_samples, len(must) + 1)]

        keys = [(prefix + n) if strip else n for n in sample]
        sd = {}
        _load_state_dict(sd, storage_reader=FileSystemReader(src), no_dist=True,
                         planner=_EmptyStateDictLoadPlanner(keys=keys))
        worst, n_params = 0.0, 0
        for n, k in zip(sample, keys):
            got, ref = f.get_tensor(n), sd[k]
            if tuple(got.shape) != tuple(ref.shape):
                raise SystemExit(f"{n}: shape {tuple(got.shape)} != shard {tuple(ref.shape)}")
            cast = ref.to(got.dtype)
            if not torch.equal(got, cast):
                d = (got.float() - cast.float()).abs().max().item()
                raise SystemExit(f"{n}: not bitwise equal to the shards (max|diff|={d:g})")
            worst = max(worst, (got.float() - ref.float()).abs().max().item())
            n_params += got.numel()
        print(f"[verify] {len(sample)} tensors bitwise equal after the cast "
              f"({n_params / 1e9:.3f} B parameters checked)")
        print(f"[verify] largest cast error vs the stored fp32: {worst:.3g}")
    print("[verify] ok")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=pathlib.Path,
                   help="the iter_XXXXXXXXX dir, or its model/ subdir")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="output file (default: beside --src)")
    p.add_argument("--weights", default="ema", choices=["ema", "live", "both"],
                   help="ema = net_ema (what the run's losses were measured on)")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--format", default="safetensors", choices=["safetensors", "pt"])
    p.add_argument("--chunk-gib", type=float, default=8.0,
                   help="source bytes held per read, to bound peak memory")
    p.add_argument("--keep-prefix", action="store_true",
                   help="leave the net./net_ema. prefix on every key")
    p.add_argument("--no-check", action="store_true", help="skip the nan/inf sweep")
    p.add_argument("--verify", type=pathlib.Path, default=None,
                   help="check an already-written file against the shards, then stop")
    p.add_argument("--samples", type=int, default=24,
                   help="tensors to compare bitwise under --verify")
    a = p.parse_args()

    src = a.src if (a.src / ".metadata").exists() else a.src / "model"
    if not (src / ".metadata").exists():
        raise SystemExit(f"no DCP .metadata under {a.src} or {a.src}/model")

    if a.verify is not None:
        which = a.weights if a.weights != "both" else "ema"
        verify(src, a.verify, which, not a.keep_prefix, a.samples)
        return

    ext = ".safetensors" if a.format == "safetensors" else ".pt"
    for which in (["ema", "live"] if a.weights == "both" else [a.weights]):
        out = a.out
        if out is None or a.weights == "both":
            stem = f"{src.parent.parent.name}-{src.parent.name}-{which}-{a.dtype}"
            out = (a.out.parent if a.out else src.parent) / (stem + ext)
        consolidate(src, out, which, DTYPES[a.dtype], a.format,
                    int(a.chunk_gib * GIB), not a.keep_prefix, not a.no_check)


if __name__ == "__main__":
    main()
