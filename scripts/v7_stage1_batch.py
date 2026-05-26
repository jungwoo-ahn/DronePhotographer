#!/usr/bin/env python3
"""Stage 1 batch sampling — run v7_sample_pairs_smoke.py over many placements.

Multi-process subprocess pool. Each worker is a Blender invocation with
``--no-render --no-viz --no-report`` so Stage 1 stays CPU-only and writes
only ``data.json`` + ``radius.csv`` per placement.

Resume-friendly: placements whose ``<out>/<name>/data.json`` already exists
are skipped on a fresh run.

Usage:
    python scripts/v7_stage1_batch.py \\
        --placements-file outputs/v7_stage1_sample/valid_placements.txt \\
        --out-dir outputs/v7_stage1_sample \\
        --workers 8

Output layout:
    <out_dir>/
        _logs/<placement_name>.log    # per-placement worker stdout+stderr
        <placement_name>/data.json    # accepted pairs + per-frame poses
        <placement_name>/radius.csv
        failed.txt                     # placements whose Blender returned != 0
        summary.json                   # aggregate stats over all data.json files
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BLENDER_BIN = REPO_ROOT / "blender" / "blender"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "v7_sample_pairs_smoke.py"


def run_one(placement_path: str, out_dir: str, log_dir: str, seed: int,
            timeout_s: int) -> dict:
    """Run Blender on a single placement. Return a result dict for the master."""
    pj = Path(placement_path)
    name = pj.stem
    log_path = Path(log_dir) / f"{name}.log"
    data_json = Path(out_dir) / name / "data.json"

    if data_json.exists():
        return {"name": name, "status": "skip", "elapsed": 0.0,
                "log": str(log_path)}

    cmd = [
        str(BLENDER_BIN), "-b", "-P", str(SMOKE_SCRIPT), "--",
        "--placement-json", str(pj),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--no-render", "--no-viz", "--no-report",
    ]
    t0 = time.time()
    try:
        with log_path.open("w") as logf:
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout",
                "elapsed": time.time() - t0, "log": str(log_path)}
    except Exception as exc:
        return {"name": name, "status": "exception",
                "error": str(exc),
                "elapsed": time.time() - t0, "log": str(log_path)}

    elapsed = time.time() - t0
    if rc != 0:
        return {"name": name, "status": "fail", "rc": rc,
                "elapsed": elapsed, "log": str(log_path)}
    if not data_json.exists():
        return {"name": name, "status": "no_output",
                "elapsed": elapsed, "log": str(log_path)}
    return {"name": name, "status": "ok", "elapsed": elapsed,
            "log": str(log_path)}


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def aggregate_summary(out_dir: Path, valid_placements: list[Path]) -> dict:
    """Walk each placement's data.json and aggregate stats."""
    n_total = len(valid_placements)
    K_dist: dict[int, int] = {}
    attempts_total = 0
    accepted_total = 0
    reject_reasons: dict[str, int] = {}
    sub_reasons: dict[str, int] = {}
    r_mins: list[float] = []
    r_maxs: list[float] = []
    time_setup_total = 0.0
    time_sample_total = 0.0
    n_with_data = 0

    for pj in valid_placements:
        dj = out_dir / pj.stem / "data.json"
        if not dj.exists():
            continue
        try:
            d = json.loads(dj.read_text())
        except Exception:
            continue
        n_with_data += 1
        k = int(d.get("K_accepted", 0))
        K_dist[k] = K_dist.get(k, 0) + 1
        attempts_total += int(d.get("attempts_used", 0))
        accepted_total += k
        time_setup_total += float(d.get("time_setup_s", 0.0))
        time_sample_total += float(d.get("time_sample_s", 0.0))
        for r, n in (d.get("rejections_by_reason") or {}).items():
            reject_reasons[r] = reject_reasons.get(r, 0) + int(n)
        for r, n in (d.get("sub_reasons") or {}).items():
            sub_reasons[r] = sub_reasons.get(r, 0) + int(n)
        for p in d.get("accepted_pairs", []) or []:
            rs = float(p["start"]["r"]); re_ = float(p["end"]["r"])
            r_mins.append(min(rs, re_))
            r_maxs.append(max(rs, re_))

    return {
        "n_total_valid": n_total,
        "n_with_data": n_with_data,
        "K_distribution": dict(sorted(K_dist.items())),
        "K_mean": (accepted_total / n_with_data) if n_with_data else 0.0,
        "attempts_total": attempts_total,
        "accepted_total": accepted_total,
        "acceptance_rate": (accepted_total / attempts_total) if attempts_total else 0.0,
        "time_setup_total_s": time_setup_total,
        "time_sample_total_s": time_sample_total,
        "rejections_by_reason": dict(sorted(reject_reasons.items(),
                                            key=lambda kv: -kv[1])),
        "sub_reasons_top20": dict(sorted(sub_reasons.items(),
                                         key=lambda kv: -kv[1])[:20]),
        "r_min_p50": _percentile(r_mins, 50.0),
        "r_max_p50": _percentile(r_maxs, 50.0),
        "r_min_p5": _percentile(r_mins, 5.0),
        "r_max_p95": _percentile(r_maxs, 95.0),
    }


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placements-file", required=True,
                    help="Path to file with one absolute placement JSON per line.")
    ap.add_argument("--out-dir", required=True,
                    help="Output base directory for per-placement subdirs and summary.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel Blender workers (default 8).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=None,
                    help="Cap total placements processed (for pilot runs).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle placement list before processing (good with --max).")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-placement Blender timeout in seconds (default 300).")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="Print progress every N completions (default 20).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    log_dir = out_dir / "_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    placements_file = Path(args.placements_file).resolve()
    placements = [
        Path(line.strip())
        for line in placements_file.read_text().splitlines()
        if line.strip()
    ]
    if args.shuffle:
        random.Random(args.shuffle_seed).shuffle(placements)
    if args.max:
        placements = placements[: args.max]

    n_total = len(placements)
    print(f"[stage1] {n_total} placements · workers={args.workers} "
          f"· timeout={args.timeout}s · out={out_dir}")

    n_skip = sum(1 for p in placements if (out_dir / p.stem / "data.json").exists())
    print(f"[stage1] resume: {n_skip} already have data.json (will skip)")

    failed_path = out_dir / "failed.txt"
    failed_path.touch()
    existing_failed = set(
        line.strip() for line in failed_path.read_text().splitlines()
        if line.strip()
    )

    t_start = time.time()
    n_done = 0
    n_ok = 0
    n_skip_runtime = 0
    n_fail_runtime = 0
    new_failed: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one,
                str(p),
                str(out_dir),
                str(log_dir),
                args.seed,
                args.timeout,
            ): p
            for p in placements
        }
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            status = res.get("status")
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip_runtime += 1
            else:
                n_fail_runtime += 1
                line = f"{res['name']}\t{status}\t{res.get('rc', '')}"
                new_failed.append(line)

            if n_done % args.progress_every == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (n_total - n_done) / rate if rate > 0 else 0
                print(
                    f"[stage1] {n_done}/{n_total} · "
                    f"ok={n_ok} skip={n_skip_runtime} fail={n_fail_runtime} · "
                    f"{rate:.2f}/s · ETA {_fmt_eta(remaining)}"
                )

    # write failed list (append new on top of existing)
    if new_failed:
        all_failed = sorted(set(list(existing_failed) + new_failed))
        failed_path.write_text("\n".join(all_failed) + "\n", encoding="utf-8")

    print(f"[stage1] done in {_fmt_eta(time.time() - t_start)}")
    print(f"[stage1] ok={n_ok} skip={n_skip_runtime} fail={n_fail_runtime}")

    # aggregate
    print(f"[stage1] aggregating summary.json over {n_total} placements...")
    summary = aggregate_summary(out_dir, placements)
    summary["run"] = {
        "n_total": n_total, "n_ok": n_ok,
        "n_skip": n_skip_runtime, "n_fail": n_fail_runtime,
        "wall_time_s": time.time() - t_start,
        "workers": args.workers,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stage1] wrote {summary_path}")
    print(f"[stage1] K_distribution: {summary['K_distribution']}")
    print(f"[stage1] K_mean: {summary['K_mean']:.2f}")
    print(f"[stage1] top rejections: {dict(list(summary['rejections_by_reason'].items())[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
