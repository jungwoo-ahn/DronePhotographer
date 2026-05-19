#!/usr/bin/env python3
"""Replicate dataset.py pair selection and visualize selected pair distance distribution.

Does NOT load images — only reads camera_position + detections from annotations.json.
Faithfully reproduces `DroneActionScoreDataset._build_pairs` logic.

Usage:
    python scripts/viz_pair_distances.py \
        --config configs/qwen35_vl_2b_4xh200_with_c2o_5k_v2.yaml \
        --output notes/pair_distance_distribution.html
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="training config YAML (reads data section)")
    p.add_argument("--annotations_path", default=None, help="override config annotations_path")
    p.add_argument("--distance_threshold", type=float, default=None, help="override")
    p.add_argument("--max_pairs_per_image", type=int, default=None, help="override")
    p.add_argument("--seed", type=int, default=None, help="override")
    p.add_argument("--output", default="notes/pair_distance_distribution.html")
    p.add_argument("--bin_width", type=float, default=0.1, help="histogram bin width (meters)")
    return p.parse_args()


def build_pairs(positions: np.ndarray, has_detection: np.ndarray,
                distance_threshold: float, max_pairs_per_image: int, seed: int):
    """Return (selected_distances, candidate_distances, per_source_valid_counts).

    Mirrors DroneActionScoreDataset._build_pairs:
      valid_j = { j : 0 < ||pos_j - pos_i|| <= threshold and has_detection[j] }
      if |valid_j| > max_pairs_per_image: random sample without replacement
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    selected = []
    candidates = []
    per_src_counts = []
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=1)
        mask = (d > 0.0) & (d <= distance_threshold)
        valid = np.where(mask)[0]
        valid = np.asarray([j for j in valid.tolist() if has_detection[j]], dtype=np.int64)
        per_src_counts.append(valid.size)
        if valid.size == 0:
            continue
        candidates.extend(d[valid].tolist())
        if max_pairs_per_image > 0 and valid.size > max_pairs_per_image:
            valid = rng.choice(valid, size=max_pairs_per_image, replace=False)
        selected.extend(d[valid].tolist())
    return np.asarray(selected), np.asarray(candidates), np.asarray(per_src_counts)


def histogram_bars(values: np.ndarray, threshold: float, bin_width: float) -> list[tuple[float, float, int]]:
    """Return list of (bin_lo, bin_hi, count). Bins cover [0, threshold]."""
    if values.size == 0:
        return []
    n_bins = max(1, int(np.ceil(threshold / bin_width)))
    edges = np.linspace(0, threshold, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return [(float(edges[i]), float(edges[i + 1]), int(counts[i])) for i in range(n_bins)]


def percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def bars_html(bars, color: str, total: int) -> str:
    if not bars:
        return "<div class='empty'>no pairs</div>"
    max_count = max(c for _, _, c in bars) or 1
    html = '<div class="hist">'
    for lo, hi, c in bars:
        h_pct = (c / max_count) * 100
        frac_pct = (c / total) * 100 if total else 0.0
        label = f"{lo:.1f}–{hi:.1f}"
        html += (
            '<div class="hcol">'
            f'<div class="hbar-wrap"><div class="hbar" style="height:{h_pct:.1f}%;background:{color}" '
            f'title="{label} m · n={c} ({frac_pct:.1f}%)"></div></div>'
            f'<div class="hx">{lo:.1f}</div>'
            f'<div class="hn">{c}</div>'
            '</div>'
        )
    html += '</div>'
    return html


def stats_html(stats: dict) -> str:
    if not stats:
        return '<div class="empty">—</div>'
    return (
        '<div class="stats-row">'
        f'<span>n={stats["n"]:,}</span>'
        f'<span>mean={stats["mean"]:.3f} m</span>'
        f'<span>median={stats["median"]:.3f} m</span>'
        f'<span>p5={stats["p05"]:.3f}</span>'
        f'<span>p95={stats["p95"]:.3f}</span>'
        f'<span>min={stats["min"]:.3f}</span>'
        f'<span>max={stats["max"]:.3f} m</span>'
        '</div>'
    )


def per_source_html(counts: np.ndarray, max_pairs: int) -> str:
    if counts.size == 0:
        return ""
    zero = int((counts == 0).sum())
    capped = int((counts > max_pairs).sum()) if max_pairs > 0 else 0
    # Bin per-source candidate counts
    max_c = int(counts.max()) if counts.size else 0
    bins = [0, 1, 5, 10, 20, 40, 64, 128, 256, 512, max(1024, max_c + 1)]
    hist_lines = ""
    total_src = counts.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        n_in = int(((counts >= lo) & (counts < hi)).sum())
        pct = (n_in / total_src) * 100
        hist_lines += f'<tr><td>{lo}–{hi-1}</td><td style="text-align:right">{n_in:,}</td><td style="text-align:right">{pct:.1f}%</td></tr>'
    return (
        f'<div class="sub-title">Candidate pairs per source frame (pre-cap, max_pairs={max_pairs})</div>'
        f'<div class="meta-line">sources with 0 candidates: {zero:,} ({zero/total_src*100:.1f}%) · sources exceeding cap: {capped:,} ({capped/total_src*100:.1f}%)</div>'
        f'<table class="dist-table"><thead><tr><th>candidates</th><th>#sources</th><th>%</th></tr></thead><tbody>{hist_lines}</tbody></table>'
    )


def build_html(meta: dict, selected, candidates, per_src_counts, bin_width: float) -> str:
    threshold = meta["distance_threshold"]
    max_pairs = meta["max_pairs_per_image"]

    sel_bars = histogram_bars(selected, threshold, bin_width)
    cand_bars = histogram_bars(candidates, threshold, bin_width)
    sel_stats = percentiles(selected)
    cand_stats = percentiles(candidates)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair distance distribution · threshold={threshold}m</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0f0f14;color:#ddd;padding:24px;max-width:1400px;margin:0 auto}}
h1{{font-size:1.4em;font-weight:600;color:#fff;margin-bottom:4px}}
.subtitle{{color:#888;font-size:.85em;margin-bottom:20px}}
.config-box{{background:#151520;border-radius:8px;padding:14px 18px;margin-bottom:24px;font-size:.85em}}
.config-box .kv{{display:grid;grid-template-columns:200px 1fr;gap:4px 12px}}
.config-box .k{{color:#888}}
.config-box .v{{color:#ddd;font-family:monospace;font-size:.95em}}
.section{{background:#151520;border-radius:10px;padding:18px 22px;margin-bottom:20px}}
.section-title{{font-size:1em;font-weight:700;color:#fff;margin-bottom:4px}}
.sub-title{{font-size:.85em;font-weight:700;color:#bbb;margin:14px 0 6px;text-transform:uppercase;letter-spacing:.5px}}
.meta-line{{font-size:.75em;color:#888;margin-bottom:8px}}
.stats-row{{display:flex;gap:18px;flex-wrap:wrap;font-size:.78em;color:#bbb;margin:4px 0 14px;padding:8px 12px;background:#1a1a28;border-radius:6px}}
.stats-row span{{font-family:monospace}}
.hist{{display:flex;align-items:flex-end;gap:2px;height:180px;padding:4px 0;background:#0f0f17;border-radius:6px;padding:10px 8px}}
.hcol{{flex:1;display:flex;flex-direction:column;align-items:center;min-width:0}}
.hbar-wrap{{width:100%;height:140px;display:flex;align-items:flex-end;justify-content:center}}
.hbar{{width:90%;border-radius:2px 2px 0 0;min-height:1px;opacity:.9}}
.hx{{font-size:.55em;color:#666;margin-top:3px}}
.hn{{font-size:.55em;color:#555}}
.empty{{color:#555;font-style:italic;padding:20px;text-align:center}}
.dist-table{{border-collapse:collapse;width:100%;max-width:420px;font-size:.8em;margin-top:6px}}
.dist-table th,.dist-table td{{padding:4px 10px;border-bottom:1px solid #2a2a3a}}
.dist-table th{{color:#888;text-align:left;font-weight:600;text-transform:uppercase;font-size:.7em;letter-spacing:.5px}}
.dist-table td{{font-family:monospace;color:#bbb}}
.note{{color:#888;font-size:.78em;line-height:1.5;padding:10px 14px;background:#1a1a28;border-radius:6px;border-left:3px solid #4fc3f7}}
</style></head><body>

<h1>Training pair distance distribution</h1>
<div class="subtitle">replicates <code>DroneActionScoreDataset._build_pairs()</code> · no images loaded</div>

<div class="config-box">
  <div class="kv">
    <div class="k">config</div><div class="v">{meta["config_path"]}</div>
    <div class="k">annotations</div><div class="v">{meta["annotations_path"]} (n={meta["n_views"]:,})</div>
    <div class="k">with detections</div><div class="v">{meta["n_detected"]:,} ({meta["n_detected"]/meta["n_views"]*100:.1f}%)</div>
    <div class="k">distance_threshold</div><div class="v">{threshold} m</div>
    <div class="k">max_pairs_per_image</div><div class="v">{max_pairs}</div>
    <div class="k">seed</div><div class="v">{meta["seed"]}</div>
    <div class="k">target detection required</div><div class="v">yes (source may be detection-free)</div>
  </div>
</div>

<div class="note">
  Filter logic: for each source frame i, a pair (i,j) is valid iff<br>
  &nbsp;&nbsp;<code>0 &lt; ‖pos_j − pos_i‖ ≤ {threshold} m</code> &nbsp;AND&nbsp; <code>j has detections</code>.<br>
  If more than {max_pairs} valid j exist, {max_pairs} are randomly sampled without replacement (seed={meta["seed"]}).
</div>

<div class="section">
  <div class="section-title">Selected pairs (what actually trains)</div>
  <div class="meta-line">After per-source cap of {max_pairs}. This is the distribution of <code>‖pos_j − pos_i‖</code> across every selected (i,j) pair.</div>
  {stats_html(sel_stats)}
  {bars_html(sel_bars, "#4fc3f7", sel_stats.get("n", 0))}
</div>

<div class="section">
  <div class="section-title">Candidate pool (all valid pairs before cap)</div>
  <div class="meta-line">Every (i,j) that satisfies the threshold + detection filter, before random downsampling.</div>
  {stats_html(cand_stats)}
  {bars_html(cand_bars, "#81c784", cand_stats.get("n", 0))}
</div>

<div class="section">
  <div class="section-title">Per-source statistics</div>
  {per_source_html(per_src_counts, max_pairs)}
</div>

</body></html>"""


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    data_cfg = cfg["data"]

    ann_path = Path(args.annotations_path or data_cfg["annotations_path"])
    threshold = args.distance_threshold if args.distance_threshold is not None else float(data_cfg["distance_threshold"])
    max_pairs = args.max_pairs_per_image if args.max_pairs_per_image is not None else int(data_cfg["max_pairs_per_image"])
    seed = args.seed if args.seed is not None else int(data_cfg.get("seed", 721))

    if not ann_path.is_absolute():
        ann_path = Path.cwd() / ann_path

    print(f"Loading annotations: {ann_path}")
    raw = json.loads(ann_path.read_text())
    print(f"  n={len(raw)}")

    positions = np.asarray([item["camera_position"] for item in raw], dtype=np.float32)
    has_detection = np.asarray([bool(item.get("detections")) for item in raw], dtype=bool)
    n_detected = int(has_detection.sum())
    print(f"  with detections: {n_detected} ({n_detected/len(raw)*100:.1f}%)")

    print(f"Building pairs: threshold={threshold}m, max_pairs={max_pairs}, seed={seed}")
    selected, candidates, per_src_counts = build_pairs(
        positions, has_detection, threshold, max_pairs, seed
    )
    print(f"  candidate pairs: {candidates.size:,}")
    print(f"  selected pairs: {selected.size:,} (after per-source cap)")

    meta = {
        "config_path": str(cfg_path),
        "annotations_path": str(ann_path.relative_to(Path.cwd()) if ann_path.is_absolute() else ann_path),
        "n_views": len(raw),
        "n_detected": n_detected,
        "distance_threshold": threshold,
        "max_pairs_per_image": max_pairs,
        "seed": seed,
    }

    html = build_html(meta, selected, candidates, per_src_counts, args.bin_width)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_kb = out.stat().st_size / 1024
    print(f"Saved: {out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
