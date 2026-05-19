"""Fine-grained pair-distance plots (no binning artifact).

Shows the raw distribution of pair distances under each sampling mode, plus
`all_valid` (every pair passing the distance + rotation + has_detection
filters, no per-bin cap) as a reference for the underlying geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualize_pair_distance_distribution import (  # type: ignore
    build_pair_distances,
    load_views,
    relative_rotation_angle_deg,
)


def all_valid_pair_distances(placements, distance_threshold, rotation_threshold_deg):
    """All (i, j) pairs that pass the filters, no per-bin sampling."""
    out = []
    for placement in placements:
        pos = placement["pos"]
        fwd = placement["fwd"]
        ups = placement["up"]
        has_det = placement["has_det"]
        n = len(pos)
        for i in range(n):
            d = np.linalg.norm(pos - pos[i], axis=1)
            a = relative_rotation_angle_deg(fwd[i], ups[i], fwd, ups)
            mask = (d > 0.0) & (d <= distance_threshold) & (a <= rotation_threshold_deg) & has_det
            out.append(d[mask])
    return np.concatenate(out) if out else np.array([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", default="outputs/v5_3090x8_260429_092917")
    parser.add_argument("--n_placements", type=int, default=100)
    parser.add_argument("--n_bins", type=int, default=10, help="sampling bins (matches dataset config)")
    parser.add_argument("--max_pairs_per_image", type=int, default=20)
    parser.add_argument("--distance_threshold", type=float, default=1.5)
    parser.add_argument("--rotation_threshold_deg", type=float, default=60.0)
    parser.add_argument("--min_pair_distance", type=float, default=0.05)
    parser.add_argument("--display_bins", type=int, default=150, help="display histogram resolution")
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--out", default="scripts/_out/pair_distance_raw.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = (repo_root / args.annotations_path).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.n_placements} placements ...", flush=True)
    placements = load_views(src_dir, args.n_placements, args.seed)
    print(f"  loaded {len(placements)} placements")

    print("[all_valid] enumerating all filter-passing pairs ...", flush=True)
    raw_all = all_valid_pair_distances(placements, args.distance_threshold, args.rotation_threshold_deg)
    print(f"  n_pairs = {len(raw_all):,}")

    modes_data = {"all_valid": raw_all}
    for mode in ["natural", "uniform", "log_uniform"]:
        print(f"[{mode}] sampling ...", flush=True)
        dists, _ = build_pair_distances(
            placements,
            mode=mode,
            distance_threshold=args.distance_threshold,
            rotation_threshold_deg=args.rotation_threshold_deg,
            n_bins=args.n_bins,
            min_pair_distance=args.min_pair_distance,
            max_pairs_per_image=args.max_pairs_per_image,
            seed=args.seed,
        )
        modes_data[mode] = dists
        print(f"  n_pairs = {len(dists):,}")

    edges = np.linspace(0.0, args.distance_threshold, args.display_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    panels = [("all_valid", "#777"), ("natural", "#7C9A3A"), ("uniform", "#4C8DAE"), ("log_uniform", "#B05A3E")]
    for ax, (mode, color) in zip(axes.flat, panels):
        dists = modes_data[mode]
        counts, _ = np.histogram(dists, bins=edges)
        ax.bar(centers, counts, width=width * 0.95, color=color, edgecolor="none")
        ax.set_title(f"{mode}  (n={len(dists):,})", fontsize=11)
        ax.set_xlabel("pair distance (m)")
        ax.set_ylabel("# pairs")
        ax.set_xlim(0, args.distance_threshold)

    fig.suptitle(
        f"pair distance — fine histogram ({args.display_bins} display bins)  "
        f"placements={args.n_placements}, max_pairs/img={args.max_pairs_per_image}, "
        f"sample_bins={args.n_bins}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")

    # Overlay (density) plot for direct shape comparison
    fig2, ax = plt.subplots(figsize=(11, 5))
    for mode, color in panels:
        dists = modes_data[mode]
        if len(dists) == 0:
            continue
        counts, _ = np.histogram(dists, bins=edges)
        density = counts / counts.sum()
        ax.plot(centers, density, color=color, label=f"{mode} (n={len(dists):,})", linewidth=1.6)
    ax.set_xlabel("pair distance (m)")
    ax.set_ylabel("normalized density")
    ax.set_title("pair distance — normalized density (overlay)")
    ax.set_xlim(0, args.distance_threshold)
    ax.legend()
    overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
    fig2.tight_layout()
    fig2.savefig(overlay_path, dpi=140)
    print(f"saved: {overlay_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({mode: {"n": int(len(d)), "min": float(d.min()) if len(d) else None, "max": float(d.max()) if len(d) else None} for mode, d in modes_data.items()}, indent=2))
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
