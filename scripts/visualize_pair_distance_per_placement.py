"""Per-placement pair-distance histograms.

Shows the bin-count distribution **within** each placement separately, so we
can see how much each placement's geometry contributes to the aggregate
shape. Defaults to a 3x3 grid of randomly chosen placements under the
`uniform` mode.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualize_pair_distance_distribution import (  # type: ignore
    build_pair_distances,
    compute_bin_edges,
    load_views,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", default="outputs/v5_3090x8_260429_092917")
    parser.add_argument("--n_placements", type=int, default=9)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--max_pairs_per_image", type=int, default=20)
    parser.add_argument("--distance_threshold", type=float, default=1.5)
    parser.add_argument("--rotation_threshold_deg", type=float, default=60.0)
    parser.add_argument("--min_pair_distance", type=float, default=0.05)
    parser.add_argument("--mode", default="uniform", choices=["uniform", "log_uniform", "natural"])
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--out", default="scripts/_out/pair_distance_per_placement.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = (repo_root / args.annotations_path).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.n_placements} placements ...", flush=True)
    placements = load_views(src_dir, args.n_placements, args.seed)
    print(f"  loaded {len(placements)} placements")

    edges = compute_bin_edges(args.mode, args.n_bins, args.min_pair_distance, args.distance_threshold)
    edges_display = edges if edges is not None else np.linspace(0.0, args.distance_threshold, args.n_bins + 1)
    centers = 0.5 * (edges_display[:-1] + edges_display[1:])
    widths = np.diff(edges_display)

    n = len(placements)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows), squeeze=False)

    per_placement_counts = []
    for ax, placement in zip(axes.flat, placements):
        dists, stats = build_pair_distances(
            [placement],
            mode=args.mode,
            distance_threshold=args.distance_threshold,
            rotation_threshold_deg=args.rotation_threshold_deg,
            n_bins=args.n_bins,
            min_pair_distance=args.min_pair_distance,
            max_pairs_per_image=args.max_pairs_per_image,
            seed=args.seed,
        )
        counts, _ = np.histogram(dists, bins=edges_display)
        per_placement_counts.append({"name": placement["name"], "counts": counts.tolist(), "total": int(len(dists))})

        ceiling = stats["target_per_bin_per_anchor"] * stats["n_anchors_with_pairs"]
        ax.bar(centers, counts, width=widths * 0.92, color="#4C8DAE", edgecolor="black", linewidth=0.5)
        if args.mode != "natural":
            ax.axhline(ceiling, color="red", linestyle="--", linewidth=1.0, label=f"ceiling={ceiling}")
        for x in edges_display:
            ax.axvline(x, color="gray", alpha=0.20, linewidth=0.5)
        for c, x in zip(counts, centers):
            if c > 0:
                ax.text(x, c, f"{c}", ha="center", va="bottom", fontsize=7)

        title = placement["name"]
        if len(title) > 36:
            title = title[:33] + "..."
        ax.set_title(f"{title}\n total={len(dists)}, anchors={stats['n_anchors_with_pairs']}", fontsize=9)
        ax.set_xlabel("pair distance (m)", fontsize=8)
        ax.set_ylabel("# pairs", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, edges_display[-1])
        if args.mode != "natural":
            ax.legend(loc="upper left", fontsize=7)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"per-placement pair distance distribution  (mode={args.mode}, "
        f"max_pairs/img={args.max_pairs_per_image}, n_bins={args.n_bins})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({"mode": args.mode, "edges": edges_display.tolist(), "placements": per_placement_counts}, indent=2))
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
