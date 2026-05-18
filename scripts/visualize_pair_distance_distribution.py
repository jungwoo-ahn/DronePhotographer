"""Visualize per-bin pair count for DroneActionScoreDataset.

Replicates the bin-sampling logic from `DroneActionScoreDataset._build_pairs`
without loading images, so it runs in seconds even on the full dataset.
For each anchor we keep at most `target_per_bin = max_pairs_per_image //
n_distance_bins` neighbors per distance bin, then histogram the resulting
pair distances against the configured bin edges.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def relative_rotation_angle_deg(fwd_i: np.ndarray, up_i: np.ndarray, fwd: np.ndarray, ups: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) angle between view i and each row in (fwd, ups), degrees.

    Mirrors `batch_relative_rotation_angle_deg`: builds camera bases (right,
    up, forward), forms R_i^T @ R_j, returns the rotation angle from the
    trace.
    """
    def basis(f, u):
        f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
        r = np.cross(f, u)
        r = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)
        u2 = np.cross(r, f)
        u2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
        return np.stack([r, u2, f], axis=-1)

    Ri = basis(fwd_i[None, :], up_i[None, :])[0]
    Rj = basis(fwd, ups)
    M = np.einsum("ab,nbc->nac", Ri.T, Rj)
    tr = np.clip((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def load_views(src_dir: Path, n_placements: int, seed: int):
    placements = sorted(p for p in src_dir.glob("p*") if (p / "annotations.json").exists())
    chosen = random.Random(seed).sample(placements, min(n_placements, len(placements)))
    per_placement = []  # list of dicts {pos, fwd, up, has_det}
    for p in chosen:
        with (p / "annotations.json").open() as f:
            items = json.load(f)
        pos = np.array([it["camera_position"] for it in items], dtype=np.float32)
        fwd = np.array([it.get("final_forward", it.get("base_forward")) for it in items], dtype=np.float32)
        ups = np.array([it.get("final_up", it.get("base_up")) for it in items], dtype=np.float32)
        has_det = np.array([bool(it.get("detections")) for it in items], dtype=bool)
        per_placement.append({"pos": pos, "fwd": fwd, "up": ups, "has_det": has_det, "name": p.name})
    return per_placement


def compute_bin_edges(mode: str, n_bins: int, min_d: float, max_d: float) -> np.ndarray | None:
    if mode == "uniform":
        return np.linspace(0.0, max_d, n_bins + 1)
    if mode == "log_uniform":
        return np.exp(np.linspace(np.log(min_d), np.log(max_d), n_bins + 1))
    return None


def build_pair_distances(
    placements,
    mode: str,
    distance_threshold: float,
    rotation_threshold_deg: float,
    n_bins: int,
    min_pair_distance: float,
    max_pairs_per_image: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Return (pair_distances, stats). Mirrors _build_pairs sampling."""
    rng = np.random.default_rng(seed)
    edges = compute_bin_edges(mode, n_bins, min_pair_distance, distance_threshold)
    target_per_bin = max(1, max_pairs_per_image // n_bins)

    all_dists = []
    n_anchors_with_pairs = 0

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
            valid = np.where(mask)[0]
            if valid.size == 0:
                continue

            if mode == "natural":
                if max_pairs_per_image > 0 and valid.size > max_pairs_per_image:
                    valid = rng.choice(valid, size=max_pairs_per_image, replace=False)
            else:
                bin_idx = np.clip(np.digitize(d[valid], edges) - 1, 0, n_bins - 1)
                chunks = []
                for b in range(n_bins):
                    in_bin = valid[bin_idx == b]
                    if in_bin.size > target_per_bin:
                        in_bin = rng.choice(in_bin, size=target_per_bin, replace=False)
                    chunks.append(in_bin)
                valid = np.concatenate(chunks) if chunks else valid[:0]
                if valid.size == 0:
                    continue

            n_anchors_with_pairs += 1
            all_dists.append(d[valid])

    dists = np.concatenate(all_dists) if all_dists else np.array([])
    stats = {
        "n_anchors_with_pairs": int(n_anchors_with_pairs),
        "target_per_bin_per_anchor": target_per_bin,
        "expected_total_if_full": target_per_bin * n_anchors_with_pairs * n_bins if mode != "natural" else None,
    }
    return dists, stats


def plot_panel(ax, mode: str, dists: np.ndarray, edges: np.ndarray | None, stats: dict, n_bins: int, max_d: float):
    if edges is None:
        edges = np.linspace(0.0, max_d, n_bins + 1)
        edge_color_hint = "(natural — edges shown for ref)"
    else:
        edge_color_hint = ""

    counts, _ = np.histogram(dists, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    bars = ax.bar(centers, counts, width=widths * 0.92, color="#4C8DAE", edgecolor="black", linewidth=0.5)

    target = stats["target_per_bin_per_anchor"] * stats["n_anchors_with_pairs"]
    ax.axhline(target, color="red", linestyle="--", linewidth=1.2, label=f"per-bin ceiling = {target}")

    for x in edges:
        ax.axvline(x, color="gray", alpha=0.25, linewidth=0.6)
    for c, x in zip(counts, centers):
        ax.text(x, c, f"{c}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{mode}  (total = {len(dists)})  {edge_color_hint}")
    ax.set_xlabel("pair distance (m)")
    ax.set_ylabel("# pairs")
    ax.set_xlim(0, edges[-1])
    ax.legend(loc="upper right", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", default="outputs/v5_3090x8_260429_092917")
    parser.add_argument("--n_placements", type=int, default=100)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--max_pairs_per_image", type=int, default=20)
    parser.add_argument("--distance_threshold", type=float, default=1.5)
    parser.add_argument("--rotation_threshold_deg", type=float, default=60.0)
    parser.add_argument("--min_pair_distance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--out", default="scripts/_out/pair_distance_distribution.png")
    parser.add_argument("--modes", nargs="+", default=["natural", "uniform", "log_uniform"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = (repo_root / args.annotations_path).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading views from {args.n_placements} placements ...", flush=True)
    placements = load_views(src_dir, args.n_placements, args.seed)
    total_views = sum(len(p["pos"]) for p in placements)
    print(f"  loaded {len(placements)} placements, {total_views} views total")

    fig, axes = plt.subplots(1, len(args.modes), figsize=(6 * len(args.modes), 4.5), squeeze=False)
    summary = []
    for ax, mode in zip(axes[0], args.modes):
        print(f"[mode={mode}] building pairs ...", flush=True)
        dists, stats = build_pair_distances(
            placements,
            mode=mode,
            distance_threshold=args.distance_threshold,
            rotation_threshold_deg=args.rotation_threshold_deg,
            n_bins=args.n_bins,
            min_pair_distance=args.min_pair_distance,
            max_pairs_per_image=args.max_pairs_per_image,
            seed=args.seed,
        )
        edges = compute_bin_edges(mode, args.n_bins, args.min_pair_distance, args.distance_threshold)
        plot_panel(ax, mode, dists, edges, stats, args.n_bins, args.distance_threshold)
        edges_for_count = edges if edges is not None else np.linspace(0.0, args.distance_threshold, args.n_bins + 1)
        counts, _ = np.histogram(dists, bins=edges_for_count)
        print(f"  n_pairs={len(dists)}  per-bin counts={counts.tolist()}")
        summary.append({
            "mode": mode,
            "n_pairs": int(len(dists)),
            "n_anchors_with_pairs": stats["n_anchors_with_pairs"],
            "target_per_bin_per_anchor": stats["target_per_bin_per_anchor"],
            "edges": (edges if edges is not None else edges_for_count).tolist(),
            "counts": counts.tolist(),
        })

    fig.suptitle(
        f"pair distance distribution  (placements={args.n_placements}, "
        f"max_pairs/img={args.max_pairs_per_image}, n_bins={args.n_bins}, "
        f"distance_threshold={args.distance_threshold}, rot_threshold={args.rotation_threshold_deg}°)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
