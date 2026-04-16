"""Visualize training data camera poses in 3D, centered on the object.

Produces three plots:
  1. 3D scatter of camera positions (object at origin) with forward arrows
  2. Azimuth-elevation heatmap (spherical distribution)
  3. Distance histogram

Usage:
    python scripts/visualize_training_data_3d.py \
        --annotations outputs/Namaqualand_namaqualand_v3_260401_024633/annotations_5k.json \
        --out visualizations/training_data_3d.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection


def load_annotations(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_camera_data(annotations: list[dict]):
    """Extract camera and object arrays from annotations."""
    cam_pos = np.array([a["camera_position"] for a in annotations], dtype=np.float64)
    obj_pos = np.array([a["object_position"] for a in annotations], dtype=np.float64)
    cam_fwd = np.array([a["final_forward"] for a in annotations], dtype=np.float64)
    cam_up = np.array([a.get("final_up", a.get("base_up")) for a in annotations], dtype=np.float64)

    elevations = np.array([a["elevation_deg"] for a in annotations], dtype=np.float64)
    azimuths = np.array([a["azimuth_deg"] for a in annotations], dtype=np.float64)
    distances = np.array([a["camera_subject_distance"] for a in annotations], dtype=np.float64)

    # Object info (same for all views in single-object scenes)
    obj_fwd = np.array(annotations[0]["object_forward"], dtype=np.float64)
    obj_up = np.array(annotations[0]["object_up"], dtype=np.float64)
    obj_center = obj_pos[0]  # use first annotation's object position

    # Relative positions: camera positions centered on the object
    rel_pos = cam_pos - obj_center

    return dict(
        cam_pos=cam_pos,
        obj_center=obj_center,
        rel_pos=rel_pos,
        cam_fwd=cam_fwd,
        cam_up=cam_up,
        obj_fwd=obj_fwd,
        obj_up=obj_up,
        elevations=elevations,
        azimuths=azimuths,
        distances=distances,
    )


def plot_3d_cameras(ax, data: dict, arrow_scale: float = 0.6, up_scale: float = 0.25, subsample: int = 400):
    """Plot camera positions with forward/up direction lines in 3D, object at origin."""
    rel = data["rel_pos"]
    fwd = data["cam_fwd"]
    up = data["cam_up"]
    elev = data["elevations"]

    norm = Normalize(vmin=elev.min(), vmax=elev.max())
    cmap = plt.cm.viridis

    # Scatter all camera positions
    sc = ax.scatter(
        rel[:, 0], rel[:, 1], rel[:, 2],
        c=elev, cmap=cmap, norm=norm, s=4, alpha=0.6, depthshade=True,
    )

    # Subsample for direction lines
    rng = np.random.default_rng(42)
    idx = rng.choice(len(rel), size=min(subsample, len(rel)), replace=False)

    # Forward direction lines (orange)
    ax.quiver(
        rel[idx, 0], rel[idx, 1], rel[idx, 2],
        fwd[idx, 0], fwd[idx, 1], fwd[idx, 2],
        length=arrow_scale, color="#ff8c42", alpha=0.5, linewidth=0.7,
        arrow_length_ratio=0.2,
    )

    # Up direction lines (cyan)
    ax.quiver(
        rel[idx, 0], rel[idx, 1], rel[idx, 2],
        up[idx, 0], up[idx, 1], up[idx, 2],
        length=up_scale, color="#42c6ff", alpha=0.35, linewidth=0.5,
        arrow_length_ratio=0.2,
    )

    # Object at origin
    ax.scatter([0], [0], [0], c="red", s=120, marker="*", zorder=10, label="Object")

    # Object forward direction
    of = data["obj_fwd"]
    ax.quiver(0, 0, 0, of[0], of[1], of[2], length=1.5, color="red", linewidth=2.5, arrow_length_ratio=0.15)

    # Object up direction
    ou = data["obj_up"]
    ax.quiver(0, 0, 0, ou[0], ou[1], ou[2], length=1.0, color="blue", linewidth=2.5, arrow_length_ratio=0.15)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Camera Positions Relative to Object (n={len(rel)})")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#ff8c42", lw=2, label="Cam Forward"),
        Line2D([0], [0], color="#42c6ff", lw=2, label="Cam Up"),
        Line2D([0], [0], color="red", lw=2, label="Obj Forward"),
        Line2D([0], [0], color="blue", lw=2, label="Obj Up"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7)

    return sc


def plot_azimuth_elevation(ax, data: dict):
    """2D heatmap of camera distribution in azimuth-elevation space."""
    az = data["azimuths"]
    el = data["elevations"]

    h = ax.hist2d(az, el, bins=[72, 36], cmap="magma")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Camera Distribution (Azimuth vs Elevation)")
    plt.colorbar(h[3], ax=ax, label="Count")


def plot_distance_histogram(ax, data: dict):
    """Histogram of camera-to-object distances."""
    d = data["distances"]
    ax.hist(d, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(d.mean(), color="red", linestyle="--", label=f"mean={d.mean():.2f}m")
    ax.axvline(np.median(d), color="orange", linestyle="--", label=f"median={np.median(d):.2f}m")
    ax.set_xlabel("Distance to Object (m)")
    ax.set_ylabel("Count")
    ax.set_title("Camera-Object Distance Distribution")
    ax.legend(fontsize=8)


def plot_pair_connections(ax, data: dict, annotations: list[dict],
                          distance_threshold: float = 1.5,
                          max_show: int = 500):
    """Show lines connecting paired cameras (within distance threshold)."""
    rel = data["rel_pos"]
    n = len(rel)

    # Find pairs (same logic as dataset)
    rng = np.random.default_rng(42)
    lines_drawn = 0
    indices = rng.permutation(n)

    for i in indices:
        if lines_drawn >= max_show:
            break
        dists = np.linalg.norm(rel - rel[i], axis=1)
        neighbors = np.where((dists > 0) & (dists <= distance_threshold))[0]
        if len(neighbors) == 0:
            continue
        # Pick one random neighbor to draw
        j = rng.choice(neighbors)
        ax.plot(
            [rel[i, 0], rel[j, 0]],
            [rel[i, 1], rel[j, 1]],
            [rel[i, 2], rel[j, 2]],
            color="gray", alpha=0.15, linewidth=0.3,
        )
        lines_drawn += 1


def main():
    parser = argparse.ArgumentParser(description="Visualize training camera poses in 3D")
    parser.add_argument(
        "--annotations",
        default="outputs/Namaqualand_namaqualand_v3_260401_024633/annotations_5k.json",
        help="Path to annotations JSON",
    )
    parser.add_argument("--out", default="visualizations/training_data_3d.png", help="Output image path")
    parser.add_argument("--arrows", type=int, default=300, help="Number of forward-direction arrows to draw")
    parser.add_argument("--pair-lines", type=int, default=500, help="Number of pair connection lines to draw")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--interactive", action="store_true", help="Show interactive matplotlib window")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    data = parse_camera_data(annotations)

    print(f"Loaded {len(annotations)} views")
    print(f"  Camera distance range: {data['distances'].min():.2f} - {data['distances'].max():.2f} m")
    print(f"  Elevation range: {data['elevations'].min():.1f} - {data['elevations'].max():.1f} deg")
    print(f"  Azimuth range: {data['azimuths'].min():.1f} - {data['azimuths'].max():.1f} deg")

    fig = plt.figure(figsize=(20, 14))

    # --- 3D scatter with pairs ---
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    sc = plot_3d_cameras(ax1, data, subsample=args.arrows)
    plot_pair_connections(ax1, data, annotations, max_show=args.pair_lines)
    fig.colorbar(sc, ax=ax1, label="Elevation (deg)", shrink=0.6, pad=0.1)

    # --- Top-down (XY) view ---
    ax2 = fig.add_subplot(2, 2, 2)
    rel = data["rel_pos"]
    sc2 = ax2.scatter(rel[:, 0], rel[:, 1], c=data["distances"], cmap="coolwarm", s=3, alpha=0.5)
    ax2.scatter([0], [0], c="red", s=100, marker="*", zorder=10)
    of = data["obj_fwd"]
    ax2.annotate("", xy=(of[0] * 1.5, of[1] * 1.5), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top-Down View (colored by distance)")
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, label="Distance (m)")

    # --- Azimuth-elevation heatmap ---
    ax3 = fig.add_subplot(2, 2, 3)
    plot_azimuth_elevation(ax3, data)

    # --- Distance histogram ---
    ax4 = fig.add_subplot(2, 2, 4)
    plot_distance_histogram(ax4, data)

    fig.suptitle("Training Data: Camera Pose Distribution (5K views)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    if args.interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
