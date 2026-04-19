#!/usr/bin/env python3
"""Add body_in_frame_ratio to existing annotations using 3D bbox projection.

body_in_frame_ratio = clipped_projected_area / full_projected_area
  - 1.0: entire body visible in frame
  - 0.5: half the body is cut off
  - 0.0: body completely outside frame

Usage:
    python scripts/add_body_in_frame_ratio.py \
        --annotations_path outputs/.../annotations.json \
        --run_info_path outputs/.../run_info.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Simple 2D convex hull (Andrew's monotone chain)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _clip_polygon_to_rect(
    polygon: list[tuple[float, float]],
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman polygon clipping to rectangle."""
    def clip_edge(poly, edge_fn, intersect_fn):
        if not poly:
            return []
        out = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            curr_inside = edge_fn(curr)
            prev_inside = edge_fn(prev)
            if curr_inside:
                if not prev_inside:
                    out.append(intersect_fn(prev, curr))
                out.append(curr)
            elif prev_inside:
                out.append(intersect_fn(prev, curr))
        return out

    def lerp(a, b, t):
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    result = list(polygon)

    # Clip left
    result = clip_edge(result,
        lambda p: p[0] >= x_min,
        lambda a, b: lerp(a, b, (x_min - a[0]) / (b[0] - a[0])) if abs(b[0] - a[0]) > 1e-12 else a)
    # Clip right
    result = clip_edge(result,
        lambda p: p[0] <= x_max,
        lambda a, b: lerp(a, b, (x_max - a[0]) / (b[0] - a[0])) if abs(b[0] - a[0]) > 1e-12 else a)
    # Clip bottom (y_min)
    result = clip_edge(result,
        lambda p: p[1] >= y_min,
        lambda a, b: lerp(a, b, (y_min - a[1]) / (b[1] - a[1])) if abs(b[1] - a[1]) > 1e-12 else a)
    # Clip top (y_max)
    result = clip_edge(result,
        lambda p: p[1] <= y_max,
        lambda a, b: lerp(a, b, (y_max - a[1]) / (b[1] - a[1])) if abs(b[1] - a[1]) > 1e-12 else a)

    return result


def _make_camera_matrix(position, forward, up):
    """Build 4x4 camera-to-world matrix from position/forward/up."""
    import numpy as np
    fwd = np.array(forward, dtype=np.float64)
    fwd /= np.linalg.norm(fwd) + 1e-12
    upn = np.array(up, dtype=np.float64)
    upn /= np.linalg.norm(upn) + 1e-12
    right = np.cross(fwd, upn)
    right /= np.linalg.norm(right) + 1e-12
    upn = np.cross(right, fwd)
    upn /= np.linalg.norm(upn) + 1e-12

    # Blender convention: camera looks -Z, +X=right, +Y=up
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = upn
    mat[:3, 2] = -fwd  # camera -Z = forward
    mat[:3, 3] = position
    return mat


def compute_body_in_frame_ratio(
    corners_3d: list[list[float]],
    camera_position: list[float],
    camera_forward: list[float],
    camera_up: list[float],
    focal_length: float,
    sensor_width: float,
    sensor_height: float,
    res_x: int,
    res_y: int,
) -> float:
    """Project 3D bbox corners to 2D, compute fraction visible in frame."""
    import numpy as np

    cam_matrix = _make_camera_matrix(camera_position, camera_forward, camera_up)
    cam_inv = np.linalg.inv(cam_matrix)

    fx = focal_length * res_x / sensor_width
    fy = focal_length * res_y / sensor_height
    cx, cy = res_x / 2.0, res_y / 2.0

    points_2d = []
    for corner in corners_3d:
        p_world = np.array([*corner, 1.0], dtype=np.float64)
        p_cam = cam_inv @ p_world
        depth = -p_cam[2]  # Blender: camera looks -Z
        if depth > 0.01:
            px = fx * p_cam[0] / depth + cx
            py = cy - fy * p_cam[1] / depth  # flip Y for pixel coords
            points_2d.append((px, py))

    if len(points_2d) < 3:
        return 0.0

    hull = _convex_hull_2d(points_2d)
    full_area = _polygon_area(hull)
    if full_area < 1e-6:
        return 0.0

    clipped = _clip_polygon_to_rect(hull, 0, 0, res_x, res_y)
    clipped_area = _polygon_area(clipped)

    return round(min(1.0, max(0.0, clipped_area / full_area)), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True)
    parser.add_argument("--run_info_path", required=True)
    parser.add_argument("--output_path", default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    with open(args.run_info_path) as f:
        run_info = json.load(f)

    corners_3d = run_info["object_3d"]["bbox_3d_corners"]
    options = run_info["options"]
    focal_length = options["focal_length"]
    sensor_width = options["sensor_width"]
    sensor_height = options["sensor_height"]
    res_x, res_y = options["resolution"]

    with open(args.annotations_path) as f:
        annotations = json.load(f)

    count = 0
    for entry in annotations:
        cam_pos = entry.get("camera_position")
        cam_fwd = entry.get("final_forward", entry.get("base_forward"))
        cam_up = entry.get("final_up", entry.get("base_up"))

        if cam_pos is None or cam_fwd is None or cam_up is None:
            entry["body_in_frame_ratio"] = 0.0
            continue

        ratio = compute_body_in_frame_ratio(
            corners_3d=corners_3d,
            camera_position=cam_pos,
            camera_forward=cam_fwd,
            camera_up=cam_up,
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            res_x=res_x,
            res_y=res_y,
        )
        entry["body_in_frame_ratio"] = ratio
        count += 1

    output_path = args.output_path or args.annotations_path
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Added body_in_frame_ratio to {count}/{len(annotations)} entries")
    print(f"Saved to {output_path}")

    # Quick stats
    ratios = [e["body_in_frame_ratio"] for e in annotations if "body_in_frame_ratio" in e]
    import numpy as np
    r = np.array(ratios)
    print(f"Stats: min={r.min():.3f} max={r.max():.3f} mean={r.mean():.3f} median={np.median(r):.3f}")
    print(f"  =1.0 (fully visible): {(r >= 0.99).sum()} ({(r >= 0.99).mean()*100:.1f}%)")
    print(f"  >0.8: {(r > 0.8).sum()} ({(r > 0.8).mean()*100:.1f}%)")
    print(f"  <0.5: {(r < 0.5).sum()} ({(r < 0.5).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
