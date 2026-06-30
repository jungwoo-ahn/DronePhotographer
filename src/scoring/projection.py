"""bpy-free geometric scoring of a camera pose against a subject.

This is the eval-time counterpart of the v7 data pipeline's stage-3 scorer. It
reproduces the *exact* shot profile stored in `data.json` render_records, so a
closed-loop rollout's achieved profile is directly comparable to the goals the
policy was trained on. Empirically validated against the dataset:

  * `compute_v5_scores(W, H, bbox_xyxy_full, az, el)` reproduces the stored
    integer `scores` on 1312/1312 frames (it IS the stage-3 scorer).
  * az/el is the **world-frame** cam->subject_center angle (azW/elW), matching
    100% of 2561 stored frames — the same formula as
    `BlenderRolloutEnv.pose_proxy_distance`.
  * `bbox_xyxy_full` is the **mesh-tight projected AABB** of all subject mesh
    vertices (unclamped) — `project_verts_to_bbox` below, lifted verbatim from
    `scripts/compute_mesh_tight_bbox.py:project_verts_vec`.

The two Blender-only inputs (`verts_world`, `frame_bounds`) are extracted once
per placement at env setup; everything here is pure numpy so per-step scoring in
the rollout loop needs no Blender call.
"""

from __future__ import annotations

import math

import numpy as np

from src.scoring.bbox_control import compute_v5_scores

# (min_x, max_x, min_y, max_y) of the camera view frame normalized to z=1.
FrameBounds = tuple[float, float, float, float]


def camera_basis(forward, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World axes (right, up, -forward) for a Blender camera (+X right, +Y up, -Z fwd)."""
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / np.linalg.norm(upv)
    nz = -fwd
    right = np.cross(upv, nz)
    right /= np.linalg.norm(right)
    ortho_up = np.cross(nz, right)
    ortho_up /= np.linalg.norm(ortho_up)
    return right, ortho_up, nz


def project_verts_to_bbox(
    verts_world: np.ndarray,
    cam_pos,
    forward,
    up,
    width: int,
    height: int,
    frame_bounds: FrameBounds,
) -> list[float] | None:
    """Vectorized projection of world verts -> mesh-tight 2D AABB [x1,y1,x2,y2].

    Unclamped (may extend beyond the image), matching the dataset's
    `bbox_xyxy_full`. Returns None if no vertex is in front of the camera.
    Verbatim port of `scripts/compute_mesh_tight_bbox.py:project_verts_vec`.
    """
    min_x, max_x, min_y, max_y = frame_bounds
    right, up_ax, nz = camera_basis(forward, up)
    cam_pos = np.asarray(cam_pos, dtype=np.float64)

    rel = np.asarray(verts_world, dtype=np.float64) - cam_pos       # (N, 3)
    co_x = rel @ right
    co_y = rel @ up_ax
    z = -(rel @ nz)                                                 # forward distance

    valid = z > 1e-6
    if not np.any(valid):
        return None
    co_x, co_y, z = co_x[valid], co_y[valid], z[valid]

    # frame corners scale linearly with depth (frame defined at z = 1).
    fmin_x, fmax_x = min_x * z, max_x * z
    fmin_y, fmax_y = min_y * z, max_y * z
    ndc_x = (co_x - fmin_x) / (fmax_x - fmin_x)
    ndc_y = (co_y - fmin_y) / (fmax_y - fmin_y)
    x_px = ndc_x * float(width)
    y_px = (1.0 - ndc_y) * float(height)
    return [float(x_px.min()), float(y_px.min()), float(x_px.max()), float(y_px.max())]


def cam_to_subject_angles(cam_pos, subject_center) -> tuple[float, float]:
    """World-frame azimuth/elevation of the cam->subject vector, in degrees.

    Matches the stored `cam_to_obj_azimuth_deg` / `cam_to_obj_elevation_deg`
    (validated 100% on real frames) and `BlenderRolloutEnv.pose_proxy_distance`.
    Elevation is negative when the camera is above the subject.
    """
    d = np.asarray(subject_center, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    az = math.degrees(math.atan2(float(d[1]), float(d[0])))
    el = math.degrees(math.atan2(float(d[2]), float(math.hypot(d[0], d[1]))))
    return az, el


def score_pose(
    cam_pos,
    forward,
    up,
    verts_world: np.ndarray,
    subject_center,
    width: int,
    height: int,
    frame_bounds: FrameBounds,
) -> dict[str, int]:
    """The 8-key V5 shot profile achieved by a camera pose (integer schema)."""
    bbox = project_verts_to_bbox(verts_world, cam_pos, forward, up, width, height, frame_bounds)
    az, el = cam_to_subject_angles(cam_pos, subject_center)
    return compute_v5_scores(width, height, bbox, az, el)
