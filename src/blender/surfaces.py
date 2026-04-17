"""Flat surface discovery via raycasting for object placement."""

from __future__ import annotations

import math
import random as _random

import bpy
from mathutils import Vector

def get_camera_ground_target(scene, depsgraph):
    """Get the center of the area the scene camera is looking at.

    This is WHERE the camera is pointed — the center of the room/scene
    that the artist designed. Objects should be placed HERE, not between
    the camera and here (which often lands outside through windows).

    Returns (target_point, camera_z) or (None, None) if no camera.
    """
    cam = scene.camera
    if cam is None:
        return None, None

    cam_pos = cam.matrix_world.translation.copy()
    fwd = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))

    # Raycast along camera forward to find where the camera looks
    hit, loc, *_ = scene.ray_cast(depsgraph, cam_pos, fwd)
    if hit:
        # Use the actual look-at point (center of the room/scene)
        # not a point between camera and wall
        target = loc.copy()
        return target, cam_pos.z

    # Fallback: point in front of camera
    if abs(fwd.z) > 0.01:
        t = -2.0 / fwd.z
        if t > 0:
            ground = cam_pos + fwd * min(t, 5.0)
            return ground, cam_pos.z

    return cam_pos + fwd * 3.0, cam_pos.z


def compute_scene_camera_view(scene, depsgraph, n_rays: int = 14):
    """Sample what the scene's authored camera sees.

    Casts an n_rays × n_rays grid through the scene camera's frustum and
    returns:
        cam_pos:        scene camera world position (or None if no camera)
        visible_hits:   list of world-space hit points (the visible surface set)
        centroid:       XY mean of all visible hits (or None if no hits)

    Used by find_flat_candidates() to (a) reject placement spots not visible
    from the scene camera and (b) bias candidates toward the centroid of the
    visible region.
    """
    cam = scene.camera
    if cam is None:
        return None, [], None

    cam_matrix = cam.matrix_world
    cam_pos = cam_matrix.translation.copy()
    cam_data = cam.data
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    if cam_data.type == "PERSP":
        half_fov_y = cam_data.angle / 2
        half_fov_x = math.atan(math.tan(half_fov_y) * aspect)
    else:
        half_fov_x = half_fov_y = 0.5

    rot = cam_matrix.to_quaternion()
    hits = []
    for iy in range(n_rays):
        for ix in range(n_rays):
            u = (ix + 0.5) / n_rays * 2 - 1
            v = (iy + 0.5) / n_rays * 2 - 1
            local_dir = Vector((
                u * math.tan(half_fov_x),
                v * math.tan(half_fov_y),
                -1.0,
            )).normalized()
            world_dir = rot @ local_dir
            hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, world_dir)
            if hit:
                hits.append(loc.copy())

    if not hits:
        return cam_pos, [], None
    cx = sum(h.x for h in hits) / len(hits)
    cy = sum(h.y for h in hits) / len(hits)
    cz = sum(h.z for h in hits) / len(hits)
    centroid = Vector((cx, cy, cz))
    return cam_pos, hits, centroid


def is_visible_from_camera(scene, depsgraph, cam_pos, target_point,
                            ignore_names=None, tolerance: float = 0.4):
    """Return True if target_point has clear line of sight from cam_pos.

    Casts a ray from cam_pos toward target_point. If the first hit is at
    or beyond the target distance (within `tolerance` meters), the target
    is considered visible. Hits on objects in `ignore_names` are skipped.
    """
    direction = target_point - cam_pos
    distance = direction.length
    if distance < 0.01:
        return True
    direction_norm = direction / distance
    hit, loc, _, _, obj, _ = scene.ray_cast(depsgraph, cam_pos, direction_norm)
    if not hit:
        return True  # nothing in the way
    if ignore_names and obj is not None and obj.name in ignore_names:
        return True
    hit_distance = (loc - cam_pos).length
    return hit_distance >= distance - tolerance


MAX_SLOPE_DEG = 12.0
HEIGHT_TOL = 0.08
FOOTPRINT_MARGIN = 0.15
CELL_SIZE = 1.0
MAX_GRID_SAMPLES = 20000
SAMPLES_PER_SIDE = 3


def iter_grid(x_min, x_max, y_min, y_max, step):
    """Yield (x, y) grid points within bounds."""
    if step <= 0:
        return
    nx = max(1, int(math.floor((x_max - x_min) / step)) + 1)
    ny = max(1, int(math.floor((y_max - y_min) / step)) + 1)
    for i in range(nx):
        x = x_min + i * step
        for j in range(ny):
            y = y_min + j * step
            yield x, y


def area_is_flat(scene, depsgraph, center, obj_size, z_top, slope_cos, ignore_names,
                 height_tol=None):
    """Check if the area around center is flat enough for placement.

    Tries both downward and upward raycasts to handle surfaces with
    inverted normals (common for table tops in imported scenes).
    """
    if height_tol is None:
        height_tol = HEIGHT_TOL
    half_x = obj_size.x * 0.5 + FOOTPRINT_MARGIN
    half_y = obj_size.y * 0.5 + FOOTPRINT_MARGIN
    heights = []
    for i in range(SAMPLES_PER_SIDE):
        for j in range(SAMPLES_PER_SIDE):
            if SAMPLES_PER_SIDE > 1:
                tx = -half_x + 2 * half_x * (i / (SAMPLES_PER_SIDE - 1))
                ty = -half_y + 2 * half_y * (j / (SAMPLES_PER_SIDE - 1))
            else:
                tx, ty = 0, 0
            # Try downward first
            origin_down = Vector((center.x + tx, center.y + ty, z_top))
            hit, loc, normal, _, obj, _ = scene.ray_cast(
                depsgraph, origin_down, Vector((0, 0, -1))
            )
            # If downward misses or hits wrong thing, try upward from below
            if not hit or obj is None or obj.name in ignore_names:
                origin_up = Vector((center.x + tx, center.y + ty, center.z - 0.3))
                hit, loc, normal, _, obj, _ = scene.ray_cast(
                    depsgraph, origin_up, Vector((0, 0, 1))
                )
            if not hit or obj is None:
                return False
            if obj.name in ignore_names:
                return False
            if abs(normal.z) < slope_cos:
                return False
            heights.append(loc.z)
    if not heights:
        return False
    return (max(heights) - min(heights)) <= height_tol


def find_flat_candidates(
    scene,
    depsgraph,
    bounds_min,
    bounds_max,
    obj_size,
    ignore_names,
    top_k=5,
    min_distance=2.0,
    height_tol=None,
    max_slope_deg=None,
    camera_target=None,
    object_seed=None,
    scene_camera_pos=None,
    visible_centroid=None,
):
    """Find top-K flat placement candidates, spread apart by min_distance."""
    if height_tol is None:
        height_tol = HEIGHT_TOL
    if max_slope_deg is None:
        max_slope_deg = MAX_SLOPE_DEG
    slope_cos = math.cos(math.radians(max_slope_deg))
    obj_radius = max(obj_size.x, obj_size.y) * 0.5

    x_min = bounds_min.x + obj_radius
    x_max = bounds_max.x - obj_radius
    y_min = bounds_min.y + obj_radius
    y_max = bounds_max.y - obj_radius
    if x_min >= x_max or y_min >= y_max:
        x_min, x_max = bounds_min.x, bounds_max.x
        y_min, y_max = bounds_min.y, bounds_max.y

    area = max(0.01, (bounds_max.x - bounds_min.x) * (bounds_max.y - bounds_min.y))
    desired_step = max(CELL_SIZE * 0.5, obj_radius * 0.5)
    min_step = math.sqrt(area / MAX_GRID_SAMPLES)
    step = max(desired_step, min_step)

    # Raycast from multiple heights AND directions to find surfaces at all levels.
    # Key insight: to find TABLE TOPS, cast UPWARD from below table height.
    # Downward rays from above a table pass through it and hit the floor.
    scene_height = bounds_max.z - bounds_min.z
    # (z_origin, ray_direction_z): -1 = down, +1 = up
    z_ray_configs = [
        # Floor: cast down from just above floor level
        (bounds_min.z + 0.3, -1),
        # Tables/counters: cast UP from below furniture height to find tops
        (bounds_min.z + 0.5, 1),     # hits table tops at ~0.75m
        (bounds_min.z + 0.9, 1),     # hits counter tops at ~1.0m
        (bounds_min.z + 1.2, 1),     # hits tall counters/shelves at ~1.3m
        # Mid-room (catches odd geometry)
        (bounds_min.z + scene_height * 0.5, -1),
        # From below floor (inverted normals)
        (bounds_min.z - 0.5, 1),
        # From above (outdoor scenes)
        (bounds_max.z + max(5.0, obj_size.z * 2.0), -1),
    ]

    cells: dict[tuple[int, int, int], dict] = {}
    for z_idx, (z_start, ray_dir_z) in enumerate(z_ray_configs):
        ray_dir = Vector((0, 0, ray_dir_z))
        for x, y in iter_grid(x_min, x_max, y_min, y_max, step):
            origin = Vector((x, y, z_start))
            hit, loc, normal, _, obj, _ = scene.ray_cast(
                depsgraph, origin, ray_dir
            )
            if not hit or obj is None:
                continue
            if obj.name in ignore_names:
                continue
            # Accept upward-facing normals AND flipped normals (abs)
            if abs(normal.z) < slope_cos:
                continue
            ix = int(math.floor((loc.x - bounds_min.x) / CELL_SIZE))
            iy = int(math.floor((loc.y - bounds_min.y) / CELL_SIZE))
            # Include z_idx in key so different ray heights don't mix
            key = (ix, iy, z_idx)
            if key not in cells:
                cells[key] = {
                    "count": 0,
                    "sum_loc": Vector((0, 0, 0)),
                    "sum_n": Vector((0, 0, 0)),
                    "heights": [],
                }
            cells[key]["count"] += 1
            cells[key]["sum_loc"] += loc
            cells[key]["sum_n"] += normal
            cells[key]["heights"].append(loc.z)

    if not cells:
        return []

    # Reference point for "central in scene-camera view" scoring.
    # Priority: visible_centroid (frustum sampling) > camera_target (single ray)
    # > geometric scene center.
    if visible_centroid is not None:
        ref_xy = Vector((visible_centroid.x, visible_centroid.y))
        ref_z = visible_centroid.z
    elif camera_target is not None:
        ref_xy = Vector((camera_target.x, camera_target.y))
        ref_z = camera_target.z
    else:
        ref_xy = Vector(
            ((bounds_min.x + bounds_max.x) / 2, (bounds_min.y + bounds_max.y) / 2)
        )
        ref_z = None

    scene_diag = math.sqrt(
        (bounds_max.x - bounds_min.x) ** 2 + (bounds_max.y - bounds_min.y) ** 2
    )

    scored = []
    rejected_invisible = 0
    for cell in cells.values():
        c = cell["count"]
        if c < 1:
            continue
        center = cell["sum_loc"] / c
        heights = cell["heights"]
        if max(heights) - min(heights) > height_tol:
            continue
        # Validate flatness: cast from just 0.3m above the surface.
        # Using a small offset avoids hitting ceilings when validating table tops.
        z_validate = center.z + 0.3
        if not area_is_flat(
            scene, depsgraph, center, obj_size, z_validate, slope_cos, ignore_names,
            height_tol=height_tol,
        ):
            continue

        # HARD VISIBILITY GATE: candidate must be visible from the scene
        # camera. Probe a point ~0.4m above the surface (where the object's
        # midsection will be) so we don't trip on the floor itself.
        if scene_camera_pos is not None:
            probe = Vector((center.x, center.y, center.z + 0.4))
            if not is_visible_from_camera(
                scene, depsgraph, scene_camera_pos, probe,
                ignore_names=ignore_names, tolerance=0.5,
            ):
                rejected_invisible += 1
                continue

        normal = cell["sum_n"].normalized()
        dist = (Vector((center.x, center.y)) - ref_xy).length
        center_score = 1.0 - min(1.0, dist / (scene_diag * 0.5 + 0.01))
        flatness_score = 1.0 - min(1.0, (max(heights) - min(heights)) / height_tol)
        density_score = min(1.0, c / 10.0)
        # Mild preference for lower surfaces (floors), but allow tables,
        # counters, shelves — small objects can sit on furniture.
        scene_height = max(0.01, bounds_max.z - bounds_min.z)
        z_normalized = (center.z - bounds_min.z) / scene_height
        floor_score = 1.0 - z_normalized
        score = (center_score * 0.4 + flatness_score * 0.2
                 + density_score * 0.1 + floor_score * 0.3)
        scored.append((score, center, normal))

    if rejected_invisible:
        print(f"  visibility filter dropped {rejected_invisible} cells "
              f"(not visible from scene camera)")

    # Filter out roof/ceiling candidates but keep furniture-height surfaces.
    # Surfaces up to 2m above the floor are plausible (tables, counters, shelves).
    # Surfaces above that are likely roofs/ceilings.
    if scored:
        floor_z = min(c.z for _, c, _ in scored)
        scene_h = bounds_max.z - bounds_min.z
        # For small scenes (<4m tall), allow up to 2m above floor.
        # For tall scenes (>4m), be more conservative to avoid roofs.
        max_surface_z = floor_z + min(2.0, scene_h * 0.6)
        scored = [(s, c, n) for s, c, n in scored if c.z <= max_surface_z]

    # Per-object diversification: add score noise so different objects
    # get different candidate orderings from the same scene.
    if object_seed is not None:
        rng = _random.Random(object_seed)
        scored = [(s + rng.uniform(-0.1, 0.1), c, n) for s, c, n in scored]

    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedy selection with min_distance, pulling from a larger pool
    pool_size = max(top_k * 4, 30)
    selected = []
    for score, center, normal in scored:
        too_close = False
        for _, prev_center, _ in selected:
            dx = center.x - prev_center.x
            dy = center.y - prev_center.y
            dz = center.z - prev_center.z
            # Use 3D distance so candidates at different heights (floor vs table) coexist
            if math.sqrt(dx * dx + dy * dy + dz * dz) < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append((score, center, normal))
            if len(selected) >= pool_size:
                break

    # From the larger pool, pick top_k with score-biased sampling.
    # Always keep the best candidate, then sample the rest by score weight.
    if object_seed is not None and len(selected) > top_k:
        rng2 = _random.Random(object_seed + 1)
        final = [selected[0]]  # always keep the best
        remaining = list(selected[1:])
        while len(final) < top_k and remaining:
            # Weight by score rank: top items more likely
            weights = [max(0.01, s) for s, _, _ in remaining]
            total_w = sum(weights)
            r = rng2.uniform(0, total_w)
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r:
                    final.append(remaining.pop(i))
                    break
        selected = final
    else:
        selected = selected[:top_k]

    # Guarantee diversity: if we have elevated candidates in the full pool
    # but none in the selection, swap one in.
    if selected:
        floor_z = min(c.z for _, c, _ in selected)
        has_elevated = any(c.z > floor_z + 0.3 for _, c, _ in selected)
        if not has_elevated:
            # Find the best elevated candidate from the full scored list
            elevated = [(s, c, n) for s, c, n in scored if c.z > floor_z + 0.3]
            if elevated:
                best_elevated = max(elevated, key=lambda x: x[0])
                selected[-1] = best_elevated  # replace the worst floor candidate

    return [
        {
            "position": [c.x, c.y, c.z],
            "normal": [n.x, n.y, n.z],
            "score": s,
        }
        for s, c, n in selected
    ]


def find_candidates_multi_height(
    scene, depsgraph, bounds_min, bounds_max, obj_size, ignore_names,
    top_k=5, min_distance=2.0, object_seed=None,
    scene_camera_pos=None, visible_centroid=None,
):
    """Try raycasting from multiple Z heights for indoor scenes with ceilings."""
    scene_height = bounds_max.z - bounds_min.z
    test_heights = [
        bounds_min.z + scene_height * f
        for f in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    ]

    all_candidates = []
    for z_start in test_heights:
        candidates = find_flat_candidates(
            scene, depsgraph, bounds_min, bounds_max, obj_size, ignore_names,
            top_k=top_k, min_distance=min_distance,
            height_tol=0.3, max_slope_deg=20.0,
            object_seed=object_seed,
            scene_camera_pos=scene_camera_pos,
            visible_centroid=visible_centroid,
        )
        if candidates:
            all_candidates.extend(candidates)

    if not all_candidates:
        return []

    selected = []
    for c in sorted(all_candidates, key=lambda x: x["score"], reverse=True):
        too_close = False
        for s in selected:
            dx = c["position"][0] - s["position"][0]
            dy = c["position"][1] - s["position"][1]
            if math.sqrt(dx * dx + dy * dy) < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(c)
            if len(selected) >= top_k:
                break
    return selected
