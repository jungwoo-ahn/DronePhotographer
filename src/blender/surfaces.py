"""Flat surface discovery via raycasting for object placement."""

from __future__ import annotations

import math
import statistics as _stats

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


# ---------------------------------------------------------------------------
# Diversity / interest helpers
# ---------------------------------------------------------------------------


def _compute_adaptive_min_distance(visible_hits, override_floor: float) -> float:
    """Scale min_distance by the visible-bbox diagonal so large scenes spread.

    Formula: clamp(vis_diag * 0.08, override_floor, 12.0).
    Falls back to override_floor when visible_hits is empty.
    """
    if not visible_hits:
        return max(override_floor, 0.8)
    xs = [h.x for h in visible_hits]
    ys = [h.y for h in visible_hits]
    vis_diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    return max(override_floor, min(12.0, vis_diag * 0.08))


def _identify_ground_mesh_names(scene_meshes, scene_height: float, vis_bbox) -> set:
    """Heuristic: a mesh is 'ground' if its Z-extent is small relative to scene
    height AND its XY footprint covers a large fraction of the visible region.

    Used to exclude ground from the variety-ring score (rays hitting ground
    don't contribute to surroundings interest).
    """
    if not scene_meshes or vis_bbox is None:
        return set()
    x_lo, x_hi, y_lo, y_hi = vis_bbox
    vis_area = max(0.01, (x_hi - x_lo) * (y_hi - y_lo))
    ground_names = set()
    for obj in scene_meshes:
        try:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        except Exception:
            continue
        if not corners:
            continue
        zs = [c.z for c in corners]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        z_ext = max(zs) - min(zs)
        footprint = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        is_flat = scene_height > 0 and z_ext < 0.3 * scene_height
        is_wide = footprint > 0.5 * vis_area
        if is_flat and is_wide:
            ground_names.add(obj.name)
    return ground_names


def _compute_variety_score(scene, depsgraph, candidate_pos: Vector,
                           ground_names: set, ignore_names,
                           n_dirs: int = 16) -> float:
    """Score visual variety of surroundings from a candidate position.

    Casts a horizontal ring of rays from eye height. More distinct non-ground
    mesh hits + greater variance in hit distances = richer surroundings.
    Returns a value in [0, 1].
    """
    eye = Vector((candidate_pos.x, candidate_pos.y, candidate_pos.z + 1.5))
    hit_meshes = set()
    hit_distances = []
    for i in range(n_dirs):
        theta = (i / n_dirs) * 2 * math.pi
        direction = Vector((math.cos(theta), math.sin(theta), 0.0))
        hit, loc, _, _, obj, _ = scene.ray_cast(depsgraph, eye, direction)
        if not hit or obj is None:
            continue
        if obj.name in ignore_names or obj.name in ground_names:
            continue
        hit_meshes.add(obj.name)
        hit_distances.append((loc - eye).length)
    n_distinct = len(hit_meshes)
    if len(hit_distances) >= 2:
        try:
            d_std = _stats.pstdev(hit_distances)
        except _stats.StatisticsError:
            d_std = 0.0
    else:
        d_std = 0.0
    return min(1.0, 0.1 * n_distinct + 0.05 * d_std)


def _compute_region_score(candidate_xy: Vector, interest_region,
                          vis_diag: float, B: int) -> float:
    """Score proximity to the VLM-identified interest region.

    `interest_region` is a dict with keys {x_lo, x_hi, y_lo, y_hi} in world
    coordinates, or None. Returns 1.0 inside the region, falls off linearly
    outside up to ~one bucket-width of grace.
    """
    if interest_region is None:
        return 1.0  # no info → neutral (don't penalize)
    x_lo = interest_region["x_lo"]
    x_hi = interest_region["x_hi"]
    y_lo = interest_region["y_lo"]
    y_hi = interest_region["y_hi"]
    # Distance from candidate to nearest point on the region rectangle
    dx = max(x_lo - candidate_xy.x, 0.0, candidate_xy.x - x_hi)
    dy = max(y_lo - candidate_xy.y, 0.0, candidate_xy.y - y_hi)
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= 0.0:
        return 1.0
    falloff = max(1.0, 2.0 * vis_diag / max(1, B))
    return max(0.0, 1.0 - dist / falloff)


def _bucket_index(center: Vector, vis_bbox, B: int) -> int:
    """Map a candidate's XY to a bucket index in a B×B grid over vis_bbox."""
    if vis_bbox is None or B <= 1:
        return 0
    x_lo, x_hi, y_lo, y_hi = vis_bbox
    # Clamp to bbox (candidates outside are rare — already filtered)
    x = max(x_lo, min(center.x, x_hi - 1e-6))
    y = max(y_lo, min(center.y, y_hi - 1e-6))
    width = max(1e-6, x_hi - x_lo)
    height = max(1e-6, y_hi - y_lo)
    ix = int((x - x_lo) / width * B)
    iy = int((y - y_lo) / height * B)
    ix = max(0, min(B - 1, ix))
    iy = max(0, min(B - 1, iy))
    return iy * B + ix


def _permuted_bucket_weights(object_seed: int, n_buckets: int) -> list:
    """Deterministic per-object bucket weights.

    Shuffles bucket indices using the seed, assigns geometric weights w_i = 0.5^i
    on the permuted order, normalizes so weights sum to n_buckets (keeping the
    overall score scale close to unperturbed for stability).

    Returns a list of length n_buckets where index = bucket id, value = weight.
    """
    if n_buckets <= 1:
        return [1.0] * max(1, n_buckets)
    import random as _random
    rng = _random.Random(object_seed)
    order = list(range(n_buckets))
    rng.shuffle(order)
    raw = [0.5 ** i for i in range(n_buckets)]
    total_raw = sum(raw)
    normalized = [w / total_raw * n_buckets for w in raw]
    out = [0.0] * n_buckets
    for rank, bucket_id in enumerate(order):
        out[bucket_id] = normalized[rank]
    return out


# ---------------------------------------------------------------------------
# Main discovery entry point
# ---------------------------------------------------------------------------


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
    visible_hits=None,
    scene_meshes=None,
    interest_region=None,
):
    """Find top-K flat placement candidates, spread apart by min_distance.

    Scoring (new): score = 0.15*center + 0.15*flatness + 0.25*floor + 0.45*interest,
    where interest combines (a) proximity to VLM-identified interest region and
    (b) a geometric "surroundings variety" from a horizontal ray ring cast at
    eye height.

    Diversity: candidates are bucketed on the visible XY bbox; each object's
    `object_seed` permutes buckets to give different objects different bucket
    preferences, so the top-1 differs per object.
    """
    if height_tol is None:
        height_tol = HEIGHT_TOL
    if max_slope_deg is None:
        max_slope_deg = MAX_SLOPE_DEG
    slope_cos = math.cos(math.radians(max_slope_deg))
    obj_radius = max(obj_size.x, obj_size.y) * 0.5

    # Adaptive min_distance scales with the visible-bbox diagonal so large
    # scenes actually spread. `min_distance` from caller becomes the floor.
    adaptive_min_distance = _compute_adaptive_min_distance(visible_hits, min_distance)

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
    scene_height_total = bounds_max.z - bounds_min.z
    z_ray_configs = [
        (bounds_min.z + 0.3, -1),
        (bounds_min.z + 0.5, 1),
        (bounds_min.z + 0.9, 1),
        (bounds_min.z + 1.2, 1),
        (bounds_min.z + scene_height_total * 0.5, -1),
        (bounds_min.z - 0.5, 1),
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
            if abs(normal.z) < slope_cos:
                continue
            ix = int(math.floor((loc.x - bounds_min.x) / CELL_SIZE))
            iy = int(math.floor((loc.y - bounds_min.y) / CELL_SIZE))
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
    if visible_centroid is not None:
        ref_xy = Vector((visible_centroid.x, visible_centroid.y))
    elif camera_target is not None:
        ref_xy = Vector((camera_target.x, camera_target.y))
    else:
        ref_xy = Vector(
            ((bounds_min.x + bounds_max.x) / 2, (bounds_min.y + bounds_max.y) / 2)
        )

    scene_diag = math.sqrt(
        (bounds_max.x - bounds_min.x) ** 2 + (bounds_max.y - bounds_min.y) ** 2
    )

    # Hard XY bbox from the scene-camera frustum hits.
    vis_bbox = None
    vis_diag = scene_diag
    if visible_hits:
        xs = [h.x for h in visible_hits]
        ys = [h.y for h in visible_hits]
        margin = max(obj_size.x, obj_size.y) * 0.5 + 1.0
        vis_bbox = (
            min(xs) - margin, max(xs) + margin,
            min(ys) - margin, max(ys) + margin,
        )
        vis_diag = math.sqrt(
            (vis_bbox[1] - vis_bbox[0]) ** 2 + (vis_bbox[3] - vis_bbox[2]) ** 2
        )

    # Bucket grid for per-object diversity. B=1 in small scenes (graceful).
    B = max(1, min(5, round(vis_diag / 3.0)))

    # Ground-mesh identification for the variety-ring score.
    scene_height_final = max(0.01, bounds_max.z - bounds_min.z)
    ground_names = _identify_ground_mesh_names(
        scene_meshes or [], scene_height_final, vis_bbox
    )

    scored = []
    rejected_invisible = 0
    rejected_out_of_frame = 0
    for cell in cells.values():
        c = cell["count"]
        if c < 1:
            continue
        center = cell["sum_loc"] / c
        # Hard frustum-bbox check.
        if vis_bbox is not None:
            x_lo, x_hi, y_lo, y_hi = vis_bbox
            if not (x_lo <= center.x <= x_hi and y_lo <= center.y <= y_hi):
                rejected_out_of_frame += 1
                continue
        heights = cell["heights"]
        if max(heights) - min(heights) > height_tol:
            continue
        z_validate = center.z + 0.3
        if not area_is_flat(
            scene, depsgraph, center, obj_size, z_validate, slope_cos, ignore_names,
            height_tol=height_tol,
        ):
            continue

        # HARD VISIBILITY GATE: probe a point ~0.4m above the surface.
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
        z_normalized = (center.z - bounds_min.z) / scene_height_final
        floor_score = 1.0 - z_normalized

        # Interest = 0.6 * (near VLM region) + 0.4 * (surroundings variety).
        region_score = _compute_region_score(
            Vector((center.x, center.y)), interest_region, vis_diag, B
        )
        variety_score = _compute_variety_score(
            scene, depsgraph, center, ground_names, ignore_names, n_dirs=16,
        )
        interest_score = 0.6 * region_score + 0.4 * variety_score

        score = (
            0.15 * center_score
            + 0.15 * flatness_score
            + 0.25 * floor_score
            + 0.45 * interest_score
        )
        scored.append((score, center, normal))

    if rejected_out_of_frame:
        print(f"  frustum filter dropped {rejected_out_of_frame} cells "
              f"(outside scene-camera view bbox)")
    if rejected_invisible:
        print(f"  visibility filter dropped {rejected_invisible} cells "
              f"(not visible from scene camera)")

    # Filter out roof/ceiling candidates.
    if scored:
        floor_z = min(c.z for _, c, _ in scored)
        max_surface_z = floor_z + min(2.0, scene_height_final * 0.6)
        scored = [(s, c, n) for s, c, n in scored if c.z <= max_surface_z]

    if not scored:
        return []

    # ---- Per-object spatial bucketing ----
    # Group scored candidates by bucket, apply per-object bucket weights to
    # re-rank. This replaces the old ±0.1 jitter + top-1-kept sampling.
    cand_with_bucket = []
    buckets_used: set = set()
    for s, c, n in scored:
        bid = _bucket_index(c, vis_bbox, B)
        cand_with_bucket.append((bid, s, c, n))
        buckets_used.add(bid)

    n_buckets_used = len(buckets_used)
    if object_seed is not None and len(cand_with_bucket) > 1:
        # Two independent per-object shuffles, multiplied:
        #   (a) BUCKET weights: spatial diversity — different objects prefer
        #       different quadrants of the scene. Softer geometric (0.7^i)
        #       so top-2 buckets are similar-weighted, letting (b) break ties.
        #   (b) CANDIDATE rank weights: within-bucket diversity — different
        #       objects prefer different positions inside their preferred area.
        # The product gives a deterministic-per-seed but well-spread ordering.
        import random as _r

        # Bucket weights (skipped if everything is in one bucket)
        if n_buckets_used > 1:
            bucket_w = _permuted_bucket_weights(object_seed, B * B)
        else:
            bucket_w = [1.0] * (B * B)

        # Candidate rank weights — shuffle candidate indices with a distinct seed
        rng2 = _r.Random(object_seed ^ 0xA5A5A5A5)
        n_cand = len(cand_with_bucket)
        order = list(range(n_cand))
        rng2.shuffle(order)
        raw = [0.7 ** i for i in range(n_cand)]
        total_raw = sum(raw)
        rank_weights = [w / total_raw * n_cand for w in raw]
        weight_per_cand = [0.0] * n_cand
        for rank, cand_idx in enumerate(order):
            weight_per_cand[cand_idx] = rank_weights[rank]

        reweighted = [
            (s * bucket_w[bid] * weight_per_cand[i], c, n, bid)
            for i, (bid, s, c, n) in enumerate(cand_with_bucket)
        ]
    else:
        reweighted = [(s, c, n, bid) for bid, s, c, n in cand_with_bucket]

    reweighted.sort(key=lambda x: x[0], reverse=True)

    # Greedy selection with adaptive min_distance + per-bucket cap.
    per_bucket_cap = max(1, math.ceil(top_k / max(1, n_buckets_used)) + 1) if n_buckets_used > 1 else top_k
    bucket_counts: dict[int, int] = {}
    selected = []
    for score, center, normal, bid in reweighted:
        if bucket_counts.get(bid, 0) >= per_bucket_cap:
            continue
        too_close = False
        for _, prev_center, _, _ in selected:
            dx = center.x - prev_center.x
            dy = center.y - prev_center.y
            dz = center.z - prev_center.z
            if math.sqrt(dx * dx + dy * dy + dz * dz) < adaptive_min_distance:
                too_close = True
                break
        if too_close:
            continue
        selected.append((score, center, normal, bid))
        bucket_counts[bid] = bucket_counts.get(bid, 0) + 1
        if len(selected) >= top_k:
            break

    # If under-filled (small scenes), relax the per-bucket cap and re-scan
    # but still respect min_distance.
    if len(selected) < top_k:
        for score, center, normal, bid in reweighted:
            if (score, center, normal, bid) in selected:
                continue
            too_close = False
            for _, prev_center, _, _ in selected:
                dx = center.x - prev_center.x
                dy = center.y - prev_center.y
                dz = center.z - prev_center.z
                if math.sqrt(dx * dx + dy * dy + dz * dz) < adaptive_min_distance:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append((score, center, normal, bid))
            if len(selected) >= top_k:
                break

    # Guarantee some elevated candidate if one exists in the scored pool.
    if selected:
        floor_z = min(c.z for _, c, _, _ in selected)
        has_elevated = any(c.z > floor_z + 0.3 for _, c, _, _ in selected)
        if not has_elevated:
            elevated = [(s, c, n, b) for s, c, n, b in reweighted
                        if c.z > floor_z + 0.3]
            if elevated:
                best_elevated = max(elevated, key=lambda x: x[0])
                selected[-1] = best_elevated

    return [
        {
            "position": [c.x, c.y, c.z],
            "normal": [n.x, n.y, n.z],
            "score": s,
        }
        for s, c, n, _ in selected
    ]


def find_candidates_multi_height(
    scene, depsgraph, bounds_min, bounds_max, obj_size, ignore_names,
    top_k=5, min_distance=2.0, object_seed=None,
    scene_camera_pos=None, visible_centroid=None, visible_hits=None,
    scene_meshes=None, interest_region=None,
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
            visible_hits=visible_hits,
            scene_meshes=scene_meshes,
            interest_region=interest_region,
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
