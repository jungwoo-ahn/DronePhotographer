"""Camera setup utilities for Blender."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector


def _compute_cam_position(target_pos, obj_size, radius, elevation_deg, azimuth_deg,
                          max_z=None):
    """Compute camera position from spherical coordinates around target.

    If max_z is set, clamp camera Z to stay below it (e.g. scene camera height).
    """
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    cam_x = target_pos.x + radius * math.cos(elev) * math.sin(azim)
    cam_y = target_pos.y - radius * math.cos(elev) * math.cos(azim)
    cam_z = target_pos.z + obj_size.z * 0.4 + radius * math.sin(elev)
    if max_z is not None and cam_z > max_z:
        cam_z = max_z
    return Vector((cam_x, cam_y, cam_z))


def is_camera_valid(cam_pos, target_pos, ignore_names=None, min_clearance=0.8,
                    check_floor=True):
    """Check if camera position is a plausible place for a photographer.

    Tests:
    1. Camera has enough open space around it (not inside walls/stairs/furniture)
    2. Floor below camera (within 3m) — skipped when ``check_floor=False``
    3. Clear line of sight from camera to target
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Test 1: Camera must have open space — like a person standing with a camera.
    # Cast rays in 14 directions; reject if too many hit close geometry.
    test_dirs = [
        Vector((1, 0, 0)), Vector((-1, 0, 0)),
        Vector((0, 1, 0)), Vector((0, -1, 0)),
        Vector((0, 0, 1)),  # above
        Vector((1, 1, 0)).normalized(), Vector((-1, 1, 0)).normalized(),
        Vector((1, -1, 0)).normalized(), Vector((-1, -1, 0)).normalized(),
        Vector((1, 0, 1)).normalized(), Vector((-1, 0, 1)).normalized(),
        Vector((0, 1, 1)).normalized(), Vector((0, -1, 1)).normalized(),
    ]
    close_hits = 0
    for d in test_dirs:
        hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, d)
        if hit and (loc - cam_pos).length < min_clearance:
            close_hits += 1
    if close_hits >= 4:
        return False  # too enclosed — inside furniture, under stairs, etc.

    # Test 2: Floor below camera (within 3m) — camera shouldn't float in void
    if check_floor:
        hit_floor, loc_floor, _, _, _, _ = scene.ray_cast(
            depsgraph, cam_pos, Vector((0, 0, -1))
        )
        if not hit_floor or (cam_pos.z - loc_floor.z) > 3.0:
            return False  # no floor below or too high above it

    # Test 3: Clear line of sight from target to camera
    direction = cam_pos - target_pos
    distance = direction.length
    if distance < 0.01:
        return False
    direction_norm = direction.normalized()

    hit, loc, _, _, obj, _ = scene.ray_cast(
        depsgraph, target_pos, direction_norm
    )
    if not hit:
        return True

    hit_distance = (loc - target_pos).length
    if hit_distance >= distance - min_clearance:
        return True

    if ignore_names and obj and obj.name in ignore_names:
        return True

    return False


def _get_scene_camera_azimuth(target_pos):
    """Compute the azimuth angle from target to the scene's own camera."""
    cam = bpy.context.scene.camera
    if cam is None:
        return None
    dx = cam.location.x - target_pos.x
    dy = cam.location.y - target_pos.y
    # atan2 gives angle; our azimuth convention: 0°=+Y, 90°=+X
    azim = math.degrees(math.atan2(dx, -dy))
    return azim % 360


def _get_scene_camera_max_z():
    """Get the scene camera's Z as a height ceiling for preview cameras."""
    cam = bpy.context.scene.camera
    if cam is None:
        return None
    return cam.location.z


def _ideal_interest_azimuth(target_pos, interest_target):
    """Azimuth that places the camera so it looks from -interest direction
    toward target (interest stays in the background of the rendered view).

    Camera convention (see _compute_cam_position): for elev=0,
        cam_xy - target_xy = radius * (sin(azim), -cos(azim)).
    For the camera to face from "opposite of interest" toward target:
        cam - target = -(interest - target) / |interest - target| * radius
    Solving:
        sin(azim) = (target.x - interest.x) / |dxy|
        cos(azim) = (interest.y - target.y) / |dxy|
        azim = atan2(target.x - interest.x, interest.y - target.y)
    """
    if interest_target is None:
        return None
    dx = target_pos.x - interest_target.x
    dy = interest_target.y - target_pos.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def find_valid_azimuths_with_elevation(
    target_pos, obj_size, radius, base_elevation_deg,
    preferred_azimuths, ignore_names=None, num_needed=4, min_separation=45.0,
    elevation_choices=None, interest_target=None,
):
    """Find diverse (azimuth, elevation) view pairs with good coverage.

    Returns list of (azimuth, elevation, coverage) tuples. Picks the elevation
    per azimuth that gives the best view coverage of scene geometry.

    If `interest_target` is provided, biases the trial order so the FIRST
    azimuth tried places the camera looking from -interest toward target
    (i.e. the interest region is in the rendered background). Offsets are
    explored from BOTH the interest direction AND the scene camera direction,
    so resulting views remain diverse but lean toward interest-revealing
    backdrops.
    """
    if elevation_choices is None:
        elevation_choices = [base_elevation_deg, base_elevation_deg + 15, base_elevation_deg - 5]

    valid = []  # list of (azimuth, elevation, coverage)
    tested = set()
    max_z = _get_scene_camera_max_z()

    def _angular_dist(a, b):
        d = abs((a % 360) - (b % 360))
        return min(d, 360 - d)

    def _is_well_separated(azim):
        return all(_angular_dist(azim, v[0]) >= min_separation for v in valid)

    def _try(azim, require_separation=True):
        azim_r = round(azim % 360, 1)
        if azim_r in tested:
            return False
        tested.add(azim_r)
        if require_separation and valid and not _is_well_separated(azim_r):
            return False
        # Try each elevation, pick the one with best coverage
        best = None
        for elev in elevation_choices:
            cam_pos = _compute_cam_position(
                target_pos, obj_size, radius, elev, azim_r, max_z=max_z
            )
            target = target_pos + Vector((0, 0, obj_size.z * 0.4))
            if not is_camera_valid(cam_pos, target, ignore_names):
                continue
            coverage = _quick_coverage(cam_pos, target, scene_cam=False)
            if best is None or coverage > best[1]:
                best = (elev, coverage)
        if best is not None and best[1] >= 0.3:
            valid.append((azim_r, best[0], best[1]))
            return True
        return False

    ideal_azim = _ideal_interest_azimuth(target_pos, interest_target)
    scene_azim = _get_scene_camera_azimuth(target_pos)

    # First view: interest direction (if available) — looks toward the
    # interesting part of the scene (interest behind target in the frame).
    if ideal_azim is not None:
        _try(ideal_azim, require_separation=False)

    # Second anchor: scene camera direction (if distinct from interest).
    if scene_azim is not None:
        _try(scene_azim, require_separation=False)

    # Offsets explored from BOTH anchors, interleaved, smaller offsets first.
    anchors = [a for a in (ideal_azim, scene_azim) if a is not None]
    if anchors and len(valid) < num_needed:
        for offset in [60, -60, 30, -30, 90, -90, 120, -120, 150, -150, 180]:
            for anchor in anchors:
                _try(anchor + offset)
                if len(valid) >= num_needed:
                    return valid

    # Fallback: preferred azimuths
    for azim in preferred_azimuths:
        _try(azim, require_separation=False)
        if len(valid) >= num_needed:
            return valid

    # Last resort: full sweep
    for azim in range(0, 360, 15):
        _try(float(azim), require_separation=False)
        if len(valid) >= num_needed:
            return valid

    return valid


def _quick_coverage(cam_pos, look_target, scene_cam=False, n_rays=6):
    """Quick coverage estimate using a small grid of rays from camera position."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Build a temporary camera frame to fire rays through
    direction = (look_target - cam_pos).normalized()
    # Frame: forward = direction, right = direction x world_up
    world_up = Vector((0, 0, 1))
    if abs(direction.dot(world_up)) > 0.95:
        world_up = Vector((0, 1, 0))
    right = direction.cross(world_up).normalized()
    up = right.cross(direction).normalized()

    # 60 degree FOV
    half_fov = math.radians(30)
    hits = 0
    total = 0
    for iy in range(n_rays):
        for ix in range(n_rays):
            u = (ix + 0.5) / n_rays * 2 - 1
            v = (iy + 0.5) / n_rays * 2 - 1
            d = (direction
                 + right * (u * math.tan(half_fov))
                 + up * (v * math.tan(half_fov))).normalized()
            hit, _, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, d)
            if hit:
                hits += 1
            total += 1
    return hits / max(1, total)


def find_valid_azimuths(target_pos, obj_size, radius, elevation_deg,
                        preferred_azimuths, ignore_names=None, num_needed=3,
                        min_separation=30.0):
    """Find azimuth angles where the camera has clear line of sight.

    Starts from the scene camera's azimuth, then searches at +120° and +240°
    for diversity. Enforces minimum angular separation between views.
    Camera Z is clamped to the scene camera's height.
    """
    valid = []
    tested = set()
    max_z = _get_scene_camera_max_z()

    def _angular_dist(a, b):
        d = abs((a % 360) - (b % 360))
        return min(d, 360 - d)

    def _is_well_separated(azim):
        return all(_angular_dist(azim, v) >= min_separation for v in valid)

    def _try(azim, require_separation=True):
        azim_r = round(azim % 360, 1)
        if azim_r in tested:
            return False
        tested.add(azim_r)
        if require_separation and valid and not _is_well_separated(azim_r):
            return False
        cam_pos = _compute_cam_position(target_pos, obj_size, radius, elevation_deg, azim_r,
                                        max_z=max_z)
        if is_camera_valid(cam_pos, target_pos + Vector((0, 0, obj_size.z * 0.4)), ignore_names):
            valid.append(azim_r)
            return True
        return False

    # First view: scene camera azimuth (no separation check needed)
    scene_azim = _get_scene_camera_azimuth(target_pos)
    if scene_azim is not None:
        _try(scene_azim, require_separation=False)
        # Try progressively wider angles, starting close to scene camera
        if len(valid) < num_needed:
            for offset in [45, -45, 90, -90, 120, -120, 30, -30, 60, -60,
                           150, -150, 180]:
                _try(scene_azim + offset)
                if len(valid) >= num_needed:
                    return valid

    # Fallback: preferred azimuths with relaxed separation
    for azim in preferred_azimuths:
        _try(azim, require_separation=False)
        if len(valid) >= num_needed:
            return valid

    # Last resort: full sweep, no separation requirement
    for azim in range(0, 360, 15):
        _try(float(azim), require_separation=False)
        if len(valid) >= num_needed:
            return valid

    return valid


def compute_view_coverage(cam_obj, n_rays: int = 12) -> float:
    """Estimate how much of the camera's view is filled with scene geometry.

    Casts a grid of rays from the camera through the image plane and counts
    how many hit any geometry. Returns ratio in [0, 1].
    A coverage < 0.3 means the camera is mostly looking at empty space.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    cam_data = cam_obj.data
    cam_matrix = cam_obj.matrix_world
    cam_pos = cam_matrix.translation

    # Compute frustum directions in camera local space
    # Camera looks down -Z, +X right, +Y up
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    if cam_data.type == "PERSP":
        half_fov_y = cam_data.angle / 2
        half_fov_x = math.atan(math.tan(half_fov_y) * aspect)
    else:
        half_fov_x = half_fov_y = 0.5  # ortho fallback

    rot = cam_matrix.to_quaternion()
    hits = 0
    total = 0
    for iy in range(n_rays):
        for ix in range(n_rays):
            # Map to normalized [-1, 1] image plane
            u = (ix + 0.5) / n_rays * 2 - 1
            v = (iy + 0.5) / n_rays * 2 - 1
            local_dir = Vector((
                u * math.tan(half_fov_x),
                v * math.tan(half_fov_y),
                -1.0,
            )).normalized()
            world_dir = rot @ local_dir
            hit, _, _, _, _, _ = scene.ray_cast(depsgraph, cam_pos, world_dir)
            if hit:
                hits += 1
            total += 1
    return hits / max(1, total)


def setup_camera(target_pos, obj_size, radius=None, elevation_deg=30.0, azimuth_deg=30.0,
                 shot_type="full"):
    """Create a camera looking at target_pos from spherical coordinates.

    shot_type controls framing:
        "full"  — standard framing showing the whole object (default)
        "wide"  — pulled back 2x to show scene context
        "mid"   — closer, focused on upper half of object
    """
    max_z = _get_scene_camera_max_z()

    # Remove only preview cameras, keep the scene's original camera
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA" and obj.name.startswith("PreviewCam"):
            bpy.data.objects.remove(obj, do_unlink=True)

    obj_diag = obj_size.length
    if radius is None or radius <= 0:
        radius = obj_diag * 3.0
    # Minimum radius: at least 2x object diagonal, but also at least 2m
    # so that wide shots of small objects actually show scene context
    radius = max(radius, obj_diag * 2.0, 2.0)

    # Adjust radius and look-target based on shot type
    if shot_type == "wide":
        radius *= 2.0
    elif shot_type == "mid":
        radius *= 0.6

    cam_pos = _compute_cam_position(target_pos, obj_size, radius, elevation_deg, azimuth_deg,
                                    max_z=max_z)

    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_obj = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = cam_pos

    # Adjust look target based on shot type
    if shot_type == "mid":
        look_target = target_pos + Vector((0, 0, obj_size.z * 0.65))  # upper body
    else:
        look_target = target_pos + Vector((0, 0, obj_size.z * 0.4))
    direction = look_target - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam_obj
    return cam_obj
