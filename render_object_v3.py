
import json
import math
import random
import sys
import warnings
from datetime import datetime
from math import atan2, degrees, sqrt
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.append(".")
from src.scoring.bbox_control import compute_v5_scores
from render_object import (
    apply_scene_scale,
    auto_place_object,
    configure_gpu,
    ensure_camera,
    parse_args,
    place_imported_object,
    prepare_camera,
    sample_pose,
    set_render_settings,
    set_sky,
    set_sky_if_no_native_lighting,
    vec,
)


# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------

def collect_object_meshes(object_name):
    """Collect all MESH objects under *object_name* (recursive hierarchy)."""
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object '{object_name}' not found in the scene.")

    def _descendants(o):
        yield o
        for child in o.children:
            yield from _descendants(child)

    meshes = [o for o in _descendants(obj) if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(
            f"No MESH objects found under '{object_name}' "
            f"(type={obj.type}, children={len(obj.children)})."
        )
    return meshes


def get_vertex_bbox(meshes):
    """Compute tight world-space AABB from actual mesh vertices (not bound_box)."""
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for m in meshes:
        mat = m.matrix_world
        for v in m.data.vertices:
            w = mat @ v.co
            min_v.x = min(min_v.x, w.x)
            min_v.y = min(min_v.y, w.y)
            min_v.z = min(min_v.z, w.z)
            max_v.x = max(max_v.x, w.x)
            max_v.y = max(max_v.y, w.y)
            max_v.z = max(max_v.z, w.z)
            found = True
    if not found:
        return None
    return min_v, max_v


def get_aabb_corners(min_v, max_v):
    """AABB min/max -> 8 world-coordinate vertices."""
    corners = []
    for x in (min_v.x, max_v.x):
        for y in (min_v.y, max_v.y):
            for z in (min_v.z, max_v.z):
                corners.append([x, y, z])
    return corners


def compute_camera_to_object_angles(cam_pos, obj_pos, object_rotation_xyz_rad):
    """Camera position expressed in object-local frame -> (azimuth_deg, elevation_deg).

    Drops camera roll. Azimuth in [0, 360), elevation in [-90, 90].
    """
    d_world = cam_pos - obj_pos
    if object_rotation_xyz_rad is None:
        d_local = d_world
    else:
        rx, ry, rz = (float(v) for v in object_rotation_xyz_rad)
        inv_rot = Euler((rx, ry, rz), "XYZ").to_matrix().transposed()
        d_local = inv_rot @ d_world
    elevation_deg = degrees(atan2(d_local.z, sqrt(d_local.x ** 2 + d_local.y ** 2)))
    azimuth_deg = degrees(atan2(d_local.y, d_local.x)) % 360
    return azimuth_deg, elevation_deg


def project_corners_to_pixels(scene, camera, corners_world, resolution):
    """Project 8 world-space AABB corners through the camera into pixel coords.

    Returns list of (x_px, y_px, depth) where x/y may extend beyond image
    bounds; depth <= 0 indicates the corner is behind the camera.
    Uses Blender's world_to_camera_view for focal length / sensor handling.
    """
    from bpy_extras.object_utils import world_to_camera_view

    res_x, res_y = (int(resolution[0]), int(resolution[1]))
    pts = []
    for corner in corners_world:
        ndc = world_to_camera_view(scene, camera, Vector(corner))
        px = float(ndc.x) * res_x
        py = (1.0 - float(ndc.y)) * res_y
        pts.append((px, py, float(ndc.z)))
    return pts


def projected_bbox_full(projected_pts):
    """From projected pixel points, return bbox [x1,y1,x2,y2] (no clamping)
    or None if all points are behind the camera (depth <= 0).
    """
    valid = [(x, y) for (x, y, z) in projected_pts if z > 0.0]
    if not valid:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_3d_metrics(cam_pos, obj_pos):
    """Camera-object geometry -> elevation, azimuth, distance."""
    d = cam_pos - obj_pos
    elevation_deg = degrees(atan2(d.z, sqrt(d.x ** 2 + d.y ** 2)))
    azimuth_deg = degrees(atan2(d.y, d.x)) % 360
    distance = d.length
    return elevation_deg, azimuth_deg, distance


def compute_object_dimensions(min_v, max_v):
    """AABB -> {width, height, depth}."""
    return {
        "width": round(float(max_v.x - min_v.x), 4),
        "height": round(float(max_v.z - min_v.z), 4),  # Z-up
        "depth": round(float(max_v.y - min_v.y), 4),
    }


# ---------------------------------------------------------------------------
# Compositor setup (mask + optional depth)
# ---------------------------------------------------------------------------

def configure_compositor(scene, mask_dir, depth_dir=None, depth_format="PNG",
                         depth_max=15.0, depth_exr_dir=None):
    """Set up compositor for Object Index mask output (+ optional depth).

    depth_exr_dir: if provided, always saves raw depth EXR here (for depth metrics).
    """
    view_layer = scene.view_layers[0]
    view_layer.use_pass_object_index = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

    # ID Mask: isolate pass_index=1
    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])

    mask_output = tree.nodes.new("CompositorNodeOutputFile")
    mask_output.base_path = str(mask_dir)
    mask_output.format.file_format = "PNG"
    mask_output.format.color_mode = "BW"
    mask_output.format.color_depth = "8"
    tree.links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])

    # Always enable depth pass for depth metrics
    view_layer.use_pass_z = True

    # Internal depth EXR (always, for depth metrics)
    depth_exr_output = None
    if depth_exr_dir is not None:
        depth_exr_output = tree.nodes.new("CompositorNodeOutputFile")
        depth_exr_output.base_path = str(depth_exr_dir)
        depth_exr_output.format.file_format = "OPEN_EXR"
        depth_exr_output.format.color_depth = "32"
        depth_exr_output.format.color_mode = "BW"
        tree.links.new(rl.outputs["Depth"], depth_exr_output.inputs[0])

    # User-facing depth output (optional, --render_depth)
    depth_output = None
    if depth_dir is not None:
        depth_output = tree.nodes.new("CompositorNodeOutputFile")
        depth_output.base_path = str(depth_dir)
        depth_output.format.color_mode = "BW"

        if depth_format == "OPEN_EXR":
            depth_output.format.file_format = "OPEN_EXR"
            depth_output.format.color_depth = "32"
            tree.links.new(rl.outputs["Depth"], depth_output.inputs[0])
        else:
            map_node = tree.nodes.new("CompositorNodeMapRange")
            map_node.inputs["From Min"].default_value = 0.0
            map_node.inputs["From Max"].default_value = depth_max
            map_node.inputs["To Min"].default_value = 1.0
            map_node.inputs["To Max"].default_value = 0.0
            if hasattr(map_node, "use_clamp"):
                map_node.use_clamp = True
            depth_output.format.file_format = "PNG"
            depth_output.format.color_depth = "16"
            tree.links.new(rl.outputs["Depth"], map_node.inputs["Value"])
            tree.links.new(map_node.outputs["Value"], depth_output.inputs[0])

    return mask_output, depth_output, depth_exr_output


# ---------------------------------------------------------------------------
# Morphological opening (numpy, no cv2/scipy dependency)
# ---------------------------------------------------------------------------

def _erode_1d(mask, radius, axis):
    """1D binary erosion along axis using shift-and-AND."""
    result = mask.copy()
    for d in range(1, radius + 1):
        fwd = np.zeros_like(mask)
        bwd = np.zeros_like(mask)
        if axis == 0:
            fwd[d:] = mask[:-d]
            bwd[:-d] = mask[d:]
        else:
            fwd[:, d:] = mask[:, :-d]
            bwd[:, :-d] = mask[:, d:]
        result &= fwd & bwd
    return result


def _dilate_1d(mask, radius, axis):
    """1D binary dilation along axis using shift-and-OR."""
    result = mask.copy()
    for d in range(1, radius + 1):
        fwd = np.zeros_like(mask)
        bwd = np.zeros_like(mask)
        if axis == 0:
            fwd[d:] = mask[:-d]
            bwd[:-d] = mask[d:]
        else:
            fwd[:, d:] = mask[:, :-d]
            bwd[:, :-d] = mask[:, d:]
        result |= fwd | bwd
    return result


def morphological_opening(mask, radius):
    """Binary opening (erosion then dilation) with separable square kernel."""
    m = mask.astype(bool)
    # Erosion: horizontal then vertical
    m = _erode_1d(m, radius, axis=1)
    m = _erode_1d(m, radius, axis=0)
    # Dilation: horizontal then vertical
    m = _dilate_1d(m, radius, axis=1)
    m = _dilate_1d(m, radius, axis=0)
    return m.astype(np.uint8)


def largest_component(mask):
    """Keep only the largest connected component (simple flood-fill)."""
    try:
        from scipy.ndimage import label as ndimage_label
        labeled, n = ndimage_label(mask)
        if n <= 1:
            return mask
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = np.argmax(sizes) + 1
        return (labeled == largest).astype(np.uint8)
    except ImportError:
        return mask  # skip if scipy unavailable


# ---------------------------------------------------------------------------
# Mask-based bbox
# ---------------------------------------------------------------------------

def _load_mask(mask_path):
    """Load mask PNG via Blender, return (h, w) uint8 array (top-down).

    Returns None if the file isn't readable (multi-worker compositor writes
    are async — caller must handle None).
    """
    try:
        mask_img = bpy.data.images.load(str(mask_path), check_existing=False)
    except RuntimeError as exc:
        print(f"  [mask] skip {Path(mask_path).name}: {exc}")
        return None
    w, h = mask_img.size
    pixels = np.array(mask_img.pixels[:])
    n_ch = mask_img.channels
    bpy.data.images.remove(mask_img)
    mask_2d = pixels.reshape(h, w, n_ch)[:, :, 0]
    mask_2d = np.flipud(mask_2d)  # Blender stores bottom-up
    return (mask_2d > 0.5).astype(np.uint8)


def compute_mask_bbox(mask_path, resolution, opening_fraction=0.03):
    """Read mask PNG, apply morphological opening.

    Returns (bbox_raw, bbox_tight, mask_stats) or (None, None, None).
    mask_stats: dict with mask_pixels, mask_pixels_raw, centroid.
    """
    mask_bin = _load_mask(mask_path)
    if mask_bin is None:
        return None, None, None

    # Raw bbox
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None, None, None
    bbox_raw = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    mask_pixels_raw = int(len(xs))

    # Morphological opening to remove thin protrusions
    short_dim = min(resolution)
    radius = max(2, int(short_dim * opening_fraction) // 2)
    mask_opened = morphological_opening(mask_bin, radius)
    mask_opened = largest_component(mask_opened)

    ys, xs = np.where(mask_opened > 0)
    if len(xs) == 0:
        # opening removed everything, fall back to raw
        ys_raw, xs_raw = np.where(mask_bin > 0)
        mask_stats = {
            "mask_pixels": mask_pixels_raw,
            "mask_pixels_raw": mask_pixels_raw,
            "centroid": [float(xs_raw.mean()), float(ys_raw.mean())],
        }
        return bbox_raw, bbox_raw, mask_stats

    bbox_tight = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    mask_stats = {
        "mask_pixels": int(len(xs)),
        "mask_pixels_raw": mask_pixels_raw,
        "centroid": [float(xs.mean()), float(ys.mean())],
    }
    return bbox_raw, bbox_tight, mask_stats


# ---------------------------------------------------------------------------
# Frame metrics (mask + composition + drone)
# ---------------------------------------------------------------------------

def compute_frame_metrics(bbox_tight, mask_stats, resolution,
                          distance, dimensions, focal_length, sensor_width):
    """Compute all per-frame metrics from mask/bbox/3D info."""
    res_x, res_y = resolution
    bbox_w = bbox_tight[2] - bbox_tight[0]
    bbox_h = bbox_tight[3] - bbox_tight[1]
    bbox_area = max(1, bbox_w * bbox_h)
    img_area = res_x * res_y

    cx_obj, cy_obj = mask_stats["centroid"]
    max_dim = max(dimensions.values()) if dimensions else 1.0

    x1, y1, x2, y2 = bbox_tight

    return {
        # Mask-based
        "visibility_ratio": round(mask_stats["mask_pixels"] / bbox_area, 4),
        "truncation": bool(x1 <= 0 or y1 <= 0 or x2 >= res_x - 1 or y2 >= res_y - 1),
        "occupancy_ratio": round(bbox_area / img_area, 4),
        "mask_pixel_ratio": round(mask_stats["mask_pixels"] / img_area, 4),
        # Margins (compatible with bbox_control.py)
        "bbox_margin_top": round(max(0.0, y1 / res_y), 4),
        "bbox_margin_bottom": round(max(0.0, (res_y - y2) / res_y), 4),
        "bbox_margin_left": round(max(0.0, x1 / res_x), 4),
        "bbox_margin_right": round(max(0.0, (res_x - x2) / res_x), 4),
        # Composition
        "object_center_pixel": [round(cx_obj, 1), round(cy_obj, 1)],
        "center_offset_normalized": round(sqrt(
            ((cx_obj - res_x / 2) / (res_x / 2)) ** 2
            + ((cy_obj - res_y / 2) / (res_y / 2)) ** 2
        ), 4),
        "bbox_aspect_ratio": round(bbox_w / max(1, bbox_h), 4),
        "bbox_centroid_offset": round(min(1.0, sqrt(
            ((cx_obj / res_x - 0.5) ** 2 + (cy_obj / res_y - 0.5) ** 2)
        ) / (0.5 * sqrt(2))), 4),
        # Drone specific
        "gsd": round(distance * sensor_width / (focal_length * res_x), 6),
        "object_angular_size_deg": round(degrees(
            2 * atan2(max_dim, 2 * distance)
        ), 2),
    }


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(depth_path, mask_path, resolution):
    """Read depth EXR, compute stats within mask region.

    Returns dict or None if depth file not found.
    """
    depth_path = Path(depth_path)
    if not depth_path.exists():
        return None

    mask_bin = _load_mask(mask_path)
    if mask_bin is None:
        return None

    # Load depth via Blender. Multi-worker EXR writes are async — the file may
    # exist but still be flushing when we try to read it. Treat any load
    # failure as "depth metrics unavailable" rather than crashing the worker
    # (depth metrics are optional and not part of the v5 score schema).
    try:
        depth_img = bpy.data.images.load(str(depth_path), check_existing=False)
    except RuntimeError as exc:
        print(f"  [depth] skip {depth_path.name}: {exc}")
        return None
    w, h = depth_img.size
    pixels = np.array(depth_img.pixels[:])
    n_ch = depth_img.channels
    bpy.data.images.remove(depth_img)

    depth_2d = pixels.reshape(h, w, n_ch)[:, :, 0]
    depth_2d = np.flipud(depth_2d)

    # Depth within mask
    mask_pixels = mask_bin > 0
    if not mask_pixels.any():
        return None

    depths_in_mask = depth_2d[mask_pixels]
    # Filter out infinite/invalid depth values
    valid = depths_in_mask[np.isfinite(depths_in_mask) & (depths_in_mask > 0) & (depths_in_mask < 1e9)]
    if len(valid) == 0:
        return None

    # Depth at object center (mask centroid)
    ys, xs = np.where(mask_pixels)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    cy = min(max(cy, 0), h - 1)
    cx = min(max(cx, 0), w - 1)
    depth_center = float(depth_2d[cy, cx])

    return {
        "depth_at_center": round(depth_center, 3),
        "depth_min": round(float(valid.min()), 3),
        "depth_max": round(float(valid.max()), 3),
        "depth_mean": round(float(valid.mean()), 3),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    args = parse_args(argv)
    random.seed(args.seed)
    blend_path = Path(args.input_scene)
    scene_stem = blend_path.stem

    # --- output dirs ---
    if args.output_run_dir:
        run_dir = Path(args.output_run_dir)
        timestamp = (
            run_dir.name.split("_")[-1]
            if "_" in run_dir.name
            else datetime.now().strftime("%y%m%d_%H%M%S")
        )
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        folder_name = f"{scene_stem}_{timestamp}"
        if args.run_name:
            folder_name = f"{scene_stem}_{args.run_name}_{timestamp}"
        run_dir = Path(args.output_dir) / folder_name
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- scene / camera / render ---
    from src.scenes.scene import open_scene

    scene = open_scene(str(blend_path))
    if getattr(args, "scene_scale", 1.0) != 1.0:
        apply_scene_scale(scene, args.scene_scale)
    camera = ensure_camera(scene)
    prepare_camera(camera, args.keep_camera_constraints)
    if args.focal_length is not None:
        camera.data.lens = args.focal_length
    if args.sensor_width is not None:
        camera.data.sensor_width = args.sensor_width
    if args.sensor_height is not None:
        camera.data.sensor_height = args.sensor_height
    if args.sky_strength > 0:
        set_sky_if_no_native_lighting(args.sky_strength)
    device_info = configure_gpu(scene, args.disable_gpu, args.gpu_backend, args.gpu_devices)
    set_render_settings(scene, args.resolution, args)

    # ------------------------------------------------------------------
    # Phase 2: Object position + 3D bbox + mask setup
    # ------------------------------------------------------------------
    has_3d_bbox = False
    min_v = max_v = None
    corners_3d = None
    dimensions = None
    object_name = args.object_name
    object_position_source = None
    mask_dir = None
    mask_output_node = None
    depth_output_node = None
    depth_exr_output_node = None
    depth_exr_dir = None

    if args.object_position is not None and args.input_object:
        # Import external object at specified position with rotation/scale
        obj_pos, placed_root = place_imported_object(
            scene, args.input_object, args.object_position,
            rotation_z_deg=args.rotation_z_deg, scale=args.scale,
            rotation_xyz_rad=getattr(args, "object_rotation_xyz_rad", None),
        )
        object_position_source = "import_at_position"
        object_name = placed_root

        meshes = collect_object_meshes(placed_root)
        bbox_result = get_vertex_bbox(meshes)
        if bbox_result:
            min_v, max_v = bbox_result
            aabb_center = (min_v + max_v) / 2
            corners_3d = get_aabb_corners(min_v, max_v)
            dimensions = compute_object_dimensions(min_v, max_v)
            has_3d_bbox = True
            print(f"3D bbox (vertex): min={vec(min_v)}, max={vec(max_v)}, center={vec(aabb_center)}")
            print(f"  dimensions: {dimensions}")
            print(f"  meshes: {len(meshes)} objects under '{placed_root}'")

        if has_3d_bbox and args.use_aabb_center:
            obj_pos = aabb_center
            object_position_source = "import_at_position_aabb_center"
            print(f"  using AABB center as orbit center: {vec(aabb_center)}")

        for m in meshes:
            m.pass_index = 1

        mask_dir = run_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        depth_exr_dir = run_dir / ".depth_exr"
        depth_exr_dir.mkdir(parents=True, exist_ok=True)
        depth_dir = None
        if args.render_depth:
            depth_dir = run_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
        mask_output_node, depth_output_node, depth_exr_output_node = configure_compositor(
            scene, mask_dir,
            depth_dir=depth_dir,
            depth_format=args.depth_format,
            depth_max=args.depth_max,
            depth_exr_dir=depth_exr_dir,
        )

    elif args.object_name:
        meshes = collect_object_meshes(args.object_name)

        # Vertex-based tight AABB (not bound_box)
        bbox_result = get_vertex_bbox(meshes)
        if bbox_result:
            min_v, max_v = bbox_result
            aabb_center = (min_v + max_v) / 2
            corners_3d = get_aabb_corners(min_v, max_v)
            dimensions = compute_object_dimensions(min_v, max_v)
            has_3d_bbox = True
            print(f"3D bbox (vertex): min={vec(min_v)}, max={vec(max_v)}, center={vec(aabb_center)}")
            print(f"  dimensions: {dimensions}")
            print(f"  meshes: {len(meshes)} objects under '{args.object_name}'")

        # Set pass_index for Object Index mask
        for m in meshes:
            m.pass_index = 1

        # Set up compositor (mask + depth EXR for metrics + optional user depth)
        mask_dir = run_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        depth_exr_dir = run_dir / ".depth_exr"
        depth_exr_dir.mkdir(parents=True, exist_ok=True)
        depth_dir = None
        if args.render_depth:
            depth_dir = run_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
        mask_output_node, depth_output_node, depth_exr_output_node = configure_compositor(
            scene, mask_dir,
            depth_dir=depth_dir,
            depth_format=args.depth_format,
            depth_max=args.depth_max,
            depth_exr_dir=depth_exr_dir,
        )

        if args.object_position is not None:
            obj_pos = Vector(args.object_position)
            object_position_source = "object_position"
        elif has_3d_bbox:
            obj_pos = aabb_center
            object_position_source = "object_name_aabb_center"
        else:
            raise RuntimeError(
                f"Object '{args.object_name}' found but could not compute bounding box."
            )

    elif args.auto_place_object:
        if not args.input_object:
            raise RuntimeError("--input_object is required when --auto_place_object is set.")
        obj_pos = auto_place_object(scene, args.input_object)
        object_position_source = "auto_place_object"
        if args.render_depth:
            from render_object import configure_depth_output
            depth_dir = run_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_output_node = configure_depth_output(scene, depth_dir, args.depth_format, args.depth_max)
        warnings.warn(
            "auto_place_object: mask/3D bbox not available (no --object_name). "
            "Mask and 3D annotation fields will be omitted.",
            stacklevel=1,
        )

    elif args.object_position is not None:
        obj_pos = Vector(args.object_position)
        object_position_source = "object_position"
        if args.render_depth:
            from render_object import configure_depth_output
            depth_dir = run_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_output_node = configure_depth_output(scene, depth_dir, args.depth_format, args.depth_max)
        warnings.warn(
            "object_position only: mask/3D bbox not available (no --object_name). "
            "Mask and 3D annotation fields will be omitted.",
            stacklevel=1,
        )

    else:
        raise RuntimeError("Provide one of --object_position, --object_name, or --auto_place_object.")

    # ------------------------------------------------------------------
    # Phase 3: Render loop
    # ------------------------------------------------------------------
    all_poses = [
        (idx, sample_pose(obj_pos, args.camera_radius_range, args.camera_direction_offsets, args.hemisphere))
        for idx in range(args.num_images)
    ]
    my_poses = [(idx, pose) for idx, pose in all_poses if idx % args.num_workers == args.worker_index]
    print(f"Worker {args.worker_index}/{args.num_workers}: rendering {len(my_poses)}/{args.num_images} images")

    annotations = []
    if args.num_workers > 1:
        ann_path = run_dir / f"annotations_worker{args.worker_index}.json"
    else:
        ann_path = run_dir / "annotations.json"

    for idx, pose in my_poses:
        scene.frame_set(idx)
        if mask_output_node is not None:
            mask_output_node.file_slots[0].path = "mask_"
        if depth_output_node is not None:
            depth_output_node.file_slots[0].path = "img_"
        if depth_exr_output_node is not None:
            depth_exr_output_node.file_slots[0].path = "depth_"

        img_ext = "jpg" if getattr(args, "image_format", "JPEG") == "JPEG" else "png"
        camera.matrix_world = Matrix.Translation(pose["cam_pos"]) @ pose["rot_matrix"].to_4x4()
        scene.render.filepath = str(images_dir / f"img_{idx:04d}.{img_ext}")
        bpy.ops.render.render(write_still=True)

        entry = {
            "image": f"images/img_{idx:04d}.{img_ext}",
            "camera_position": vec(pose["cam_pos"]),
            "radius": pose["radius"],
            "object_position": vec(obj_pos),
            "base_forward": vec(pose["base_forward"]),
            "base_up": vec(pose["base_up"]),
            "offsets_deg": pose["offsets"],
            "final_forward": vec(pose["final_forward"]),
            "final_up": vec(pose["final_up"]),
        }
        if args.render_depth:
            depth_ext = "exr" if args.depth_format == "OPEN_EXR" else "png"
            entry["depth"] = f"depth/img_{idx:04d}.{depth_ext}"

        # --- 3D metrics ---
        dist = None
        if has_3d_bbox:
            elev, azi, dist = compute_3d_metrics(pose["cam_pos"], obj_pos)
            # camera_pitch: angle between final_forward and horizontal plane
            fwd = pose["final_forward"]
            pitch_deg = degrees(atan2(-fwd.z, sqrt(fwd.x ** 2 + fwd.y ** 2)))
            entry["elevation_deg"] = round(elev, 2)
            entry["azimuth_deg"] = round(azi, 2)
            entry["camera_subject_distance"] = round(dist, 3)
            entry["camera_pitch_deg"] = round(pitch_deg, 2)

        # --- Object orientation (coordinate frame) ---
        if args.rotation_z_deg is not None:
            from math import radians, cos, sin
            rot_z = radians(args.rotation_z_deg)
            obj_fwd = Vector((-sin(rot_z), cos(rot_z), 0.0))
            obj_up = Vector((0.0, 0.0, 1.0))
            entry["object_rotation_z_deg"] = args.rotation_z_deg
            entry["object_forward"] = vec(obj_fwd)
            entry["object_up"] = vec(obj_up)

            # Camera-Object relative rotation (6D)
            cam_fwd = pose["final_forward"]
            cam_up = pose["final_up"]
            cam_right = cam_fwd.cross(cam_up)
            obj_fwd_in_cam = Vector((
                cam_right.dot(obj_fwd),
                cam_fwd.dot(obj_fwd),
                cam_up.dot(obj_fwd),
            ))
            obj_up_in_cam = Vector((
                cam_right.dot(obj_up),
                cam_fwd.dot(obj_up),
                cam_up.dot(obj_up),
            ))
            entry["camera_to_object_6d"] = vec(obj_fwd_in_cam) + vec(obj_up_in_cam)

        # --- Mask-based tight bbox + all frame metrics ---
        if mask_dir is not None:
            mask_path = mask_dir / f"mask_{idx:04d}.png"
            entry["mask"] = f"masks/mask_{idx:04d}.png"

            if mask_path.exists():
                bbox_raw, bbox_tight, mask_stats = compute_mask_bbox(
                    mask_path, args.resolution, opening_fraction=0.03,
                )
                if bbox_tight is not None and mask_stats is not None:
                    entry["bbox_2d"] = bbox_tight
                    entry["bbox_2d_raw"] = bbox_raw
                    # Backward compat
                    entry["detections"] = [
                        {
                            "label": "mask_tight",
                            "score": 1.0,
                            "bbox_xyxy": bbox_tight,
                        }
                    ]

                    # Frame metrics (mask + composition + drone)
                    if dist is not None and dimensions is not None:
                        metrics = compute_frame_metrics(
                            bbox_tight, mask_stats, args.resolution,
                            dist, dimensions,
                            camera.data.lens, camera.data.sensor_width,
                        )
                        entry.update(metrics)

                    # Depth metrics
                    if depth_exr_dir is not None:
                        depth_exr_path = depth_exr_dir / f"depth_{idx:04d}.exr"
                        depth_metrics = compute_depth_metrics(
                            depth_exr_path, mask_path, args.resolution,
                        )
                        if depth_metrics is not None:
                            entry.update(depth_metrics)

        # --- v5 score fields (integer schema; full-projected bbox allows off-image centers) ---
        if has_3d_bbox and corners_3d is not None:
            projected = project_corners_to_pixels(
                scene, camera, corners_3d, args.resolution,
            )
            bbox_full_px = projected_bbox_full(projected)
            obj_rot_xyz = getattr(args, "object_rotation_xyz_rad", None)
            if obj_rot_xyz is None:
                obj_rot_xyz = (0.0, 0.0, math.radians(args.rotation_z_deg or 0.0))
            v5_az, v5_el = compute_camera_to_object_angles(
                pose["cam_pos"], obj_pos, obj_rot_xyz,
            )
            v5 = compute_v5_scores(
                image_width=int(args.resolution[0]),
                image_height=int(args.resolution[1]),
                bbox_full=bbox_full_px,
                azimuth_deg=v5_az,
                elevation_deg=v5_el,
            )
            for k, val in v5.items():
                entry[f"score_{k}"] = int(val)
            if bbox_full_px is not None:
                entry["bbox_2d_full_projected"] = [
                    round(bbox_full_px[0], 2),
                    round(bbox_full_px[1], 2),
                    round(bbox_full_px[2], 2),
                    round(bbox_full_px[3], 2),
                ]

        annotations.append(entry)

        # Incremental save — prevents annotation loss on crash
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

    # ------------------------------------------------------------------
    # Phase 4: run_info.json
    # ------------------------------------------------------------------
    run_info = {
        "input_scene": str(blend_path),
        "output_dir": str(run_dir),
        "scene_stem": scene_stem,
        "object_name": object_name,
        "run_name": args.run_name,
        "created_at": timestamp,
        "input_object": args.input_object,
        "scale": args.scale,
        "rotation_z_deg": args.rotation_z_deg,
        "options": {
            "sky_strength": args.sky_strength,
            "object_position": vec(obj_pos),
            "object_position_source": object_position_source,
            "hemisphere": args.hemisphere,
            "camera_radius_range": args.camera_radius_range,
            "camera_direction_offsets": args.camera_direction_offsets,
            "num_images": args.num_images,
            "seed": args.seed,
            "focal_length": args.focal_length,
            "sensor_width": args.sensor_width,
            "sensor_height": args.sensor_height,
            "resolution": args.resolution,
            "render_device": device_info["device"],
            "render_devices": device_info["devices"],
            "gpu_backend": args.gpu_backend,
            "gpu_devices": args.gpu_devices,
            "samples": args.samples,
            "adaptive_sampling": args.adaptive_sampling,
            "adaptive_threshold": args.adaptive_threshold,
            "max_bounces": args.max_bounces,
            "diffuse_bounces": args.diffuse_bounces,
            "glossy_bounces": args.glossy_bounces,
            "transmission_bounces": args.transmission_bounces,
            "volume_bounces": args.volume_bounces,
            "transparent_max_bounces": args.transparent_max_bounces,
            "persistent_data": args.persistent_data,
            "blender_threads": args.blender_threads,
            "render_depth": args.render_depth,
            "depth_format": args.depth_format,
            "depth_max": args.depth_max,
        },
    }

    if has_3d_bbox:
        run_info["object_3d"] = {
            "bbox_3d_corners": corners_3d,
            "bbox_3d_min": vec(min_v),
            "bbox_3d_max": vec(max_v),
            "dimensions": dimensions,
        }

    # ------------------------------------------------------------------
    # Phase 5: Save (run_info + final annotations already saved incrementally)
    # ------------------------------------------------------------------
    if args.num_workers > 1:
        if args.worker_index == 0:
            with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)
        print(f"Worker {args.worker_index}: saved {len(annotations)} images to {ann_path}")
    else:
        with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        print(f"Saved {len(annotations)} images to {run_dir}")

    # Cleanup internal depth EXR files (metrics already extracted)
    if depth_exr_dir is not None and depth_exr_dir.exists():
        import shutil
        shutil.rmtree(depth_exr_dir, ignore_errors=True)


if __name__ == "__main__":
    raw_argv = getattr(bpy.app, "argv", None)
    if raw_argv is None:
        raw_argv = sys.argv
    raw_argv = list(raw_argv)
    if "--" not in raw_argv and raw_argv and raw_argv[0].endswith(".py"):
        raw_argv = raw_argv[1:]
    main(raw_argv)
