# test_auto_place.py
# Headless-friendly test:
# 1) load a scene (.blend)
# 2) import an object
# 3) find a suitable flat placement (floor/road) via ray-casting
# 4) render a few images around the object
#
# Run:
#   blender -b -P test_auto_place.py

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

# -------------------------
# HARD-CODED PATHS (as requested)
# -------------------------
# SCENE_PATH = Path("assets/scenes/attic-demo-ccby.blend")
SCENE_PATH = Path("assets/scenes/medieval-christmas-scene")
# SCENE_PATH = Path("assets/scenes/road-through-mountains-scene.blend")
# OBJECT_PATH = Path("assets/objects/rp_posedplus_00068_18_100k")
OBJECT_PATH = Path("assets/objects/standing-cool-ba_0fdffc77-d514-49c3-b84f-a38513031536")
# OBJECT_PATH = Path("assets/objects/snow-man_f667c23f-d220-4faf-91bf-7bd339148bfb")
# OBJECT_PATH = Path("assets/objects/luke_4bea31a4-e972-4bb9-a62b-4d1b9093d66e")

OUT_DIR = Path("outputs/test_auto_place")

# -------------------------
# Tuning
# -------------------------
PICK_EXTS_SCENE = [".blend"]
PICK_EXTS_OBJ = [".glb", ".gltf", ".fbx", ".obj", ".blend"]

MAX_SLOPE_DEG = 12.0
HEIGHT_TOL = 0.08
FOOTPRINT_MARGIN = 0.15
GRID_STEP_BASE = 1.0
CELL_SIZE = 1.0
MAX_GRID_SAMPLES = 20000
SAMPLES_PER_SIDE = 3
CAM_VIEWS = 4

# -------------------------
# Helpers
# -------------------------
REPO_ROOT = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def pick_first_file(path: Path, exts: list[str]) -> Path | None:
    path = resolve_path(path)
    if path.is_file():
        return path
    if not path.exists():
        return None
    for ext in exts:
        for cand in sorted(path.rglob(f"*{ext}")):
            return cand
    return None


def open_scene(scene_file: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(scene_file))


def import_object(obj_file: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    ext = obj_file.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(obj_file))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(obj_file))
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(obj_file))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(obj_file), link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported object file: {obj_file}")
    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    return imported


def parent_imported_objects(imported: list[bpy.types.Object]) -> bpy.types.Object:
    root = bpy.data.objects.new("AutoPlacedRoot", None)
    bpy.context.scene.collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent in imported_set:
            continue
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    return root


def get_world_bbox(objs: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in objs:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, world.x)
            min_v.y = min(min_v.y, world.y)
            min_v.z = min(min_v.z, world.z)
            max_v.x = max(max_v.x, world.x)
            max_v.y = max(max_v.y, world.y)
            max_v.z = max(max_v.z, world.z)
            found = True
    if not found:
        return None
    return min_v, max_v


def move_group_bottom_center_to_origin(root: bpy.types.Object, imported: list[bpy.types.Object]) -> None:
    bbox = get_world_bbox(imported)
    if bbox is None:
        return
    min_v, max_v = bbox
    bottom_center = Vector(((min_v.x + max_v.x) * 0.5, (min_v.y + max_v.y) * 0.5, min_v.z))
    root.location = root.location - bottom_center


def set_hidden(objs: list[bpy.types.Object], hidden: bool) -> list[tuple[bpy.types.Object, bool, bool]]:
    prev = []
    for obj in objs:
        prev.append((obj, bool(obj.hide_viewport), bool(obj.hide_render)))
        try:
            obj.hide_set(hidden)
        except Exception:
            obj.hide_viewport = hidden
        obj.hide_render = hidden
    return prev


def restore_hidden(prev: list[tuple[bpy.types.Object, bool, bool]]) -> None:
    for obj, hv, hr in prev:
        try:
            obj.hide_set(hv)
        except Exception:
            obj.hide_viewport = hv
        obj.hide_render = hr


def iter_grid(x_min: float, x_max: float, y_min: float, y_max: float, step: float):
    if step <= 0:
        return
    nx = max(1, int(math.floor((x_max - x_min) / step)) + 1)
    ny = max(1, int(math.floor((y_max - y_min) / step)) + 1)
    for i in range(nx):
        x = x_min + i * step
        for j in range(ny):
            y = y_min + j * step
            yield x, y


def compute_grid_step(bounds_min: Vector, bounds_max: Vector, desired_step: float) -> float:
    width = max(0.01, bounds_max.x - bounds_min.x)
    depth = max(0.01, bounds_max.y - bounds_min.y)
    area = width * depth
    min_step = math.sqrt(area / MAX_GRID_SAMPLES)
    return max(desired_step, min_step)


def area_is_flat(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    center: Vector,
    obj_size: Vector,
    z_top: float,
    slope_cos: float,
) -> bool:
    half_x = obj_size.x * 0.5 + FOOTPRINT_MARGIN
    half_y = obj_size.y * 0.5 + FOOTPRINT_MARGIN
    heights = []
    if SAMPLES_PER_SIDE <= 1:
        samples = [center]
    else:
        samples = []
        for i in range(SAMPLES_PER_SIDE):
            for j in range(SAMPLES_PER_SIDE):
                tx = -half_x + (2 * half_x) * (i / (SAMPLES_PER_SIDE - 1))
                ty = -half_y + (2 * half_y) * (j / (SAMPLES_PER_SIDE - 1))
                samples.append(Vector((center.x + tx, center.y + ty, center.z)))
    for sample in samples:
        origin = Vector((sample.x, sample.y, z_top))
        direction = Vector((0.0, 0.0, -1.0))
        hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or obj is None:
            return False
        if normal.z < slope_cos:
            return False
        heights.append(loc.z)
    if not heights:
        return False
    if max(heights) - min(heights) > HEIGHT_TOL:
        return False
    return True


def find_best_flat_location(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    bounds_min: Vector,
    bounds_max: Vector,
    obj_size: Vector,
    ignore_names: set[str],
) -> tuple[Vector, Vector] | None:
    slope_cos = math.cos(math.radians(MAX_SLOPE_DEG))
    obj_radius = max(obj_size.x, obj_size.y) * 0.5

    x_min = bounds_min.x + obj_radius
    x_max = bounds_max.x - obj_radius
    y_min = bounds_min.y + obj_radius
    y_max = bounds_max.y - obj_radius
    if x_min >= x_max or y_min >= y_max:
        x_min, x_max = bounds_min.x, bounds_max.x
        y_min, y_max = bounds_min.y, bounds_max.y

    step = compute_grid_step(bounds_min, bounds_max, max(GRID_STEP_BASE, obj_radius * 0.5))
    z_top = bounds_max.z + max(5.0, obj_size.z * 2.0)

    cells: dict[tuple[int, int], dict[str, object]] = {}
    for x, y in iter_grid(x_min, x_max, y_min, y_max, step):
        origin = Vector((x, y, z_top))
        direction = Vector((0.0, 0.0, -1.0))
        hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or obj is None:
            continue
        if obj.name in ignore_names:
            continue
        if normal.z < slope_cos:
            continue
        ix = int(math.floor((loc.x - bounds_min.x) / CELL_SIZE))
        iy = int(math.floor((loc.y - bounds_min.y) / CELL_SIZE))
        key = (ix, iy)
        cell = cells.get(key)
        if cell is None:
            cell = {"count": 0, "sum_loc": Vector((0.0, 0.0, 0.0)), "sum_n": Vector((0.0, 0.0, 0.0))}
            cells[key] = cell
        cell["count"] += 1
        cell["sum_loc"] += loc
        cell["sum_n"] += normal

    if not cells:
        return None

    ranked = sorted(cells.values(), key=lambda c: c["count"], reverse=True)
    for cell in ranked:
        count = cell["count"]
        if count <= 0:
            continue
        center = cell["sum_loc"] / count
        normal = cell["sum_n"].normalized()
        if area_is_flat(scene, depsgraph, center, obj_size, z_top, slope_cos):
            return center, normal

    return None


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    if direction.length == 0:
        return
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()


def ensure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    if scene.camera is not None:
        return scene.camera
    cam_data = bpy.data.cameras.new("AutoCamera")
    cam_obj = bpy.data.objects.new("AutoCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def configure_fast_render(scene: bpy.types.Scene) -> None:
    # Workbench is the fastest and ignores heavy materials/textures.
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
    except Exception:
        return
    scene.render.use_simplify = True
    if hasattr(scene.render, "simplify_subdivision"):
        scene.render.simplify_subdivision = 0
    if hasattr(scene.render, "simplify_child_particles"):
        scene.render.simplify_child_particles = 0
    if hasattr(scene.render, "simplify_volumes"):
        scene.render.simplify_volumes = 0
    if hasattr(scene.render, "simplify_shadow_samples"):
        scene.render.simplify_shadow_samples = 1
    scene.render.resolution_percentage = 50
    try:
        shading = scene.display.shading
        shading.light = "FLAT"
        shading.color_type = "SINGLE"
        shading.single_color = (0.8, 0.8, 0.8)
        shading.show_shadows = False
        shading.show_cavity = False
    except Exception:
        pass


def render_orbit(scene: bpy.types.Scene, target: Vector, obj_size: Vector) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_fast_render(scene)
    cam = ensure_camera(scene)
    radius = max(obj_size.x, obj_size.y) * 2.5 + 2.0
    height = max(1.5, obj_size.z * 1.5 + 1.5)
    for i in range(CAM_VIEWS):
        angle = (2 * math.pi / CAM_VIEWS) * i
        cam.location = Vector((
            target.x + radius * math.cos(angle),
            target.y + radius * math.sin(angle),
            target.z + height,
        ))
        look_at(cam, target + Vector((0.0, 0.0, obj_size.z * 0.5)))
        scene.render.filepath = str((OUT_DIR / f"render_{i:02d}.png").resolve())
        bpy.ops.render.render(write_still=True)


# -------------------------
# Main
# -------------------------
scene_file = pick_first_file(SCENE_PATH, PICK_EXTS_SCENE)
obj_file = pick_first_file(OBJECT_PATH, PICK_EXTS_OBJ)

if scene_file is None:
    raise FileNotFoundError(f"Scene not found: {SCENE_PATH}")
if obj_file is None:
    raise FileNotFoundError(f"Object not found: {OBJECT_PATH}")

log(f"Scene: {scene_file}")
log(f"Object: {obj_file}")

open_scene(scene_file)
scene = bpy.context.scene

imported_objs = import_object(obj_file)
if not imported_objs:
    raise RuntimeError("No objects were imported.")

root = parent_imported_objects(imported_objs)
move_group_bottom_center_to_origin(root, imported_objs)
bpy.context.view_layer.update()

bbox = get_world_bbox(imported_objs)
if bbox is None:
    raise RuntimeError("Could not compute bounding box for imported objects.")
obj_min, obj_max = bbox
obj_size = obj_max - obj_min
log(f"Object size (m): {obj_size.x:.3f} x {obj_size.y:.3f} x {obj_size.z:.3f}")

# Temporarily hide imported objects to avoid ray-cast self hits
hidden_state = set_hidden(imported_objs, True)
try:
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene_meshes = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj.name not in {o.name for o in imported_objs}
    ]
    bounds = get_world_bbox(scene_meshes)
    if bounds is None:
        raise RuntimeError("No mesh geometry found in the scene.")
    bounds_min, bounds_max = bounds
    scene_size = bounds_max - bounds_min
    log(
        "Scene bounds (m): "
        f"min=({bounds_min.x:.2f}, {bounds_min.y:.2f}, {bounds_min.z:.2f}) "
        f"max=({bounds_max.x:.2f}, {bounds_max.y:.2f}, {bounds_max.z:.2f}) "
        f"size=({scene_size.x:.2f}, {scene_size.y:.2f}, {scene_size.z:.2f})"
    )
    found = find_best_flat_location(
        scene,
        depsgraph,
        bounds_min,
        bounds_max,
        obj_size,
        ignore_names={o.name for o in imported_objs},
    )
finally:
    restore_hidden(hidden_state)
    bpy.context.view_layer.update()

if found is None:
    log("No suitable flat location found. Falling back to scene center.")
    bounds_min, bounds_max = get_world_bbox([obj for obj in scene.objects if obj.type == "MESH"]) or (
        Vector((0.0, 0.0, 0.0)),
        Vector((0.0, 0.0, 0.0)),
    )
    target_loc = Vector(((bounds_min.x + bounds_max.x) * 0.5, (bounds_min.y + bounds_max.y) * 0.5, bounds_max.z))
    target_normal = Vector((0.0, 0.0, 1.0))
else:
    target_loc, target_normal = found

# Align root to surface
up = Vector((0.0, 0.0, 1.0))
rot = up.rotation_difference(target_normal)
root.rotation_euler = rot.to_euler()
root.location = target_loc + target_normal * 0.02

bpy.context.view_layer.update()

log(f"Placed at: {root.location}, normal: {target_normal}")
render_orbit(scene, root.location, obj_size)
log(f"Renders saved to: {OUT_DIR.resolve()}")

log("=== Summary ===")
log(f"Scene file: {scene_file}")
log(f"Object file: {obj_file}")
log(f"Object size (m): {obj_size.x:.3f} x {obj_size.y:.3f} x {obj_size.z:.3f}")
if found is None:
    log("Placement: fallback to scene center")
else:
    log("Placement: flat surface candidate")
log(f"Placement location: ({root.location.x:.3f}, {root.location.y:.3f}, {root.location.z:.3f})")
log(f"Placement normal: ({target_normal.x:.3f}, {target_normal.y:.3f}, {target_normal.z:.3f})")
log(f"Render views: {CAM_VIEWS}")
log(f"Render output: {OUT_DIR.resolve()}")
