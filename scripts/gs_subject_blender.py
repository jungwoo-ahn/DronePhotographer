"""The subject half of a 3DGS composite, inside Blender: dump its geometry, or serve renders.

A 3DGS placement (`data/gs_scenes/composite_v5/<scene>/jobs.json`, built by the 3DGS session's
`_tools/composite/prepare_jobs.py`) lives in a local metric frame L: origin at the subject's feet,
z up, +x toward the placement's preview camera. The subject is imported exactly as that
pipeline's `blender_subjects.py` imports it (shared `render_object` importer, bottom-centre to
the origin, turned by `rotation_z_deg`), stands on a shadow-catcher plane at z = 0 and is lit
only by the splat's own 360-degree environment map. Both modes below build that scene.

  dump   -> the subject's vertices in L plus the v7 camera frame bounds, as the npz
            `src.goal_authoring.pose_solver.PlacementGeometry` loads. It scores any camera pose
            with the stage-3 scorer that labelled the training data, so a 3DGS goal gets exactly
            the kind of shot profile the policies were trained on.
  serve  -> a persistent render loop: RGBA (straight alpha; the shadow arrives as black with
            partial alpha) plus a Z pass in metres, for `scripts/gs_render_server.py` to
            composite over the splat. File IPC, the protocol of `blender_render_server.py`.

  blender -b -P scripts/gs_subject_blender.py -- dump  SPEC.json OUT.npz
  blender -b -P scripts/gs_subject_blender.py -- serve SPEC.json COMM_DIR
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, SHARED)
import render_object as ro  # noqa: E402  (the training data's own importer)

LENS_MM, SENSOR_W, SENSOR_H = 24.0, 12.8, 9.6          # the v7 camera
RES = (1024, 768)


def build_scene(spec: dict, *, lit: bool) -> tuple[bpy.types.Object, list]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    objs = ro.import_object(ro.pick_first_file(Path(spec["blend_dir"]), ro.PICK_EXTS_OBJ))
    root = ro.parent_imported_objects(objs)
    ro.move_group_bottom_center_to_origin(root, objs)
    root.rotation_euler = (0.0, 0.0, math.radians(spec["rotation_z_deg"]))
    root.location = (0.0, 0.0, 0.0)
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width, cd.sensor_height = LENS_MM, SENSOR_W, SENSOR_H
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start = 0.05
    cam = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    if lit:
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        bpy.context.active_object.is_shadow_catcher = True
        world = bpy.data.worlds.new("env")
        scene.world = world
        world.use_nodes = True
        light = os.environ.get("GS_SUBJECT_LIGHT", "env")       # diagnostic switch; default = the pipeline
        bg = world.node_tree.nodes["Background"]
        if light == "env":
            env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            env.image = bpy.data.images.load(spec["env"])
            world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
        elif light == "uniform":
            bg.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1.0)
            bg.inputs["Strength"].default_value = 1.0
        else:
            raise SystemExit(f"unknown GS_SUBJECT_LIGHT {light!r}")
    bpy.context.view_layer.update()
    return cam, objs


def dump(spec: dict, out: str) -> None:
    cam, objs = build_scene(spec, lit=False)
    chunks = []
    for o in objs:
        if o.type != "MESH" or o.hide_render:
            continue
        mw = np.array(o.matrix_world, dtype=np.float64)
        local = np.array([v.co[:] for v in o.data.vertices], dtype=np.float64)
        if len(local):
            chunks.append(local @ mw[:3, :3].T + mw[:3, 3])
    verts = np.concatenate(chunks)
    lo, hi = verts.min(0), verts.max(0)
    rng = np.random.default_rng(0)
    if len(verts) > 20000:                                   # the scorer only needs the hull's spread
        verts = verts[rng.choice(len(verts), 20000, replace=False)]
    frame = cam.data.view_frame(scene=bpy.context.scene)      # v7's frame-bounds rule, verbatim
    vz = abs(frame[0].z) or 1.0
    xs, ys = [f.x / vz for f in frame], [f.y / vz for f in frame]
    np.savez_compressed(
        out, placement=np.array(spec["placement"]), verts=verts,
        frame_bounds=np.array([min(xs), max(xs), min(ys), max(ys)], dtype=np.float64),
        resolution=np.array(RES), subject_center=(lo + hi) / 2.0,
        subject_foot=np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]]),
        subject_height=np.float64(hi[2] - lo[2]))
    print(f"[gs-dump] {spec['placement']} verts={len(verts)} height={hi[2] - lo[2]:.3f} m", flush=True)


def setup_render(spec: dict, comm: Path):
    """The lit scene, on the GPU, writing RGBA + a Z pass into `comm`. Returns (camera, file
    output node, the device in use)."""
    cam, _ = build_scene(spec, lit=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    device = "CPU"
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:  # noqa: BLE001
            continue
        if [d for d in prefs.devices if d.type == backend]:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            device = backend
            break
    scene.cycles.samples = int(spec.get("samples", 64))
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = True
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = os.environ.get("GS_SUBJECT_VIEW", "Standard")
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    tree.links.new(rl.outputs["Image"], tree.nodes.new("CompositorNodeComposite").inputs["Image"])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = str(comm)
    fo.format.file_format, fo.format.color_depth = "OPEN_EXR", "32"
    tree.links.new(rl.outputs["Depth"], fo.inputs[0])
    s = scene.render.image_settings
    s.file_format, s.color_mode, s.color_depth = "PNG", "RGBA", "16"
    return cam, fo, device


def serve(spec: dict, comm: Path) -> None:
    cam, fo, device = setup_render(spec, comm)
    scene = bpy.context.scene
    print(f"[gs-subject] {spec['placement']} on {device}, {scene.render.threads} threads",
          file=sys.stderr, flush=True)
    (comm / "ready").write_text("1")
    req, resp, stop = comm / "request.json", comm / "response.json", comm / "stop"
    # Exit with the server that spawned us, too: a server killed mid-run (preemption, a hung
    # render) never writes `stop`, and an orphaned Blender would hold its GPU until the pod dies.
    parent = os.getppid()
    while not stop.exists() and os.getppid() == parent:
        if not req.exists():
            time.sleep(0.02)
            continue
        try:
            r = json.loads(req.read_text())
        except Exception:  # noqa: BLE001
            time.sleep(0.02)
            continue
        req.unlink(missing_ok=True)
        try:
            cam.matrix_world = (Matrix.Translation(Vector(r["location"]))
                                @ Matrix(r["rotation"]).to_4x4())
            stem = Path(r["fg"]).stem
            fo.file_slots[0].path = stem + "_z_"
            scene.frame_current = 1
            scene.render.filepath = r["fg"]
            bpy.ops.render.render(write_still=True)
            payload = {"ok": True, "fg": r["fg"], "z": str(comm / f"{stem}_z_0001.exr")}
        except Exception as e:  # noqa: BLE001  (keep serving across one bad frame)
            payload = {"ok": False, "error": repr(e)}
        tmp = resp.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(resp)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    mode, spec_path, target = argv[0], argv[1], argv[2]
    spec = json.loads(Path(spec_path).read_text())
    if mode == "dump":
        dump(spec, target)
    elif mode == "serve":
        serve(spec, Path(target))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
