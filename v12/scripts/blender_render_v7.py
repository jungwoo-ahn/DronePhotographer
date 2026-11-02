"""Run the shared repo's Blender render server / pose renderer the way v7 rendered its frames.

The v7 training frames were rendered by `scripts/v7_reference/v7_sample_pairs_smoke.
configure_renderer`, which sets exactly this and leaves the rest as the .blend has it:
  * a NEW camera (lens 24, sensor 12.8 x 9.6) -- so no depth of field, default clipping
  * Cycles at 32 samples, bounces 4 / diffuse 2 / glossy 2 / transmission 2
  * `use_denoising = True`
  * a Nishita sky at 0.1 (deterministic sun)
The rollout renderers (`blender_render_server.py`, `blender_render_pose.py` in the shared repo)
build the scene through `BlenderDrone`, which differs in four ways that each show up in the
picture: it renders through the scene's OWN camera (Desert-Roadside's has depth of field on), it
forces `transparent_max_bounces` to 4 (Desert-Roadside's is 32 -- stacked alpha-card foliage
turns black), it also forces volume bounces and adaptive sampling, and it has no denoise option
at all (Modern-Office came out with 62x the training frames' pixel noise). The sky it does take,
from `options.sky_strength`.

With `options["v7_renderer"]` set, this wrapper undoes all four without editing the shared repo:
it wraps `_set_render_settings` to restore the scene's own values for the settings v7 never
touched, swap in a fresh camera datablock carrying the same lens and sensor, and denoise -- then
hands control to the shared script unchanged. Absent the option, nothing differs from running
the shared script directly. Measured on Desert-Roadside: MAE vs the training frame 16.1 with
sky + denoise alone, 4.6 with the camera and bounces restored too.

One file for both entry points; which one is meant is read off the arguments, because the two
take disjoint flags (`--comm_dir` for the server, `--output_image` for the per-frame renderer):

    blender/blender -b -P <V12>/scripts/blender_render_v7.py -- --run_info_path ri --comm_dir d
    blender/blender -b -P <V12>/scripts/blender_render_v7.py -- --run_info_path ri --output_image f ...

`src.common.blender_env` launches it in place of the shared scripts when a rollout asks for the
`v7` render profile (`src.common.run_info.RENDER_PROFILES`).
"""
from __future__ import annotations

import runpy
import sys

SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
SERVER = f"{SHARED}/scripts/blender_render_server.py"
POSE = f"{SHARED}/scripts/blender_render_pose.py"


def target_script(argv: list[str]) -> str:
    """The shared script these arguments belong to."""
    tail = argv[argv.index("--") + 1:] if "--" in argv else argv
    if "--comm_dir" in tail:
        return SERVER
    if "--output_image" in tail:
        return POSE
    raise SystemExit(f"blender_render_v7: cannot tell server from pose renderer in {tail!r}")


# Cycles settings BlenderDrone forces and v7's configure_renderer never touched.
SCENE_OWNED = ("transparent_max_bounces", "volume_bounces", "use_adaptive_sampling",
               "adaptive_threshold")


def main() -> None:
    import bpy  # (only exists inside Blender)

    sys.path.insert(0, SHARED)
    import src.drones.blender_drone as drone

    base = drone._set_render_settings

    def _set_render_settings(scene, options):
        if not options.get("v7_renderer"):
            return base(scene, options)
        own = {k: getattr(scene.cycles, k) for k in SCENE_OWNED if hasattr(scene.cycles, k)}
        base(scene, options)
        for k, v in own.items():
            setattr(scene.cycles, k, v)
        scene.cycles.use_denoising = True
        cam = scene.camera
        if cam is not None and cam.type == "CAMERA":
            fresh = bpy.data.cameras.new("V7Camera")
            fresh.lens = cam.data.lens
            fresh.sensor_width, fresh.sensor_height = cam.data.sensor_width, cam.data.sensor_height
            cam.data = fresh

    # BlenderDrone looks the function up in its module globals at call time, and the shared
    # script imports that same module object from sys.modules, so the patch is what it calls.
    drone._set_render_settings = _set_render_settings
    argv = list(getattr(bpy.app, "argv", sys.argv))
    runpy.run_path(target_script(argv), run_name="__main__")


if __name__ == "__main__":
    main()
