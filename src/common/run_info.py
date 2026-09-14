"""Translate a v7 `data.json` into the run_info the Blender pose renderer expects.

Shared by every rollout script. This lives in one place on purpose: it reconstructs
the *placement transform* (scene scale, object scale, object rotation, object
position) that the training frames were rendered with, and any script that gets it
wrong renders a scene where the subject sits somewhere else — every distance-to-goal
then measures the wrong thing while looking perfectly healthy. A second copy of this
function is a second chance for that to drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH


def _reduced_aspect() -> tuple[int, int]:
    """The training render aspect in lowest terms (1024x768 -> 4:3)."""
    import math

    w, h = int(RENDER_WIDTH), int(RENDER_HEIGHT)
    g = math.gcd(w, h)
    return w // g, h // g


def to_policy_frame(image, square_size: int):
    """Render frame -> the square uint8 array the policy consumes.

    The other half of `render_resolution`'s contract: render at the training aspect, then
    squash to the square. Training did the squash in ffmpeg (`scale=256:256` over a
    1024x768 render); this is the same non-aspect-preserving resize, so an inference frame
    and a training frame of the same pose show the subject at the same size.
    """
    import numpy as np
    from PIL import Image

    img = np.asarray(image)
    if img.shape[0] == square_size and img.shape[1] == square_size:
        return img
    return np.asarray(Image.fromarray(img).resize((square_size,) * 2, Image.BILINEAR))


def render_resolution(square_size: int) -> tuple[int, int]:
    """(width, height) to RENDER so that squashing to `square_size` reproduces training.

    Snapped to an exact multiple of the reduced training aspect, and never smaller than
    `square_size`, so the squash only ever downsamples.
    """
    aw, ah = _reduced_aspect()
    k = -(-int(square_size) // ah)             # ceil
    return aw * k, ah * k


def write_run_info(
    placement: str,
    data_path: str | Path,
    out_dir: Path,
    *,
    shared_root: str | Path,
    resolution: int | tuple[int, int] = 256,
    extra_options: dict | None = None,
    v6_dir: str = "data/vlm_object_placing_v6_260428_061326",
) -> str:
    """Write the run_info JSON for `placement` and return its path.

    `resolution` may be a single int — the size of the SQUARE frame the policy consumes —
    or an explicit (width, height) pair. An int is rendered at the training aspect
    (RENDER_WIDTH:RENDER_HEIGHT = 4:3) and squashed to the square by the caller's resize,
    which is exactly what produced the training frames: v7 rendered 1024x768 and
    `lerobot_export._encode_mp4` ran `scale=256:256`, a non-aspect-preserving squash.

    Rendering the square NATIVELY instead is not the same picture. Blender's AUTO sensor fit
    keys off the long side, so a 256x256 render keeps the horizontal FOV and widens the
    vertical one by 4/3 (frame half-height 0.26667 vs 0.2) — the subject comes out 0.75x as
    tall as the policy ever saw it in training, on every frame of every rollout. That is what
    an int used to do here, and the effect is a systematic "subject too small" bias in the one
    quantity the goal profile is mostly about.

    `extra_options` is merged into `options` — `BlenderDrone` reads its render settings from
    there (`render_device`, `samples`, `blender_threads`, ...) and defaults to CPU with 64
    samples, which is right for a 256x256 rollout frame and far too slow for a full-resolution
    one. Callers that render at training resolution pass their own.

    `BlenderDrone.from_run_info` wants `input_scene` / `input_object` /
    `options.object_position` / `scene_scale` / `rotation_xyz_rad` / `scale`; v7
    stores `scene_file` / `object_file` / `subject_foot` and keeps the placement
    transform in the v6 placement JSON, so the transform is read back from there.
    Falls back to the v7 `subject_foot` with identity transform only when the v6
    file is missing.
    """
    shared = Path(shared_root)
    data = json.loads(Path(data_path).read_text())
    v6_path = shared / v6_dir / f"{placement}.json"

    scene_scale, scale, rotation = 1.0, 1.0, [0.0, 0.0, 0.0]
    position = data.get("subject_foot", [0.0, 0.0, 0.0])
    if v6_path.exists():
        v6 = json.loads(v6_path.read_text())
        chosen = v6.get("placements", [{}])[int(data.get("placement_idx", 0))]
        scene_scale = float(v6.get("scene_scale", 1.0))
        scale = float(chosen.get("scale", 1.0))
        rotation = [float(v) for v in chosen.get("rotation", [0.0, 0.0, 0.0])]
        position = [float(v) for v in chosen.get("position", position)]

    if isinstance(resolution, (int, float)):
        # Square policy input -> render at the training aspect; the caller's resize squashes.
        # Snap to an EXACT multiple of the reduced aspect (1024:768 -> 4:3) rather than
        # rounding width to `h * 4/3`: at h=256 rounding gives 341x256, whose frame
        # half-height is 0.20020 instead of 0.2. Small, but the bug this replaced was also
        # just a number that looked plausible.
        res = list(render_resolution(int(resolution)))
    else:
        res = [int(resolution[0]), int(resolution[1])]
    run_info = {
        "input_scene": str(shared / data["scene_file"]),
        "input_object": str(shared / data["object_file"]),
        "scene_scale": scene_scale,
        "scale": scale,
        "rotation_xyz_rad": rotation,
        "options": {"object_position": position, "resolution": res, **(extra_options or {})},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{placement}.run_info.json"
    path.write_text(json.dumps(run_info, indent=1))
    return str(path)


__all__ = ["write_run_info", "render_resolution", "to_policy_frame"]
