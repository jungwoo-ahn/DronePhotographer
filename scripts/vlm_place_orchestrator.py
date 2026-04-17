"""VLM-in-the-loop object placement orchestrator.

For each (scene, object) pair:
  1. Call Blender worker to discover N candidate flat positions.
  2. For each candidate, render an EEVEE preview and ask a VLM to evaluate.
  3. If VLM quality < threshold, apply suggested adjustment and re-render.
  4. Save per-pair JSON results.

Usage:
    python scripts/vlm_place_orchestrator.py --config configs/vlm_placement.json

    # Single pair test:
    python scripts/vlm_place_orchestrator.py \
        --config configs/vlm_placement.json \
        --scene_file /path/to/scene.blend \
        --object_file /path/to/object.blend \
        --candidates 1 --max_iterations 1
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Allow importing from the project root (parent of scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assets import (
    OBJECT_EXTS,
    SCENE_EXTS,
    discover_files,
    filter_scenes_by_texture,
    find_scene_blend,
    object_name as _object_name,
    scene_name as _scene_name,
)
from src.vlm.api import (
    call_vlm,
    check_compatibility,
    estimate_scale_from_placement,
)
from src.vlm.transforms import camera_to_world_adjustment

log = logging.getLogger("vlm_placement")
log.setLevel(logging.DEBUG)

# Console handler: minimal progress only
_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)  # only progress (WARNING+) goes to terminal
_console.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_console)

# Progress logger — separate so we can control its level independently
progress = logging.getLogger("vlm_placement.progress")
progress.setLevel(logging.INFO)
_progress_handler = logging.StreamHandler()
_progress_handler.setFormatter(logging.Formatter("%(message)s"))
progress.addHandler(_progress_handler)
progress.propagate = False

# Suppress noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _setup_file_logging(output_dir: Path):
    """Add a file handler for detailed logs."""
    log_path = output_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(fh)

RESULT_PREFIX = "###RESULT###"


def _f(val, default: float = 0.0) -> float:
    """Safely coerce a value to float, returning default for None / non-numeric."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Blender subprocess
# ---------------------------------------------------------------------------


def call_blender_worker(
    blender_bin: str,
    scene_blend: Path,
    mode: str,
    object_file: str,
    extra_args: list[str] | None = None,
    timeout: int = 600,
) -> dict | None:
    worker_script = Path(__file__).resolve().parent / "vlm_place_worker.py"
    cmd = [
        blender_bin,
        "--background",
        str(scene_blend),
        "--python",
        str(worker_script),
        "--",
        "--mode", mode,
        "--object_file", object_file,
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.debug(f"Blender cmd: --mode {mode}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        log.error(f"Blender timed out after {timeout}s")
        return None

    for line in result.stdout.split("\n"):
        if line.startswith(RESULT_PREFIX):
            try:
                data = json.loads(line[len(RESULT_PREFIX):])
                if data.get("status") == "error":
                    log.error(f"Blender error: {data.get('error')}")
                    return None
                return data
            except json.JSONDecodeError:
                log.error(f"Failed to parse worker JSON")
                return None

    stderr_tail = result.stderr[-500:] if result.stderr else ""
    log.error(f"No ###RESULT### in Blender output. Exit code: {result.returncode}")
    if stderr_tail:
        log.error(f"stderr: {stderr_tail}")
    return None


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


# Azimuths are now computed dynamically by the worker using find_valid_azimuths


def _build_render_args(position, scale, rotation, placement_cfg, render_cfg, img_path,
                       gpu_backend="AUTO", gpu_devices=None, multi_view=False,
                       scene_preview_path=None, shot_types=None, scene_scale=1.0):
    """Build extra_args for call_blender_worker place_and_render."""
    args = [
        "--position", *[f"{p:.6f}" for p in position],
        "--camera_radius", str(placement_cfg.get("camera_radius", 0)),
        "--camera_elevation", str(placement_cfg.get("camera_elevation_deg", 30.0)),
        "--output_image", img_path,
        "--engine", render_cfg.get("engine", "CYCLES"),
        "--resolution", *[str(r) for r in render_cfg.get("resolution", [512, 384])],
        "--samples", str(render_cfg.get("samples", 32)),
        "--sky_strength", str(placement_cfg.get("sky_strength", 0.3)),
        "--gpu_backend", gpu_backend,
    ]
    if scene_preview_path:
        args.extend(["--scene_preview_path", scene_preview_path])
    if multi_view:
        # Pass placeholder azimuths — worker will find valid ones dynamically
        args.extend(["--camera_azimuths", "0", "120", "240"])
    if gpu_devices:
        args.extend(["--gpu_devices", *[str(d) for d in gpu_devices]])
    if shot_types:
        args.extend(["--shot_types", *shot_types])
    if abs(scene_scale - 1.0) > 0.01:
        args.extend(["--scene_scale", f"{scene_scale:.6f}"])
    if scale > 0:
        args.extend(["--scale", f"{scale:.6f}"])
    if rotation:
        rx, ry, rz = rotation
        if abs(rx) > 0.01:
            args.extend(["--rotation_x", f"{rx:.2f}"])
        if abs(ry) > 0.01:
            args.extend(["--rotation_y", f"{ry:.2f}"])
        if abs(rz) > 0.01:
            args.extend(["--rotation_z", f"{rz:.2f}"])
    return args


def process_candidate(
    blender_bin: str,
    scene_blend: Path,
    object_file: str,
    candidate: dict,
    scene_name: str,
    object_name: str,
    candidate_idx: int,
    config: dict,
    output_dir: Path,
    initial_scale: float = 0.0,
    initial_rotation: list[float] | None = None,
    scene_preview_path: str | None = None,
    previous_attempts: list | None = None,
    defer_cleanup: bool = False,
) -> dict:
    placement_cfg = config["placement"]
    vlm_cfg = config["vlm"]
    preview_cfg = config["preview"]
    max_iter = placement_cfg.get("max_iterations", 3)
    threshold = placement_cfg.get("quality_threshold", 7.0)
    adj_scale = placement_cfg.get("adjustment_scale", 0.5)
    scene_scale = config.get("scene_scale", 1.0)
    # Once the focused scale check has fixed the scene scale, disable further
    # in-loop scale adjustments so we only refine position/rotation.
    scale_locked = bool(config.get("_scale_locked"))
    gpu_backend = config.get("gpu_backend", "AUTO")
    gpu_devices = config.get("gpu_devices")

    position = list(candidate["position"])
    current_scale = initial_scale  # 0 = auto
    current_rotation = list(initial_rotation or [0.0, 0.0, 0.0])
    best_quality = -1.0
    best_result = None
    attempts = []

    for iteration in range(max_iter):
        img_name = f"{scene_name}_{object_name}_c{candidate_idx}_iter{iteration}.png"
        img_path = str(output_dir / "previews" / img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        render_result = call_blender_worker(
            blender_bin,
            scene_blend,
            mode="place_and_render",
            object_file=object_file,
            extra_args=_build_render_args(
                position, current_scale, current_rotation,
                placement_cfg, preview_cfg, img_path,
                gpu_backend=gpu_backend, gpu_devices=gpu_devices,
                multi_view=True,
                scene_preview_path=scene_preview_path,
                scene_scale=scene_scale,
            ),
        )

        if render_result is None:
            log.warning(f"  Render failed at iteration {iteration}")
            attempts.append({
                "iteration": iteration, "position": position, "status": "render_failed",
            })
            break

        # Collect non-skipped views (skipped views have image_path=None due to low coverage)
        all_views = render_result.get("views", [])
        valid_views = [v for v in all_views if v.get("image_path") and not v.get("skipped")]
        view_images = [v["image_path"] for v in valid_views]
        view_meta = valid_views

        collisions = render_result.get("collisions", [])
        if collisions:
            log.info(f"  Collisions: {[c['object'] for c in collisions[:3]]}")
        log.info(f"  Rendered {len(view_images)}/{len(all_views)} valid views for iter {iteration}")

        # Programmatic grounding gate: reject early if floating/embedded
        grounding_info = render_result.get("grounding", {})
        if grounding_info and not grounding_info.get("grounded", True):
            gap = grounding_info.get("gap_min", 0)
            log.info(f"  REJECT: not grounded (gap={gap:.3f}m)")
            attempts.append({
                "iteration": iteration,
                "position": list(position),
                "status": "not_grounded",
                "vlm_quality": 0,
                "vlm_issues": [f"floating/embedded (gap={gap:.3f}m)"],
            })
            break

        # Need at least 1 valid view (we'll require higher quality if only 1)
        if len(view_images) < 1:
            log.info(f"  REJECT: no valid views (all cameras out of bounds)")
            attempts.append({
                "iteration": iteration,
                "position": list(position),
                "status": "no_valid_views",
                "vlm_quality": 0,
                "vlm_issues": ["camera could not see scene"],
            })
            break

        # Build extra context for VLM
        obj_size = render_result.get("object_size", [0, 0, 0])
        grounding = render_result.get("grounding", {})
        # Filter out meaningless collisions (entire room meshes, floors)
        real_collisions = [
            c for c in collisions
            if c.get("overlap_ratio", 0) < 0.9
        ]
        extra_context = ""
        if real_collisions:
            collision_names = [c["object"] for c in real_collisions[:3]]
            extra_context += (
                f"\nNote: Object may be overlapping with: {', '.join(collision_names)}. "
                f"Check visually if there is actual clipping."
            )
        from src.assets.object_sizes import expected_object_height_m as _eoh
        _exp_h, _label = _eoh(object_file)
        extra_context += (
            f"\nObject: {_label}, real-world height = {_exp_h:.2f}m."
            f"\nCRITICAL — CHECK RELATIVE SCALE:"
            f"\n  The {_label} is {_exp_h:.1f}m tall in real life."
        )
        if _exp_h > 1.0:
            extra_context += (
                f"\n  - Kitchen counters should be at the person's WAIST (~0.9m)."
                f"\n  - Door handles should be at the person's HAND level."
                f"\n  - Chairs should reach the person's KNEE to WAIST."
                f"\n  - If counters are at HEAD level or above, the scene is TOO BIG → score ≤ 3, set scale_factor > 1."
                f"\n  - If the person towers over doors/furniture, the scene is TOO SMALL → score ≤ 3, set scale_factor < 1."
            )
        else:
            extra_context += (
                f"\n  - A {_exp_h:.1f}m {_label} should be smaller than a typical chair (~0.8m)."
                f"\n  - On a table, it should be noticeably smaller than a teapot or monitor."
                f"\n  - If it looks like a speck on the floor, the scene is TOO BIG → score ≤ 3."
            )
        # Programmatic grounding info (authoritative)
        if grounding:
            gap = grounding.get("gap_min")
            if gap is not None:
                if gap > 0.08:
                    extra_context += f"\nWARNING: Object is FLOATING {gap:.2f}m above the floor."
                elif gap < -0.05:
                    extra_context += f"\nWARNING: Object is EMBEDDED {abs(gap):.2f}m into the floor."
                else:
                    extra_context += f"\nVerified: object is grounded (gap={gap:.3f}m)."

        # Cross-iteration / cross-candidate context
        if previous_attempts:
            extra_context += "\n\nPrevious attempts on this scene/object:"
            for pa in previous_attempts[-3:]:
                pos = pa.get("position", [0, 0, 0])
                q = pa.get("quality", 0)
                issues = pa.get("issues", [])[:2]
                extra_context += (
                    f"\n  - pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                    f"quality={q:.1f} issues={issues}"
                )
            extra_context += "\nUse this history to suggest a meaningfully different position."

        # Evaluate each view with VLM
        view_qualities = []
        view_scores = []  # (quality, azimuth, response) for tracking best view
        view_evaluated = []  # [{image, quality, visible}] — non-broken views only
        all_issues = []
        worst_response = None
        for vi, vimg in enumerate(view_images):
            scene_preview = render_result.get("scene_preview")
            if scene_preview and not os.path.exists(scene_preview):
                scene_preview = None
            vr = call_vlm(
                image_path=vimg,
                object_name=object_name,
                scene_name=scene_name,
                camera_radius=_f(render_result.get("camera_radius"), 5.0),
                camera_elevation=_f(placement_cfg.get("camera_elevation_deg"), 30.0),
                vlm_config=vlm_cfg,
                extra_context=extra_context,
                scene_context_image=scene_preview,
            )
            if vr is None:
                continue
            vq = _f(vr.get("quality"), 0)
            vr_issues = vr.get("issues", [])
            view_azim = view_meta[vi].get("azimuth", "?") if vi < len(view_meta) else "?"
            view_elev = view_meta[vi].get("elevation", "?") if vi < len(view_meta) else "?"
            view_cov = view_meta[vi].get("coverage", 0.0) if vi < len(view_meta) else 0.0

            # Skip broken renders (black/no object) — don't let them tank the score
            is_broken = vq == 0 and any(
                k in str(vr_issues).lower() for k in ["black", "no object visible", "broken render"]
            )
            if is_broken:
                log.info(f"    view {vi} (azim={view_azim} elev={view_elev}): BROKEN — skipping")
                continue

            view_qualities.append(vq)
            view_scores.append((vq, view_azim, view_elev, vr))
            view_evaluated.append({
                "image": vimg,
                "quality": vq,
                "visible": bool(vr.get("fully_visible", False)),
            })
            all_issues.extend(vr_issues)
            log.info(
                f"    view {vi} (azim={view_azim} elev={view_elev} cov={view_cov:.2f}): "
                f"quality={vq:.1f} visible={vr.get('fully_visible', '?')} "
                f"grounded={vr.get('grounded', '?')}"
            )
            if worst_response is None or vq < _f(worst_response.get("quality"), 10):
                worst_response = vr

        # Use worst view for adjustment suggestions
        vlm_response = worst_response

        if vlm_response is None:
            log.warning(f"  VLM failed at iteration {iteration}")
            attempts.append({
                "iteration": iteration, "position": position,
                "preview_image": img_path, "status": "vlm_failed",
            })
            break

        if len(view_qualities) < 1:
            log.info(f"  REJECT: zero non-broken VLM evaluations")
            attempts.append({
                "iteration": iteration,
                "position": list(position),
                "status": "no_valid_views",
                "vlm_quality": 0,
                "vlm_issues": ["no valid VLM evaluations"],
            })
            break

        # Quality scoring: use the BEST view's score AMONG VISIBLE views.
        # A view where the VLM couldn't see the object shouldn't contribute
        # to the candidate's quality — only visible views count.
        visible_qualities = [
            v["quality"] for v in view_evaluated if v.get("visible")
        ]
        if visible_qualities:
            quality = max(visible_qualities)
        else:
            # No view had the object visible. Cap quality so the candidate
            # can't be accepted, but keep it non-zero so we still record it.
            quality = min(max(view_qualities), 2.0)
            log.info(f"  No visible views (all cropped/occluded) — capping quality at 2.0")
        issues = list(set(all_issues)) if all_issues else vlm_response.get("issues", [])
        issues_str = str(issues).lower()
        has_scale_problem = any(k in issues_str for k in [
            "scale mismatch", "implausible scale", "incorrect scale",
            "too small", "too large", "tiny", "giant", "doll",
        ])
        if has_scale_problem and quality > 3.0 and not scale_locked:
            log.info(f"  Scale issue detected in issues — capping quality from {quality:.1f} to 3.0")
            quality = 3.0

        adjustment = vlm_response.get("adjustment", {})
        vlm_scale = _f(vlm_response.get("scale_factor"), 1.0)
        # If scale issues detected but VLM didn't suggest a scale_factor,
        # default to shrinking the scene by 2x
        if has_scale_problem and abs(vlm_scale - 1.0) < 0.05:
            vlm_scale = 2.0  # suggest "object too small" → shrink scene
            log.info(f"  Forcing scale_factor={vlm_scale} due to detected scale issues")
        vlm_rotation = vlm_response.get("rotation", {})
        reasoning = vlm_response.get("reasoning", "")

        effective_scale = current_scale if current_scale > 0 else render_result.get("applied_scale", 1.0)

        attempts.append({
            "iteration": iteration,
            "position": list(position),
            "rotation": list(current_rotation),
            "scale": effective_scale,
            "preview_image": img_path,
            "view_images": view_images,
            "view_qualities": view_qualities,
            "view_evaluated": view_evaluated,
            "vlm_quality": quality,
            "vlm_issues": issues,
            "vlm_adjustment": adjustment,
            "vlm_scale_factor": vlm_scale,
            "vlm_rotation": vlm_rotation,
            "vlm_reasoning": reasoning,
        })

        log.info(
            f"  iter {iteration}: quality={quality:.1f} "
            f"scale={vlm_scale:.2f} rot={vlm_rotation} issues={issues}"
        )

        # Find the best-scoring view (azimuth + elevation) for final render
        best_view_azim = None
        best_view_elev = None
        if view_scores:
            best_vs = max(view_scores, key=lambda x: x[0])
            best_view_azim = best_vs[1]
            best_view_elev = best_vs[2]

        if quality > best_quality:
            best_quality = quality
            best_result = {
                "position": list(position),
                "rotation": list(current_rotation),
                "scale": effective_scale,
                "quality": quality,
                "iteration": iteration,
                "preview_image": img_path,
                "render_result": render_result,
                "best_azimuth": best_view_azim,
                "best_elevation": best_view_elev,
            }

        if quality >= threshold:
            log.info(f"  Accepted at iteration {iteration} (quality={quality:.1f})")
            break

        if iteration < max_iter - 1:
            # Apply position adjustment
            adj = adjustment
            fwd = _f(adj.get("forward"), 0)
            right = _f(adj.get("right"), 0)
            up = _f(adj.get("up"), 0)
            has_pos_adj = abs(fwd) + abs(right) + abs(up) > 0.01
            has_scale_adj = abs(vlm_scale - 1.0) > 0.05
            has_rot_adj = bool(vlm_rotation) and any(
                abs(_f(vlm_rotation.get(k), 0)) > 1.0 for k in ("x", "y", "z")
            )

            if not has_pos_adj and not has_scale_adj and not has_rot_adj:
                log.info("  VLM suggested no changes, stopping early")
                break

            if has_pos_adj:
                world_delta = camera_to_world_adjustment(
                    fwd, right, up,
                    camera_forward=render_result["camera_forward"],
                    camera_up=render_result["camera_up"],
                )
                position = [
                    position[i] + world_delta[i] * adj_scale for i in range(3)
                ]

            # Apply scale adjustment — VLM says scale_factor for the OBJECT,
            # but we invert it to scale the SCENE instead.
            # If VLM says "object too small, scale_factor: 2.0" → shrink scene by 0.5
            # Skipped when scale_locked (focused scale check already fixed it).
            if has_scale_adj and not scale_locked:
                scene_scale_adj = 1.0 / vlm_scale
                scene_scale *= scene_scale_adj
                config["scene_scale"] = scene_scale
                log.info(f"  VLM says scale_factor={vlm_scale:.2f} → "
                         f"scene_scale now {scene_scale:.4f}")
            elif has_scale_adj and scale_locked:
                log.info(f"  Ignoring VLM scale suggestion (scale locked at {scene_scale:.4f})")

            # Apply rotation adjustment
            if has_rot_adj:
                current_rotation = [
                    current_rotation[0] + _f(vlm_rotation.get("x"), 0),
                    current_rotation[1] + _f(vlm_rotation.get("y"), 0),
                    current_rotation[2] + _f(vlm_rotation.get("z"), 0),
                ]
                log.info(f"  Adjusting rotation: {current_rotation}")

    accepted = best_quality >= threshold

    # Cache the scene scale if it changed (so other objects in the same scene use it)
    if accepted and abs(scene_scale - 1.0) > 0.01:
        import json as _json
        sc_path = Path("outputs/_scene_preview_cache") / f"{scene_name}_scale.json"
        sc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sc_path, "w") as f:
            _json.dump({"scale_factor": scene_scale}, f)
        log.info(f"  Cached scene_scale={scene_scale:.4f} for {scene_name}")

    result = {
        "candidate_idx": candidate_idx,
        "accepted": accepted,
        "position": best_result["position"] if best_result else position,
        "rotation": best_result["rotation"] if best_result else current_rotation,
        "scale": best_result["scale"] if best_result else current_scale,
        "attempts": attempts,
    }

    # Final high-quality renders for accepted placements.
    # Camera positions are stored per final image (not preview).
    # Render ALL preview views that scored well — each becomes a final image
    # with its own framing (close-up, standard, wide, etc.)
    if accepted and best_result:
        final_cfg = config.get("final", {})
        if final_cfg:
            os.makedirs(str(output_dir / "images"), exist_ok=True)
            final_images = []

            # Only consider views the preview VLM marked fully_visible AND scored
            # above the minimum quality. No visibility = no final render.
            min_final_quality = 5.0
            candidates_ranked = sorted(
                [
                    (vq, va, ve)
                    for vq, va, ve, vr in view_scores
                    if vq >= min_final_quality and vr.get("fully_visible")
                ] if view_scores else [],
                key=lambda x: x[0], reverse=True,
            )

            # Enforce angular diversity: pick top view, then next-best that's
            # at least 60° away in azimuth. Keeps outputs from being two
            # near-identical shots.
            def _ang_dist(a, b):
                try:
                    d = abs((float(a) % 360) - (float(b) % 360))
                    return min(d, 360 - d)
                except (TypeError, ValueError):
                    return 180.0

            good_views = []
            for v in candidates_ranked:
                if not good_views:
                    good_views.append(v)
                    continue
                if all(_ang_dist(v[1], g[1]) >= 60.0 for g in good_views):
                    good_views.append(v)
                if len(good_views) >= 2:
                    break
            # If diversity filter left us with only 1, fall back to 2 best regardless
            if len(good_views) < 2 and len(candidates_ranked) >= 2:
                good_views = candidates_ranked[:2]

            if not good_views:
                best_azim = best_result.get("best_azimuth")
                best_elev = best_result.get("best_elevation")
                if best_azim is not None:
                    good_views = [(best_quality, best_azim, best_elev)]

            for vi, (vq, va, ve) in enumerate(good_views):
                final_name = f"{scene_name}_{object_name}_c{candidate_idx}_v{vi}.png"
                final_path = str(output_dir / "images" / final_name)

                final_placement_cfg = dict(placement_cfg)
                ve_numeric = None
                if ve is not None:
                    try:
                        ve_numeric = float(ve)
                    except (TypeError, ValueError):
                        ve_numeric = None
                if ve_numeric is not None:
                    final_placement_cfg["camera_elevation_deg"] = ve_numeric
                final_extra = _build_render_args(
                    best_result["position"],
                    best_result["scale"],
                    best_result["rotation"],
                    final_placement_cfg,
                    final_cfg,
                    final_path,
                    gpu_backend=gpu_backend, gpu_devices=gpu_devices,
                    scene_preview_path=scene_preview_path,
                    scene_scale=scene_scale,
                )
                final_extra.extend(["--camera_azimuths", str(va)])
                log.info(f"  Rendering final v{vi} (azim={va} elev={ve} preview_q={vq:.1f})...")
                final_render = call_blender_worker(
                    blender_bin, scene_blend,
                    mode="place_and_render",
                    object_file=object_file,
                    extra_args=final_extra,
                )
                if final_render and os.path.exists(final_path):
                    # VLM validation: object must be identifiable in the render
                    sp = final_render.get("scene_preview") or scene_preview_path
                    if sp and not os.path.exists(sp):
                        sp = None
                    check = call_vlm(
                        image_path=final_path,
                        object_name=object_name,
                        scene_name=scene_name,
                        camera_radius=_f(final_render.get("camera_radius"), 5.0),
                        camera_elevation=_f(ve, 15.0),
                        vlm_config=vlm_cfg,
                        scene_context_image=sp,
                    )
                    check_q = _f(check.get("quality"), 0) if check else 0
                    check_vis = check.get("fully_visible", False) if check else False
                    check_grounded = check.get("grounded", False) if check else False
                    # Keep only if the object is mostly visible AND the
                    # placement was scored acceptably.
                    if check_vis and check_q >= 5.0:
                        # Get camera info from the final render result
                        cam_info = {}
                        fr_views = final_render.get("views", [])
                        if fr_views:
                            v0 = fr_views[0]
                            cam_info = {
                                "camera_position": v0.get("camera_position"),
                                "camera_forward": v0.get("camera_forward"),
                                "camera_up": v0.get("camera_up"),
                                "camera_radius": v0.get("camera_radius"),
                            }
                        final_images.append({
                            "image": final_path,
                            "azimuth": va,
                            "elevation": ve,
                            "preview_quality": vq,
                            "final_quality": check_q,
                            **cam_info,
                        })
                        log.info(f"    v{vi}: saved (q={check_q:.1f} vis={check_vis})")
                    else:
                        os.remove(final_path)
                        log.info(f"    v{vi}: SKIPPED (q={check_q:.1f} vis={check_vis})")

            result["final_images"] = final_images
            log.info(f"  Saved {len(final_images)}/{len(good_views)} final renders")

    # Clean up preview images for rejected placements
    if not accepted and not defer_cleanup:
        for attempt in attempts:
            for img in attempt.get("view_images", []):
                try:
                    os.remove(img)
                except OSError:
                    pass

    return result


def process_pair(
    blender_bin: str,
    scene_blend: Path,
    object_file: str,
    scene_name: str,
    object_name: str,
    config: dict,
    output_dir: Path,
) -> dict:
    """Process one (scene, object) pair. Returns a result dict saved as per-pair JSON."""
    placement_cfg = config["placement"]
    top_k = placement_cfg.get("candidates_per_scene", 5)
    min_dist = placement_cfg.get("min_candidate_distance", 2.0)

    # --- Step 0: Scene preview cache + VLM scale estimation ---
    cache_root = Path("outputs/_scene_preview_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    scene_preview_path = str(cache_root / f"{scene_name}.png")
    scale_cache_path = cache_root / f"{scene_name}_scale.json"

    # Get expected object size
    from src.assets.object_sizes import expected_object_height_m
    expected_h, obj_label = expected_object_height_m(object_file)

    # Scene scale: use cached value if available, otherwise start at 1.0.
    # The VLM will judge proportions during placement evaluation and suggest
    # scale_factor corrections via the iteration loop.
    if scale_cache_path.exists():
        import json as _json
        cached = _json.load(open(scale_cache_path))
        scene_scale = cached.get("scale_factor", 1.0)
        log.info(f"Cached scene scale: {scene_scale:.2f}")
    else:
        scene_scale = 1.0

    config["scene_scale"] = scene_scale

    # --- Step 1: Discover candidates (at the estimated scene scale) ---
    log.info(f"Discovering candidates for {scene_name} + {object_name} "
             f"(scene_scale={scene_scale:.2f})...")

    disc_args = [
        "--top_k", str(top_k),
        "--min_candidate_distance", str(min_dist),
    ]
    if abs(scene_scale - 1.0) > 0.01:
        disc_args.extend(["--scene_scale", str(scene_scale)])

    discovery = call_blender_worker(
        blender_bin,
        scene_blend,
        mode="discover_candidates",
        object_file=object_file,
        extra_args=disc_args,
    )

    candidates = []
    scene_bounds = None
    if discovery:
        candidates = discovery.get("candidates", [])
        scene_bounds = discovery.get("scene_bounds")

    # If this was the first pair for this scene, the place_and_render call
    # will generate the scene preview. After that, we can ask VLM for scale
    # and re-discover if needed. This is handled by the retry logic below.

    if not candidates and scene_bounds:
        sb_min, sb_max = scene_bounds
        center = [(sb_min[i] + sb_max[i]) / 2 for i in range(3)]
        center[2] = sb_min[2]
        candidates = [{"position": center, "normal": [0, 0, 1], "score": 0.0}]
        log.warning(f"No flat surfaces found, using scene center fallback")
    elif not candidates:
        log.warning(f"No candidates and no scene bounds for {scene_name}")
        return {
            "scene_file": str(scene_blend),
            "object_file": object_file,
            "scene_name": scene_name,
            "object_name": object_name,
            "status": "no_candidates",
            "placements": [],
        }

    initial_scale = discovery.get("applied_scale", 0.0) if discovery else 0.0
    log.info(f"Found {len(candidates)} candidates")

    def _best_view_image(result_dict):
        """Pick the best view for scale judgment.

        Prefers views where the VLM said the object was visible (fully_visible=True).
        Falls back to any on-disk view image.
        """
        # First pass: find a visible view (across all attempts)
        for attempt in result_dict.get("attempts", []):
            evals = attempt.get("view_evaluated", [])
            visible = [v for v in evals if v.get("visible") and Path(v["image"]).exists()]
            if visible:
                visible.sort(key=lambda v: v.get("quality", 0), reverse=True)
                return visible[0]["image"]
        # Second pass: any on-disk image
        for attempt in result_dict.get("attempts", []):
            for v in attempt.get("view_images", []):
                if v and Path(v).exists():
                    return v
            if attempt.get("preview_image") and Path(attempt["preview_image"]).exists():
                return attempt["preview_image"]
        return None

    def _any_view_visible(result_dict):
        """True if any view in the result had the object visible."""
        for attempt in result_dict.get("attempts", []):
            for v in attempt.get("view_evaluated", []):
                if v.get("visible"):
                    return True
        return False

    placements = []
    skip_remaining = False
    previous_attempts_log = []  # rolling context across candidates
    scale_check_done = bool(scale_cache_path.exists())  # skip if we already cached
    if scale_check_done:
        config["_scale_locked"] = True
    scale_retries = 0
    max_scale_retries = 2

    outer_retry = True
    while outer_retry:
        outer_retry = False
        for idx, candidate in enumerate(candidates):
            if skip_remaining:
                log.info(f"Candidate {idx + 1}/{len(candidates)}: SKIP (incompatible pair)")
                continue

            log.info(
                f"Candidate {idx + 1}/{len(candidates)}: "
                f"pos={[f'{p:.2f}' for p in candidate['position']]}"
            )
            # Keep preview images around for the first candidate so the
            # focused scale check (below) can read the best view.
            defer = (idx == 0 and not scale_check_done)
            result = process_candidate(
                blender_bin=blender_bin,
                scene_blend=scene_blend,
                object_file=str(object_file),
                candidate=candidate,
                scene_name=scene_name,
                object_name=object_name,
                candidate_idx=idx,
                config=config,
                output_dir=output_dir,
                initial_scale=initial_scale,
                scene_preview_path=scene_preview_path,
                previous_attempts=list(previous_attempts_log),
                defer_cleanup=defer,
            )
            placements.append(result)
            # Track for next candidate
            if result.get("attempts"):
                for a in result["attempts"]:
                    if a.get("vlm_quality") is not None:
                        previous_attempts_log.append({
                            "position": a.get("position"),
                            "quality": a.get("vlm_quality"),
                            "issues": a.get("vlm_issues", []),
                        })
                # Keep only last 3
                previous_attempts_log = previous_attempts_log[-3:]

            # --- Focused scale check (only on first candidate, once per scene) ---
            if idx == 0 and not scale_check_done and scale_retries < max_scale_retries:
                scale_check_done = True
                config["_scale_locked"] = True
                preview_img = _best_view_image(result)
                any_visible = _any_view_visible(result)
                if not any_visible and preview_img:
                    # No view rendered a visible object: strong evidence scene is too big.
                    # Skip the VLM (it'll just say "not visible") and downscale aggressively.
                    vlm_scale = 0.3
                    reasoning = "no view showed the object — scene likely too big, forcing 0.3"
                    log.warning(
                        f"  Scale check: no visible views across {len(result.get('attempts', []))} "
                        f"attempts — treating as scene-too-big, applying x{vlm_scale}"
                    )
                elif preview_img:
                    log.info(f"  Running focused scale check on {preview_img}")
                    vlm_scale, reasoning = estimate_scale_from_placement(
                        image_path=preview_img,
                        object_label=obj_label,
                        object_height_m=expected_h,
                        scene_name=scene_name,
                        vlm_config=config["vlm"],
                    )
                else:
                    vlm_scale, reasoning = 1.0, "no preview image"

                if preview_img:
                    if abs(vlm_scale - 1.0) > 0.15:
                        new_scale = scene_scale * vlm_scale
                        # Clamp to reasonable range
                        new_scale = max(0.05, min(new_scale, 10.0))
                        log.info(
                            f"  Scale check: VLM says scene x{vlm_scale:.2f} "
                            f"({reasoning}). scene_scale {scene_scale:.3f} → {new_scale:.3f}. "
                            f"Re-discovering candidates."
                        )
                        scene_scale = new_scale
                        config["scene_scale"] = scene_scale
                        scale_retries += 1

                        # Discard current placements and their preview images
                        for p in placements:
                            for a in p.get("attempts", []):
                                for img in a.get("view_images", []):
                                    try:
                                        os.remove(img)
                                    except OSError:
                                        pass
                        placements = []
                        previous_attempts_log = []
                        skip_remaining = False

                        # Re-discover candidates at the new scale
                        disc_args2 = [
                            "--top_k", str(top_k),
                            "--min_candidate_distance", str(min_dist),
                            "--scene_scale", str(scene_scale),
                        ]
                        discovery2 = call_blender_worker(
                            blender_bin,
                            scene_blend,
                            mode="discover_candidates",
                            object_file=str(object_file),
                            extra_args=disc_args2,
                        )
                        if discovery2 and discovery2.get("candidates"):
                            candidates = discovery2.get("candidates", [])
                            initial_scale = discovery2.get("applied_scale", initial_scale)
                            log.info(f"  Re-discovered {len(candidates)} candidates at scale {scene_scale:.3f}")
                            outer_retry = True
                            break
                        else:
                            log.warning(f"  Re-discovery failed at scale {scene_scale:.3f}, giving up on this scene")
                            break
                    else:
                        log.info(
                            f"  Scale check: scene scale OK (VLM x{vlm_scale:.2f}: {reasoning})"
                        )
                else:
                    log.warning(
                        f"  Scale check: no preview image available for candidate 0"
                    )
                # Deferred cleanup for this candidate's previews if it was rejected
                if not result.get("accepted"):
                    for a in result.get("attempts", []):
                        for img in a.get("view_images", []):
                            try:
                                os.remove(img)
                            except OSError:
                                pass

            # After the first candidate, check if this object belongs in this scene.
            # Use the actual rendered preview image for an informed compatibility check.
            if idx == 0 and not config.get("skip_compatibility"):
                first_preview = _best_view_image(result)
                if first_preview:
                    compat = check_compatibility(
                        scene_preview_path=first_preview,
                        object_name=object_name,
                        scene_name=scene_name,
                        vlm_config=config["vlm"],
                    )
                    if compat and not compat.get("compatible", True):
                        log.info(
                            f"  INCOMPATIBLE — skipping remaining candidates: "
                            f"{compat.get('reasoning', '')}"
                        )
                        result["incompatible"] = True
                        result["incompatible_reason"] = compat.get("reasoning", "")
                        skip_remaining = True

    # Cache the working scale (even if 1.0) so future pairs for this scene skip the check
    if scale_retries > 0 and any(p.get("accepted") for p in placements):
        import json as _json
        with open(scale_cache_path, "w") as f:
            _json.dump({"scale_factor": scene_scale}, f)
        log.info(f"  Cached scene_scale={scene_scale:.4f} for {scene_name}")

    # Only keep accepted placements in the output
    accepted_placements = [p for p in placements if p.get("accepted")]

    pair_result = {
        "scene": scene_name,
        "scene_file": str(scene_blend),
        "object": object_name,
        "object_file": object_file,
        "incompatible": skip_remaining,
        "placements": accepted_placements,
    }

    # Save per-pair JSON
    pair_json_name = f"{scene_name}__{object_name}.json"
    pair_json_path = output_dir / pair_json_name
    with open(pair_json_path, "w") as f:
        json.dump(pair_result, f, indent=2, default=str)
    log.info(f"Saved pair result: {pair_json_path}")

    return pair_result


def _run_pair(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    blender_bin, scene_blend, obj_file, sname, obj_name, worker_config, output_dir = args_tuple
    return process_pair(
        blender_bin=blender_bin,
        scene_blend=Path(scene_blend),
        object_file=obj_file,
        scene_name=sname,
        object_name=obj_name,
        config=worker_config,
        output_dir=Path(output_dir),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM-in-the-loop object placement")
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--scene_file", help="Override: single scene .blend file")
    p.add_argument("--object_file", help="Override: single object file")
    p.add_argument("--candidates", type=int, help="Override candidates_per_scene")
    p.add_argument("--max_iterations", type=int, help="Override max_iterations")
    p.add_argument("--run_name", help="Override run name")
    p.add_argument("--skip_compatibility", action="store_true",
                   help="Skip VLM scene-object compatibility check after first render")
    p.add_argument("--gpu_backend", default="AUTO",
                   choices=["AUTO", "OPTIX", "CUDA", "CPU"],
                   help="GPU backend for Cycles rendering (default: AUTO)")
    p.add_argument("--gpu_devices", type=int, nargs="*", default=None,
                   help="GPU device indices to use (default: all available)")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Number of parallel Blender workers, each on its own GPU")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.candidates:
        config["placement"]["candidates_per_scene"] = args.candidates
    if args.max_iterations:
        config["placement"]["max_iterations"] = args.max_iterations
    config["gpu_backend"] = args.gpu_backend
    config["gpu_devices"] = args.gpu_devices

    blender_bin = config["blender"]["binary"]
    if not Path(blender_bin).is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        blender_bin = str(repo_root / blender_bin)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = args.run_name or "vlm_placement"
    output_dir = Path(config["output"]["output_dir"]) / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    log.info(f"Output: {output_dir}")

    # Discover assets
    if args.scene_file:
        scene_blend = find_scene_blend(Path(args.scene_file))
        if scene_blend is None:
            log.error(f"Scene not found: {args.scene_file}")
            sys.exit(1)
        scene_files = [scene_blend]
    else:
        scenes_dir = config["assets"]["scenes_dir"]
        all_scenes = discover_files(scenes_dir, SCENE_EXTS)
        scenes_dir_path = Path(scenes_dir)
        try:
            for p in sorted(scenes_dir_path.iterdir()):
                if p.is_dir():
                    blends = sorted(p.rglob("*.blend"))
                    if blends:
                        all_scenes.append(blends[0])
        except PermissionError:
            pass
        all_scenes = sorted(set(all_scenes))
        scene_files = filter_scenes_by_texture(
            all_scenes,
            config["assets"].get("texture_check"),
            config["assets"].get("quality_filter", ["perfect", "mostly_ok"]),
        )

    if args.object_file:
        object_files = [Path(args.object_file)]
    else:
        object_files = discover_files(
            config["assets"]["objects_dir"], OBJECT_EXTS
        )

    log.info(f"Scenes: {len(scene_files)}, Objects: {len(object_files)}")
    log.info(f"Total pairs: {len(scene_files) * len(object_files)}")

    if args.skip_compatibility:
        config["skip_compatibility"] = True

    # Build list of all (scene_blend, object_path) pairs
    pairs = []
    for scene_path in scene_files:
        scene_blend = find_scene_blend(scene_path)
        if scene_blend is None:
            continue
        sname = _scene_name(scene_blend)
        for obj_path in object_files:
            obj_name = _object_name(obj_path)
            pairs.append((scene_blend, str(obj_path), sname, obj_name))

    total_pairs = len(pairs)

    # Setup file logging
    _setup_file_logging(output_dir)

    progress.info(f"Output: {output_dir}")
    progress.info(f"Pairs: {total_pairs} ({len(scene_files)} scenes x {len(object_files)} objects)")

    # Determine GPU assignment for workers
    num_workers = min(args.num_workers, total_pairs)
    gpu_devices = args.gpu_devices or list(range(8))

    all_pair_results = []
    accepted_total = 0
    start_time = time.time()

    def _log_progress(pair_idx, sname, obj_name, result):
        nonlocal accepted_total
        n_accepted = len(result.get("placements", []))
        accepted_total += n_accepted
        elapsed = time.time() - start_time
        avg = elapsed / pair_idx if pair_idx else 0
        eta = avg * (total_pairs - pair_idx)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta > 0 else "done"
        status = f"{n_accepted} placed" if n_accepted else "skipped"
        if result.get("incompatible"):
            status = "incompatible"
        progress.info(
            f"[{pair_idx:03d}/{total_pairs:03d}] {sname} + {obj_name} -> "
            f"{status} | total={accepted_total} | ETA {eta_str}"
        )

    if num_workers <= 1:
        for pair_idx, (scene_blend, obj_file, sname, obj_name) in enumerate(pairs, 1):
            log.info(f"Pair {pair_idx}/{total_pairs}: {sname} + {obj_name}")
            pair_result = process_pair(
                blender_bin=blender_bin,
                scene_blend=scene_blend,
                object_file=obj_file,
                scene_name=sname,
                object_name=obj_name,
                config=config,
                output_dir=output_dir,
            )
            all_pair_results.append(pair_result)
            _log_progress(pair_idx, sname, obj_name, pair_result)
    else:
        progress.info(f"Running {num_workers} workers on GPUs {gpu_devices[:num_workers]}")

        work_items = []
        for idx, (scene_blend, obj_file, sname, obj_name) in enumerate(pairs):
            gpu_id = gpu_devices[idx % len(gpu_devices[:num_workers])]
            worker_config = copy.deepcopy(config)
            worker_config["gpu_backend"] = args.gpu_backend
            worker_config["gpu_devices"] = [gpu_id]
            work_items.append((
                blender_bin, str(scene_blend), obj_file, sname, obj_name,
                worker_config, str(output_dir),
            ))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_run_pair, item): item for item in work_items
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                item = futures[future]
                _, _, _, sname, obj_name, wc, _ = item
                try:
                    result = future.result()
                    all_pair_results.append(result)
                    _log_progress(completed, sname, obj_name, result)
                except Exception as e:
                    log.error(f"Pair {sname} + {obj_name} failed: {e}")
                    all_pair_results.append({
                        "scene_name": sname,
                        "object_name": obj_name,
                        "status": "error",
                        "error": str(e),
                        "placements": [],
                    })
                    _log_progress(completed, sname, obj_name, {"placements": []})

    # Summary
    total_accepted = sum(r.get("accepted_count", 0) for r in all_pair_results)
    total_placements = sum(len(r.get("placements", [])) for r in all_pair_results)
    summary = {
        "created_at": datetime.now().isoformat(),
        "config": config,
        "total_pairs": len(all_pair_results),
        "total_placements": total_placements,
        "total_accepted": total_accepted,
        "pairs": [
            {
                "scene": r.get("scene_name"),
                "object": r.get("object_name"),
                "accepted": r.get("accepted_count", 0),
                "total": len(r.get("placements", [])),
            }
            for r in all_pair_results
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
    incompat_count = sum(1 for r in all_pair_results if r.get("incompatible"))
    progress.info(
        f"\nDone! {accepted_total} placements from {len(all_pair_results)} pairs "
        f"({incompat_count} incompatible) in {elapsed_str}. "
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
