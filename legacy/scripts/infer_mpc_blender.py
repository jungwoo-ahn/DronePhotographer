from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scoring.bbox_control import RULE_BASED_SCORE_KEYS, compute_rule_based_scores
from src.scoring.evaluator import ALL_SUPPORTED_SCORE_KEYS
from src.vlm_qwen25.mpc import (
    generate_local_candidate_actions,
    score_action_candidates,
    score_action_candidates_vllm,
    score_with_lookahead_vllm,
    write_rollout_summary_image,
    write_rollout_video,
)
from src.vlm_qwen25.objective import (
    available_target_presets,
    build_target_objective,
    build_target_objective_schedule,
    objective_for_step,
    preset_specs,
    weighted_target_error,
)


def _load_inference_config(path: str) -> dict:
    """Load a YAML inference preset and flatten it to argparse-style overrides.

    YAML schema:
        name: human-readable name
        description: shot description
        target: {center_x, center_y, occupancy, body_in_frame_ratio, ...}
        weights: {key: float, ...}     # optional
        mpc:
          num_steps: 50
          translation_values_m: [-0.2, -0.1, 0, 0.1, 0.2]
          rotation_values_deg: [-5, 0, 5]
          max_translation_norm_m: 0.3
          max_rotation_norm_deg: 7.5
          max_candidates: 720
          max_new_tokens: 256
          disable_roll: false        # optional
          # ... any other --flag from infer_mpc_blender.py
    """
    cfg = yaml.safe_load(open(path))
    overrides: dict = {}
    if cfg.get("target") is not None:
        overrides["target_json"] = json.dumps(cfg["target"], separators=(",", ":"))
        # YAML target is authoritative — clear the default --target_preset so we
        # don't pull in legacy bbox_* keys the v6 model doesn't predict.
        overrides["target_preset"] = None
    if cfg.get("weights") is not None:
        overrides["score_weights_json"] = json.dumps(cfg["weights"], separators=(",", ":"))
    mpc = cfg.get("mpc") or {}
    for k, v in mpc.items():
        if k in ("translation_values_m", "rotation_values_deg") and isinstance(v, (list, tuple)):
            overrides[k] = ",".join(str(x) for x in v)
        else:
            overrides[k] = v
    return overrides


def parse_args() -> argparse.Namespace:
    # Pre-parse just --inference_config so we can use it to set defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--inference_config", type=str, default=None,
                     help="YAML preset that supplies target/weights/MPC defaults")
    pre_args, _ = pre.parse_known_args()
    yaml_defaults = _load_inference_config(pre_args.inference_config) if pre_args.inference_config else {}

    parser = argparse.ArgumentParser(description="True Blender-in-the-loop MPC rollout.")
    parser.add_argument("--inference_config", type=str, default=None,
                        help="YAML preset that supplies target/weights/MPC defaults")
    parser.add_argument("--run_dir", type=str, required=True, help="original rendered scene run directory")
    parser.add_argument("--model_path", type=str, required=True, help="trained model directory, usually runs/<run>/final")
    parser.add_argument("--config", type=str, default=None, help="optional training config override")
    parser.add_argument("--blender_bin", type=str, default="blender/blender")
    parser.add_argument("--initial_seed", type=int, default=721)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--translation_values_m", type=str, default="-0.12,0,0.12")
    parser.add_argument("--rotation_values_deg", type=str, default="-3,0,3")
    parser.add_argument("--max_translation_norm_m", type=float, default=0.18)
    parser.add_argument("--max_rotation_norm_deg", type=float, default=4.5)
    parser.add_argument("--max_candidates", type=int, default=720)
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=int(os.environ.get("CANDIDATE_BATCH_SIZE", 96)),
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--target_preset", type=str, default="centered_50")
    parser.add_argument("--target_json", type=str, default=None)
    parser.add_argument("--score_weights_json", type=str, default=None)
    parser.add_argument(
        "--objective_schedule_json",
        type=str,
        default=None,
        help=(
            "Optional JSON list of phases (inline or path). Each phase: "
            "{'until_step': int, 'score_weights': {key: weight, ...}}. "
            "Overrides --score_weights_json when provided; phases apply sequentially by step index."
        ),
    )
    parser.add_argument(
        "--candidate_mode",
        type=str,
        choices=["grid", "axis"],
        default="grid",
        help="'grid' = full cross-product over axes (default); 'axis' = one axis at a time for finer per-axis resolution.",
    )
    parser.add_argument("--translation_penalty_weight", type=float, default=0.0)
    parser.add_argument("--rotation_penalty_weight", type=float, default=0.0)
    parser.add_argument("--parse_failure_penalty", type=float, default=10.0)
    parser.add_argument("--top_k_to_log", type=int, default=10)
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="runs/infer_mpc_blender")
    parser.add_argument("--caption", type=str, default=None, help="GroundingDINO caption; default is taken from annotations")
    parser.add_argument(
        "--evaluate_with_detector",
        action="store_true",
        help="run GroundingDINO on rendered frames to log actual bbox-based scores",
    )
    parser.add_argument(
        "--dino_config",
        type=str,
        default="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--detector_device", type=str, default="cuda")
    parser.add_argument("--list_target_presets", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="vLLM tensor parallel size (default: number of visible GPUs)")
    parser.add_argument("--lookahead_top_k", type=int, default=0,
                        help="Enable 2-step lookahead MPC. Top-K first actions to expand (0=disabled, 30 recommended)")
    parser.add_argument("--disable_adaptive_step", action="store_true",
                        help="Disable distance-based adaptive step size scaling")
    parser.add_argument("--disable_spike_detection", action="store_true",
                        help="Disable c2o spike detection safety guard")
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=4,
        help="CPU thread limit for each Blender subprocess (-t flag)",
    )
    parser.add_argument(
        "--disable_roll",
        action="store_true",
        help="Fix rz=0 in candidate rotation (no roll, pitch+yaw only)",
    )
    parser.add_argument(
        "--force_template",
        action="store_true",
        help="Force the model to emit a valid JSON template via prefix_allowed_tokens_fn "
             "(digits-only at value positions, fixed scaffolding tokens elsewhere). "
             "Skips chain-of-thought entirely and guarantees parseable output.",
    )
    # YAML preset values become the defaults; explicit CLI args still override.
    if yaml_defaults:
        parser.set_defaults(**yaml_defaults)
    return parser.parse_args()


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def torch_dtype_from_name(name: str):
    import torch

    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def load_json(path: str | Path) -> dict | list:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def path_has_processor_files(path: str | Path) -> bool:
    path = Path(path)
    required_names = [
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]
    return any((path / name).exists() for name in required_names)


def resolve_training_config(model_path: str | Path, explicit_config: str | None) -> Path:
    if explicit_config:
        config_path = Path(explicit_config)
        if not config_path.exists():
            raise FileNotFoundError(f"training config not found: {config_path}")
        return config_path

    model_path = Path(model_path)
    if model_path.exists():
        resolved = model_path.resolve()
        candidate_dirs = [resolved]
        candidate_dirs.extend(resolved.parents[:4])
        for directory in candidate_dirs:
            candidate = directory / "config.yaml"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not locate training config automatically. "
        "Pass --config explicitly or use a local model path inside a saved run, "
        "such as runs/<run>/final or runs/<run>/checkpoints/checkpoint-XXXX."
    )


def resolve_processor_source(model_path: str | Path, model_cfg: dict[str, object]) -> str:
    model_path = Path(model_path)
    if model_path.exists() and path_has_processor_files(model_path):
        return str(model_path)
    return str(model_cfg["name_or_path"])


def load_detection_caption(run_dir: Path, explicit_caption: str | None) -> str:
    if explicit_caption:
        return str(explicit_caption)

    for filename in ["annotations_detected.json", "annotations.json"]:
        path = run_dir / filename
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, list) and payload:
            prompt = payload[0].get("prompt")
            if prompt:
                return str(prompt)

    raise ValueError(
        "Could not find a detection caption in the run directory. "
        "Pass --caption explicitly."
    )


def run_blender_render(
    *,
    blender_bin: str,
    run_info_path: Path,
    output_image: Path,
    output_json: Path,
    blender_threads: int = 4,
    sample_initial_pose: bool = False,
    seed: int | None = None,
    position: list[float] | None = None,
    forward: list[float] | None = None,
    up: list[float] | None = None,
) -> dict[str, object]:
    cmd = [
        blender_bin,
        "-b",
        "-t", str(max(1, int(blender_threads))),
        "-P",
        str(REPO_ROOT / "scripts" / "blender_render_pose.py"),
        "--",
        "--run_info_path",
        str(run_info_path),
        "--output_image",
        str(output_image),
        "--output_json",
        str(output_json),
    ]

    if sample_initial_pose:
        cmd.append("--sample_initial_pose")
        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])
    else:
        if position is None or forward is None or up is None:
            raise ValueError("position, forward, and up are required for exact-pose rendering")
        cmd.extend(["--position", *[str(value) for value in position]])
        cmd.extend(["--forward", *[str(value) for value in forward]])
        cmd.extend(["--up", *[str(value) for value in up]])

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Blender render failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return load_json(output_json)


def wait_for_image_ready(image_path: str | Path, timeout_sec: float = 15.0, poll_sec: float = 0.25) -> None:
    image_path = Path(image_path)
    deadline = time.time() + float(timeout_sec)
    last_error: Exception | None = None
    while time.time() < deadline:
        if image_path.exists() and image_path.stat().st_size > 0:
            try:
                with Image.open(image_path) as image:
                    image.load()
                return
            except OSError as exc:
                last_error = exc
        time.sleep(float(poll_sec))
    raise RuntimeError(f"rendered image is not readable after waiting: {image_path}") from last_error


def detect_and_score_image(
    *,
    detector,
    image_path: Path,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    score_keys: list[str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    detections = detector.detect_file(
        image_path=image_path,
        caption=caption,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
    )
    detection_dicts = [det.as_dict() for det in detections]
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    rule_scores = compute_rule_based_scores(
        image_width=image_width,
        image_height=image_height,
        detections=detection_dicts,
    )
    actual_scores = {key: float(rule_scores[key]) for key in score_keys}
    return detection_dicts, actual_scores


def filter_candidates_by_environment(
    *,
    candidates,
    object_position: np.ndarray,
    camera_radius_range: list[float],
    hemisphere: bool,
    max_view_angle_deg: float,
):
    filtered = []
    radius_min, radius_max = sorted(float(value) for value in camera_radius_range)
    for candidate in candidates:
        position = np.asarray(candidate.target_position_world, dtype=np.float32)
        forward = np.asarray(candidate.target_forward_world, dtype=np.float32)
        rel = position - object_position
        radius = float(np.linalg.norm(rel))
        if radius < radius_min or radius > radius_max:
            continue
        if hemisphere and rel[2] < 0.0:
            continue

        target_dir = object_position - position
        target_dir_norm = float(np.linalg.norm(target_dir))
        if target_dir_norm < 1e-8:
            continue
        target_dir = target_dir / target_dir_norm
        dot = float(np.clip(np.dot(forward, target_dir), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        if angle_deg > max_view_angle_deg + 1e-6:
            continue
        filtered.append(candidate)
    return filtered


def limit_candidates_evenly(candidates, max_candidates: int):
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return list(candidates)
    if max_candidates == 1:
        return [candidates[len(candidates) // 2]]

    indices = np.linspace(0, len(candidates) - 1, num=max_candidates)
    chosen_indices = []
    seen = set()
    for value in indices:
        index = int(round(float(value)))
        index = max(0, min(index, len(candidates) - 1))
        if index in seen:
            continue
        seen.add(index)
        chosen_indices.append(index)
    return [candidates[index] for index in chosen_indices]


def mean_absolute_error(a: dict[str, float] | None, b: dict[str, float]) -> float | None:
    if b is None:
        return None
    if a is None:
        return None
    keys = [key for key in b if key in a]
    if not keys:
        return None
    return sum(abs(float(a[key]) - float(b[key])) for key in keys) / len(keys)


def main() -> None:
    args = parse_args()
    if args.list_target_presets:
        print(
            json.dumps(
                {
                    "target_presets": available_target_presets(),
                    "preset_specs": preset_specs(),
                },
                indent=2,
            )
        )
        return

    run_dir = Path(args.run_dir)
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(f"run_info.json not found under {run_dir}")

    config_path = resolve_training_config(args.model_path, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    action_frame = str(data_cfg.get("action_frame", "camera_local"))
    rotation_representation = str(data_cfg.get("rotation_representation", "orientation_6d"))
    score_keys = list(data_cfg.get("target_score_keys", []))
    unsupported_score_keys = [key for key in score_keys if key not in ALL_SUPPORTED_SCORE_KEYS]
    if unsupported_score_keys:
        raise ValueError(f"Unsupported score keys: {', '.join(unsupported_score_keys)}")

    run_info = load_json(run_info_path)
    if not isinstance(run_info, dict):
        raise ValueError(f"invalid run_info payload: {run_info_path}")
    options = dict(run_info.get("options", {}))
    resolution = options.get("resolution", [1024, 768])
    frame_aspect_ratio = float(resolution[0]) / float(resolution[1])

    frame_resolution = (int(resolution[0]), int(resolution[1]))
    if args.objective_schedule_json:
        objective_schedule = build_target_objective_schedule(
            preset_name=args.target_preset,
            target_json_text=args.target_json,
            schedule_json_text=args.objective_schedule_json,
            default_weights_json_text=args.score_weights_json,
            frame_aspect_ratio=frame_aspect_ratio,
            score_keys_hint=score_keys,
            frame_resolution=frame_resolution,
        )
        target_objective = objective_schedule[0][1]
    else:
        target_objective = build_target_objective(
            preset_name=args.target_preset,
            target_json_text=args.target_json,
            score_weights_json_text=args.score_weights_json,
            frame_aspect_ratio=frame_aspect_ratio,
            score_keys_hint=score_keys,
            frame_resolution=frame_resolution,
        )
        objective_schedule = [(int(args.num_steps) + 1, target_objective)]

    # Auto-add body_in_frame_ratio target if model supports it. Apply to every phase.
    if "body_in_frame_ratio" in score_keys:
        for _, phase_objective in objective_schedule:
            if "body_in_frame_ratio" not in phase_objective.score_targets:
                phase_objective.score_targets["body_in_frame_ratio"] = 1.0
                phase_objective.score_weights["body_in_frame_ratio"] = (
                    phase_objective.score_weights.get("body_in_frame_ratio", 1.0)
                )
    unsupported_targets = sorted(set(target_objective.score_targets) - set(score_keys))
    if unsupported_targets:
        raise ValueError(
            "target objective uses keys not predicted by this model config: "
            + ", ".join(unsupported_targets)
        )
    object_position = np.asarray(options["object_position"], dtype=np.float32)
    camera_radius_range = [float(value) for value in options.get("camera_radius_range", [2.0, 10.0])]
    hemisphere = bool(options.get("hemisphere", False))
    yaw_range, pitch_range, _ = [float(value) for value in options.get("camera_direction_offsets", [30.0, 30.0, 30.0])]
    max_view_angle_deg = math.sqrt(yaw_range * yaw_range + pitch_range * pitch_range) + 1.0

    translation_values = parse_csv_floats(args.translation_values_m)
    rotation_values_deg = parse_csv_floats(args.rotation_values_deg)
    rotation_values_rad = [math.radians(value) for value in rotation_values_deg]
    max_rotation_norm_rad = math.radians(float(args.max_rotation_norm_deg))

    import torch
    from transformers import AutoProcessor

    trust_remote_code = bool(model_cfg.get("trust_remote_code", False) or args.trust_remote_code)
    processor_source = resolve_processor_source(args.model_path, model_cfg)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=trust_remote_code,
    )

    use_vllm = args.use_vllm
    if use_vllm:
        from vllm import LLM

        tp_size = args.tensor_parallel_size
        if tp_size is None:
            tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        model = LLM(
            model=args.model_path,
            dtype=model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16")),
            tensor_parallel_size=tp_size,
            trust_remote_code=trust_remote_code,
            max_model_len=int(cfg.get("training", {}).get("max_length", 2048)),
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.9,
        )
        print(f"vLLM loaded (tensor_parallel_size={tp_size})", flush=True)
    else:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch_dtype_from_name(model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16"))),
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        if os.environ.get("DRONE_TORCH_COMPILE", "0") == "1":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile applied (mode=reduce-overhead)", flush=True)
            except Exception as e:
                print(f"torch.compile skipped: {e}", flush=True)

    caption = None
    detector = None
    if args.evaluate_with_detector:
        from src.detectors.detector import GroundingDINODetector

        caption = load_detection_caption(run_dir, args.caption)
        detector = GroundingDINODetector(
            model_config_path=args.dino_config,
            model_checkpoint_path=args.dino_weights,
            device=args.detector_device,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{timestamp}_mpc_blender"
    frames_dir = output_dir / "frames"
    blender_dir = output_dir / "blender"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    blender_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving rollout artifacts under {output_dir}", flush=True)

    source_run_info_copy_path = output_dir / "source_run_info.json"
    with source_run_info_copy_path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    inference_run_info = {
        "created_at": timestamp,
        "command": sys.argv,
        "paths": {
            "run_dir": str(run_dir),
            "source_run_info_path": str(run_info_path),
            "source_run_info_copy_path": str(source_run_info_copy_path),
            "training_config_path": str(config_path),
            "model_path": str(args.model_path),
            "processor_source": str(processor_source),
            "blender_bin": str(args.blender_bin),
            "blender_threads": int(args.blender_threads),
            "output_dir": str(output_dir),
        },
        "planner": {
            "action_frame": action_frame,
            "rotation_representation": rotation_representation,
            "score_keys": score_keys,
            "target_objective": {
                "name": target_objective.name,
                "raw_spec": target_objective.raw_spec,
                "score_targets": target_objective.score_targets,
                "score_weights": target_objective.score_weights,
            },
        },
        "rollout": {
            "initial_seed": int(args.initial_seed),
            "num_steps": int(args.num_steps),
            "translation_values_m": translation_values,
            "rotation_values_deg": rotation_values_deg,
            "max_translation_norm_m": float(args.max_translation_norm_m),
            "max_rotation_norm_deg": float(args.max_rotation_norm_deg),
            "max_candidates": int(args.max_candidates),
            "candidate_batch_size": int(args.candidate_batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "translation_penalty_weight": float(args.translation_penalty_weight),
            "rotation_penalty_weight": float(args.rotation_penalty_weight),
            "parse_failure_penalty": float(args.parse_failure_penalty),
        },
        "detector": {
            "enabled": bool(args.evaluate_with_detector),
            "caption": caption,
            "config_path": str(args.dino_config),
            "weights_path": str(args.dino_weights),
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "device": str(args.detector_device),
        },
    }
    inference_run_info_path = output_dir / "inference_run_info.json"
    with inference_run_info_path.open("w", encoding="utf-8") as f:
        json.dump(inference_run_info, f, indent=2)

    initial_image_path = frames_dir / "frame_0000.png"
    initial_response_path = blender_dir / "frame_0000.json"
    initial_render = run_blender_render(
        blender_bin=args.blender_bin,
        run_info_path=run_info_path,
        output_image=initial_image_path,
        output_json=initial_response_path,
        blender_threads=args.blender_threads,
        sample_initial_pose=True,
        seed=args.initial_seed,
    )
    wait_for_image_ready(initial_image_path)
    current_pose = dict(initial_render["pose"])
    initial_detections = None
    initial_scores = None
    initial_target_error = None
    initial_error_by_key = None
    if detector is not None and caption is not None:
        initial_detections, initial_scores = detect_and_score_image(
            detector=detector,
            image_path=initial_image_path,
            caption=caption,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            score_keys=score_keys,
        )
        initial_target_error, initial_error_by_key = weighted_target_error(
            initial_scores,
            objective_for_step(objective_schedule, 0).score_targets,
            objective_for_step(objective_schedule, 0).score_weights,
        )

    frame_paths = [str(initial_image_path)]
    steps: list[dict[str, object]] = []
    current_image_path = initial_image_path
    current_scores = initial_scores
    current_detections = initial_detections
    prev_best_c2o = None  # for spike detection (fix #1)

    for step_idx in range(int(args.num_steps)):
        step_objective = objective_for_step(objective_schedule, step_idx)
        # --- Fix #4: adaptive step size based on distance to subject ---
        cam_pos = np.asarray(current_pose["position"], dtype=np.float32)
        dist_to_obj = float(np.linalg.norm(cam_pos - object_position))
        if args.disable_adaptive_step:
            step_scale = 1.0
        else:
            step_scale = float(np.clip(dist_to_obj / 4.0, 0.3, 1.0))
        scaled_translation_values = [v * step_scale for v in translation_values]
        scaled_max_translation_norm = float(args.max_translation_norm_m) * step_scale

        with Image.open(current_image_path) as current_image_raw:
            current_image = current_image_raw.convert("RGB")
            candidates = generate_local_candidate_actions(
                position=cam_pos,
                forward=np.asarray(current_pose["forward"], dtype=np.float32),
                up=np.asarray(current_pose["up"], dtype=np.float32),
                translation_values=scaled_translation_values,
                rotation_values_rad=rotation_values_rad,
                max_translation_norm=scaled_max_translation_norm,
                max_rotation_norm_rad=max_rotation_norm_rad,
                action_frame=action_frame,
                disable_roll=bool(args.disable_roll),
                rotation_representation=rotation_representation,
                mode=str(args.candidate_mode),
            )
            candidates = filter_candidates_by_environment(
                candidates=candidates,
                object_position=object_position,
                camera_radius_range=camera_radius_range,
                hemisphere=hemisphere,
                max_view_angle_deg=max_view_angle_deg,
            )
            candidates_before_limit = len(candidates)
            candidates = limit_candidates_evenly(candidates, int(args.max_candidates))
            if not candidates:
                raise RuntimeError("no valid candidates remain after environment filtering")

            if use_vllm:
                scored_candidates = score_action_candidates_vllm(
                    llm=model,
                    processor=processor,
                    image=current_image,
                    candidates=candidates,
                    target_score_keys=score_keys,
                    action_frame=action_frame,
                    rotation_representation=rotation_representation,
                    max_new_tokens=int(args.max_new_tokens),
                )
            else:
                scored_candidates = score_action_candidates(
                    model=model,
                    processor=processor,
                    image=current_image,
                    candidates=candidates,
                    target_score_keys=score_keys,
                    action_frame=action_frame,
                    rotation_representation=rotation_representation,
                    max_new_tokens=int(args.max_new_tokens),
                    candidate_batch_size=int(args.candidate_batch_size),
                    force_template=bool(args.force_template),
                )
        parse_success_count = sum(1 for item in scored_candidates if item.predicted_scores is not None)
        if parse_success_count == 0:
            raise RuntimeError(
                "Failed to parse every candidate prediction. "
                "This usually means the model output was truncated or not valid JSON. "
                "Try increasing --max_new_tokens, or pass --force_template."
            )

        ranked_candidates: list[dict[str, object]] = []
        for item in scored_candidates:
            parse_success = item.predicted_scores is not None
            if parse_success:
                target_error, error_by_key = weighted_target_error(
                    item.predicted_scores,
                    step_objective.score_targets,
                    step_objective.score_weights,
                )
            else:
                target_error = float(args.parse_failure_penalty)
                error_by_key = {}

            objective_value = (
                target_error
                + float(args.translation_penalty_weight) * float(item.candidate.translation_norm)
                + float(args.rotation_penalty_weight) * float(item.candidate.rotation_norm_rad)
            )
            ranked_candidates.append(
                {
                    "action_text": item.candidate.action_text,
                    "delta_position_local": list(item.candidate.delta_position_local),
                    "delta_rotation_local_rad": list(item.candidate.delta_rotation_local),
                    "target_position_world": list(item.candidate.target_position_world),
                    "target_forward_world": list(item.candidate.target_forward_world),
                    "target_up_world": list(item.candidate.target_up_world),
                    "translation_norm": float(item.candidate.translation_norm),
                    "rotation_norm_rad": float(item.candidate.rotation_norm_rad),
                    "generated_text": item.generated_text,
                    "parse_success": parse_success,
                    "predicted_scores": item.predicted_scores,
                    "target_error": float(target_error),
                    "error_by_key": error_by_key,
                    "objective_value": float(objective_value),
                }
            )

        ranked_candidates.sort(key=lambda row: float(row["objective_value"]))

        # --- Multi-step lookahead MPC ---
        if use_vllm and args.lookahead_top_k > 0:
            lookahead_errors = score_with_lookahead_vllm(
                llm=model,
                processor=processor,
                image=current_image,
                first_step_scores=scored_candidates,
                current_position=cam_pos,
                current_forward=np.asarray(current_pose["forward"], dtype=np.float32),
                current_up=np.asarray(current_pose["up"], dtype=np.float32),
                target_score_keys=score_keys,
                score_targets=step_objective.score_targets,
                score_weights=step_objective.score_weights,
                action_frame=action_frame,
                rotation_representation=rotation_representation,
                max_new_tokens=int(args.max_new_tokens),
                top_k=int(args.lookahead_top_k),
            )
            if lookahead_errors:
                # Re-rank: blend 1-step error with lookahead error (uniform scale)
                for rc in ranked_candidates:
                    la_err = lookahead_errors.get(rc["action_text"])
                    if la_err is not None:
                        rc["lookahead_error"] = la_err
                    else:
                        la_err = rc["target_error"]  # conservative: assume no improvement
                    rc["objective_value"] = 0.5 * rc["target_error"] + 0.5 * la_err
                ranked_candidates.sort(key=lambda row: float(row["objective_value"]))
                print(f"  [lookahead] re-ranked with {len(lookahead_errors)} lookahead scores", flush=True)

        # --- Fix #3: margin lower bound filter ---
        # Reject candidates where person is predicted to be cut off
        MIN_MARGIN = 0.03
        margin_keys = [k for k in ["bbox_margin_top", "bbox_margin_bottom"] if k in score_keys]
        if margin_keys:
            safe_candidates = [
                rc for rc in ranked_candidates
                if rc["parse_success"] and all(
                    rc["predicted_scores"].get(k, 1.0) >= MIN_MARGIN for k in margin_keys
                )
            ]
            if safe_candidates:
                ranked_candidates = safe_candidates

        best = ranked_candidates[0]

        # --- Fix #1: prediction spike detection + rollback ---
        # If c2o prediction changes drastically from previous step,
        # the model is likely confused — prefer a retreat action
        C2O_SPIKE_THRESHOLD = 0.5
        if not args.disable_spike_detection and prev_best_c2o and best["predicted_scores"] is not None:
            c2o_keys = ["camera_to_object_fx", "camera_to_object_fy", "camera_to_object_fz"]
            c2o_keys_present = [k for k in c2o_keys if k in best["predicted_scores"]]
            max_delta = max(
                (abs(best["predicted_scores"].get(k, 0) - prev_best_c2o.get(k, 0))
                 for k in c2o_keys_present),
                default=0.0,
            )
            if max_delta > C2O_SPIKE_THRESHOLD:
                # Find the "stay" candidate (smallest translation norm)
                stay_candidates = sorted(
                    [rc for rc in ranked_candidates if rc["parse_success"]],
                    key=lambda rc: rc["translation_norm"],
                )
                if stay_candidates:
                    best = stay_candidates[0]
                    print(f"  [safety] c2o spike detected (delta={max_delta:.2f}), using minimal-move action", flush=True)

        if best["predicted_scores"] is not None:
            prev_best_c2o = {k: v for k, v in best["predicted_scores"].items()
                             if k.startswith("camera_to_object_")}

        next_image_path = frames_dir / f"frame_{step_idx + 1:04d}.png"
        next_response_path = blender_dir / f"frame_{step_idx + 1:04d}.json"
        next_render = run_blender_render(
            blender_bin=args.blender_bin,
            run_info_path=run_info_path,
            output_image=next_image_path,
            output_json=next_response_path,
            blender_threads=args.blender_threads,
            position=best["target_position_world"],
            forward=best["target_forward_world"],
            up=best["target_up_world"],
        )
        wait_for_image_ready(next_image_path)
        next_pose = dict(next_render["pose"])
        next_detections = None
        next_scores = None
        actual_target_error = None
        actual_error_by_key = None
        if detector is not None and caption is not None:
            next_detections, next_scores = detect_and_score_image(
                detector=detector,
                image_path=next_image_path,
                caption=caption,
                box_threshold=float(args.box_threshold),
                text_threshold=float(args.text_threshold),
                score_keys=score_keys,
            )
            actual_target_error, actual_error_by_key = weighted_target_error(
                next_scores,
                step_objective.score_targets,
                step_objective.score_weights,
            )

        steps.append(
            {
                "step_index": step_idx,
                "objective_name": step_objective.name,
                "objective_weights": step_objective.score_weights,
                "current_image_path": str(current_image_path),
                "current_pose": current_pose,
                "current_actual_scores": current_scores,
                "current_detections": current_detections,
                "candidate_count_before_limit": int(candidates_before_limit),
                "candidate_count_scored": int(len(scored_candidates)),
                "parse_success_count": int(parse_success_count),
                "chosen_action": best,
                "next_image_path": str(next_image_path),
                "next_pose": next_pose,
                "next_actual_scores": next_scores,
                "next_detections": next_detections,
                "next_target_error": None if actual_target_error is None else float(actual_target_error),
                "next_error_by_key": actual_error_by_key,
                "prediction_vs_actual_mae": mean_absolute_error(best["predicted_scores"], next_scores),
                "top_candidates": ranked_candidates[: max(1, int(args.top_k_to_log))],
            }
        )

        current_image_path = next_image_path
        current_pose = next_pose
        current_scores = next_scores
        current_detections = next_detections
        frame_paths.append(str(next_image_path))

    video_info = write_rollout_video(
        frame_paths=frame_paths,
        output_dir=output_dir,
        fps=int(args.video_fps),
    )

    final_objective = objective_for_step(objective_schedule, max(0, int(args.num_steps) - 1))
    summary_image_path = write_rollout_summary_image(
        frame_paths=frame_paths,
        steps=steps,
        score_targets=final_objective.score_targets,
        score_weights=final_objective.score_weights,
        output_dir=output_dir,
    )
    if summary_image_path:
        print(f"Summary image saved: {summary_image_path}", flush=True)

    final_target_error = None
    final_error_by_key = None
    if current_scores is not None:
        final_target_error, final_error_by_key = weighted_target_error(
            current_scores,
            final_objective.score_targets,
            final_objective.score_weights,
        )
    summary = {
        "run_dir": str(run_dir),
        "run_info_path": str(run_info_path),
        "scene_path": str(run_info["input_scene"]),
        "training_config_path": str(config_path),
        "model_path": str(args.model_path),
        "processor_source": str(processor_source),
        "blender_bin": str(args.blender_bin),
        "action_frame": action_frame,
        "rotation_representation": rotation_representation,
        "score_keys": score_keys,
        "evaluate_with_detector": bool(args.evaluate_with_detector),
        "detection_caption": caption,
        "candidate_mode": str(args.candidate_mode),
        "target_objective": {
            "name": target_objective.name,
            "raw_spec": target_objective.raw_spec,
            "score_targets": target_objective.score_targets,
            "score_weights": target_objective.score_weights,
        },
        "objective_schedule": [
            {
                "until_step": until_step,
                "name": objective.name,
                "score_weights": objective.score_weights,
            }
            for until_step, objective in objective_schedule
        ],
        "environment": {
            "object_position": [float(value) for value in object_position.tolist()],
            "camera_radius_range": camera_radius_range,
            "hemisphere": hemisphere,
            "camera_direction_offsets_deg": options.get("camera_direction_offsets"),
            "max_view_angle_deg": float(max_view_angle_deg),
        },
        "rollout": {
            "initial_seed": int(args.initial_seed),
            "num_steps": int(args.num_steps),
            "translation_values_m": translation_values,
            "rotation_values_deg": rotation_values_deg,
            "max_translation_norm_m": float(args.max_translation_norm_m),
            "max_rotation_norm_deg": float(args.max_rotation_norm_deg),
            "max_candidates": int(args.max_candidates),
            "candidate_batch_size": int(args.candidate_batch_size),
            "translation_penalty_weight": float(args.translation_penalty_weight),
            "rotation_penalty_weight": float(args.rotation_penalty_weight),
            "parse_failure_penalty": float(args.parse_failure_penalty),
            "initial_pose": initial_render["pose"],
            "initial_actual_scores": initial_scores,
            "initial_detections": initial_detections,
            "initial_target_error": None if initial_target_error is None else float(initial_target_error),
            "initial_error_by_key": initial_error_by_key,
            "final_pose": current_pose,
            "final_actual_scores": current_scores,
            "final_detections": current_detections,
            "final_target_error": None if final_target_error is None else float(final_target_error),
            "final_error_by_key": final_error_by_key,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "trajectory.json"),
            "inference_run_info_path": str(inference_run_info_path),
            "source_run_info_copy_path": str(source_run_info_copy_path),
            "frames_dir": str(frames_dir),
            "frame_paths": frame_paths,
            "blender_dir": str(blender_dir),
            **video_info,
        },
        "steps": steps,
    }

    summary_path = output_dir / "trajectory.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
