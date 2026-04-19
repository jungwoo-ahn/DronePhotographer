from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.mpc import (
    find_nearest_view,
    generate_local_candidate_actions,
    load_planner_views,
    save_frame_copy,
    score_action_candidates,
    score_action_candidates_vllm,
    write_rollout_video,
)
from src.vlm_qwen25.objective import (
    available_target_presets,
    build_target_objective,
    preset_specs,
    weighted_target_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen35_vl_2b_1xh200.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--translation_values_m", type=str, default="-0.25,0,0.25")
    parser.add_argument("--rotation_values_deg", type=str, default="-6,0,6")
    parser.add_argument("--max_translation_norm_m", type=float, default=0.5)
    parser.add_argument("--max_rotation_norm_deg", type=float, default=10.0)
    parser.add_argument(
        "--candidate_batch_size",
        type=int,
        default=int(os.environ.get("CANDIDATE_BATCH_SIZE", 96)),
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--target_preset", type=str, default="centered_50")
    parser.add_argument("--target_json", type=str, default=None)
    parser.add_argument("--score_weights_json", type=str, default=None)
    parser.add_argument("--translation_penalty_weight", type=float, default=0.05)
    parser.add_argument("--rotation_penalty_weight", type=float, default=0.05)
    parser.add_argument("--parse_failure_penalty", type=float, default=10.0)
    parser.add_argument("--snap_position_weight", type=float, default=1.0)
    parser.add_argument("--snap_rotation_weight", type=float, default=0.35)
    parser.add_argument("--top_k_to_log", type=int, default=10)
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="runs/infer_mpc")
    parser.add_argument("--list_target_presets", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="vLLM tensor parallel size (default: number of visible GPUs)")
    return parser.parse_args()


def parse_csv_floats(text: str) -> list[float]:
    values = [part.strip() for part in str(text).split(",")]
    return [float(value) for value in values if value]


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

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import torch
    from transformers import AutoProcessor

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    action_frame = str(data_cfg.get("action_frame", "camera_local"))
    rotation_representation = str(data_cfg.get("rotation_representation", "orientation_6d"))
    score_keys = list(data_cfg.get("target_score_keys", []))
    target_objective = build_target_objective(
        preset_name=args.target_preset,
        target_json_text=args.target_json,
        score_weights_json_text=args.score_weights_json,
    )

    unsupported_targets = sorted(set(target_objective.score_targets) - set(score_keys))
    if unsupported_targets:
        raise ValueError(
            "target objective uses keys not predicted by this config: "
            + ", ".join(unsupported_targets)
        )

    translation_values = parse_csv_floats(args.translation_values_m)
    rotation_values_rad = [math.radians(value) for value in parse_csv_floats(args.rotation_values_deg)]
    max_rotation_norm_rad = math.radians(float(args.max_rotation_norm_deg))

    views = load_planner_views(
        annotations_path=data_cfg["annotations_path"],
        image_root=data_cfg.get("image_root"),
        target_score_keys=score_keys,
    )
    if args.start_index < 0 or args.start_index >= len(views):
        raise IndexError(f"start_index {args.start_index} is out of range for {len(views)} views")

    trust_remote_code = bool(model_cfg.get("trust_remote_code", False) or args.trust_remote_code)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
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
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile applied (mode=reduce-overhead)", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{timestamp}_mpc_rollout"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    current_view = views[args.start_index]
    frame_paths: list[str] = []
    save_frame_copy(current_view.image_path, frames_dir / "frame_0000.png")
    frame_paths.append(str(frames_dir / "frame_0000.png"))

    steps: list[dict[str, object]] = []
    initial_error, initial_error_by_key = weighted_target_error(
        current_view.scores,
        target_objective.score_targets,
        target_objective.score_weights,
    )

    for step_idx in range(int(args.num_steps)):
        with Image.open(current_view.image_path) as current_image_raw:
            current_image = current_image_raw.convert("RGB")
            candidates = generate_local_candidate_actions(
                position=current_view.camera_position,
                forward=current_view.camera_forward,
                up=current_view.camera_up,
                translation_values=translation_values,
                rotation_values_rad=rotation_values_rad,
                max_translation_norm=float(args.max_translation_norm_m),
                max_rotation_norm_rad=max_rotation_norm_rad,
                action_frame=action_frame,
                rotation_representation=rotation_representation,
            )
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
                )

        ranked_candidates: list[dict[str, object]] = []
        for item in scored_candidates:
            parse_success = item.predicted_scores is not None
            if parse_success:
                target_error, error_by_key = weighted_target_error(
                    item.predicted_scores,
                    target_objective.score_targets,
                    target_objective.score_weights,
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
        best = ranked_candidates[0]

        snap_index, snap_cost = find_nearest_view(
            views=views,
            target_position=best["target_position_world"],
            target_forward=best["target_forward_world"],
            target_up=best["target_up_world"],
            position_weight=float(args.snap_position_weight),
            rotation_weight=float(args.snap_rotation_weight),
        )
        next_view = views[snap_index]
        actual_error, actual_error_by_key = weighted_target_error(
            next_view.scores,
            target_objective.score_targets,
            target_objective.score_weights,
        )

        frame_path = frames_dir / f"frame_{step_idx + 1:04d}.png"
        save_frame_copy(next_view.image_path, frame_path)
        frame_paths.append(str(frame_path))

        steps.append(
            {
                "step_index": step_idx,
                "current_view_index": current_view.index,
                "current_image_path": str(current_view.image_path),
                "current_actual_scores": current_view.scores,
                "chosen_action": best,
                "snapped_next_view_index": next_view.index,
                "snapped_next_image_path": str(next_view.image_path),
                "snapped_next_actual_scores": next_view.scores,
                "snapped_next_target_error": float(actual_error),
                "snapped_next_error_by_key": actual_error_by_key,
                "snap_cost": float(snap_cost),
                "top_candidates": ranked_candidates[: max(1, int(args.top_k_to_log))],
            }
        )
        current_view = next_view

    video_info = write_rollout_video(
        frame_paths=frame_paths,
        output_dir=output_dir,
        fps=int(args.video_fps),
    )

    summary = {
        "config_path": str(args.config),
        "model_path": str(args.model_path),
        "action_frame": action_frame,
        "rotation_representation": rotation_representation,
        "score_keys": score_keys,
        "target_objective": {
            "name": target_objective.name,
            "raw_spec": target_objective.raw_spec,
            "score_targets": target_objective.score_targets,
            "score_weights": target_objective.score_weights,
        },
        "rollout": {
            "start_index": int(args.start_index),
            "num_steps": int(args.num_steps),
            "translation_values_m": translation_values,
            "rotation_values_deg": parse_csv_floats(args.rotation_values_deg),
            "max_translation_norm_m": float(args.max_translation_norm_m),
            "max_rotation_norm_deg": float(args.max_rotation_norm_deg),
            "candidate_batch_size": int(args.candidate_batch_size),
            "translation_penalty_weight": float(args.translation_penalty_weight),
            "rotation_penalty_weight": float(args.rotation_penalty_weight),
            "parse_failure_penalty": float(args.parse_failure_penalty),
            "snap_position_weight": float(args.snap_position_weight),
            "snap_rotation_weight": float(args.snap_rotation_weight),
            "initial_view_index": int(views[args.start_index].index),
            "initial_image_path": str(views[args.start_index].image_path),
            "initial_actual_scores": views[args.start_index].scores,
            "initial_target_error": float(initial_error),
            "initial_error_by_key": initial_error_by_key,
            "final_view_index": int(current_view.index),
            "final_image_path": str(current_view.image_path),
            "final_actual_scores": current_view.scores,
            "final_target_error": float(
                weighted_target_error(
                    current_view.scores,
                    target_objective.score_targets,
                    target_objective.score_weights,
                )[0]
            ),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "trajectory.json"),
            "frames_dir": str(frames_dir),
            "frame_paths": frame_paths,
            **video_info,
        },
        "steps": steps,
    }

    summary_path = output_dir / "trajectory.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
