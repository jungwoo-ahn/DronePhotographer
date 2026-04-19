from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .prompt import build_action_text, build_user_prompt
from .rotation_utils import (
    apply_camera_local_action,
    make_camera_basis_from_forward_up,
    orthonormalize_forward_up,
    relative_translation_camera_local,
    target_orientation_forward_up_camera_local,
    target_orientation_forward_up_world,
)
from .schema import extract_scores_from_annotation, parse_scores_from_text


@dataclass(frozen=True)
class PlannerView:
    index: int
    image_path: Path
    camera_position: np.ndarray
    camera_forward: np.ndarray
    camera_up: np.ndarray
    has_detection: bool
    scores: dict[str, float]


@dataclass(frozen=True)
class CandidateAction:
    action_text: str
    delta_position_local: tuple[float, float, float]
    delta_rotation_local: tuple[float, float, float]
    target_position_world: tuple[float, float, float]
    target_forward_world: tuple[float, float, float]
    target_up_world: tuple[float, float, float]
    translation_norm: float
    rotation_norm_rad: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: CandidateAction
    generated_text: str
    predicted_scores: dict[str, float] | None


def load_planner_views(
    annotations_path: str | Path,
    image_root: str | Path | None,
    target_score_keys: Sequence[str],
) -> list[PlannerView]:
    annotations_path = Path(annotations_path)
    image_root_path = Path(image_root) if image_root is not None else annotations_path.parent
    with annotations_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    views: list[PlannerView] = []
    for index, item in enumerate(raw):
        image_path = image_root_path / str(item["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"image missing: {image_path}")
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        scores = extract_scores_from_annotation(
            annotation=item,
            image_width=image_width,
            image_height=image_height,
            score_keys=target_score_keys,
        )
        forward, up = orthonormalize_forward_up(
            np.asarray(item.get("final_forward", item.get("base_forward")), dtype=np.float32),
            np.asarray(item.get("final_up", item.get("base_up")), dtype=np.float32),
        )
        views.append(
            PlannerView(
                index=index,
                image_path=image_path,
                camera_position=np.asarray(item["camera_position"], dtype=np.float32),
                camera_forward=forward,
                camera_up=up,
                has_detection=bool(item.get("detections")),
                scores=scores,
            )
        )
    return views


def _rotation_local_to_world(rotvec_local: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    basis = make_camera_basis_from_forward_up(forward, up)
    return (basis @ np.asarray(rotvec_local, dtype=np.float32)).astype(np.float32)


def generate_local_candidate_actions(
    *,
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    translation_values: Sequence[float],
    rotation_values_rad: Sequence[float],
    max_translation_norm: float,
    max_rotation_norm_rad: float,
    action_frame: str,
    rotation_representation: str,
    disable_roll: bool = False,
) -> list[CandidateAction]:
    translation_values = [float(value) for value in translation_values]
    rotation_values_rad = [float(value) for value in rotation_values_rad]
    if not any(abs(value) < 1e-8 for value in translation_values):
        translation_values.append(0.0)
    if not any(abs(value) < 1e-8 for value in rotation_values_rad):
        rotation_values_rad.append(0.0)
    translation_values = sorted(set(translation_values))
    rotation_values_rad = sorted(set(rotation_values_rad))
    if action_frame not in {"camera_local", "world"}:
        raise ValueError("action_frame must be 'camera_local' or 'world'")
    if rotation_representation not in {"orientation_6d", "rotvec"}:
        raise ValueError("rotation_representation must be 'orientation_6d' or 'rotvec'")

    dedup: dict[str, CandidateAction] = {}
    for dx in translation_values:
        for dy in translation_values:
            for dz in translation_values:
                delta_position_local = np.asarray([dx, dy, dz], dtype=np.float32)
                translation_norm = float(np.linalg.norm(delta_position_local))
                if translation_norm > max_translation_norm + 1e-8:
                    continue
                roll_values = [0.0] if disable_roll else rotation_values_rad
                for rx in rotation_values_rad:
                    for ry in rotation_values_rad:
                        for rz in roll_values:
                            delta_rotation_local = np.asarray([rx, ry, rz], dtype=np.float32)
                            rotation_norm_rad = float(np.linalg.norm(delta_rotation_local))
                            if rotation_norm_rad > max_rotation_norm_rad + 1e-8:
                                continue

                            next_position_world, next_forward_world, next_up_world = apply_camera_local_action(
                                position=position,
                                forward=forward,
                                up=up,
                                delta_position_local=delta_position_local,
                                delta_rotation_local=delta_rotation_local,
                            )

                            if action_frame == "camera_local":
                                delta_position_for_text = tuple(delta_position_local.tolist())
                                if rotation_representation == "orientation_6d":
                                    target_forward, target_up = target_orientation_forward_up_camera_local(
                                        forward,
                                        up,
                                        next_forward_world,
                                        next_up_world,
                                    )
                                    action_text = build_action_text(
                                        delta_position=delta_position_for_text,
                                        action_frame=action_frame,
                                        rotation_representation=rotation_representation,
                                        target_forward=tuple(target_forward.tolist()),
                                        target_up=tuple(target_up.tolist()),
                                    )
                                else:
                                    action_text = build_action_text(
                                        delta_position=delta_position_for_text,
                                        action_frame=action_frame,
                                        rotation_representation=rotation_representation,
                                        delta_rotation=tuple(delta_rotation_local.tolist()),
                                    )
                            else:
                                basis = make_camera_basis_from_forward_up(forward, up)
                                delta_position_world = (basis @ delta_position_local).astype(np.float32)
                                if rotation_representation == "orientation_6d":
                                    target_forward, target_up = target_orientation_forward_up_world(
                                        next_forward_world,
                                        next_up_world,
                                    )
                                    action_text = build_action_text(
                                        delta_position=tuple(delta_position_world.tolist()),
                                        action_frame=action_frame,
                                        rotation_representation=rotation_representation,
                                        target_forward=tuple(target_forward.tolist()),
                                        target_up=tuple(target_up.tolist()),
                                    )
                                else:
                                    delta_rotation_world = _rotation_local_to_world(delta_rotation_local, forward, up)
                                    action_text = build_action_text(
                                        delta_position=tuple(delta_position_world.tolist()),
                                        action_frame=action_frame,
                                        rotation_representation=rotation_representation,
                                        delta_rotation=tuple(delta_rotation_world.tolist()),
                                    )

                            dedup[action_text] = CandidateAction(
                                action_text=action_text,
                                delta_position_local=tuple(float(v) for v in delta_position_local.tolist()),
                                delta_rotation_local=tuple(float(v) for v in delta_rotation_local.tolist()),
                                target_position_world=tuple(float(v) for v in next_position_world.tolist()),
                                target_forward_world=tuple(float(v) for v in next_forward_world.tolist()),
                                target_up_world=tuple(float(v) for v in next_up_world.tolist()),
                                translation_norm=translation_norm,
                                rotation_norm_rad=rotation_norm_rad,
                            )
    return list(dedup.values())


def score_action_candidates(
    *,
    model,
    processor,
    image: Image.Image,
    candidates: Sequence[CandidateAction],
    target_score_keys: Sequence[str],
    action_frame: str,
    rotation_representation: str,
    max_new_tokens: int,
    candidate_batch_size: int,
) -> list[CandidateScore]:
    import torch

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if not candidates:
        raise ValueError("candidate list is empty")
    scored: list[CandidateScore] = []
    for start in range(0, len(candidates), candidate_batch_size):
        batch_candidates = list(candidates[start : start + candidate_batch_size])
        texts: list[str] = []
        for candidate in batch_candidates:
            user_prompt = build_user_prompt(
                candidate.action_text,
                target_score_keys=target_score_keys,
                action_frame=action_frame,
                rotation_representation=rotation_representation,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        batch_images = [image] * len(batch_candidates)
        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(processor.tokenizer, "eos_token_id", None)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=pad_token_id,
            )

        for row, candidate in enumerate(batch_candidates):
            prompt_len = int(inputs["attention_mask"][row].sum().item())
            generated_text = processor.decode(
                generated_ids[row, prompt_len:],
                skip_special_tokens=True,
            )
            predicted_scores = parse_scores_from_text(generated_text, score_keys=target_score_keys)
            scored.append(
                CandidateScore(
                    candidate=candidate,
                    generated_text=generated_text,
                    predicted_scores=predicted_scores,
                )
            )
    return scored


def score_action_candidates_vllm(
    *,
    llm,
    processor,
    image: Image.Image,
    candidates: Sequence[CandidateAction],
    target_score_keys: Sequence[str],
    action_frame: str,
    rotation_representation: str,
    max_new_tokens: int,
) -> list[CandidateScore]:
    """Score candidates using vLLM for faster inference."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=int(max_new_tokens),
    )

    prompts = []
    for candidate in candidates:
        user_prompt = build_user_prompt(
            candidate.action_text,
            target_score_keys=target_score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append({
            "prompt": text,
            "multi_modal_data": {"image": image},
        })

    outputs = llm.generate(prompts, sampling_params)

    scored: list[CandidateScore] = []
    for output, candidate in zip(outputs, candidates):
        generated_text = output.outputs[0].text
        predicted_scores = parse_scores_from_text(generated_text, score_keys=target_score_keys)
        scored.append(
            CandidateScore(
                candidate=candidate,
                generated_text=generated_text,
                predicted_scores=predicted_scores,
            )
        )
    return scored


def score_with_lookahead_vllm(
    *,
    llm,
    processor,
    image: Image.Image,
    first_step_scores: list[CandidateScore],
    current_position: np.ndarray,
    current_forward: np.ndarray,
    current_up: np.ndarray,
    target_score_keys: Sequence[str],
    score_targets: dict[str, float],
    score_weights: dict[str, float],
    action_frame: str,
    rotation_representation: str,
    max_new_tokens: int,
    top_k: int = 30,
    followup_per_candidate: int = 50,
    followup_translation_values: Sequence[float] = (-0.15, 0.0, 0.15),
    followup_rotation_values_deg: Sequence[float] = (-4.0, 0.0, 4.0),
) -> dict[str, float]:
    """2-step lookahead: score composed actions and return best lookahead error per first action.

    Returns a dict mapping first-step action_text → best 2-step weighted target error.
    """
    import math
    from vllm import SamplingParams
    from .objective import weighted_target_error

    sampling_params = SamplingParams(temperature=0, max_tokens=int(max_new_tokens))

    # Take top-K first-step candidates by 1-step weighted target error
    scored_with_err = []
    for s in first_step_scores:
        if s.predicted_scores is not None:
            err, _ = weighted_target_error(s.predicted_scores, score_targets, score_weights)
            scored_with_err.append((err, s))
    scored_with_err.sort(key=lambda x: x[0])
    top_candidates = [s for _, s in scored_with_err[:top_k]]
    if not top_candidates:
        return {}

    followup_rotation_rad = [math.radians(d) for d in followup_rotation_values_deg]
    followup_translation = list(followup_translation_values)

    # Generate follow-up candidates from each top-K resulting pose
    composed_prompts = []
    composed_map: list[tuple[str, CandidateAction]] = []  # (first_action_text, followup_candidate)

    for first in top_candidates:
        fc = first.candidate
        mid_pos = np.asarray(fc.target_position_world, dtype=np.float32)
        mid_fwd = np.asarray(fc.target_forward_world, dtype=np.float32)
        mid_up = np.asarray(fc.target_up_world, dtype=np.float32)

        followups = generate_local_candidate_actions(
            position=mid_pos,
            forward=mid_fwd,
            up=mid_up,
            translation_values=followup_translation,
            rotation_values_rad=followup_rotation_rad,
            max_translation_norm=0.25,
            max_rotation_norm_rad=math.radians(6.0),
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            disable_roll=True,
        )
        # Subsample if too many
        if len(followups) > followup_per_candidate:
            step = max(1, len(followups) // followup_per_candidate)
            followups = followups[::step][:followup_per_candidate]

        for fu in followups:
            # Compose: current pose → final pose (fu's target) as single action
            final_pos = np.asarray(fu.target_position_world, dtype=np.float32)
            final_fwd = np.asarray(fu.target_forward_world, dtype=np.float32)
            final_up = np.asarray(fu.target_up_world, dtype=np.float32)

            if action_frame == "camera_local":
                delta = relative_translation_camera_local(
                    current_position, final_pos, current_forward, current_up,
                )
                target_fwd_out, target_up_out = target_orientation_forward_up_camera_local(
                    current_forward, current_up, final_fwd, final_up,
                )
            else:  # world
                delta = (final_pos - current_position).astype(np.float32)
                target_fwd_out, target_up_out = target_orientation_forward_up_world(
                    final_fwd, final_up,
                )

            composed_text = build_action_text(
                delta_position=tuple(float(v) for v in delta.tolist()),
                action_frame=action_frame,
                rotation_representation=rotation_representation,
                target_forward=tuple(float(v) for v in target_fwd_out.tolist()),
                target_up=tuple(float(v) for v in target_up_out.tolist()),
            )

            user_prompt = build_user_prompt(
                composed_text,
                target_score_keys=target_score_keys,
                action_frame=action_frame,
                rotation_representation=rotation_representation,
            )
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            composed_prompts.append({
                "prompt": text,
                "multi_modal_data": {"image": image},
            })
            composed_map.append((fc.action_text, fu))

    if not composed_prompts:
        return {}

    # Score all composed actions in one batch
    outputs = llm.generate(composed_prompts, sampling_params)

    # For each first action, find the best 2-step weighted target error
    best_lookahead: dict[str, float] = {}
    for (first_action_text, _fu), output in zip(composed_map, outputs):
        gen_text = output.outputs[0].text
        pred = parse_scores_from_text(gen_text, score_keys=target_score_keys)
        if pred is None:
            continue
        err, _ = weighted_target_error(pred, score_targets, score_weights)
        if first_action_text not in best_lookahead or err < best_lookahead[first_action_text]:
            best_lookahead[first_action_text] = err

    return best_lookahead


def find_nearest_view(
    *,
    views: Sequence[PlannerView],
    target_position: np.ndarray,
    target_forward: np.ndarray,
    target_up: np.ndarray,
    position_weight: float = 1.0,
    rotation_weight: float = 0.35,
) -> tuple[int, float]:
    positions = np.stack([view.camera_position for view in views], axis=0)
    forwards = np.stack([view.camera_forward for view in views], axis=0)
    ups = np.stack([view.camera_up for view in views], axis=0)

    target_position = np.asarray(target_position, dtype=np.float32)
    target_forward = np.asarray(target_forward, dtype=np.float32)
    target_up = np.asarray(target_up, dtype=np.float32)

    position_cost = np.linalg.norm(positions - target_position[None, :], axis=1)
    forward_dot = np.clip(forwards @ target_forward, -1.0, 1.0)
    up_dot = np.clip(ups @ target_up, -1.0, 1.0)
    rotation_cost = np.arccos(forward_dot) + 0.5 * np.arccos(up_dot)
    total_cost = float(position_weight) * position_cost + float(rotation_weight) * rotation_cost
    best_index = int(np.argmin(total_cost))
    return best_index, float(total_cost[best_index])


def _describe_targets(score_targets: dict[str, float], score_weights: dict[str, float]) -> str:
    """Generate a human-readable one-line description of the target composition."""
    active = {k: v for k, v in score_targets.items() if score_weights.get(k, 0) > 0}
    parts: list[str] = []

    # Bbox description
    occ = active.get("bbox_occupancy_ratio")
    if occ is not None:
        parts.append(f"size {occ:.0%}")
    off = active.get("bbox_centroid_offset")
    if off is not None and off < 0.05:
        parts.append("centered")

    # C2O description
    fy = active.get("camera_to_object_fy")
    fx = active.get("camera_to_object_fx")
    fz = active.get("camera_to_object_fz")
    uy = active.get("camera_to_object_uy")

    if fy is not None:
        if fy > 0.5:
            facing = "front"
        elif fy < -0.5:
            facing = "back"
        else:
            facing = ""
        # obj_fwd points backward, so fx>0 means camera sees left profile
        if fx is not None and abs(fx) > 0.3:
            side = "left-" if fx > 0 else "right-"
            facing = side + facing if facing else ("left side" if fx > 0 else "right side")
        if facing:
            parts.append(facing)

    if uy is not None and uy < -0.5:
        parts.append("top-down")
    elif fz is not None:
        if fz < -0.2:
            parts.append("from above")
        elif fz > 0.2:
            parts.append("from below")
        else:
            parts.append("eye-level")

    return "Target: " + ", ".join(parts) if parts else ""


def write_rollout_summary_image(
    *,
    frame_paths: Sequence[str | Path],
    steps: Sequence[dict],
    score_targets: dict[str, float],
    score_weights: dict[str, float],
    output_dir: str | Path,
) -> str:
    """Create a single summary image: frame strip on top, error graphs below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    output_dir = Path(output_dir)
    frame_paths = [Path(p) for p in frame_paths]
    num_frames = len(frame_paths)
    if num_frames == 0:
        return ""

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    fw, fh = frames[0].size

    # --- Collect per-step data ---
    step_indices = list(range(len(steps)))
    target_errors = [float(s["chosen_action"]["target_error"]) for s in steps]

    active_keys = [k for k in score_targets if score_weights.get(k, 0) > 0]
    bbox_keys = [k for k in active_keys if k.startswith("bbox_")]
    c2o_keys = [k for k in active_keys if k.startswith("camera_")]

    per_key_predicted: dict[str, list[float]] = {k: [] for k in active_keys}
    for s in steps:
        pred = s["chosen_action"].get("predicted_scores") or {}
        for k in active_keys:
            per_key_predicted[k].append(float(pred.get(k, 0.0)))

    # --- Layout ---
    has_bbox = bool(bbox_keys)
    has_c2o = bool(c2o_keys)
    c2o_f_keys = [k for k in c2o_keys if "_f" in k]
    c2o_u_keys = [k for k in c2o_keys if "_u" in k]
    n_c2o_rows = int(bool(c2o_f_keys)) + int(bool(c2o_u_keys))
    n_graph_rows = int(has_bbox) + n_c2o_rows + 1  # +1 for total error

    max_strip_frames = 20
    if num_frames > max_strip_frames:
        sample_indices = [0] + [
            int(round(i * (num_frames - 1) / (max_strip_frames - 1)))
            for i in range(1, max_strip_frames)
        ]
        sample_indices = sorted(set(sample_indices))
    else:
        sample_indices = list(range(num_frames))
    sampled_frames = [frames[i] for i in sample_indices]

    thumb_h = 160
    thumb_w = int(thumb_h * fw / fh)
    n_sampled = len(sampled_frames)

    fig = plt.figure(figsize=(max(14, n_sampled * 1.2), 3 + n_graph_rows * 2.5))
    gs = GridSpec(1 + n_graph_rows, 1, height_ratios=[3] + [2.5] * n_graph_rows, hspace=0.35)

    # --- Top: frame strip (sampled) ---
    ax_frames = fig.add_subplot(gs[0])
    strip_w = thumb_w * n_sampled
    strip = Image.new("RGB", (strip_w, thumb_h))
    for j, frame in enumerate(sampled_frames):
        strip.paste(frame.resize((thumb_w, thumb_h), Image.LANCZOS), (j * thumb_w, 0))
    ax_frames.imshow(np.array(strip))
    for j, orig_idx in enumerate(sample_indices):
        ax_frames.axvline(x=j * thumb_w, color="white", linewidth=0.5, alpha=0.5)
        ax_frames.text(
            j * thumb_w + thumb_w // 2, thumb_h - 8, str(orig_idx),
            ha="center", va="bottom", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6),
        )
    ax_frames.set_xlim(0, strip_w)
    ax_frames.set_ylim(thumb_h, 0)
    ax_frames.axis("off")
    title_suffix = f" ({n_sampled}/{num_frames} sampled)" if n_sampled < num_frames else ""
    target_desc = _describe_targets(score_targets, score_weights)
    ax_frames.set_title(f"Rollout Frames{title_suffix}", fontsize=10, fontweight="bold")
    if target_desc:
        fig.text(0.5, 0.98, target_desc, ha="center", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.9))

    row = 1

    # --- Total target error ---
    ax_err = fig.add_subplot(gs[row])
    ax_err.plot(step_indices, target_errors, "o-", color="black", markersize=4, linewidth=1.5)
    ax_err.set_ylabel("Weighted Error")
    ax_err.set_xlabel("Step")
    ax_err.set_title("Target Error", fontsize=10, fontweight="bold")
    ax_err.grid(True, alpha=0.3)
    row += 1

    # --- Bbox keys ---
    if has_bbox:
        ax_bbox = fig.add_subplot(gs[row])
        for k in bbox_keys:
            short = k.replace("bbox_", "")
            ax_bbox.plot(step_indices, per_key_predicted[k], "o-", markersize=3, linewidth=1, label=short)
            target_val = score_targets[k]
            ax_bbox.axhline(y=target_val, linestyle="--", alpha=0.4, linewidth=0.8)
        ax_bbox.set_ylabel("Score")
        ax_bbox.set_xlabel("Step")
        ax_bbox.set_title("Bbox Scores (predicted vs target ─ ─)", fontsize=10, fontweight="bold")
        ax_bbox.legend(fontsize=7, ncol=4, loc="upper right")
        ax_bbox.grid(True, alpha=0.3)
        row += 1

    # --- C2O keys: split into f (forward) and u (up) subplots ---
    if has_c2o:
        f_keys = [k for k in c2o_keys if "_f" in k]
        u_keys = [k for k in c2o_keys if "_u" in k]
        c2o_groups = []
        if f_keys:
            c2o_groups.append(("obj_forward in cam (fx, fy, fz)", f_keys))
        if u_keys:
            c2o_groups.append(("obj_up in cam (ux, uy, uz)", u_keys))

        for title, keys in c2o_groups:
            ax_c2o = fig.add_subplot(gs[row])
            for k in keys:
                short = k.replace("camera_to_object_", "")
                line, = ax_c2o.plot(step_indices, per_key_predicted[k], "o-",
                                    markersize=3, linewidth=1.5, label=short)
                target_val = score_targets[k]
                ax_c2o.axhline(y=target_val, linestyle="--", color=line.get_color(),
                               alpha=0.6, linewidth=1.2, label=f"{short} target={target_val:+.1f}")
            ax_c2o.set_ylabel("Score")
            ax_c2o.set_xlabel("Step")
            ax_c2o.set_ylim(-1.15, 1.15)
            ax_c2o.set_title(title, fontsize=10, fontweight="bold")
            ax_c2o.legend(fontsize=7, ncol=3, loc="upper right")
            ax_c2o.grid(True, alpha=0.3)
            row += 1

    out_path = output_dir / "rollout_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def save_frame_copy(source_path: str | Path, destination_path: str | Path) -> None:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.convert("RGB").save(destination_path)


def write_rollout_video(
    *,
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    fps: int,
) -> dict[str, str | int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [Path(path) for path in frame_paths]
    frame_dir = frame_paths[0].parent if frame_paths else output_dir
    mp4_path = output_dir / "rollout.mp4"

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path and frame_paths:
        frame_pattern = str(frame_dir / "frame_%04d.png")
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            frame_pattern,
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {
            "video_path": str(mp4_path),
            "video_format": "mp4",
            "video_backend": "ffmpeg",
            "fps": int(fps),
        }

    try:
        import imageio_ffmpeg  # noqa: F401

        with imageio.get_writer(mp4_path, fps=int(fps)) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        return {
            "video_path": str(mp4_path),
            "video_format": "mp4",
            "video_backend": "imageio_ffmpeg",
            "fps": int(fps),
        }
    except Exception:
        gif_path = output_dir / "rollout.gif"
        frames = [imageio.imread(frame_path) for frame_path in frame_paths]
        if frames:
            imageio.mimsave(gif_path, frames, duration=1.0 / max(1, int(fps)))
        return {
            "video_path": str(gif_path),
            "video_format": "gif",
            "video_backend": "imageio_gif_fallback",
            "fps": int(fps),
        }
