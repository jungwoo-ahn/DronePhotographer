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
                for rx in rotation_values_rad:
                    for ry in rotation_values_rad:
                        for rz in rotation_values_rad:
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
