"""VLADroneDataset — v7 windows for the VLA baseline.

Thin wrapper over `common.dataset_base.BasePolicyDataset` (same windows, same HER
goal sampling, same clamped-goal filter as the Cosmos dataset — so the two train
on the identical sample distribution). Emits only what the VLA needs:

  - `state_image`: (3, H, W) in [-1, 1] — the window's start frame.
  - `goal_vec`:    (D_goal,) normalized to [-1, 1].
  - `action_chunk`: (chunk_size, 5) normalized.
  - `meta`: dict for debugging.

Dropped vs CosmosDroneDataset: `next_state_image` (no world model → no
future-frame target) and `value_target` (no value head). The VLM's own image
processor turns `state_image` into pixel tensors at train time (in the policy's
`prepare_inputs`), so we keep the image as a plain [-1, 1] CHW tensor here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.policy.common.action_repr import normalize_action_5d
from src.policy.common.dataset_base import BasePolicyDataset
from src.policy.common.goal_space import normalize_goal


def _load_image_as_tensor(image_path: Path, target_resolution: tuple[int, int]) -> torch.Tensor:
    """Load JPEG/PNG, resize to (H, W), return (3, H, W) in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (target_resolution[1], target_resolution[0]):
        img = img.resize((target_resolution[1], target_resolution[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_vlm_inputs(processor, prompt, images: torch.Tensor) -> dict:
    """Run the Qwen3-VL processor on a batch of [-1,1] CHW images + prompt(s).

    `prompt` may be a single string (soft-token conditioning: a fixed prompt, goal
    enters as soft tokens) or a per-sample list of strings (NL conditioning: each
    prompt IS the goal, via `goal_text.goal_prompt`). Shared by the training collate
    (runs in dataloader workers → overlaps GPU compute) and by eval.
    """
    from PIL import Image

    pil = []
    for im in images:
        arr = ((im.float().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        pil.append(Image.fromarray(arr))
    prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt] * len(pil)
    messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
    text = [processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
    proc = processor(text=text, images=pil, return_tensors="pt", padding=True)
    return dict(proc)


def goal_to_prompt(goal_raw: "np.ndarray", object_key: str | None = None) -> str:
    """NL conditioning prompt for one sample. The VLA goal space now uses
    `subject_bearing_deg` (its goal_score_keys == goal_text.NL_GOAL_KEYS order), so the raw
    goal vector serializes directly — the subject-relative bearing is already in it (derived
    in annotations._frame_to_view from the facing map). `object_key` is accepted for
    signature compatibility but unused."""
    from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt

    return goal_prompt(np.asarray(goal_raw, dtype=np.float32), NL_GOAL_KEYS)


class VLACollate:
    """Picklable collate_fn that does Qwen preprocessing in the dataloader worker.

    Bound with the processor + prompt; the DataLoader pickles it to each worker,
    so the (CPU-heavy) image processing overlaps GPU compute instead of
    serializing in the training loop. Returns the model-ready batch.
    """

    def __init__(self, processor, prompt: str, goal_conditioning: str = "soft_token") -> None:
        self.processor = processor
        self.prompt = prompt
        self.goal_conditioning = goal_conditioning   # "soft_token" | "text"

    def __call__(self, samples: list[dict]) -> dict:
        images = torch.stack([s["state_image"] for s in samples])
        if self.goal_conditioning == "text":
            # The goal IS the prompt (per sample). goal_vec is then unused by the model;
            # pass zeros so the batch shape is unchanged.
            prompts = [goal_to_prompt(s["goal_raw"].numpy(), s["meta"]["object"]) for s in samples]
            vlm_inputs = build_vlm_inputs(self.processor, prompts, images)
            goal_vec = torch.zeros_like(torch.stack([s["goal_vec"] for s in samples]))
        else:
            vlm_inputs = build_vlm_inputs(self.processor, self.prompt, images)
            goal_vec = torch.stack([s["goal_vec"] for s in samples])
        return {
            "vlm_inputs": vlm_inputs,
            "goal_vec": goal_vec,
            "action_chunk": torch.stack([s["action_chunk"] for s in samples]),
            "meta": [s["meta"] for s in samples],
        }


class VLADroneDataset(Dataset):
    def __init__(
        self,
        annotation_roots: Sequence[str | Path],
        *,
        goal_score_keys: Sequence[str] | None = None,
        chunk_size: int = 8,
        stride: int = 1,
        max_samples: int | None = None,
        target_resolution: tuple[int, int] = (480, 720),
        normalize_goal_to_unit_cube: bool = True,
        normalize_actions: bool = True,
        action_scale=None,
        filter_clamped_goals: bool = True,
        goal_sampling: str = "uniform_future",
        sampling_scheme: str = "sliding_window",
        offsets: Sequence[int] = (8, 16, 24),
        goal_start_max_per_pair: int = 24,
        goal_start_seed: int = 0,
        val_pair_stride: int = 0,
        val_split_level: str = "pair",
        val_names: Sequence[str] | None = None,
        split: str = "train",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.target_resolution = target_resolution
        self.normalize = normalize_goal_to_unit_cube
        self.normalize_actions = normalize_actions
        self.action_scale = action_scale
        self.base = BasePolicyDataset(
            annotation_roots,
            goal_score_keys=goal_score_keys,
            chunk_size=chunk_size,
            stride=stride,
            max_samples=max_samples,
            filter_clamped_goals=filter_clamped_goals,
            goal_sampling=goal_sampling,
            sampling_scheme=sampling_scheme,
            offsets=offsets,
            goal_start_max_per_pair=goal_start_max_per_pair,
            goal_start_seed=goal_start_seed,
            val_pair_stride=val_pair_stride,
            val_split_level=val_split_level,
            val_names=val_names,
            split=split,
            cache_dir=cache_dir,
        )
        if len(self.base):
            p = Path(self.base[0].start.image)
            if not p.exists():
                raise FileNotFoundError(f"missing rendered image: {p}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        s = self.base[idx]
        img = _load_image_as_tensor(Path(s.start.image), self.target_resolution)
        goal = s.goal_vec
        if self.normalize:
            goal = normalize_goal(goal, self.base.goal_keys)
        action_chunk = s.action_chunk
        if self.normalize_actions:
            action_chunk = normalize_action_5d(action_chunk, self.action_scale)
        return {
            "state_image": img,
            "goal_vec": torch.from_numpy(goal),
            # Raw (unnormalized) goal + object: the NL collate serializes these into the
            # goal_prompt text (needs degrees/%/px + the object's facing for bearing).
            "goal_raw": torch.from_numpy(np.asarray(s.goal_vec, dtype=np.float32)),
            "action_chunk": torch.from_numpy(action_chunk),
            "meta": {
                "annotation_path": str(s.start.annotation_path),
                "pair_idx": s.start.pair_idx,
                "start_frame_idx": s.start.frame_idx,
                "goal_frame_idx": s.goal.frame_idx,
                "scene": s.start.scene,
                "object": s.goal.object,
            },
        }


__all__ = ["VLADroneDataset", "VLACollate", "build_vlm_inputs"]
