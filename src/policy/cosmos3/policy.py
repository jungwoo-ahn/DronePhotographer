"""The cosmos3 camera policy: one RGB frame and a shot profile in, 8 pose actions out.

Run from the root of `repos/cosmos-framework`, because the framework resolves several model
configs by relative path; `scripts/serve_cosmos_policy.sh` sets the environment up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.inference.action import build_action_batch
from cosmos_framework.inference.args import ModelMode, OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import maybe_init_distributed
from PIL import Image

from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt
from src.policy.cosmos3.recipe import CAMERA_ACTION_DIM, CHUNK_LENGTH, DOMAIN_NAME, EXPERIMENT

IMAGE_SIZE = 256
FPS = 30                # conditioning metadata only; the renders have no clock
CROP_KEYS = ("top_cut_frac", "bot_cut_frac", "visible_frac")


def to_policy_frame(image: np.ndarray) -> np.ndarray:
    """Squash to the square, ignoring aspect, as the training export did with ffmpeg."""
    if image.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE):
        return image
    return np.asarray(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))


def profile_prompt(profile: dict) -> str:
    """A complete shot profile as the sentence the policy trained on.

    Refuses a partial one. A missing key would read as 0, which the sentence turns into
    "extreme wide shot" or "top-left corner", and the policy would obey it.
    """
    missing = [k for k in NL_GOAL_KEYS if k not in profile]
    if missing:
        raise ValueError(f"shot profile is missing {missing}")
    crop = {k: float(profile[k]) for k in CROP_KEYS if k in profile} or None
    return goal_prompt(np.array([float(profile[k]) for k in NL_GOAL_KEYS]), crop=crop)


class Cosmos3CameraPolicy:
    def __init__(self, checkpoint: str, *, vae_path: str, output_dir: str, num_steps: int = 30,
                 samples: int = 4, guidance: float = 1.0, seed: int = 0):
        self.num_steps, self.samples, self.guidance = num_steps, samples, guidance
        self.seed = seed
        maybe_init_distributed()
        setup = OmniSetupOverrides.model_validate({
            "checkpoint_path": checkpoint,
            "config_file": "cosmos_framework/configs/base/config.py",
            "experiment": EXPERIMENT,
            "experiment_overrides": [f"model.config.tokenizer.vae_path={vae_path}"],
            "output_dir": str(Path(output_dir)),
            "sampler": "unipc", "guardrails": False, "use_ema_weights": True,
        }).build_setup()
        init_output_dir(setup.output_dir)
        self.model = OmniInference.create(setup).model
        self.model.eval()

    def format_prompt(self, sentence: str) -> str:
        sample = {"ai_caption": sentence, "viewpoint": "ego_view",
                  "conditioning_fps": torch.tensor(FPS, dtype=torch.long),
                  "image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
                  "mode": "policy",
                  "action": torch.zeros(CHUNK_LENGTH, 1, dtype=torch.float32),
                  "video": torch.zeros(3, CHUNK_LENGTH + 1, IMAGE_SIZE, IMAGE_SIZE,
                                       dtype=torch.uint8),
                  "idle_frames": torch.tensor(0)}
        out = ActionPromptJsonFormatter()(sample)["ai_caption"]
        return out if isinstance(out, str) else json.dumps(out)

    def predict_chunk(self, image: np.ndarray, profile: dict) -> np.ndarray:
        """`(CHUNK_LENGTH, 10)` raw actions, averaged over `samples` draws."""
        prompt = self.format_prompt(profile_prompt(profile))
        frame = torch.from_numpy(to_policy_frame(image)).permute(2, 0, 1).contiguous()
        video = frame.unsqueeze(1).repeat(1, CHUNK_LENGTH + 1, 1, 1)
        batch = build_action_batch(
            video=video, action=torch.zeros(CHUNK_LENGTH, 64, dtype=torch.float32),
            raw_action_dim=CAMERA_ACTION_DIM, prompt=prompt, view_point="ego_view",
            domain_name=DOMAIN_NAME, model_mode=ModelMode.POLICY,
            action_chunk_size=CHUNK_LENGTH, fps=FPS,
            input_video_key=self.model.config.input_video_key, batch_size=1, device="cuda")
        batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
        draws = []
        with torch.no_grad():
            for k in range(max(1, self.samples)):
                out = self.model.generate_samples_from_batch(
                    batch, guidance=self.guidance, num_steps=self.num_steps,
                    seed=[self.seed + 100_000 * k])
                draws.append(out["action"][0].float().cpu().numpy())
        return np.mean(draws, axis=0)[:, :CAMERA_ACTION_DIM]
