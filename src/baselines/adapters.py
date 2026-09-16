"""One rollout loop, N policies.

`scripts/closed_loop_eval.py` renders, steps the camera, decides when to stop and scores the
result. The only thing that differs between our policy and a baseline is how a frame and a
goal become an action chunk, so that is the whole adapter contract:

    act(frame_uint8, goal_vec, goal_raw, seed) -> (chunk_size, 10) float32

Everything a policy needs beyond that -- its own input resolution, its own goal encoding,
its own venv -- lives inside its adapter. The loop never learns which policy it is running,
which is exactly what makes the numbers comparable: same renderer, same stop rule, same
episodes, same metric code, by construction rather than by port.

Frames arrive at the TRAINING ASPECT (4:3, see `src.common.run_info.render_resolution`),
not the policy's square. Every policy here was trained on a 1024x768 render squashed to its
own input size -- 256x256 for Cosmos (`lerobot_export`), 224x224 for the VLAs (the source
repo's `export_lerobot.py --resize 224`), 480x720 for DP (`data.target_resolution`) -- so
each adapter does its own squash from the same picture.

Goal vectors arrive in `src.common.goal_space.DEFAULT_GOAL_KEYS` order. That is also the
order the baselines were trained with (`goal_text.NL_GOAL_KEYS` for the VLAs, the DP
config's `goal_score_keys`), which each adapter asserts at load time rather than assumes:
the source repo's `goal_space.DEFAULT_GOAL_KEYS` is a DIFFERENT list (world azimuth, not
subject bearing) that neither trained baseline uses.

Imports are lazy on purpose. Cosmos and LeRobot cannot share a venv (transformers 5 vs the
framework's pin), so the script has to be importable in either, and only the adapter that
is actually constructed pulls in its stack.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.common.goal_space import DEFAULT_GOAL_KEYS

CHUNK_DIM = 10          # 9 pose + shoot, every policy in this comparison


class PolicyAdapter(Protocol):
    name: str

    def act(self, frame_uint8: np.ndarray, goal_vec: np.ndarray, goal_raw: dict,
            seed: int) -> np.ndarray: ...


def _squash(frame_uint8: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """The non-aspect-preserving resize training did. (w, h) order, PIL convention."""
    from PIL import Image
    img = np.asarray(frame_uint8)
    if img.shape[1] == size_wh[0] and img.shape[0] == size_wh[1]:
        return img
    return np.asarray(Image.fromarray(img).resize(size_wh, Image.BILINEAR))


def _check_chunk(chunk: np.ndarray, who: str) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != CHUNK_DIM:
        raise RuntimeError(f"{who} returned a chunk of shape {chunk.shape}; expected (T, {CHUNK_DIM})")
    if not np.isfinite(chunk).all():
        raise RuntimeError(f"{who} returned non-finite actions")
    return chunk


# ---------------------------------------------------------------------------- Cosmos (ours)
class CosmosAdapter:
    """The V12 policy. Moved here verbatim from `closed_loop_eval.py` so the loop can be
    policy-agnostic; the prompt/batch subtleties documented inline are the ones that cost
    rollouts before (double-formatted prompt, truncated shoot channel, idle_frame leak)."""

    name = "cosmos"

    def __init__(self, checkpoint: str, out_dir: Path, *, image_size: int = 256,
                 chunk_size: int = 8, fps: int = 30, num_steps: int = 30,
                 guidance: float = 1.0, samples: int = 1, use_ema: bool = True,
                 idle_frames: int | None = 0):
        # `init_script` must run before any other cosmos import; it only insists that
        # nothing from `imaginaire` is loaded yet, so torch and our own modules are fine.
        from cosmos_framework.inference.common.init import init_script
        init_script()
        import torch
        from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode, OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import maybe_init_distributed
        from src.common.run_info import to_policy_frame
        from src.goal_authoring.serialize import goal_prompt

        self._torch, self._fmt, self._build = torch, ActionPromptJsonFormatter(), build_action_batch
        self._mode, self._to_policy_frame, self._goal_prompt = ModelMode, to_policy_frame, goal_prompt
        self.image_size, self.chunk_size, self.fps = int(image_size), int(chunk_size), int(fps)
        self.num_steps, self.guidance, self.samples = int(num_steps), float(guidance), max(1, int(samples))
        self.idle_frames = idle_frames
        # The domain the checkpoint was trained under. Mirrors
        # src/data/cosmos_camera_dataset.DOMAIN_NAME without importing that module (it pulls
        # cosmos in at import time); tests/test_policy_adapters.py pins the two together.
        self.domain_name = "camera_pose_shoot"

        maybe_init_distributed()
        wan = os.environ.get("WAN_VAE_PATH",
                             "/home/nas_main/jungwooahn/projects/DronePhotographerV12/repos/"
                             "cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth")
        overrides = OmniSetupOverrides.model_validate({
            "checkpoint_path": checkpoint,
            "config_file": "cosmos_framework/configs/base/config.py",
            "experiment": "action_policy_camera_nano",
            "experiment_overrides": [f"model.config.tokenizer.vae_path={wan}"],
            "output_dir": str(Path(out_dir) / "_infer"),
            "sampler": "unipc",
            "guardrails": False,          # a rollout loop must not pay for the safety models
            "use_ema_weights": bool(use_ema),
        })
        setup = overrides.build_setup()
        init_output_dir(setup.output_dir)
        pipe = OmniInference.create(setup)
        pipe.model.eval()
        self.model = pipe.model

    def make_prompt(self, goal_vec: np.ndarray, crop=None) -> str:
        """The goal as the JSON string the model was trained on.

        `crop` is not optional in practice: the exporter always passes it, so every training
        prompt ends with a crop phrase and a `visible` number. `idle_frames` LEAKS the shoot
        label (idle_frames > 0 => shoot=1 in 896/896 exported episodes); 0 is the
        training-modal value (81%), None omits the field -- the ablation that separates "the
        policy learned to declare arrival" from "the policy is reading the leak".
        """
        torch = self._torch
        sample = {
            "ai_caption": self._goal_prompt(goal_vec, crop=crop),
            "viewpoint": "ego_view",
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "image_size": torch.tensor([self.image_size, self.image_size]),
            "mode": "policy",
            # `action` is here ONLY so the formatter can count total frames: without it
            # idle_frame reads "0." where training says "0 out of 8."
            "action": torch.zeros(self.chunk_size, 1, dtype=torch.float32),
            "video": torch.zeros(3, self.chunk_size + 1, self.image_size, self.image_size,
                                 dtype=torch.uint8),
        }
        if self.idle_frames is not None:
            sample["idle_frames"] = torch.tensor(int(self.idle_frames))
        out = self._fmt(sample)["ai_caption"]
        return out if isinstance(out, str) else json.dumps(out)

    def act(self, frame_uint8, goal_vec, goal_raw, seed):
        torch = self._torch
        prompt = self.make_prompt(goal_vec, crop=goal_raw)
        frame = self._to_policy_frame(frame_uint8, self.image_size)          # 4:3 -> square
        video = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).contiguous()
        video = video.unsqueeze(1).repeat(1, self.chunk_size + 1, 1, 1)      # [C,T,H,W]
        # Rebuilt every call: the model normalizes video in place and stamps
        # `is_preprocessed`, so a reused dict silently takes the float branch.
        batch = self._build(
            video=video,
            action=torch.zeros(self.chunk_size, 64, dtype=torch.float32),
            raw_action_dim=CHUNK_DIM,
            prompt=prompt,
            view_point="ego_view",
            domain_name=self.domain_name,
            model_mode=self._mode.POLICY,       # the enum: the builder reads `.value`
            action_chunk_size=self.chunk_size,
            fps=self.fps,
            input_video_key=self.model.config.input_video_key,
            batch_size=1,
            device="cuda",
        )
        # `build_action_batch` formats whatever it is handed, and `prompt` is ALREADY the
        # formatted JSON; letting it wrap again produced an 868-char prompt training never saw.
        batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
        with torch.no_grad():
            draws = [self.model.generate_samples_from_batch(
                         batch, guidance=self.guidance, num_steps=self.num_steps,
                         seed=[seed + 100_000 * k])["action"][0].float().cpu().numpy()
                     for k in range(self.samples)]
        # Mean of K, shoot dim included: its output is effectively binary (90.5% < 0.05,
        # 9.5% > 0.95 on 300 held-out windows), so the mean is the fraction of draws that
        # voted to stop and thresholding it at 0.5 is a majority vote.
        chunk = draws[0] if len(draws) == 1 else np.mean(draws, axis=0)
        # Slice to the FULL width. Truncating to 9 here once dropped the shoot channel
        # before the caller ever saw it, and termination silently never fired.
        return _check_chunk(chunk[:, :CHUNK_DIM], self.name)


# ---------------------------------------------------------------------------- LeRobot VLAs
class LeRobotAdapter:
    """pi0.5 / GR00T through LeRobot, as trained in the source repo.

    Goal is the source repo's NL prompt (`src.baselines.goal_text.goal_prompt`), NOT ours --
    that is the string these checkpoints were fine-tuned on. Frame is the 4:3 render squashed
    to 224x224, the same resize their `export_lerobot.py` applied to the training frames.
    Zero proprio state, as in training (pi0.5 ignores it; GR00T requires the key).
    """

    def __init__(self, kind: str, checkpoint: str, *, resize: int = 224, state_dim: int = 9,
                 device: str = "cuda"):
        assert kind in ("pi05", "groot"), kind
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        if kind == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
        else:
            from lerobot.policies.groot.modeling_groot import GrootPolicy as P
        from src.baselines.goal_text import NL_GOAL_KEYS, goal_prompt
        assert tuple(NL_GOAL_KEYS) == tuple(DEFAULT_GOAL_KEYS), (
            "the loop's goal_vec is in DEFAULT_GOAL_KEYS order; the VLA prompt expects "
            f"NL_GOAL_KEYS -- they differ:\n {DEFAULT_GOAL_KEYS}\n {NL_GOAL_KEYS}")
        self.name, self._torch, self._keys, self._goal_prompt = kind, torch, NL_GOAL_KEYS, goal_prompt
        self.policy = P.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, checkpoint)
        self.resize = int(resize)
        self.state = torch.zeros(state_dim, dtype=torch.float32)

    def act(self, frame_uint8, goal_vec, goal_raw, seed):
        torch = self._torch
        task = self._goal_prompt(np.asarray(goal_vec, dtype=np.float32), self._keys, crop=goal_raw)
        a = _squash(frame_uint8, (self.resize, self.resize)).astype(np.float32) / 255.0
        batch = {"observation.images.image": torch.from_numpy(a).permute(2, 0, 1),
                 "observation.state": self.state, "task": task}
        torch.manual_seed(int(seed))               # the action expert samples noise
        with torch.no_grad():
            chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        chunk = chunk.squeeze(0).float().cpu().numpy()
        return _check_chunk(chunk[:, :CHUNK_DIM], self.name)


# ---------------------------------------------------------------------------- Diffusion Policy
class DPAdapter:
    """Diffusion Policy (frozen DINOv2-L -> FiLM ConditionalUnet1D, DDPM/DDIM), as trained.

    Goal is the NORMALIZED VECTOR over the config's `goal_score_keys`, in the source repo's
    ranges. Frame is the 4:3 render squashed to `data.target_resolution` (480x720), the
    resize its dataset loader applied. `sample(denormalize=True)` returns raw pose units and
    passes the shoot dim through.
    """

    def __init__(self, checkpoint: str, config_path: str, *, n_steps: int = 16,
                 chunk_size: int = 8, device: str = "cuda"):
        import torch
        import yaml
        from transformers import AutoImageProcessor, AutoModel
        from src.baselines.dp_model import DiffusionPolicy, build_obs_inputs
        from src.baselines.goal_space import goal_keys, normalize_goal

        cfg = yaml.safe_load(Path(config_path).read_text())
        data, m = cfg.get("data", {}), cfg.get("model", {})
        self.keys = goal_keys(data.get("goal_score_keys"))
        assert tuple(self.keys) == tuple(DEFAULT_GOAL_KEYS), (
            "the loop's goal_vec is in DEFAULT_GOAL_KEYS order; this DP config trained on "
            f"a different key order:\n {DEFAULT_GOAL_KEYS}\n {self.keys}")
        h, w = data.get("target_resolution", [480, 720])
        self.res_wh, self.n_steps, self.chunk = (int(w), int(h)), int(n_steps), int(chunk_size)
        self.name, self._torch, self._normalize = "dp", torch, normalize_goal
        self._build_obs, self.dt, self.dev = build_obs_inputs, torch.bfloat16, torch.device(device)

        bb = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=self.dt)
        pr = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.policy = DiffusionPolicy(
            bb, goal_dim=len(self.keys), chunk_size=self.chunk, processor=pr,
            goal_embed_dim=m.get("goal_embed_dim", 128),
            down_dims=tuple(m.get("down_dims", [128, 256, 512])),
            diffusion_step_embed_dim=m.get("diffusion_step_embed_dim", 128),
            num_train_timesteps=m.get("num_train_timesteps", 100),
            beta_schedule=m.get("beta_schedule", "squaredcos_cap_v2"),
        ).to(self.dev).eval()
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = self.policy.load_state_dict(sd["policy_state"], strict=False)
        bad = [k for k in list(missing) + list(unexpected) if "backbone" not in k]
        if bad:
            raise RuntimeError(f"DP checkpoint/model MISMATCH: {bad[:5]} -- wrong --policy-config?")

    def act(self, frame_uint8, goal_vec, goal_raw, seed):
        torch = self._torch
        # EXACTLY what `_load_image_as_tensor` did to the training frames: LANCZOS to (W, H),
        # then [-1, 1] via /127.5 - 1. Not interchangeable with a BILINEAR [0, 1] tensor --
        # `build_obs_inputs` maps its input back with `x * 0.5 + 0.5`, so feeding [0, 1] lands
        # the whole image in the upper half of the range before DINOv2 ever sees it.
        from PIL import Image
        pil = Image.fromarray(np.asarray(frame_uint8)).convert("RGB")
        if pil.size != self.res_wh:
            pil = pil.resize(self.res_wh, Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 127.5 - 1.0
        img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()
        g = self._normalize(np.asarray(goal_vec, dtype=np.float32), self.keys)
        batch = {"state_image": img.unsqueeze(0), "goal_vec": torch.from_numpy(g).unsqueeze(0),
                 "action_chunk": torch.zeros(1, self.chunk, CHUNK_DIM)}
        torch.manual_seed(int(seed))               # DDIM starts from noise
        with torch.no_grad(), torch.amp.autocast(self.dev.type, dtype=self.dt):
            oi, gv, _ = self.policy.prepare_inputs(batch, self.dev, self.dt)
            chunk = self.policy.sample(oi, gv, n_steps=self.n_steps).pred_action_chunk
        chunk = chunk.squeeze(0).float().cpu().numpy()
        return _check_chunk(chunk[:, :CHUNK_DIM], self.name)


# ---------------------------------------------------------------------------- AutoPhoto (RL)
class AutoPhotoAdapter:
    """AutoPhoto: RecurrentPPO (actor-critic + LSTM) over a pretrained aesthetic reward.

    THE FLOOR ROW, and it is here to be one. Two things make it structurally unable to do this
    task, and saying so is the point of running it:

    * **It never sees the goal.** Its observation is 512-d features from AutoPhoto's aesthetic
      ResNet18, and its reward was that network's score. Nothing about a requested shot reaches
      it. A goal-conditioned metric should therefore score it near the no-op, and if it does
      not, the metric is measuring "moving the camera helps" rather than "following the goal".
    * **Its action space cannot express the goal space.** The nine discrete actions are dolly
      forward/back and yaw by +-10/30/90 degrees, plus terminate. In the 5-D delta they map to,
      `dx`, `dy` and `dpitch` are ALWAYS zero -- no lateral, no vertical, no tilt. Elevation and
      the vertical half of `d_aim` are unreachable by construction.

    Fitted to the shared contract rather than given its own loop: it emits one discrete action
    per step, so the adapter runs it `chunk_size` times from a single observation, carrying the
    LSTM state across chunks, and each discrete move is converted to the 9-D action the rollout
    env applies. That is exactly the deal every other policy gets -- eight steps from one frame
    -- and it is a worse deal for a reactive policy than for a chunk predictor. Its `terminate`
    action drives the shoot channel, so it stops when it says it is done.
    """

    name = "autophoto"

    def __init__(self, checkpoint: str, reward_ckpt: str, *, chunk_size: int = 8,
                 device: str = "cuda"):
        import torch
        from sb3_contrib import RecurrentPPO
        from src.baselines.autophoto_vendor import AestheticReward
        from src.common.action_repr import apply_action_5d, encode_action_9d

        self._torch = torch
        self._apply5d, self._encode9d = apply_action_5d, encode_action_9d
        self.chunk_size = int(chunk_size)
        self.model = RecurrentPPO.load(checkpoint, device=device)
        self.reward = AestheticReward(reward_ckpt, device=device)
        self._lstm = None            # carried across chunks, as in AutoPhoto's own eval loop
        self._episode = None
        # The action table, copied from the source repo's PhotoEnv rather than imported: that
        # module pulls in gymnasium, its Blender env and its renderer, none of which this needs.
        _F, _T10, _T30, _T90 = 0.25, np.deg2rad(10), np.deg2rad(30), np.deg2rad(90)
        self.actions = [
            (0, 0, _F, 0, 0), (0, 0, 0, -_T30, 0), (0, 0, 0, _T30, 0),
            None,                                            # terminate
            (0, 0, -_F, 0, 0), (0, 0, 0, -_T10, 0), (0, 0, 0, _T10, 0),
            (0, 0, 0, -_T90, 0), (0, 0, 0, _T90, 0),
        ]

    def act(self, frame_uint8, goal_vec, goal_raw, seed):
        """Eight discrete moves from one observation, as 9-D pose actions + a shoot channel.

        `goal_vec` and `goal_raw` are accepted and ignored — deliberately, and it is the whole
        finding: this policy has no way to take them.
        """
        from PIL import Image
        # New episode when the loop's seed base changes; the LSTM state must not leak across.
        ep = int(seed) // 100
        if ep != self._episode:
            self._episode, self._lstm = ep, None

        _, obs = self.reward.score_and_features(Image.fromarray(np.asarray(frame_uint8)))
        obs = np.asarray(obs, dtype=np.float32)

        # Integrate the discrete moves on a local pose, then encode each step as the 9-D
        # action the env applies. A local basis is enough: apply_action_5d/encode_action_9d
        # are both relative, so the chunk is the same wherever the camera actually is.
        pos = np.zeros(3, np.float32)
        fwd = np.array([0, 0, 1], np.float32)
        up = np.array([0, 1, 0], np.float32)
        chunk = np.zeros((self.chunk_size, CHUNK_DIM), np.float32)
        for t in range(self.chunk_size):
            a, self._lstm = self.model.predict(obs, state=self._lstm, deterministic=True)
            spec = self.actions[int(a)]
            if spec is None:                       # terminate -> latch the shoot channel
                chunk[t:, 9] = 1.0
                break
            np_, fwd_, up_ = self._apply5d(pos, fwd, up, np.asarray(spec, np.float32))
            chunk[t, :9] = self._encode9d(pos, fwd, up, np_, fwd_, up_)
            pos, fwd, up = np_, fwd_, up_
        return _check_chunk(chunk, self.name)


# ---------------------------------------------------------------------------- factory
def make_adapter(args: Any, out_dir: Path) -> PolicyAdapter:
    """Build the adapter `--policy` names from the eval script's parsed args."""
    if args.policy == "cosmos":
        return CosmosAdapter(args.checkpoint, out_dir, image_size=args.image_size,
                             chunk_size=args.chunk_size, fps=args.fps, num_steps=args.num_steps,
                             guidance=args.guidance, samples=args.samples,
                             use_ema=not args.no_ema, idle_frames=getattr(args, "idle_frames", 0))
    if args.policy in ("pi05", "groot"):
        return LeRobotAdapter(args.policy, args.checkpoint)
    if args.policy == "autophoto":
        if not args.policy_config:
            raise SystemExit("--policy autophoto needs --policy-config <aesthetic resnet .pt>")
        return AutoPhotoAdapter(args.checkpoint, args.policy_config, chunk_size=args.chunk_size)
    if args.policy == "dp":
        if not args.policy_config:
            raise SystemExit("--policy dp needs --policy-config <config.yaml> from the training run")
        return DPAdapter(args.checkpoint, args.policy_config, chunk_size=args.chunk_size)
    raise SystemExit(f"unknown --policy {args.policy!r}")


__all__ = ["PolicyAdapter", "CosmosAdapter", "LeRobotAdapter", "DPAdapter", "AutoPhotoAdapter", "make_adapter", "CHUNK_DIM"]
