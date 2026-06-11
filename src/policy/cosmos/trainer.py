"""CosmosPolicyTrainer: pure flow-matching training loop.

With the latent-frame action/value design, the joint loss collapses to a single
flow-matching MSE over the entire latent sequence (image + action + value
frames). No separate action / value MSE terms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.vae import CosmosVAEWrapper


@dataclass
class TrainerConfig:
    output_root: Path
    run_name: str
    max_iter: int = 5000
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_iter: int = 500
    save_iter: int = 1000
    log_iter: int = 50
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    # EMA horizon (in iterations) for the loss estimate that decides ckpt_best.pt
    best_ema_beta: float = 0.98


class CosmosPolicyTrainer:
    def __init__(
        self,
        policy: CosmosWorldActionPolicy,
        vae: CosmosVAEWrapper,
        config: TrainerConfig,
    ) -> None:
        self.policy = policy
        self.vae = vae
        self.config = config
        self.run_dir = self._make_run_dir()
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

    def _make_run_dir(self) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_root / f"{ts}_{self.config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_optimizer(self) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        warmup = max(1, self.config.warmup_iter)

        def lr_lambda(step: int) -> float:
            return min(1.0, step / warmup)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return opt, sched

    def fit(self, dataloader_train: DataLoader, dataloader_val: Optional[DataLoader] = None) -> None:
        cfg = self.config
        self.policy.to(self.device)
        self.vae.to(self.device)
        opt, sched = self._build_optimizer()
        scaler = torch.amp.GradScaler(self.device.type) if self.dtype == torch.float16 else None

        log_path = self.run_dir / "train.log"
        log_f = open(log_path, "a")

        iteration = 0
        accum = 0
        opt.zero_grad(set_to_none=True)
        last_log = time.time()
        # Keep only best + last checkpoints. "Best" = lowest EMA of the total
        # loss (per-iter loss is too noisy under EDM sigma sampling to compare raw).
        loss_ema: Optional[float] = None
        best_ema = float("inf")

        while iteration < cfg.max_iter:
            for batch in dataloader_train:
                if iteration >= cfg.max_iter:
                    break
                state_img = batch["state_image"].to(self.device, dtype=self.dtype)
                next_img = batch["next_state_image"].to(self.device, dtype=self.dtype)
                goal = batch["goal_vec"].to(self.device, dtype=self.dtype)
                action_chunk = batch["action_chunk"].to(self.device, dtype=self.dtype)
                value_target = batch["value_target"].to(self.device, dtype=self.dtype)

                with torch.amp.autocast(self.device.type, dtype=self.dtype):
                    # Encode state and goal frames separately (T=1 each — the Wan
                    # VAE needs (T-1)%4==0) and concat: ALOHA-style T_img=2 latents.
                    image_latent = self.vae.encode_pair_frames(state_img, next_img)  # (B, 16, 2, h, w)
                    loss_out = self.policy.compute_loss(
                        image_latent=image_latent,
                        action_chunk=action_chunk,
                        goal_vec=goal,
                        value_target=value_target,
                    )
                    loss = loss_out.total / cfg.grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum += 1
                cur = float(loss_out.total.detach())
                loss_ema = cur if loss_ema is None else cfg.best_ema_beta * loss_ema + (1 - cfg.best_ema_beta) * cur
                if accum >= cfg.grad_accum:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    accum = 0
                    iteration += 1

                    if iteration % cfg.log_iter == 0:
                        dt = time.time() - last_log
                        last_log = time.time()
                        parts = loss_out.detach_dict()
                        line = (
                            f"iter={iteration} "
                            f"total={parts['total']:.4f} "
                            f"world={parts['world']:.4f} "
                            f"action={parts['action']:.4f} "
                            + (f"value={parts.get('value', 0):.4f} " if 'value' in parts else "")
                            + f"lr={sched.get_last_lr()[0]:.2e} "
                            f"{cfg.log_iter/dt:.2f}it/s"
                        )
                        print(line)
                        log_f.write(line + "\n")
                        log_f.flush()

                    if iteration % cfg.save_iter == 0:
                        self.save_checkpoint(iteration, name="ckpt_last.pt")
                        if loss_ema is not None and loss_ema < best_ema:
                            best_ema = loss_ema
                            self.save_checkpoint(iteration, name="ckpt_best.pt")
                            log_f.write(f"iter={iteration} new best (loss EMA {loss_ema:.4f})\n")

        self.save_checkpoint(iteration, name="ckpt_last.pt")
        log_f.close()

    def save_checkpoint(self, iteration: int, name: Optional[str] = None) -> Path:
        path = self.run_dir / (name or f"ckpt_iter{iteration:07d}.pt")
        torch.save(
            {
                "iteration": iteration,
                "policy_state": self.policy.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )
        return path
