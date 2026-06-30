"""DPTrainer — lean DDPM training loop for the Diffusion Policy baseline.

Mirrors `vla/trainer.py` (DDP, grad accumulation, LR warmup, EMA-selected
best/last checkpoints, TensorBoard, held-out validation) but with a frozen-vision
obs encoder + DDPM epsilon loss instead of the flow-matching action loss. Per our
families-self-contained convention it's a separate ~140-line loop, not a shared
base.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.policy.diffusion_policy.model import DiffusionPolicy


class _LossForward(nn.Module):
    """DDP needs grads to flow through forward; expose compute_loss as forward."""

    def __init__(self, policy: DiffusionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **kwargs):
        return self.policy.compute_loss(**kwargs)


@dataclass
class DPTrainerConfig:
    output_root: Path
    run_name: str
    max_iter: int = 50000
    batch_size: int = 32
    grad_accum: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_iter: int = 500
    save_iter: int = 2000
    log_iter: int = 100
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    best_ema_beta: float = 0.98
    # Validation (rank 0). 0 = off. ckpt_best is selected by the val metric when a
    # val loader is given, else by training-loss EMA.
    val_iter: int = 0
    # val_sample_steps > 0 also runs the (slow) sampler for action_mse; 0 = val
    # LOSS ONLY (no per-step simulation) — the selection/early-stop metric then
    # falls back to the fixed-timestep noise loss.
    val_sample_steps: int = 16
    # Resume policy (+ optimizer if present) from this checkpoint and continue
    # the iteration counter. LR stays at its post-warmup constant.
    resume_from: Optional[str] = None
    # Stop if the val metric hasn't improved for this many validations (0 = off).
    early_stop_patience: int = 0


class DPTrainer:
    def __init__(self, policy: DiffusionPolicy, config: DPTrainerConfig) -> None:
        self.policy = policy
        self.config = config
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = self.rank == 0
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        self.run_dir = self._make_run_dir()
        torch.manual_seed(config.seed + self.rank)

    def _make_run_dir(self) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.distributed:
            obj = [ts if self.is_main else None]
            dist.broadcast_object_list(obj, src=0)
            ts = obj[0]
        run_dir = self.config.output_root / f"{ts}_{self.config.run_name}"
        if self.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_optimizer(self):
        # Frozen backbone -> only the goal embed + denoiser params carry grad.
        params = [p for p in self.policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        warmup = max(1, self.config.warmup_iter)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / warmup))
        return opt, sched

    def _batch_to_device(self, batch: dict):
        """Move a DPCollate batch (obs_inputs dict + goal + action) to device."""
        obs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch["obs_inputs"].items()}
        goal = batch["goal_vec"].to(self.device, self.dtype)
        action = batch["action_chunk"].to(self.device, self.dtype)
        return obs, goal, action

    def fit(self, dataloader_train: DataLoader, dataloader_val: Optional[DataLoader] = None) -> None:
        cfg = self.config
        self.policy.to(self.device)
        loss_module: nn.Module = _LossForward(self.policy)
        if self.distributed:
            loss_module = nn.parallel.DistributedDataParallel(
                loss_module, device_ids=[self.device.index], output_device=self.device.index,
                find_unused_parameters=False,
            )
        opt, sched = self._build_optimizer()

        # Resume: load weights (+ optimizer if the ckpt has it) and continue the
        # iteration counter. Past warmup the LR is constant, so we just advance
        # the scheduler's epoch to match.
        start_iter = 0
        if cfg.resume_from:
            ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
            self.policy.load_state_dict(ckpt["policy_state"])
            self.policy.to(self.device)
            start_iter = int(ckpt.get("iteration", 0))
            if "optimizer_state" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state"])
            sched.last_epoch = start_iter
            if self.is_main:
                print(f"resumed from {cfg.resume_from} at iter {start_iter}"
                      f" ({'with' if 'optimizer_state' in ckpt else 'NO'} optimizer state)", flush=True)

        log_f = open(self.run_dir / "train.log", "a") if self.is_main else None
        tb = None
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        iteration = start_iter
        accum = 0
        epoch = 0
        opt.zero_grad(set_to_none=True)
        last_log = time.time()
        loss_ema: Optional[float] = None
        best_metric = float("inf")
        no_improve = 0
        stop = False
        have_val = dataloader_val is not None

        while iteration < cfg.max_iter and not stop:
            if isinstance(getattr(dataloader_train, "sampler", None), DistributedSampler):
                dataloader_train.sampler.set_epoch(epoch)
            epoch += 1
            for batch in dataloader_train:
                if iteration >= cfg.max_iter or stop:
                    break
                obs_inputs, goal, action_chunk = self._batch_to_device(batch)

                is_sync = accum + 1 >= cfg.grad_accum
                sync_ctx = (loss_module.no_sync() if self.distributed and not is_sync else contextlib.nullcontext())
                with sync_ctx, torch.amp.autocast(self.device.type, dtype=self.dtype):
                    loss_out = loss_module(obs_inputs=obs_inputs, goal_vec=goal, action_chunk=action_chunk)
                    loss = loss_out.total / cfg.grad_accum
                loss.backward()

                accum += 1
                cur = float(loss_out.total.detach())
                loss_ema = cur if loss_ema is None else cfg.best_ema_beta * loss_ema + (1 - cfg.best_ema_beta) * cur
                if accum >= cfg.grad_accum:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    accum = 0
                    iteration += 1

                    if tb is not None:
                        for k, v in loss_out.detach_dict().items():
                            tb.add_scalar(f"loss/{k}", v, iteration)
                        tb.add_scalar("loss/total_ema", loss_ema, iteration)
                        tb.add_scalar("lr", sched.get_last_lr()[0], iteration)
                    if iteration % cfg.log_iter == 0 and self.is_main:
                        dt = time.time() - last_log
                        last_log = time.time()
                        line = (f"iter={iteration} total={cur:.4f} ema={loss_ema:.4f} "
                                f"lr={sched.get_last_lr()[0]:.2e} {cfg.log_iter/dt:.2f}it/s")
                        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

                    # Rank-agnostic: depends only on iteration (synced across ranks)
                    # and the config — NOT on have_val (the val loader lives on rank 0
                    # only, so gating on it would make the early-stop broadcast
                    # asymmetric and deadlock the collectives).
                    is_val_iter = bool(cfg.val_iter and iteration % cfg.val_iter == 0)
                    metric = loss_ema
                    improved = False
                    if is_val_iter and self.is_main and have_val:
                        vmetrics = self.validate(dataloader_val, iteration, tb)
                        # Selection/early-stop metric: action_mse if the sampler ran,
                        # else the fixed-timestep noise loss (loss-only validation).
                        metric = vmetrics.get("val/action_mse")
                        if metric is None or metric != metric:  # absent or NaN
                            metric = vmetrics["val/noise_loss_mean"]
                        improved = metric < best_metric - 1e-5
                        if improved:
                            best_metric = metric
                            no_improve = 0
                        else:
                            no_improve += 1
                        line = "iter=%d VAL " % iteration + " ".join(
                            f"{k.removeprefix('val/')}={v:.4f}" for k, v in vmetrics.items())
                        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

                    if iteration % cfg.save_iter == 0 and self.is_main:
                        self.save_checkpoint(iteration, opt, "ckpt_last.pt")
                        # ckpt_best by val metric (or training EMA if no val loader).
                        if not have_val and metric is not None and metric < best_metric:
                            best_metric = metric
                            improved = True
                        if improved:
                            self.save_checkpoint(iteration, opt, "ckpt_best.pt")
                            log_f.write(f"iter={iteration} new best ({'val' if have_val else 'loss_ema'}={best_metric:.4f})\n")
                            log_f.flush()

                    # Early stopping: rank 0 decides on val iters; broadcast so all
                    # ranks leave together (no half-collective hang).
                    if is_val_iter and cfg.early_stop_patience > 0:
                        if self.is_main and no_improve >= cfg.early_stop_patience:
                            msg = (f"iter={iteration} EARLY STOP: val metric did not improve for "
                                   f"{no_improve} validations (best={best_metric:.4f})")
                            print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
                            stop = True
                        if self.distributed:
                            flag = torch.tensor([1 if stop else 0], device=self.device)
                            dist.broadcast(flag, src=0)
                            stop = bool(flag.item())

        if self.is_main:
            self.save_checkpoint(iteration, opt, "ckpt_last.pt")
            log_f.close()
            if tb is not None:
                tb.close()
        if self.distributed:
            dist.barrier()

    # Fixed timestep grid (fractions of the DDPM horizon) for the val noise-pred
    # loss; fixed seed -> curves comparable across checkpoints.
    VAL_TIMESTEP_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)

    @torch.no_grad()
    def validate(self, dataloader_val: DataLoader, iteration: int, tb=None) -> dict:
        cfg = self.config
        self.policy.eval()
        torch.manual_seed(cfg.seed + 7777)
        T = self.policy.num_train_timesteps
        grid = [max(0, min(T - 1, int(round(f * T)))) for f in self.VAL_TIMESTEP_FRACS]
        t_losses = {t: [] for t in grid}
        action_mse = []
        for batch in dataloader_val:
            obs, goal, action = self._batch_to_device(batch)
            b = action.shape[0]
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                for t in grid:
                    out = self.policy.compute_loss(obs, goal, action, timesteps=torch.full((b,), t, device=self.device))
                    t_losses[t].append(float(out.total))
                # Sampler-based action_mse is the (slow) per-step simulation; skip
                # it unless val_sample_steps > 0. Loss-only validation otherwise.
                if cfg.val_sample_steps > 0:
                    pred = self.policy.sample(obs, goal, n_steps=cfg.val_sample_steps, denormalize=False)
                    action_mse.append(float(((pred.pred_action_chunk.float() - action.float()) ** 2).mean()))
        metrics = {f"val/noise_loss_t{t}": sum(v) / len(v) for t, v in t_losses.items() if v}
        metrics["val/noise_loss_mean"] = sum(metrics.values()) / len(metrics) if metrics else float("nan")
        if action_mse:
            metrics["val/action_mse"] = sum(action_mse) / len(action_mse)
        if tb is not None:
            for k, v in metrics.items():
                tb.add_scalar(k, v, iteration)
        self.policy.train()
        if self.policy.freeze_backbone:
            self.policy.backbone.eval()
        return metrics

    def save_checkpoint(self, iteration: int, opt=None, name: str = "ckpt_last.pt") -> Path:
        path = self.run_dir / name
        blob = {"iteration": iteration, "policy_state": self.policy.state_dict(),
                "config": self.config.__dict__}
        if opt is not None:
            blob["optimizer_state"] = opt.state_dict()
        torch.save(blob, path)
        return path


__all__ = ["DPTrainer", "DPTrainerConfig"]
