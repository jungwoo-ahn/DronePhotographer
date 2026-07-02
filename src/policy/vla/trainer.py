"""VLATrainer — lean flow-matching training loop for the VLA baseline.

Mirrors the proven structure of `cosmos/trainer.py` (DDP, grad accumulation, LR
warmup, EMA-selected best/last checkpoints, TensorBoard) but with VLM input-prep
instead of VAE encoding, and a single action-flow loss. Per our families-self-
contained convention it's a separate ~130-line loop rather than a shared base.
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

from src.policy.vla.model import VLAActionPolicy


class _LossForward(nn.Module):
    """DDP needs grads to flow through forward; expose compute_loss as forward."""

    def __init__(self, policy: VLAActionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **kwargs):
        return self.policy.compute_loss(**kwargs)


@dataclass
class VLATrainerConfig:
    output_root: Path
    run_name: str
    max_iter: int = 50000
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_iter: int = 1000
    save_iter: int = 2000
    log_iter: int = 100
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    best_ema_beta: float = 0.98
    # Validation (rank 0). 0 = off. ckpt_best is selected by val action MSE when
    # a val loader is given, else by training-loss EMA.
    val_iter: int = 0
    # >0 also runs the slow sampler for action_mse; 0 = val LOSS ONLY.
    val_sample_steps: int = 10
    # Stop if the val metric hasn't improved for this many validations (0 = off).
    early_stop_patience: int = 0


class VLATrainer:
    def __init__(self, policy: VLAActionPolicy, config: VLATrainerConfig) -> None:
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
        params = [p for p in self.policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        warmup = max(1, self.config.warmup_iter)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / warmup))
        return opt, sched

    def _batch_to_device(self, batch: dict):
        """Move a VLACollate batch (vlm_inputs dict + goal + action) to device."""
        vlm = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch["vlm_inputs"].items()}
        goal = batch["goal_vec"].to(self.device, self.dtype)
        action = batch["action_chunk"].to(self.device, self.dtype)
        return vlm, goal, action

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

        log_f = open(self.run_dir / "train.log", "a") if self.is_main else None
        tb = None
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        iteration = 0
        accum = 0
        epoch = 0
        opt.zero_grad(set_to_none=True)
        last_log = time.time()
        loss_ema: Optional[float] = None
        best_metric = float("inf")          # val metric if val present, else loss EMA
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
                vlm_inputs, goal, action_chunk = self._batch_to_device(batch)

                is_sync = accum + 1 >= cfg.grad_accum
                sync_ctx = (loss_module.no_sync() if self.distributed and not is_sync else contextlib.nullcontext())
                with sync_ctx, torch.amp.autocast(self.device.type, dtype=self.dtype):
                    loss_out = loss_module(vlm_inputs=vlm_inputs, goal_vec=goal, action_chunk=action_chunk)
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
                        # Goal-conditioning engagement (matches WAM's cond/goal_proj_norm):
                        # weight norm of the goal->soft-token projection.
                        tb.add_scalar("cond/goal_proj_norm",
                                      float(self.policy.goal_proj.weight.detach().norm()), iteration)
                    if iteration % cfg.log_iter == 0 and self.is_main:
                        dt = time.time() - last_log
                        last_log = time.time()
                        line = (f"iter={iteration} total={cur:.4f} ema={loss_ema:.4f} "
                                f"lr={sched.get_last_lr()[0]:.2e} {cfg.log_iter/dt:.2f}it/s")
                        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

                    # Rank-agnostic (iteration is synced); never gate on have_val,
                    # which is rank-0-only and would desync the early-stop broadcast.
                    is_val_iter = bool(cfg.val_iter and iteration % cfg.val_iter == 0)
                    metric = loss_ema
                    improved = False
                    if is_val_iter and self.is_main and have_val:
                        vmetrics = self.validate(dataloader_val, iteration, tb)
                        # action_mse if the sampler ran, else the fixed-sigma flow loss.
                        metric = vmetrics.get("val/action_mse")
                        if metric is None or metric != metric:
                            metric = vmetrics["val/flow_loss_mean"]
                        improved = metric < best_metric - 1e-5
                        if improved:
                            best_metric = metric; no_improve = 0
                        else:
                            no_improve += 1
                        line = "iter=%d VAL " % iteration + " ".join(
                            f"{k.removeprefix('val/')}={v:.4f}" for k, v in vmetrics.items())
                        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

                    if iteration % cfg.save_iter == 0 and self.is_main:
                        self.save_checkpoint(iteration, "ckpt_last.pt")
                        if not have_val and metric is not None and metric < best_metric:
                            best_metric = metric; improved = True
                        if improved:
                            self.save_checkpoint(iteration, "ckpt_best.pt")
                            log_f.write(f"iter={iteration} new best ({'val' if have_val else 'loss_ema'}={best_metric:.4f})\n")
                            log_f.flush()

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
            self.save_checkpoint(iteration, "ckpt_last.pt")
            log_f.close()
            if tb is not None:
                tb.close()
        if self.distributed:
            dist.barrier()

    # Fixed sigma grid for the val flow loss (random per-batch training sigma is
    # high-variance); fixed noise seed → curves comparable across checkpoints.
    VAL_SIGMA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)

    @torch.no_grad()
    def validate(self, dataloader_val: DataLoader, iteration: int, tb=None) -> dict:
        cfg = self.config
        self.policy.eval()
        torch.manual_seed(cfg.seed + 7777)
        sigma_losses = {s: [] for s in self.VAL_SIGMA_GRID}
        action_mse = []
        for batch in dataloader_val:
            vlm, goal, action = self._batch_to_device(batch)
            b = action.shape[0]
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                for s in self.VAL_SIGMA_GRID:
                    out = self.policy.compute_loss(vlm, goal, action, sigma=torch.full((b,), s, device=self.device))
                    sigma_losses[s].append(float(out.total))
                # Sampler-based action_mse is slow (forward through the 2B VLM per
                # step); skip unless val_sample_steps > 0 (loss-only validation).
                if cfg.val_sample_steps > 0:
                    pred = self.policy.sample(vlm, goal, n_steps=cfg.val_sample_steps, denormalize=False)
                    action_mse.append(float(((pred.pred_action_chunk.float() - action.float()) ** 2).mean()))
        metrics = {f"val/flow_loss_sigma_{s}": sum(v) / len(v) for s, v in sigma_losses.items() if v}
        metrics["val/flow_loss_mean"] = sum(metrics.values()) / len(metrics) if metrics else float("nan")
        if action_mse:
            metrics["val/action_mse"] = sum(action_mse) / len(action_mse)
        if tb is not None:
            for k, v in metrics.items():
                tb.add_scalar(k, v, iteration)
        self.policy.train()
        return metrics

    def save_checkpoint(self, iteration: int, name: str) -> Path:
        path = self.run_dir / name
        torch.save({"iteration": iteration, "policy_state": self.policy.state_dict(),
                    "config": self.config.__dict__}, path)
        return path


__all__ = ["VLATrainer", "VLATrainerConfig"]
