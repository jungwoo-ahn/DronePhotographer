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

    def fit(self, dataloader_train: DataLoader) -> None:
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
        best_ema = float("inf")

        while iteration < cfg.max_iter:
            if isinstance(getattr(dataloader_train, "sampler", None), DistributedSampler):
                dataloader_train.sampler.set_epoch(epoch)
            epoch += 1
            for batch in dataloader_train:
                if iteration >= cfg.max_iter:
                    break
                vlm_inputs, goal, action_chunk = self.policy.prepare_inputs(batch, self.device, self.dtype)

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
                    if iteration % cfg.log_iter == 0 and self.is_main:
                        dt = time.time() - last_log
                        last_log = time.time()
                        line = (f"iter={iteration} total={cur:.4f} ema={loss_ema:.4f} "
                                f"lr={sched.get_last_lr()[0]:.2e} {cfg.log_iter/dt:.2f}it/s")
                        print(line); log_f.write(line + "\n"); log_f.flush()
                    if iteration % cfg.save_iter == 0 and self.is_main:
                        self.save_checkpoint(iteration, "ckpt_last.pt")
                        if loss_ema is not None and loss_ema < best_ema:
                            best_ema = loss_ema
                            self.save_checkpoint(iteration, "ckpt_best.pt")

        if self.is_main:
            self.save_checkpoint(iteration, "ckpt_last.pt")
            log_f.close()
            if tb is not None:
                tb.close()
        if self.distributed:
            dist.barrier()

    def save_checkpoint(self, iteration: int, name: str) -> Path:
        path = self.run_dir / name
        torch.save({"iteration": iteration, "policy_state": self.policy.state_dict(),
                    "config": self.config.__dict__}, path)
        return path


__all__ = ["VLATrainer", "VLATrainerConfig"]
