"""CosmosPolicyTrainer: pure flow-matching training loop.

With the latent-frame action/value design, the joint loss collapses to a single
flow-matching MSE over the entire latent sequence (image + action + value
frames). No separate action / value MSE terms.

Adds to the bare loop:
  - first-batch sanity print (shapes / ranges / finite check)
  - TensorBoard scalar logging (loss components, lr, gate, grad norm)
  - per-component val loss every `val_iter` (if a val loader is passed)
  - slim checkpoints — only conditioner trainables + buffers + action_scale
    (frozen backbone NOT re-saved each ckpt; ~1 MB vs ~4 GB)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    SummaryWriter = None
    _HAS_TB = False

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
    val_iter: int = 0                  # 0 disables val
    max_val_batches: int = 50          # cap so val stays fast
    # Checkpoint retention: keep the most recent `keep_last_n` periodic ckpts
    # (iter_*.pt) plus ckpt_best.pt (lowest val/total) plus ckpt_last.pt.
    keep_last_n: int = 3
    best_metric: str = "total"         # which val component decides "best"
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"


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

    # ------------------------------------------------------------------
    # Sanity / logging helpers
    # ------------------------------------------------------------------

    def _print_first_batch(self, batch: dict, log_f) -> None:
        """One-time dump of shapes, dtypes, ranges so dataset bugs surface fast."""
        lines = ["[sanity] first batch:"]
        for k, v in batch.items():
            if torch.is_tensor(v):
                vf = v.float()
                finite = bool(torch.isfinite(vf).all().item())
                lines.append(
                    f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype} "
                    f"min={vf.min().item():.4g} max={vf.max().item():.4g} "
                    f"mean={vf.mean().item():.4g} finite={finite}"
                )
            elif isinstance(v, dict):
                sample = {kk: (vv[0] if isinstance(vv, list) and vv else vv) for kk, vv in v.items()}
                lines.append(f"  {k:20s} dict (sample item 0): {sample}")
        for ln in lines:
            print(ln)
            log_f.write(ln + "\n")
        log_f.flush()

    def _grad_norm(self) -> float:
        """L2 norm of grads on trainable params (for monitoring)."""
        total_sq = 0.0
        for p in self.policy.parameters():
            if p.requires_grad and p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum().item())
        return total_sq ** 0.5

    def _gate_value(self) -> float:
        """Read the ShotProfileVectorConditioner.gate scalar (ramps from 0)."""
        try:
            return float(self.policy.conditioner.gate.detach().item())
        except (AttributeError, RuntimeError):
            return float("nan")

    @torch.no_grad()
    def _validate(self, dataloader_val: DataLoader, max_batches: int) -> dict:
        """Return mean per-component val loss."""
        self.policy.eval()
        sums = {"total": 0.0, "world": 0.0, "action": 0.0, "value": 0.0}
        n = 0
        for i, batch in enumerate(dataloader_val):
            if i >= max_batches:
                break
            state_img = batch["state_image"].to(self.device, dtype=self.dtype)
            next_img = batch["next_state_image"].to(self.device, dtype=self.dtype)
            goal = batch["goal_vec"].to(self.device, dtype=self.dtype)
            action_chunk = batch["action_chunk"].to(self.device, dtype=self.dtype)
            value_target = batch["value_target"].to(self.device, dtype=self.dtype)
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                clip = self.vae.assemble_clip(state_img, next_img)
                image_latent = self.vae.encode(clip)
                lo = self.policy.compute_loss(
                    image_latent=image_latent,
                    action_chunk=action_chunk,
                    goal_vec=goal,
                    value_target=value_target,
                )
            d = lo.detach_dict()
            for k in sums:
                if k in d:
                    sums[k] += d[k]
            n += 1
        self.policy.train()
        return {k: (v / max(1, n)) for k, v in sums.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self, dataloader_train: DataLoader, dataloader_val: Optional[DataLoader] = None) -> None:
        cfg = self.config
        self.policy.to(self.device)
        self.vae.to(self.device)
        opt, sched = self._build_optimizer()
        scaler = torch.amp.GradScaler(self.device.type) if self.dtype == torch.float16 else None

        log_path = self.run_dir / "train.log"
        log_f = open(log_path, "a")

        tb_writer = None
        if _HAS_TB:
            tb_dir = self.run_dir / "tb"
            tb_dir.mkdir(exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"[trainer] tensorboard logdir = {tb_dir}")
            log_f.write(f"[trainer] tensorboard logdir = {tb_dir}\n")

        # Report trainable param count up front so the user sees how many params
        # actually update (vs the frozen backbone).
        n_trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.policy.parameters())
        msg = f"[trainer] params: trainable={n_trainable:,}  total={n_total:,}"
        print(msg); log_f.write(msg + "\n"); log_f.flush()

        iteration = 0
        accum = 0
        opt.zero_grad(set_to_none=True)
        last_log = time.time()
        first_batch_printed = False
        best_val: Optional[float] = None

        while iteration < cfg.max_iter:
            for batch in dataloader_train:
                if iteration >= cfg.max_iter:
                    break

                if not first_batch_printed:
                    self._print_first_batch(batch, log_f)
                    first_batch_printed = True

                state_img = batch["state_image"].to(self.device, dtype=self.dtype)
                next_img = batch["next_state_image"].to(self.device, dtype=self.dtype)
                goal = batch["goal_vec"].to(self.device, dtype=self.dtype)
                action_chunk = batch["action_chunk"].to(self.device, dtype=self.dtype)
                value_target = batch["value_target"].to(self.device, dtype=self.dtype)

                with torch.amp.autocast(self.device.type, dtype=self.dtype):
                    clip = self.vae.assemble_clip(state_img, next_img)        # (B, C, T=4, H, W)
                    image_latent = self.vae.encode(clip)                       # (B, 16, T_lat, H_lat, W_lat)
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
                if accum >= cfg.grad_accum:
                    grad_norm = self._grad_norm()
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
                        gate = self._gate_value()
                        lr = sched.get_last_lr()[0]
                        line = (
                            f"iter={iteration} "
                            f"total={parts['total']:.4f} "
                            f"world={parts['world']:.4f} "
                            f"action={parts['action']:.4f} "
                            + (f"value={parts.get('value', 0):.4f} " if 'value' in parts else "")
                            + f"gate={gate:+.4f} "
                            f"|g|={grad_norm:.2e} "
                            f"lr={lr:.2e} "
                            f"{cfg.log_iter/dt:.2f}it/s"
                        )
                        print(line)
                        log_f.write(line + "\n")
                        log_f.flush()
                        if tb_writer is not None:
                            tb_writer.add_scalar("train/total", parts["total"], iteration)
                            tb_writer.add_scalar("train/world", parts["world"], iteration)
                            tb_writer.add_scalar("train/action", parts["action"], iteration)
                            if "value" in parts:
                                tb_writer.add_scalar("train/value", parts["value"], iteration)
                            tb_writer.add_scalar("train/lr", lr, iteration)
                            tb_writer.add_scalar("train/gate", gate, iteration)
                            tb_writer.add_scalar("train/grad_norm", grad_norm, iteration)
                            tb_writer.add_scalar("train/it_per_s", cfg.log_iter / max(dt, 1e-6), iteration)

                    if cfg.val_iter > 0 and dataloader_val is not None and iteration % cfg.val_iter == 0:
                        val = self._validate(dataloader_val, cfg.max_val_batches)
                        vline = (f"[val] iter={iteration} "
                                 f"total={val['total']:.4f} world={val['world']:.4f} "
                                 f"action={val['action']:.4f} value={val['value']:.4f}")
                        print(vline); log_f.write(vline + "\n"); log_f.flush()
                        if tb_writer is not None:
                            tb_writer.add_scalar("val/total", val["total"], iteration)
                            tb_writer.add_scalar("val/world", val["world"], iteration)
                            tb_writer.add_scalar("val/action", val["action"], iteration)
                            tb_writer.add_scalar("val/value", val["value"], iteration)

                        # Track best val and snapshot when it improves.
                        metric_value = val.get(cfg.best_metric, val["total"])
                        if best_val is None or metric_value < best_val:
                            best_val = metric_value
                            self.save_checkpoint(iteration, name="ckpt_best.pt")
                            bline = f"[val] iter={iteration} new best {cfg.best_metric}={best_val:.4f} → ckpt_best.pt"
                            print(bline); log_f.write(bline + "\n"); log_f.flush()
                            if tb_writer is not None:
                                tb_writer.add_scalar("val/best_total", best_val, iteration)

                    if iteration % cfg.save_iter == 0:
                        self.save_checkpoint(iteration)
                        self._prune_periodic_checkpoints(cfg.keep_last_n)

        self.save_checkpoint(iteration, name="ckpt_last.pt")
        if tb_writer is not None:
            tb_writer.close()
        log_f.close()

    # ------------------------------------------------------------------
    # Checkpoints — slim, only what's actually learned / needed
    # ------------------------------------------------------------------

    def save_checkpoint(self, iteration: int, name: Optional[str] = None) -> Path:
        """Save only conditioner + action_scale. Backbone is frozen and lives in
        the HF cache, so re-saving it every checkpoint wastes ~4 GB / file.
        """
        path = self.run_dir / (name or f"ckpt_iter{iteration:07d}.pt")
        full_sd = self.policy.state_dict()
        slim_sd = {
            k: v for k, v in full_sd.items()
            if not k.startswith("transformer.")
        }
        torch.save(
            {
                "iteration": iteration,
                "policy_state": slim_sd,            # ~1 MB instead of ~4 GB
                "config": self.config.__dict__,
                "trainable_param_names": [
                    n for n, p in self.policy.named_parameters() if p.requires_grad
                ],
            },
            path,
        )
        return path

    def _prune_periodic_checkpoints(self, keep_last_n: int) -> None:
        """Keep only the most recent `keep_last_n` periodic `ckpt_iter*.pt`
        files. `ckpt_best.pt` and `ckpt_last.pt` are NEVER deleted.
        """
        if keep_last_n <= 0:
            return
        periodic = sorted(self.run_dir.glob("ckpt_iter*.pt"))
        for old in periodic[:-keep_last_n]:
            try:
                old.unlink()
            except OSError:
                pass
