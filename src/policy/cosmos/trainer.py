"""CosmosPolicyTrainer: pure flow-matching training loop.

With the latent-frame action/value design, the joint loss collapses to a single
flow-matching MSE over the entire latent sequence (image + action + value
frames). No separate action / value MSE terms.
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

from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.vae import CosmosVAEWrapper


class _LossForward(nn.Module):
    """DDP needs gradients to flow through `forward`; the policy exposes
    `compute_loss` instead, so this shim makes that the forward."""

    def __init__(self, policy: CosmosWorldActionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **kwargs):
        return self.policy.compute_loss(**kwargs)


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
    # Validate every N iters (0 = off). Runs on rank 0 only.
    val_iter: int = 0
    # Euler steps for the sampling-based val metrics (action MSE / value MAE).
    val_sample_steps: int = 8
    # Denoising-sample viz: every N iters dump [state|pred-world|gt-world] grids +
    # action/value pred-vs-GT json to <run>/samples/iter<N>/ (0 = off, rank 0 only).
    viz_iter: int = 0
    viz_samples: int = 4
    viz_steps: int = 16
    # Resume from a run dir or checkpoint .pt (restores optimizer/scheduler/
    # iteration/RNG so a stopped run continues exactly). None = fresh run.
    resume_from: Optional[str] = None


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
        # Distributed state (torchrun sets these up in the entry script).
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = self.rank == 0
        # Resolve a resume checkpoint (a run dir or a .pt file) before choosing
        # the run dir: a resumed run continues in its original dir.
        self.resume_ckpt_path = self._resolve_resume(config.resume_from)
        self.run_dir = self._make_run_dir()
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        # Per-rank seed so EDM/flow sigma draws decorrelate across ranks.
        torch.manual_seed(config.seed + self.rank)

    def _resolve_resume(self, resume_from: Optional[str]) -> Optional[Path]:
        if not resume_from:
            return None
        p = Path(resume_from)
        ckpt = p / "ckpt_last.pt" if p.is_dir() else p
        if not ckpt.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt}")
        return ckpt

    def _make_run_dir(self) -> Path:
        # Resuming: continue in the checkpoint's own run dir so TensorBoard,
        # train.log and checkpoints stay contiguous. Every rank derives the same
        # path from the same config, so no broadcast is needed.
        if self.resume_ckpt_path is not None:
            run_dir = self.resume_ckpt_path.parent
            if self.is_main:
                run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        # Fresh run: rank 0 names the dir by timestamp; other ranks receive it by
        # broadcast so a tick boundary can't split the run across two directories.
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.distributed:
            obj = [ts if self.is_main else None]
            dist.broadcast_object_list(obj, src=0)
            ts = obj[0]
        run_dir = self.config.output_root / f"{ts}_{self.config.run_name}"
        if self.is_main:
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

        # Resume: restore policy weights before the DDP wrap so every rank enters
        # the all-reduce with identical parameters (all ranks read the same NAS
        # checkpoint). Optimizer/scheduler/loop state is restored below, once they
        # are built. weights_only=False: it's our own trusted checkpoint and the
        # saved `config` holds non-tensor objects (Path), which the torch>=2.6
        # default (weights_only=True) would refuse to unpickle.
        resume_state = None
        if self.resume_ckpt_path is not None:
            resume_state = torch.load(
                self.resume_ckpt_path, map_location=self.device, weights_only=False,
            )
            self.policy.load_state_dict(resume_state["policy_state"])

        # DDP wraps a forward shim around compute_loss; self.policy stays the
        # raw module (checkpoints save un-prefixed state dicts).
        loss_module: nn.Module = _LossForward(self.policy)
        if self.distributed:
            loss_module = nn.parallel.DistributedDataParallel(
                loss_module, device_ids=[self.device.index], output_device=self.device.index,
            )

        opt, sched = self._build_optimizer()
        scaler = torch.amp.GradScaler(self.device.type) if self.dtype == torch.float16 else None

        log_f = open(self.run_dir / "train.log", "a") if self.is_main else None
        tb = None
        if self.is_main:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        iteration = 0
        accum = 0
        epoch = 0
        # Keep only best + last checkpoints. "Best" = lowest EMA of the total
        # loss (per-iter loss is too noisy under EDM sigma sampling to compare raw).
        loss_ema: Optional[float] = None
        best_ema = float("inf")

        # Resume: restore optimizer/scheduler/scaler + loop counters and RNG so
        # the continued run faithfully extends the original.
        if resume_state is not None:
            if "optimizer_state" in resume_state:
                opt.load_state_dict(resume_state["optimizer_state"])
            if "scheduler_state" in resume_state:
                sched.load_state_dict(resume_state["scheduler_state"])
            if scaler is not None and "scaler_state" in resume_state:
                scaler.load_state_dict(resume_state["scaler_state"])
            iteration = int(resume_state.get("iteration", 0))
            epoch = int(resume_state.get("epoch", 0))
            loss_ema = resume_state.get("loss_ema", None)
            best_ema = resume_state.get("best_ema", float("inf"))
            if "torch_rng_state" in resume_state:
                torch.set_rng_state(resume_state["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in resume_state:
                torch.cuda.set_rng_state(resume_state["cuda_rng_state"].cpu(), self.device)
            if self.is_main:
                print(f"[resume] {self.resume_ckpt_path} -> continuing at "
                      f"iter {iteration}/{cfg.max_iter}")
                if log_f is not None:
                    log_f.write(f"[resume] from {self.resume_ckpt_path} at iter {iteration}\n")
                    log_f.flush()

        opt.zero_grad(set_to_none=True)
        last_log = time.time()

        # Fixed viz batch: the first `viz_samples` val examples, collated once so the
        # same states/goals are re-sampled every viz step (progress is comparable across
        # iters). val_loader is batch_size=1, so we pull and concat a few batches.
        viz_batch = None
        if cfg.viz_iter and dataloader_val is not None and self.is_main:
            collected = []
            for vb in dataloader_val:
                collected.append(vb)
                if len(collected) >= cfg.viz_samples:
                    break
            if collected:
                keys = ("state_image", "next_state_image", "goal_vec", "action_chunk", "value_target")
                viz_batch = {k: torch.cat([c[k] for c in collected], dim=0) for k in keys}

        while iteration < cfg.max_iter:
            if isinstance(getattr(dataloader_train, "sampler", None), DistributedSampler):
                dataloader_train.sampler.set_epoch(epoch)
            epoch += 1
            for batch in dataloader_train:
                if iteration >= cfg.max_iter:
                    break
                state_img = batch["state_image"].to(self.device, dtype=self.dtype)
                next_img = batch["next_state_image"].to(self.device, dtype=self.dtype)
                goal = batch["goal_vec"].to(self.device, dtype=self.dtype)
                action_chunk = batch["action_chunk"].to(self.device, dtype=self.dtype)
                value_target = batch["value_target"].to(self.device, dtype=self.dtype)

                # Under DDP, skip gradient all-reduce on non-final micro-steps.
                is_sync_step = accum + 1 >= cfg.grad_accum
                sync_ctx = (
                    loss_module.no_sync()
                    if self.distributed and not is_sync_step
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    with torch.amp.autocast(self.device.type, dtype=self.dtype):
                        # Encode state and goal frames separately (T=1 each — the Wan
                        # VAE needs (T-1)%4==0) and concat: ALOHA-style T_img=2 latents.
                        image_latent = self.vae.encode_pair_frames(state_img, next_img)  # (B, 16, 2, h, w)
                        loss_out = loss_module(
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

                    if tb is not None:
                        parts = loss_out.detach_dict()   # total, world, action, value
                        for k, v in parts.items():
                            tb.add_scalar(f"train/{k}", v, iteration)
                        tb.add_scalar("train/lr", sched.get_last_lr()[0], iteration)
                        tb.add_scalar("train/total_ema", loss_ema, iteration)

                    if iteration % cfg.log_iter == 0 and self.is_main:
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

                    if (
                        cfg.val_iter
                        and dataloader_val is not None
                        and iteration % cfg.val_iter == 0
                        and self.is_main
                    ):
                        metrics = self.validate(dataloader_val, iteration, tb)
                        line = f"iter={iteration} VAL " + " ".join(
                            f"{k.removeprefix('val/')}={v:.4f}" for k, v in metrics.items()
                        )
                        print(line)
                        log_f.write(line + "\n")
                        log_f.flush()

                    # Denoising-sample viz dump (rank 0). Wrapped so a viz/decode failure
                    # never kills the run — it just logs and continues training.
                    if (
                        cfg.viz_iter
                        and viz_batch is not None
                        and iteration % cfg.viz_iter == 0
                        and self.is_main
                    ):
                        from src.policy.cosmos.sample_logging import log_denoise_samples

                        try:
                            s = log_denoise_samples(
                                self.policy, self.vae, viz_batch,
                                self.run_dir / "samples" / f"iter{iteration:06d}",
                                device=self.device, dtype=self.dtype,
                                n_steps=cfg.viz_steps, max_samples=cfg.viz_samples,
                            )
                            log_f.write(f"iter={iteration} VIZ samples/iter{iteration:06d}/ "
                                        f"action_rmse_mean={s.get('action_rmse_mean')}\n")
                            log_f.flush()
                        except Exception as e:  # pragma: no cover - viz must never crash training
                            print(f"[viz] iter {iteration} failed: {e!r}")

                    if iteration % cfg.save_iter == 0 and self.is_main:
                        ckpt_kw = dict(
                            optimizer=opt, scheduler=sched, scaler=scaler,
                            loss_ema=loss_ema, best_ema=best_ema, epoch=epoch,
                        )
                        self.save_checkpoint(iteration, name="ckpt_last.pt", **ckpt_kw)
                        if loss_ema is not None and loss_ema < best_ema:
                            best_ema = loss_ema
                            ckpt_kw["best_ema"] = best_ema
                            self.save_checkpoint(iteration, name="ckpt_best.pt", **ckpt_kw)
                            log_f.write(f"iter={iteration} new best (loss EMA {loss_ema:.4f})\n")

        if self.is_main:
            self.save_checkpoint(
                iteration, name="ckpt_last.pt", optimizer=opt, scheduler=sched,
                scaler=scaler, loss_ema=loss_ema, best_ema=best_ema, epoch=epoch,
            )
            log_f.close()
            if tb is not None:
                tb.close()
        if self.distributed:
            dist.barrier()

    # Fixed sigma grid for the validation flow loss: the training loss draws a
    # random sigma per batch (high variance by construction), so val evaluates
    # every batch at these fixed noise levels with a fixed noise seed — curves
    # are comparable across checkpoints.
    VAL_SIGMA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)

    @torch.no_grad()
    def validate(self, dataloader_val: DataLoader, iteration: int, tb=None) -> dict:
        cfg = self.config
        self.policy.eval()
        # Fixed noise per validation pass -> the only thing that changes between
        # evals is the model.
        torch.manual_seed(cfg.seed + 7777)

        comp = {"world": [], "action": [], "value": []}   # teacher-forced flow loss per component
        action_mse, value_mae, n = [], [], 0
        for batch in dataloader_val:
            state_img = batch["state_image"].to(self.device, dtype=self.dtype)
            next_img = batch["next_state_image"].to(self.device, dtype=self.dtype)
            goal = batch["goal_vec"].to(self.device, dtype=self.dtype)
            action_chunk = batch["action_chunk"].to(self.device, dtype=self.dtype)
            value_target = batch["value_target"].to(self.device, dtype=self.dtype)
            b = state_img.shape[0]

            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                image_latent = self.vae.encode_pair_frames(state_img, next_img)
                # (a) flow loss on the fixed sigma grid
                for s in self.VAL_SIGMA_GRID:
                    out = self.policy.compute_loss(
                        image_latent=image_latent,
                        action_chunk=action_chunk,
                        goal_vec=goal,
                        value_target=value_target,
                        sigma=torch.full((b,), s),
                    )
                    comp["world"].append(float(out.world))
                    comp["action"].append(float(out.action))
                    if out.value is not None:
                        comp["value"].append(float(out.value))
                # (b) sample-quality: run the Euler sampler, compare to ground truth
                pred = self.policy.sample(
                    image_latent=image_latent, goal_vec=goal,
                    n_steps=cfg.val_sample_steps, denormalize=False,
                )
            action_mse.append(float(((pred.pred_action_chunk.float() - action_chunk.float()) ** 2).mean()))
            if pred.pred_value is not None:
                value_mae.append(float((pred.pred_value.float() - value_target.float()).abs().mean()))
            n += b

        # Flow loss per component (teacher-forced, averaged over the fixed-sigma grid) + the
        # combined total — SAME names as train/{world,action,value,total} so they overlay in TB.
        metrics = {f"val/{name}": sum(vals) / len(vals) for name, vals in comp.items() if vals}
        metrics["val/total"] = sum(metrics.get(f"val/{k}", 0.0) for k in ("world", "action", "value"))
        # Sampled-quality (run the Euler sampler): val/action_mse is the DEPLOYED action error,
        # distinct from val/action (the single-step denoising loss above).
        if action_mse:
            metrics["val/action_mse"] = sum(action_mse) / len(action_mse)
        if value_mae:
            metrics["val/value_mae"] = sum(value_mae) / len(value_mae)
        if tb is not None:
            for k, v in metrics.items():
                tb.add_scalar(k, v, iteration)
        self.policy.train()
        return metrics

    def save_checkpoint(
        self,
        iteration: int,
        name: Optional[str] = None,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler=None,
        loss_ema: Optional[float] = None,
        best_ema: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> Path:
        path = self.run_dir / (name or f"ckpt_iter{iteration:07d}.pt")
        ckpt = {
            "iteration": iteration,
            "policy_state": self.policy.state_dict(),
            "config": self.config.__dict__,
        }
        # Full training state — passed in from the training loop so a preempted /
        # time-limited / crashed run can resume exactly (see --resume). Absent
        # when callers ask for a weights-only snapshot.
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state"] = scheduler.state_dict()
        if scaler is not None:
            ckpt["scaler_state"] = scaler.state_dict()
        if loss_ema is not None:
            ckpt["loss_ema"] = loss_ema
        if best_ema is not None:
            ckpt["best_ema"] = best_ema
        if epoch is not None:
            ckpt["epoch"] = epoch
        ckpt["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            ckpt["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        torch.save(ckpt, path)
        return path
