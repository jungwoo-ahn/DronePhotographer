"""Rewrite a run's TensorBoard scalars onto the repo-canonical tag scheme.

Different trainers logged the same quantity under different names (DP: `loss/total`, `lr`;
SB3/AutoPhoto: `train/loss`, `train/learning_rate`; scripts/tb_from_log.py: `train/loss`,
`train/lr`), so nothing overlaid in `tensorboard --logdir runs`. This maps them all onto:

    train/loss   train/lr   train/grad_norm        val/loss  val/recon_cm  val/recon_deg ...

Unmapped tags pass through unchanged (run-specific extras are preserved, not dropped).
The original event files are moved to a backup dir OUTSIDE runs/ so TensorBoard does not
re-scan them as phantom runs.

  python scripts/tb_align.py runs/<run>/tb [...]        # never pass a RUNNING run's tb dir
"""
import shutil
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

CANON = {
    "loss": "train/loss", "loss/total": "train/loss", "train/loss": "train/loss",
    "loss/total_ema": "train/loss_ema", "loss/action": "train/loss_action",
    "lr": "train/lr", "learning_rate": "train/lr", "train/learning_rate": "train/lr",
    "train/lr": "train/lr",
    "grad_norm": "train/grad_norm", "train/grad_norm": "train/grad_norm",
    "val/noise_loss_mean": "val/loss",
}
BACKUP = Path("/home/nas5/jooyeolyun/tb_raw_backup")


def align(tbdir: Path) -> None:
    ea = EventAccumulator(str(tbdir)); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"{tbdir}: no scalars, skipped"); return

    data = {t: [(s.step, s.value) for s in ea.Scalars(t)] for t in tags}
    renamed = {t: CANON.get(t, t) for t in tags}

    dest = BACKUP / tbdir.parts[-2]; dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for ev in list(tbdir.glob("events.out.tfevents*")):
        shutil.move(str(ev), str(dest / ev.name)); moved += 1

    w = SummaryWriter(str(tbdir))
    for t, pts in data.items():
        for step, val in pts:
            w.add_scalar(renamed[t], val, step)
    w.flush(); w.close()

    changed = {k: v for k, v in renamed.items() if k != v}
    print(f"{tbdir}: {len(tags)} tags, {sum(len(v) for v in data.values())} points; "
          f"{moved} raw file(s) -> {dest}")
    for k, v in sorted(changed.items()):
        print(f"    {k}  ->  {v}")
    if not changed:
        print("    (already canonical)")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        align(Path(d))
