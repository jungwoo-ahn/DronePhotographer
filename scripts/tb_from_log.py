"""Write TensorBoard scalars from lerobot-train stdout logs (no wandb, no restart needed).

lerobot 0.6.2 has no native TB tracker and we run wandb-off, so parse the training log:
each logged line carries the EXACT step in its tqdm prefix ("... | 1400/300000 ...") and the
metrics in the appended INFO ("... loss:0.197 grdn:2.437 lr:1.1e-05 ..."). Emit loss/grad_norm/
lr to <run>/tb/. Backfills existing history, then tails for new lines. Handles runs that start
later (e.g. groot_fair). Run in an env with tensorboard (drone).

  python scripts/tb_from_log.py /tmp/pi05_fair.log:runs/pi05_fair/tb  /tmp/groot_fair.log:runs/groot_fair/tb
"""
import json, re, sys, time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

PAIRS = [a.split(":", 1) for a in sys.argv[1:]]
STEP = re.compile(r"(\d+)/(\d+)\s*\[")               # tqdm prefix "1400/300000 [" -> (done, total)
MET = {k: re.compile(rf"{k}:([0-9.eE+-]+)") for k in ("loss", "grdn", "lr")}
writers, seen, _full = {}, {}, {}


def _offset(tbdir: str, bar_total: int) -> int:
    """Global-step offset for a resumed run: full run length minus this bar's total."""
    if tbdir not in _full:
        cfg = Path(tbdir).parent / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        try:
            _full[tbdir] = int(json.loads(cfg.read_text())["steps"])
        except Exception:
            _full[tbdir] = 0
    full = _full[tbdir]
    return full - bar_total if 0 < bar_total < full else 0

def process(logf, tbdir):
    p = Path(logf)
    if not p.exists():
        return
    # Recreate the writer if the dir is missing: a cached SummaryWriter keeps an open handle to a
    # deleted inode, so if the run dir is ever rm -rf'd (e.g. relaunching a run) every subsequent
    # write silently goes nowhere. Resetting `seen` re-backfills the whole log.
    if tbdir not in writers or not Path(tbdir).exists():
        Path(tbdir).mkdir(parents=True, exist_ok=True)
        writers[tbdir] = SummaryWriter(tbdir); seen[tbdir] = -1
    w = writers[tbdir]
    text = p.read_text(errors="ignore").replace("\r", "\n")
    last = seen[tbdir]
    for line in text.split("\n"):
        if "loss:" not in line:
            continue
        ms = STEP.search(line)
        ml = MET["loss"].search(line)
        if not (ms and ml):
            continue
        # After a resume, lerobot's bar restarts at 0 and counts only the REMAINING steps
        # (e.g. "0/160000" when resuming a 300k run at 140k). Taking the bar step as the global
        # step rewinds the curve and makes TensorBoard purge the pre-resume history as an
        # orphaned restart, so shift by (full_run_steps - bar_total).
        step = int(ms.group(1)) + _offset(tbdir, int(ms.group(2)))
        if step <= last:
            continue
        w.add_scalar("train/loss", float(ml.group(1)), step)
        for k, tag in (("grdn", "train/grad_norm"), ("lr", "train/lr")):
            m = MET[k].search(line)
            if m:
                w.add_scalar(tag, float(m.group(1)), step)
        last = step
    if last > seen[tbdir]:
        seen[tbdir] = last; w.flush()

while True:
    for logf, tbdir in PAIRS:
        try:
            process(logf, tbdir)
        except Exception as e:
            print(f"[tb] {logf}: {e}", flush=True)
    time.sleep(60)
