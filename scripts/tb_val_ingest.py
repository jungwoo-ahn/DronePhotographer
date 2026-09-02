"""Parse val_sweep.sh output -> print the validation curve + write TensorBoard val/* scalars.

Canonical tag scheme (shared by every run in this repo so curves overlay in TensorBoard):
    train/loss  train/lr  train/grad_norm       (see scripts/tb_from_log.py, scripts/tb_align.py)
    val/recon_cm  val/recon_deg  val/within_20cm

  python scripts/tb_val_ingest.py <run_name> <sweep_dir>
"""
import re
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

run, sweep = sys.argv[1], Path(sys.argv[2])
REC = re.compile(r"->\s*end:\s*([\d.]+)\s*cm\s+([\d.]+)\s*deg.*?median\s*([\d.]+)")
WITHIN = re.compile(r"within 20cm:\s*(\d+)%")

rows = []
for f in sorted(sweep.glob("*.txt")):
    txt = f.read_text()
    m, w = REC.search(txt), WITHIN.search(txt)
    if not m:
        continue
    rows.append((int(f.stem), float(m.group(1)), float(m.group(2)),
                 float(m.group(3)), int(w.group(1)) if w else -1))

if not rows:
    print(f"[{run}] no parseable sweep results in {sweep}")
    sys.exit(0)

tb = Path("runs") / run / "tb"
tb.mkdir(parents=True, exist_ok=True)
w = SummaryWriter(str(tb))
for step, cm, deg, med, pct in rows:
    w.add_scalar("val/recon_cm", cm, step)
    w.add_scalar("val/recon_deg", deg, step)
    if pct >= 0:
        w.add_scalar("val/within_20cm", pct, step)
w.flush(); w.close()

best = min(rows, key=lambda r: r[1])
print(f"\n=== {run} validation curve (held-out scenes) ===")
print(f"{'step':>8} {'recon_cm':>9} {'median':>8} {'deg':>7} {'<20cm':>6}")
for step, cm, deg, med, pct in rows:
    print(f"{step:>8} {cm:>9.1f} {med:>8.1f} {deg:>7.2f} {pct:>5}%"
          + ("   <-- BEST" if step == best[0] else ""))
print(f"\nBEST checkpoint: {best[0]}  ({best[1]:.1f} cm / {best[2]:.2f} deg)")
print(f"TensorBoard val/* written to {tb}")
