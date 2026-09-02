"""Turn a closed-loop rollout (frames in <dir>/ctl/frame_NNN.jpg + rollout.json) into an
annotated GIF. No GPU needed — pure PIL over already-rendered frames.

  python scripts/make_rollout_gif.py <rollout_dir> <out.gif> [--width 512] [--ms 260]
"""
import argparse, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("rollout_dir")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--ms", type=int, default=260)
a = ap.parse_args()

rd = Path(a.rollout_dir)
frames = sorted((rd / "ctl").glob("frame_*.jpg"))
if not frames:
    sys.exit(f"no frames in {rd/'ctl'}")
rj = rd / "rollout.json"
traj = json.loads(rj.read_text())["trajectory"] if rj.exists() else []
gd = [t["goal_dist"] for t in traj]

imgs = []
for i, fp in enumerate(frames):
    im = Image.open(fp).convert("RGB")
    w = a.width; h = int(im.height * w / im.width)
    im = im.resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    g = gd[i] if i < len(gd) else None
    label = f"step {i:02d}   goal-dist {g:.3f}" if g is not None else f"step {i:02d}"
    # readable text: black outline + white fill, top-left
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            d.text((10 + dx, 8 + dy), label, font=font, fill=(0, 0, 0))
    d.text((10, 8), label, font=font, fill=(255, 255, 255))
    imgs.append(im)

# hold the last frame ~6x so the converged shot lingers
durations = [a.ms] * (len(imgs) - 1) + [a.ms * 6]
imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
print(f"wrote {a.out}  ({len(imgs)} frames, {Path(a.out).stat().st_size//1024} KB)")
