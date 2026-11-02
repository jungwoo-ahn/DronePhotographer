"""Swap re-placed subjects into a placement.

  merge_redo.py --base DIR --cands DIR [DIR ...] --out DIR --pick SCENE:i=k,j=k ... [SCENE:...]

Every scene of --base is written to --out; in the picked scenes subject i is replaced by its
candidate k from --cands (prepare_jobs.py --redo; a later directory wins for a scene both hold),
and the overview is redrawn on the candidates' clean overview render. Jobs keep their absolute
image paths, so nothing is copied but the overview.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from gscommon import draw_spots

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--cands", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pick", nargs="+", required=True)
a = ap.parse_args()

picks = {}
for p in a.pick:
    s, rest = p.split(":")
    picks[s] = {int(i): int(k) for i, k in (x.split("=") for x in rest.split(","))}

for sd in sorted(Path(a.base).iterdir()):
    if not (sd / "jobs.json").exists():
        continue
    od = Path(a.out) / sd.name; od.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((sd / "jobs.json").read_text())
    if sd.name not in picks:
        shutil.copy(sd / "overview.jpg", od / "overview.jpg")
    else:
        cd = [Path(c) / sd.name for c in a.cands if (Path(c) / sd.name / "jobs.json").exists()][-1]
        cands = json.loads((cd / "jobs.json").read_text())
        for i, k in picks[sd.name].items():
            new = next(c for c in cands if c["index"] == i and c["cand"] == k)
            jobs = [new if j["index"] == i else j for j in jobs]
        cam = json.loads((cd / "overview_cam.json").read_text())
        img = Image.open(cd / "overview_clean.jpg").convert("RGB")
        draw_spots(img, np.array(cam["c2w"]), np.array(cam["K"]),
                   [(np.array(j["spot_splat"]), str(j["index"]), (255, 40, 40)) for j in jobs])
        img.save(od / "overview.jpg", quality=88)
    (od / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"{sd.name}: {len(jobs)} jobs -> {od}")
