"""Pack everything bench_v2 needs that is NOT in git into one tar, for running off-cluster.

    python scripts/photoflow_make_bundle.py --out /path/photoflow_bench_v2_bundle.tar

Run on the cluster. The tar has two folders:
  shared/  -> DP_SHARED_ROOT: Blender 4.5 (+ its syslibs), the shared repo's two render scripts and
              the src/ modules they import, and only the scenes / subjects / placement JSONs the manifest uses
  v12/     -> copy INTO the V12 checkout (DP_V12_ROOT): data.json + start/goal frames per
              placement, and the placement geometry cache runs/bench_v2/_geom
About 6 GB. See docs/photoflow_bench_v2_runbook.md.
"""
import argparse
import json
import re
import tarfile
from pathlib import Path

V12 = Path("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
SHARED = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")
V6 = "data/vlm_object_placing_v6_260428_061326"
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(V12 / "configs/eval_episodes_v2.json"))
ap.add_argument("--out", required=True)
args = ap.parse_args()

eps = json.loads(Path(args.manifest).read_text())["episodes"]
# Only what the Blender render scripts import. NOT the whole shared src/: other modules there
# (src/vlm) carry hardcoded API keys, and a bundle gets copied to machines we do not control.
CODE = ["render_object.py", "scripts/blender_render_server.py", "scripts/blender_render_pose.py", "src/drones",
        "src/scenes", "src/utils", "src/blender", "src/detectors"]
CODE += [str(p.relative_to(SHARED)) for p in (SHARED / "src").glob("*.py")]
KEY = re.compile(rb"sk-[A-Za-z0-9_-]{15,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}")
for rel in CODE:
    root = SHARED / rel
    for f in ([root] if root.is_file() else root.rglob("*.py")):
        if KEY.search(f.read_bytes()):
            raise SystemExit(f"refusing to bundle {f}: it contains something that looks like an API key")
shared, v12 = {"blender", *CODE}, set()
for e in eps:
    dj = Path(e["data_json"])
    d = json.loads(dj.read_text())
    shared |= {str(Path(d["scene_file"]).parent), str(Path(d["object_file"]).parent),
               f"{V6}/{e['placement']}.json"}
    v12 |= {str(p.relative_to(V12)) for p in (dj, Path(e["start_image"]), Path(e["goal_image"]))}
    v12.add(f"runs/bench_v2/_geom/{e['placement']}.npz")


def keep(ti):
    return None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti


with tarfile.open(args.out, "w") as tar:
    for rel in sorted(shared):
        tar.add(SHARED / rel, arcname=f"shared/{rel}", filter=keep)
    for rel in sorted(v12):
        tar.add(V12 / rel, arcname=f"v12/{rel}", filter=keep)
print(f"wrote {args.out}: {len(shared)} shared entries, {len(v12)} v12 files, "
      f"{Path(args.out).stat().st_size / 1e9:.2f} GB")
