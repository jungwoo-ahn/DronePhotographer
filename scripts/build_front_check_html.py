"""Driver: render canonical FRONT/RIGHT/BACK/LEFT views for every unique
object referenced in placement json files, then build a single HTML page
showing all of them so a human can verify which assets actually face +Y
in their default orientation (the renderer's assumed "forward").

Usage:
    python scripts/build_front_check_html.py \
        --placements_dir data/vlm_object_placing \
        --output_dir outputs/front_check \
        [--limit 25] [--blender_bin blender/blender] \
        [--resolution 512] [--samples 16] [--skip_existing]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def collect_unique_objects(placements_dir: Path) -> dict[str, str]:
    """Return {object_file: object_name}. First occurrence wins."""
    seen: dict[str, str] = {}
    for f in sorted(placements_dir.glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        obj_file = d.get("object_file")
        obj_name = d.get("object")
        if obj_file and obj_file not in seen:
            seen[obj_file] = obj_name or Path(obj_file).stem
    return seen


def slug(name: str, max_len: int = 50) -> str:
    return name[:max_len].replace("/", "_")


def render_one(blender_bin: str, script: str, obj_file: str, out_dir: Path,
               resolution: int, samples: int, timeout: int) -> bool:
    cmd = [
        blender_bin, "--background", "--python", script, "--",
        "--object_file", obj_file,
        "--output_dir", str(out_dir),
        "--resolution", str(resolution),
        "--samples", str(samples),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    if r.returncode != 0:
        # Save stderr tail for debugging
        log_path = out_dir / "render.err.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text((r.stdout or "") + "\n---STDERR---\n" + (r.stderr or ""))
        print(f"  FAIL (exit {r.returncode}); see {log_path}")
        return False
    return True


def build_html(rendered: list[tuple[str, Path]], out_root: Path) -> Path:
    sections = []
    toc_links = []
    for i, (obj_name, per_obj_dir) in enumerate(rendered):
        rel = per_obj_dir.relative_to(out_root)
        cards = []
        for label in ("FRONT", "RIGHT", "BACK", "LEFT"):
            img = per_obj_dir / f"{label.lower()}.png"
            if not img.exists():
                cards.append(
                    f'<div class="card missing"><div class="cap"><span class="lbl lbl-{label.lower()}">{label}</span></div>(missing)</div>'
                )
                continue
            cards.append(f"""
            <div class="card">
              <div class="cap"><span class="lbl lbl-{label.lower()}">{label}</span></div>
              <img src="{rel}/{label.lower()}.png" loading="lazy" />
            </div>
            """)
        anchor = f"obj-{i}"
        toc_links.append(f'<a href="#{anchor}">{obj_name}</a>')
        sections.append(f"""
        <section id="{anchor}">
          <h2>{obj_name}</h2>
          <div class="grid">{''.join(cards)}</div>
        </section>
        """)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>object front check (canonical renders)</title>
<style>
  body {{ font-family: ui-monospace, Menlo, monospace; background:#0b0b0b; color:#ddd;
         margin:18px 24px }}
  h1 {{ font-size:16px; margin:0 0 6px }}
  h2 {{ font-size:13px; color:#7df; margin:14px 0 6px }}
  section {{ border-top:1px solid #222; padding:8px 0 14px }}
  .grid {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:8px }}
  .card {{ background:#181818; border:1px solid #2a2a2a; border-radius:5px; padding:6px;
           position:relative }}
  .card img {{ width:100%; display:block; border-radius:3px }}
  .card.missing {{ display:flex; align-items:center; justify-content:center;
                   color:#666; aspect-ratio:1; font-size:12px }}
  .cap {{ font-size:11px; color:#aaa; margin-bottom:4px }}
  .lbl {{ display:inline-block; padding:2px 8px; border-radius:3px; font-weight:bold; font-size:11px }}
  .lbl-front {{ background:#ff3030; color:white }}
  .lbl-back  {{ background:#3060ff; color:white }}
  .lbl-right {{ background:#ffaa00; color:black }}
  .lbl-left  {{ background:#00cc88; color:black }}
  .legend {{ font-size:11px; color:#aaa; margin-bottom:14px; line-height:1.6 }}
  .toc {{ font-size:11px; color:#888; margin-bottom:18px; line-height:1.7 }}
  .toc a {{ color:#7df; margin-right:10px; text-decoration:none }}
  .toc a:hover {{ text-decoration:underline }}
</style></head><body>
<h1>object front orientation — {len(sections)} unique object(s)</h1>
<div class="legend">
  Each row is one object rendered fresh with rotation_z_deg=0 from 4 canonical
  camera angles. The renderer assumes the object's local +Y is its front; the
  red 3D arrow on the ground points +Y so you can see which way the assumed
  front goes.<br>
  ✓ <b>FRONT</b> view shows a face → assumption is correct.<br>
  ✗ FRONT shows the back → that asset needs <code>rotation_z_deg=180</code>
  (the actual face is in the BACK column).<br>
  ✗ FRONT shows a side → asset needs <code>rotation_z_deg=±90</code>.
</div>
<div class="toc">{' · '.join(toc_links)}</div>
{''.join(sections)}
</body></html>
"""
    out_path = out_root / "front_check.html"
    out_path.write_text(html)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--placements_dir", default="data/vlm_object_placing")
    p.add_argument("--output_dir", default="outputs/front_check")
    p.add_argument("--limit", type=int, default=999)
    p.add_argument("--blender_bin", default="blender/blender")
    p.add_argument("--script", default="scripts/render_object_canonical_views.py")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--skip_existing", action="store_true",
                   help="skip objects whose meta.json already exists")
    args = p.parse_args()

    placements_dir = Path(args.placements_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    objs = collect_unique_objects(placements_dir)
    items = list(objs.items())[:args.limit]
    if not items:
        raise SystemExit(f"no objects found under {placements_dir}")
    print(f"Found {len(objs)} unique objects; rendering first {len(items)}")

    rendered: list[tuple[str, Path]] = []
    fails = 0
    for i, (obj_file, obj_name) in enumerate(items):
        per_obj_dir = out_root / slug(obj_name)
        if args.skip_existing and (per_obj_dir / "meta.json").exists():
            print(f"[{i+1}/{len(items)}] skip (exists): {obj_name[:60]}")
            rendered.append((obj_name, per_obj_dir))
            continue
        per_obj_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(items)}] rendering: {obj_name[:60]}")
        ok = render_one(
            args.blender_bin, args.script, obj_file, per_obj_dir,
            args.resolution, args.samples, args.timeout,
        )
        if ok:
            rendered.append((obj_name, per_obj_dir))
        else:
            fails += 1

    out_path = build_html(rendered, out_root)
    print(f"\nrendered {len(rendered)} / {len(items)} objects ({fails} failed)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
