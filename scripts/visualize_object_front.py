"""Render a self-contained HTML page that overlays a *3D* arrow lying on the
floor of the scene at the object's footprint, pointing in the stored
``object_forward`` direction. The arrow is projected into each rendered
frame as a filled, shaded polygon so it reads as a real arrow lying on
the ground from any drone-camera viewpoint.

Use this to verify, by eye, that ``object_forward`` actually points to
the front of the rendered character. If the red arrow comes out of the
chest / nose, orientation is correct. If it points out the back of the
head, the stored ``object_rotation_z_deg`` is wrong.

Usage (single placement):
    python scripts/visualize_object_front.py \
        --placement_dir outputs/<run>/p0_<scene>_<obj> \
        [--n_samples 6]

Usage (many placements at once):
    python scripts/visualize_object_front.py \
        --placements_root outputs/<run> \
        [--limit 12] [--n_samples 6]

Open the resulting HTML in a browser. Without --embed, image src uses
relative paths so the HTML lives in --placements_root.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- geometry


def cross(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def project(point_world, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy):
    """Project a world-space point into pixel coords. Returns (px, py, depth)
    or None if behind camera. Uses the renderer's right = fwd × up."""
    rel = np.asarray(point_world, dtype=float) - np.asarray(cam_pos, dtype=float)
    cam_right = cross(cam_fwd, cam_up)
    x = float(np.dot(cam_right, rel))
    y = float(np.dot(cam_fwd, rel))
    z = float(np.dot(cam_up, rel))
    if y <= 1e-6:
        return None
    return cx + fx * x / y, cy - fy * z / y, y


def ground_arrow_vertices(base_xyz, fwd_xy, length, half_width_shaft, half_width_head, head_frac=0.4):
    """Return 7 ordered XYZ verts of a flat arrow lying at z=base_xyz[2],
    pointing along fwd_xy (already normalised, with z≈0). Order traces the
    outline counterclockwise as seen from above:

        v6 (back-left)  ─── v5 (shaft-mid-left)  ─ v4 (head-base-left)
        |                                                               ╲
        v0 (back-right)─── v1 (shaft-mid-right) ─ v2 (head-base-right) ─ v3 (tip)
    """
    fwd = np.array([fwd_xy[0], fwd_xy[1], 0.0])
    n = np.linalg.norm(fwd[:2])
    if n < 1e-8:
        return None
    fwd /= n
    right = np.array([-fwd[1], fwd[0], 0.0])  # left turn = -right
    base = np.array([base_xyz[0], base_xyz[1], base_xyz[2]], dtype=float)

    shaft_len = length * (1.0 - head_frac)
    s = half_width_shaft
    h = half_width_head

    v0 = base - right * s
    v1 = base - right * s + fwd * shaft_len
    v2 = base - right * h + fwd * shaft_len
    v3 = base + fwd * length
    v4 = base + right * h + fwd * shaft_len
    v5 = base + right * s + fwd * shaft_len
    v6 = base + right * s
    return [v0, v1, v2, v3, v4, v5, v6]


def project_polygon(verts, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy):
    pts = [project(v, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy) for v in verts]
    if any(p is None for p in pts):
        return None
    return [(p[0], p[1]) for p in pts], np.mean([p[2] for p in pts])


# ---------------------------------------------------------------- sampling


def sample_indices(entries, n):
    """Spread n samples evenly over azimuth, prefer un-truncated frames."""
    pool = [
        (i, e.get("azimuth_deg", 0.0))
        for i, e in enumerate(entries)
        if not e.get("truncation", False) and e.get("visibility_ratio", 1.0) > 0.7
    ]
    if len(pool) < n:
        pool = [(i, e.get("azimuth_deg", 0.0)) for i, e in enumerate(entries)]
    pool.sort(key=lambda t: t[1])
    if len(pool) <= n:
        return [i for i, _ in pool]
    step = len(pool) / n
    return [pool[int(round(k * step))][0] for k in range(n)]


def find_canonical_views(entries, obj_fwd, obj_up):
    """For each of FRONT/BACK/LEFT/RIGHT, find the frame whose camera is best
    positioned to capture that side of the object. Definitions:

      view_dir = (cam_pos - obj_pos) / ||·||  (= world dir from object to camera)
      front view = camera in front of object → view_dir aligns with object_forward
      back  view = camera behind object      → view_dir anti-aligns
      right view = camera on object's right  → view_dir aligns with object_right
                   where object_right = object_forward × object_up

    Prefers near-horizontal elevation and high object visibility. Falls back
    if no clean candidate exists.
    """
    fwd = np.array(obj_fwd, dtype=float)
    up = np.array(obj_up, dtype=float)
    fwd /= max(np.linalg.norm(fwd), 1e-8)
    up /= max(np.linalg.norm(up), 1e-8)
    right = cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-8)

    sides = {
        "FRONT": fwd,    # see the face
        "BACK":  -fwd,   # see the back
        "RIGHT": right,  # see the object's right side
        "LEFT":  -right, # see the object's left side
    }

    # Two passes: tight filter first, fall back if empty.
    for tight in (True, False):
        scored = {k: [] for k in sides}
        for i, e in enumerate(entries):
            if tight:
                if e.get("truncation", False):
                    continue
                if e.get("visibility_ratio", 1.0) < 0.85:
                    continue
                if abs(e.get("elevation_deg", 0.0)) > 35:
                    continue
            cam_pos = np.array(e["camera_position"], dtype=float)
            obj_pos = np.array(e["object_position"], dtype=float)
            view = cam_pos - obj_pos
            n = np.linalg.norm(view)
            if n < 1e-6:
                continue
            view = view / n
            for label, target in sides.items():
                # Bonus for low elevation magnitude (cleaner profile shots).
                el = abs(e.get("elevation_deg", 0.0))
                el_penalty = 0.002 * el
                scored[label].append((float(np.dot(view, target)) - el_penalty, i))
        if all(scored[k] for k in sides):
            chosen = {}
            for label in sides:
                scored[label].sort(key=lambda t: t[0], reverse=True)
                chosen[label] = scored[label][0][1]
            return chosen
    # absolute fallback
    return {k: 0 for k in sides}


# ---------------------------------------------------------------- per-frame card


def render_frame_card(entry, idx, intrinsics, ground_z, arrow_len, src_url, view_label=None):
    fx, fy, cx, cy = intrinsics
    res_x, res_y = 2 * cx, 2 * cy

    cam_pos = entry["camera_position"]
    cam_fwd = entry["final_forward"]
    cam_up = entry["final_up"]
    obj_pos = entry["object_position"]
    obj_fwd = entry["object_forward"]
    obj_up = entry["object_up"]

    # Build a flat arrow at the object's footprint (z = ground_z) in XY.
    base = (obj_pos[0], obj_pos[1], ground_z)
    fwd_xy = (obj_fwd[0], obj_fwd[1])
    verts = ground_arrow_vertices(
        base_xyz=base, fwd_xy=fwd_xy, length=arrow_len,
        half_width_shaft=arrow_len * 0.07,
        half_width_head=arrow_len * 0.18,
    )
    proj_polygon = None
    arrow_depth = None
    if verts is not None:
        out = project_polygon(verts, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy)
        if out is not None:
            proj_polygon, arrow_depth = out

    # Also compute a small upright forward axis at the object center for context.
    p_center = project(obj_pos, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy)
    up_tip = [obj_pos[i] + 0.6 * obj_up[i] for i in range(3)]
    p_up = project(up_tip, cam_pos, cam_fwd, cam_up, fx, fy, cx, cy)

    bbox = entry.get("bbox_2d")
    az = entry.get("azimuth_deg", 0.0)
    el = entry.get("elevation_deg", 0.0)
    dist = entry.get("camera_subject_distance", 0.0)

    overlay = []
    if bbox:
        x0, y0, x1, y1 = bbox
        overlay.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'fill="none" stroke="#00ff88" stroke-width="3"/>'
        )

    if proj_polygon is not None:
        # Shade the arrow based on viewing angle vs ground normal — flatter
        # views darken the arrow, steeper views brighten it. Pure cosmetic.
        ground_normal = np.array([0, 0, 1.0])
        view_dir = np.array(cam_fwd)
        cosang = abs(float(np.dot(view_dir, ground_normal)))
        intensity = 0.55 + 0.45 * cosang  # 0.55 .. 1.0
        r = int(255 * intensity)
        fill = f"rgb({r},{int(40 * intensity)},{int(40 * intensity)})"
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in proj_polygon)
        overlay.append(
            f'<polygon points="{pts}" fill="{fill}" fill-opacity="0.85" '
            f'stroke="black" stroke-width="2.5" stroke-linejoin="round"/>'
        )
        # FRONT label at the tip vertex (index 3).
        tip = proj_polygon[3]
        overlay.append(
            f'<text x="{tip[0] + 8:.1f}" y="{tip[1] - 6:.1f}" '
            f'fill="#ff3030" stroke="black" stroke-width="3" paint-order="stroke" '
            f'font-size="18" font-weight="bold">FRONT</text>'
        )

    if p_center is not None and p_up is not None:
        overlay.append(
            f'<line x1="{p_center[0]:.1f}" y1="{p_center[1]:.1f}" '
            f'x2="{p_up[0]:.1f}" y2="{p_up[1]:.1f}" '
            f'stroke="#3060ff" stroke-width="3"/>'
        )
    if p_center is not None:
        overlay.append(
            f'<circle cx="{p_center[0]:.1f}" cy="{p_center[1]:.1f}" r="5" '
            f'fill="#ffff00" stroke="black" stroke-width="1"/>'
        )

    inside = ""
    if p_center is not None and bbox is not None:
        x0, y0, x1, y1 = bbox
        ok = x0 <= p_center[0] <= x1 and y0 <= p_center[1] <= y1
        inside = (
            f'<span class="{"ok" if ok else "bad"}">'
            f'{"center inside bbox" if ok else "center OUTSIDE bbox"}</span>'
        )

    label_html = ""
    expectation = ""
    if view_label is not None:
        expectations = {
            "FRONT": "should see the face",
            "BACK":  "should see the back of the head",
            "RIGHT": "should see the right side",
            "LEFT":  "should see the left side",
        }
        label_html = f'<div class="view-label view-{view_label.lower()}">{view_label} view</div>'
        expectation = (
            f'<span class="expect">→ {expectations.get(view_label, "")}</span>'
        )

    return f"""
    <div class="card">
      <div class="caption">
        {label_html}
        <b>frame {idx}</b> · az={az:.0f}° el={el:.0f}° dist={dist:.2f} {inside} {expectation}
      </div>
      <div class="frame" style="aspect-ratio:{res_x}/{res_y}">
        <img src="{src_url}" loading="lazy" />
        <svg viewBox="0 0 {res_x} {res_y}" preserveAspectRatio="none">
          {''.join(overlay)}
        </svg>
      </div>
    </div>
    """


def render_topdown(entries, sampled_idx, obj_pos, obj_fwd):
    pts_x = [e["camera_position"][0] for e in entries] + [obj_pos[0]]
    pts_y = [e["camera_position"][1] for e in entries] + [obj_pos[1]]
    cx_min, cx_max = min(pts_x), max(pts_x)
    cy_min, cy_max = min(pts_y), max(pts_y)
    pad = max(cx_max - cx_min, cy_max - cy_min) * 0.1 + 0.5
    cx_min -= pad; cx_max += pad
    cy_min -= pad; cy_max += pad
    W, H = 320, 320
    sx = lambda x: (x - cx_min) / (cx_max - cx_min) * W
    sy = lambda y: H - (y - cy_min) / (cy_max - cy_min) * H

    parts = [f'<svg width="{W}" height="{H}" style="background:#111;border:1px solid #444">']
    for e in entries:
        cp = e["camera_position"]
        parts.append(f'<circle cx="{sx(cp[0]):.1f}" cy="{sy(cp[1]):.1f}" r="1" fill="#444"/>')
    for i in sampled_idx:
        cp = entries[i]["camera_position"]
        parts.append(
            f'<circle cx="{sx(cp[0]):.1f}" cy="{sy(cp[1]):.1f}" r="3.5" fill="#00ff88"/>'
            f'<text x="{sx(cp[0]) + 5:.1f}" y="{sy(cp[1]) + 4:.1f}" fill="#00ff88" font-size="10">{i}</text>'
        )
    ox, oy = sx(obj_pos[0]), sy(obj_pos[1])
    parts.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="6" fill="#ffff00" stroke="black"/>')
    L = 1.5
    fx_w = obj_pos[0] + L * obj_fwd[0]
    fy_w = obj_pos[1] + L * obj_fwd[1]
    parts.append(
        f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{sx(fx_w):.1f}" y2="{sy(fy_w):.1f}" '
        f'stroke="#ff3030" stroke-width="3"/>'
        f'<text x="{sx(fx_w) + 4:.1f}" y="{sy(fy_w):.1f}" fill="#ff3030" font-size="11">front</text>'
    )
    parts.append('</svg>')
    return ''.join(parts)


# ---------------------------------------------------------------- per-placement


def process_placement(pdir, n_samples, root_for_links, embed):
    ann_path = pdir / "annotations.json"
    info_path = pdir / "run_info.json"
    if not ann_path.exists() or not info_path.exists():
        return None
    ann = json.load(open(ann_path))
    info = json.load(open(info_path))
    if not ann:
        return None

    opt = info["options"]
    fl = opt["focal_length"]; sw = opt["sensor_width"]; sh = opt["sensor_height"]
    res_x, res_y = opt["resolution"]
    fx = fl / sw * res_x
    fy = fl / sh * res_y
    cx, cy = res_x / 2, res_y / 2
    intr = (fx, fy, cx, cy)

    # ground level = AABB min z
    ground_z = info.get("object_3d", {}).get("bbox_3d_min", [0, 0, 0])[2]
    dims = info.get("object_3d", {}).get("dimensions", {})
    if isinstance(dims, dict):
        # dimensions = {"width": x, "depth": y, "height": z}
        xy_size = max(dims.get("width", 1.0), dims.get("depth", 1.0))
    else:
        xy_size = max(dims[0], dims[1])
    arrow_len = max(0.8, 0.9 * xy_size)

    obj_pos = ann[0]["object_position"]
    obj_fwd = ann[0]["object_forward"]
    obj_up = ann[0]["object_up"]
    rot_z = ann[0].get("object_rotation_z_deg", 0.0)
    scene = info.get("scene_stem", "?")
    obj_blend = Path(info.get("input_object", "?")).stem

    canonical = find_canonical_views(ann, obj_fwd, obj_up)
    extra = sample_indices(ann, n_samples) if n_samples > 0 else []

    def src_for(entry):
        if embed:
            p = pdir / entry["image"]
            try:
                return f"data:image/png;base64,{base64.b64encode(open(p, 'rb').read()).decode()}"
            except FileNotFoundError:
                return ""
        return f"{pdir.relative_to(root_for_links)}/{entry['image']}"

    canonical_cards = "\n".join(
        render_frame_card(ann[canonical[label]], canonical[label],
                          intr, ground_z, arrow_len,
                          src_for(ann[canonical[label]]), view_label=label)
        for label in ("FRONT", "RIGHT", "BACK", "LEFT")
    )
    extra_cards = "\n".join(
        render_frame_card(ann[i], i, intr, ground_z, arrow_len, src_for(ann[i]))
        for i in extra
    )
    extra_block = (
        f'<details><summary>+{len(extra)} extra random views</summary>'
        f'<div class="grid">{extra_cards}</div></details>'
    ) if extra else ""

    sampled_for_topdown = list(canonical.values()) + list(extra)
    topdown = render_topdown(ann, sampled_for_topdown, obj_pos, obj_fwd)

    return f"""
    <section id="{pdir.name}">
      <h2>{pdir.name}</h2>
      <div class="meta">
        scene <b>{scene}</b> · object <b>{obj_blend}</b> ·
        rotation_z_deg <b>{rot_z}</b> ·
        forward (world) <b>[{obj_fwd[0]:.2f}, {obj_fwd[1]:.2f}, {obj_fwd[2]:.2f}]</b> ·
        ground_z <b>{ground_z:.2f}</b> · arrow_len <b>{arrow_len:.2f}</b> m
      </div>
      <div class="layout">
        <div>{topdown}</div>
        <div class="canon-grid">{canonical_cards}</div>
      </div>
      {extra_block}
    </section>
    """


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement_dir", action="append", default=[],
                    help="explicit placement dir (repeatable)")
    ap.add_argument("--placements_root",
                    help="parent dir; visualize all placement subdirs inside")
    ap.add_argument("--limit", type=int, default=12,
                    help="cap on placements when using --placements_root")
    ap.add_argument("--n_samples", type=int, default=0,
                    help="extra random frames per placement on top of the 4 canonical views")
    ap.add_argument("--output", default=None)
    ap.add_argument("--embed", action="store_true",
                    help="base64-embed images for a single-file html (much larger)")
    args = ap.parse_args()

    placements = [Path(p).resolve() for p in args.placement_dir]
    if args.placements_root:
        root = Path(args.placements_root).resolve()
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "annotations.json").exists():
                placements.append(child)
    placements = list(dict.fromkeys(placements))[:args.limit]
    if not placements:
        raise SystemExit("no placements found; pass --placement_dir or --placements_root")

    # All placements should share a common ancestor for relative image links.
    common = Path(*Path.cwd().parts)  # placeholder
    if not args.embed:
        # use longest common ancestor
        parts_list = [list(p.parts) for p in placements]
        common_parts = []
        for tup in zip(*parts_list):
            if len(set(tup)) == 1:
                common_parts.append(tup[0])
            else:
                break
        common = Path(*common_parts) if common_parts else Path("/")

    out_path = Path(args.output) if args.output else \
        (Path(args.placements_root).resolve() if args.placements_root else placements[0].parent) / "object_front_check.html"

    sections = []
    toc = []
    for p in placements:
        print(f"  processing {p.name}...")
        section = process_placement(p, args.n_samples, common, args.embed)
        if section is None:
            continue
        sections.append(section)
        toc.append(f'<a href="#{p.name}">{p.name}</a>')

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>object front check</title>
<style>
  body {{ font-family: ui-monospace, Menlo, monospace; background:#0b0b0b; color:#ddd;
         margin:0; padding:18px 24px }}
  h1 {{ font-size:16px; margin:0 0 4px }}
  h2 {{ font-size:13px; margin:0 0 6px; color:#7df }}
  .toc {{ font-size:11px; color:#888; margin-bottom:18px; line-height:1.7 }}
  .toc a {{ color:#7df; margin-right:10px; text-decoration:none }}
  .toc a:hover {{ text-decoration:underline }}
  .legend {{ font-size:11px; color:#aaa; margin:8px 0 18px }}
  .legend span {{ display:inline-block; margin-right:18px }}
  .swatch {{ display:inline-block; width:14px; height:6px; vertical-align:middle; margin-right:5px }}
  section {{ border-top:1px solid #222; padding:14px 0 22px }}
  .meta {{ color:#888; font-size:11px; margin-bottom:10px }}
  .meta b {{ color:#ddd }}
  .layout {{ display:grid; grid-template-columns:340px 1fr; gap:14px; align-items:start }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:10px }}
  .canon-grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:10px }}
  .view-label {{ display:inline-block; padding:2px 8px; border-radius:3px;
                 font-weight:bold; font-size:11px; margin-right:8px }}
  .view-front {{ background:#ff3030; color:white }}
  .view-back  {{ background:#3060ff; color:white }}
  .view-right {{ background:#ffaa00; color:black }}
  .view-left  {{ background:#00cc88; color:black }}
  .expect {{ color:#888; font-size:10px }}
  details {{ margin-top:12px; color:#888 }}
  details summary {{ cursor:pointer; font-size:11px; margin-bottom:8px }}
  .card {{ background:#181818; border:1px solid #2a2a2a; border-radius:5px; padding:6px }}
  .frame {{ position:relative; width:100% }}
  .frame img {{ width:100%; display:block; border-radius:3px }}
  .frame svg {{ position:absolute; inset:0; width:100%; height:100% }}
  .caption {{ font-size:10px; color:#aaa; margin-bottom:4px }}
  .ok {{ color:#00ff88 }}
  .bad {{ color:#ff5050 }}
</style></head><body>
<h1>object front orientation — {len(sections)} placement(s)</h1>
<div class="legend">
  Each placement shows 4 canonical views — the renderer's existing frames whose camera
  is closest to being directly in front / behind / right / left of the object.
  Just check: in the <b>FRONT view</b>, do you see a face? If yes, ✓ orientation is right.
  <br/>
  <span><span class="swatch" style="background:#ff3030"></span>3D ground arrow = object_forward</span>
  <span><span class="swatch" style="background:#3060ff"></span>up axis</span>
  <span><span class="swatch" style="background:#ffff00;height:8px;width:8px;border-radius:50%"></span>projected object center</span>
  <span><span class="swatch" style="background:#00ff88"></span>tight bbox</span>
</div>
<div class="toc">{' '.join(toc)}</div>
{''.join(sections)}
</body></html>
"""
    out_path.write_text(html)
    print(f"\nwrote {out_path}")
    print(f"  {out_path.stat().st_size / 1024:.0f} KB · {len(sections)} placement(s) · {args.n_samples} frame(s) each")


if __name__ == "__main__":
    main()
