#!/usr/bin/env python3
"""Generate a self-contained HTML report for a v6 local-dense placement run.

Given an output run dir (e.g. outputs/smoke_v6_local_dense/p0_*), emits
report.html in that dir containing:
  - An interactive Plotly 3D scatter of: object position, discovered anchor
    positions, and every rendered camera position (colored by anchor).
  - A contact-sheet grid of all rendered thumbnails, grouped by anchor_id,
    with each frame's key metadata. Images are base64-embedded so the file
    is self-contained.

Usage:
    python scripts/make_local_dense_report.py outputs/smoke_v6_local_dense/p0_*/
    # optional flags:
    #   --thumb_width 512   (px; default 512 — smaller -> tinier HTML)
    #   --quality 82
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from html import escape
from pathlib import Path

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


# 10 distinct, colorblind-tolerant colors — covers up to 10 anchors comfortably.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def _safe_round(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def _color_for(anchor_id: int) -> str:
    if anchor_id is None or anchor_id < 0:
        return "#999999"
    return _PALETTE[anchor_id % len(_PALETTE)]


def _embed_image(img_path: Path, thumb_width: int, quality: int) -> str:
    """Return a data: URI for img_path, optionally resized."""
    if not img_path.exists():
        return ""
    if Image is None:
        # Pillow missing — embed original bytes verbatim.
        data = img_path.read_bytes()
        mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    im = Image.open(img_path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if thumb_width and im.width > thumb_width:
        ratio = thumb_width / im.width
        new_size = (thumb_width, int(im.height * ratio))
        im = im.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def render_report(run_dir: Path, thumb_width: int, quality: int) -> Path:
    run_dir = run_dir.resolve()
    ann_path = run_dir / "annotations.json"
    anc_path = run_dir / "anchors.json"
    place_path = run_dir / "placement.json"

    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found in {run_dir}")
    ann = json.loads(ann_path.read_text())
    anchors_doc = json.loads(anc_path.read_text()) if anc_path.exists() else None
    placement = json.loads(place_path.read_text()) if place_path.exists() else None

    by_anchor: dict[int, list[dict]] = {}
    for row in ann:
        a = row.get("anchor_id")
        if a is None:
            a = -1
        by_anchor.setdefault(a, []).append(row)
    for v in by_anchor.values():
        v.sort(key=lambda r: r.get("image", ""))

    n_imgs = len(ann)
    n_anchors = len(by_anchor)
    scene = (placement or {}).get("scene", "?")
    obj = (placement or {}).get("object", "?")
    obj_pos = (anchors_doc or {}).get("object_position") \
        or (placement or {}).get("position") \
        or (ann[0].get("object_position") if ann else None)
    discovery = anchors_doc or {}
    anchor_positions = (anchors_doc or {}).get("anchors") or []

    # ---- Build Plotly traces ----
    traces = []
    if obj_pos is not None:
        traces.append({
            "x": [obj_pos[0]], "y": [obj_pos[1]], "z": [obj_pos[2]],
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 6, "color": "#000", "symbol": "diamond"},
            "name": "object",
        })
    for i, a in enumerate(anchor_positions):
        col = _color_for(i)
        traces.append({
            "x": [a[0]], "y": [a[1]], "z": [a[2]],
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 10, "color": col, "line": {"color": "#222", "width": 1}},
            "name": f"anchor {i}",
        })
    for a_id in sorted(by_anchor.keys()):
        rows = by_anchor[a_id]
        xs, ys, zs, texts = [], [], [], []
        for r in rows:
            cp = r.get("camera_position") or [None, None, None]
            xs.append(cp[0]); ys.append(cp[1]); zs.append(cp[2])
            texts.append(Path(r.get("image", "")).name)
        col = _color_for(a_id)
        traces.append({
            "x": xs, "y": ys, "z": zs,
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 3, "color": col, "opacity": 0.7},
            "name": f"cam pos (a{a_id}, n={len(rows)})",
            "text": texts,
            "hovertemplate": "%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        })

    layout = {
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "zaxis": {"title": "z"},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.2}},
        },
        "margin": {"l": 0, "r": 0, "t": 24, "b": 0},
        "legend": {"itemsizing": "constant"},
        "height": 600,
    }

    # ---- 2D scatter: radius vs applied camera pitch (offsets.roll) ----
    # The Euler XYZ packing means offsets.roll is the local-X rotation amount,
    # which physically is the camera up/down tilt.
    pitch_traces = []
    for a_id in sorted(by_anchor.keys()):
        rows = by_anchor[a_id]
        xs = [r.get("radius") for r in rows]
        ys = [(r.get("offsets_deg") or {}).get("roll") for r in rows]
        pitch_traces.append({
            "x": xs, "y": ys,
            "mode": "markers", "type": "scatter",
            "marker": {"size": 6, "color": _color_for(a_id), "opacity": 0.75},
            "name": f"a{a_id}",
        })
    # Lerp envelope (read from run_info.json if present)
    run_info_path = run_dir / "run_info.json"
    envelope = None
    if run_info_path.exists():
        try:
            ri = json.loads(run_info_path.read_text())
            near = ri.get("options", {}).get("pitch_lerp_near")
            far = ri.get("options", {}).get("pitch_lerp_far")
            if near and far:
                envelope = (near, far)
        except Exception:
            envelope = None
    if envelope is not None:
        near, far = envelope
        rs = [near[0], far[0]]
        pitch_traces.append({
            "x": rs, "y": [near[1], far[1]],
            "mode": "lines", "type": "scatter",
            "line": {"color": "#222", "dash": "dash"},
            "name": "lower bound",
        })
        pitch_traces.append({
            "x": rs, "y": [near[2], far[2]],
            "mode": "lines", "type": "scatter",
            "line": {"color": "#222", "dash": "dash"},
            "name": "upper bound",
        })
    pitch_layout = {
        "xaxis": {"title": "radius (m)  =  cam-to-object distance"},
        "yaxis": {"title": "applied pitch (deg, offsets.roll = local-X rotation)"},
        "margin": {"l": 60, "r": 20, "t": 24, "b": 50},
        "legend": {"itemsizing": "constant"},
        "height": 360,
    }

    # ---- HTML scaffolding ----
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>v6 local-dense report — {escape(run_dir.name)}</title>")
    parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
    parts.append("<style>")
    parts.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 24px; background: #fafafa; color: #222; }
        h1 { font-size: 20px; margin: 0 0 4px; }
        h2 { font-size: 16px; margin: 24px 0 8px; padding-top: 8px;
             border-top: 1px solid #ddd; }
        .meta { font-size: 12px; color: #555; margin-bottom: 16px; line-height: 1.4; }
        .meta code { background: #eee; padding: 1px 4px; border-radius: 3px; }
        #plot3d { background: white; border: 1px solid #ddd; border-radius: 4px;
                  margin: 12px 0 24px; }
        .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
        .card { background: white; border: 1px solid #ddd; border-radius: 4px;
                overflow: hidden; }
        .card img { width: 100%; height: auto; display: block; }
        .card .info { font-size: 10px; padding: 4px 6px; color: #333;
                      font-family: ui-monospace, SFMono-Regular, monospace;
                      line-height: 1.35; }
        .info .k { color: #888; }
        .anchor-card { padding: 8px 12px; margin-bottom: 12px; border-radius: 4px;
                       font-size: 12px; font-family: ui-monospace, monospace;
                       border-left: 4px solid #888; background: #f0f4ff; }
        @media (max-width: 1200px) { .grid { grid-template-columns: repeat(4, 1fr); } }
        @media (max-width: 900px)  { .grid { grid-template-columns: repeat(3, 1fr); } }
    """)
    parts.append("</style></head><body>")

    parts.append(f"<h1>v6 local-dense — {escape(run_dir.name)}</h1>")
    parts.append("<div class='meta'>")
    parts.append(f"Scene: <code>{escape(str(scene))}</code> · "
                 f"Object: <code>{escape(str(obj))}</code><br>")
    if obj_pos is not None:
        parts.append(f"Object position: <code>{[_safe_round(v) for v in obj_pos]}</code><br>")
    parts.append(f"Anchors: {n_anchors} · Images: {n_imgs}")
    if discovery:
        parts.append(f" · radius range: {discovery.get('radius_range')} m"
                     f" · ball radius: {discovery.get('ball_radius')} m"
                     f" · clearance: {discovery.get('min_clearance')} m"
                     f" · discovery attempts: {discovery.get('discovery_attempts')}"
                     f" / accepted {discovery.get('discovered')}/{discovery.get('target')}")
    parts.append("</div>")

    # ---- 3D plot ----
    parts.append("<div id='plot3d'></div>")
    parts.append("<script>")
    parts.append("var traces = " + json.dumps(traces) + ";")
    parts.append("var layout = " + json.dumps(layout) + ";")
    parts.append("Plotly.newPlot('plot3d', traces, layout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("</script>")

    # ---- 2D pitch-vs-radius plot ----
    parts.append("<div id='plot_pitch' style='background:white;border:1px solid #ddd;"
                 "border-radius:4px;margin:12px 0 24px;'></div>")
    parts.append("<script>")
    parts.append("var pitch_traces = " + json.dumps(pitch_traces) + ";")
    parts.append("var pitch_layout = " + json.dumps(pitch_layout) + ";")
    parts.append("Plotly.newPlot('plot_pitch', pitch_traces, pitch_layout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("</script>")

    # ---- Per-anchor sections with inline thumbnails ----
    for a_id in sorted(by_anchor.keys()):
        rows = by_anchor[a_id]
        anchor_pos = None
        if anchors_doc and 0 <= a_id < len(anchors_doc.get("anchors", [])):
            anchor_pos = anchors_doc["anchors"][a_id]
        elif rows and "anchor_position" in rows[0]:
            anchor_pos = rows[0]["anchor_position"]
        col = _color_for(a_id)
        title = (f"Anchor {a_id} — {len(rows)} images"
                 if a_id >= 0 else f"(no anchor) — {len(rows)} images")
        parts.append(f"<h2 style='color: {col}'>{escape(title)}</h2>")
        if anchor_pos is not None:
            parts.append(
                f"<div class='anchor-card' style='border-left-color: {col}'>"
                f"anchor_position = {[_safe_round(v) for v in anchor_pos]}"
                f"</div>"
            )
        parts.append("<div class='grid'>")
        for row in rows:
            img_rel = row.get("image", "")
            img_path = run_dir / img_rel
            data_uri = _embed_image(img_path, thumb_width, quality)
            cam = [_safe_round(v) for v in (row.get("camera_position") or [])]
            rad = _safe_round(row.get("radius"))
            off = row.get("offsets_deg") or {}
            elev = _safe_round(row.get("elevation_deg"))
            azim = _safe_round(row.get("azimuth_deg"))
            pitch = _safe_round(row.get("camera_pitch_deg"))
            occ = _safe_round(row.get("occupancy_ratio"))
            trunc = row.get("truncation")
            parts.append("<div class='card'>")
            parts.append(f"<img src='{data_uri}' loading='lazy' "
                         f"alt='{escape(Path(img_rel).name)}'>")
            parts.append("<div class='info'>")
            parts.append(f"<span class='k'>{escape(Path(img_rel).name)}</span><br>")
            parts.append(f"<span class='k'>cam</span> {cam}<br>")
            parts.append(f"<span class='k'>r</span> {rad}  "
                         f"<span class='k'>elev/azi</span> {elev}/{azim}<br>")
            parts.append(f"<span class='k'>pitch</span> {pitch}  "
                         f"<span class='k'>occ</span> {occ}"
                         + ("  <span class='k'>TRUNC</span>" if trunc else "")
                         + "<br>")
            parts.append(f"<span class='k'>jitter</span> "
                         f"y{_safe_round(off.get('yaw'), 1)}/"
                         f"p{_safe_round(off.get('pitch'), 1)}/"
                         f"r{_safe_round(off.get('roll'), 1)}")
            parts.append("</div></div>")
        parts.append("</div>")

    parts.append("</body></html>")
    out = run_dir / "report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="placement run dir containing images/, annotations.json")
    p.add_argument("--thumb_width", type=int, default=512,
                   help="resize embedded thumbnails to this width (default 512 px)")
    p.add_argument("--quality", type=int, default=82, help="JPEG quality 1-95 (default 82)")
    args = p.parse_args()
    out = render_report(Path(args.run_dir), args.thumb_width, args.quality)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main() or 0)
