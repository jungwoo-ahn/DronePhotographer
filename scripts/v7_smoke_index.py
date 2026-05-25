#!/usr/bin/env python3
"""Build an index.html linking to every per-placement report in a v7 smoke
sweep directory. Reads each subdir's data.json for summary stats.

Usage:
    python scripts/v7_smoke_index.py outputs/v7_pair_smoke_7run_v2
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from html import escape
from pathlib import Path


def _embed_thumb(p: Path, max_bytes: int = 250_000) -> str:
    """Return data: URI for a JPEG thumbnail, or '' if too big / missing."""
    if not p.exists():
        return ""
    try:
        data = p.read_bytes()
    except OSError:
        return ""
    if len(data) > max_bytes:
        return ""
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _first_render(placement_dir: Path) -> Path | None:
    renders = sorted((placement_dir / "renders").glob("pair_*_frame_*.jpg"))
    return renders[0] if renders else None


def build_index(sweep_dir: Path) -> Path:
    sweep_dir = sweep_dir.resolve()
    entries: list[dict] = []
    for placement_dir in sorted(sweep_dir.iterdir()):
        if not placement_dir.is_dir() or placement_dir.name.startswith("_"):
            continue
        data_path = placement_dir / "data.json"
        if not data_path.exists():
            continue
        try:
            data = json.loads(data_path.read_text())
        except Exception:
            continue
        report_path = placement_dir / f"{placement_dir.name}.html"
        report_rel = (
            report_path.relative_to(sweep_dir).as_posix()
            if report_path.exists() else ""
        )
        thumb_src = ""
        thumb = _first_render(placement_dir)
        if thumb is not None:
            thumb_src = _embed_thumb(thumb)
        entries.append({
            "name": data.get("placement", placement_dir.name),
            "report_rel": report_rel,
            "scene": data.get("scene_file", "?"),
            "object": data.get("object_file", "?"),
            "K_target": data.get("K_target", 0),
            "K_accepted": data.get("K_accepted", 0),
            "attempts": data.get("attempts_used", 0),
            "rejections": data.get("rejections_by_reason", {}) or {},
            "time_setup": data.get("time_setup_s", 0.0),
            "time_sample": data.get("time_sample_s", 0.0),
            "time_render": data.get("time_render_s", 0.0),
            "thumb_src": thumb_src,
        })

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>v7 pair-smoke sweep — {escape(sweep_dir.name)}</title>")
    parts.append("""<style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 24px; background: #fafafa; color: #222; }
        h1 { font-size: 22px; margin: 0 0 8px; }
        .meta { font-size: 12px; color: #555; margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                gap: 16px; }
        .card { background: white; border: 1px solid #ddd; border-radius: 6px;
                padding: 14px; text-decoration: none; color: inherit;
                display: flex; flex-direction: column; gap: 8px;
                transition: box-shadow 0.15s; }
        .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .card h2 { font-size: 13px; font-family: ui-monospace, monospace;
                   margin: 0; word-break: break-all; line-height: 1.3; }
        .thumb { width: 100%; height: 180px; object-fit: cover; border-radius: 3px;
                 border: 1px solid #ccc; background: #f5f5f5; }
        .no-thumb { width: 100%; height: 180px; background: #eee; border-radius: 3px;
                    border: 1px dashed #ccc; display: flex; align-items: center;
                    justify-content: center; color: #999; font-size: 12px; }
        .stat-row { display: flex; gap: 6px; flex-wrap: wrap; font-size: 12px;
                    font-family: ui-monospace, monospace; }
        .badge { padding: 2px 8px; border-radius: 12px; background: #eef;
                 color: #335; }
        .badge.ok  { background: #dff2dd; color: #2a7a2a; }
        .badge.mid { background: #fff3cc; color: #8a6d00; }
        .badge.bad { background: #f7dddd; color: #b04040; }
        .badge.gray { background: #eee; color: #555; }
        .meta-row { font-size: 11px; color: #777; font-family: ui-monospace, monospace;
                    line-height: 1.4; }
        summary { font-size: 22px; margin: 24px 0 8px; cursor: pointer; }
    </style></head><body>""")

    parts.append(f"<h1>v7 pair-smoke sweep — {escape(sweep_dir.name)}</h1>")
    n_total = len(entries)
    n_full = sum(1 for e in entries if e["K_accepted"] >= e["K_target"])
    parts.append(f"<div class='meta'>{n_total} placements · "
                 f"{n_full} reached K_target · "
                 f"out_dir = <code>{escape(str(sweep_dir))}</code></div>")

    parts.append("<div class='grid'>")
    for e in entries:
        if e["K_target"]:
            ratio = e["K_accepted"] / e["K_target"]
        else:
            ratio = 0.0
        if ratio >= 1.0: cls = "ok"
        elif ratio >= 0.5: cls = "mid"
        else: cls = "bad"
        href = e["report_rel"] or "#"
        parts.append(f"<a class='card' href='{escape(href)}'>")
        parts.append(f"<h2>{escape(e['name'])}</h2>")
        if e["thumb_src"]:
            parts.append(f"<img class='thumb' src='{e['thumb_src']}' alt='first render'>")
        else:
            parts.append("<div class='no-thumb'>(no render)</div>")
        parts.append("<div class='stat-row'>")
        parts.append(f"<span class='badge {cls}'>{e['K_accepted']}/{e['K_target']}</span>")
        attempts = e["attempts"] or 1
        rate = e["K_accepted"] / attempts
        parts.append(f"<span class='badge gray'>rate {rate:.0%}</span>")
        parts.append(f"<span class='badge gray'>{attempts} attempts</span>")
        parts.append("</div>")
        # rejection summary
        rej_items = sorted(e["rejections"].items(), key=lambda kv: -kv[1])
        if rej_items:
            top = ", ".join(f"{k}={v}" for k, v in rej_items[:3])
            parts.append(f"<div class='meta-row'>rejections: {escape(top)}</div>")
        parts.append("<div class='meta-row'>"
                     f"setup={e['time_setup']:.1f}s · "
                     f"sample={e['time_sample']:.1f}s · "
                     f"render={e['time_render']:.1f}s"
                     "</div>")
        parts.append(f"<div class='meta-row'>scene: {escape(e['scene'])}</div>")
        parts.append("</a>")
    parts.append("</div></body></html>")

    out_path = sweep_dir / "index.html"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir")
    args = ap.parse_args()
    out = build_index(Path(args.sweep_dir))
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
