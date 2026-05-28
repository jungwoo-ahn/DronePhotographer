#!/usr/bin/env python3
"""Render a self-contained HTML report for one v7 pair-sampling smoke run.

Reads ``<smoke_dir>/data.json`` (the output of scripts/v7_sample_pairs_smoke.py
when run with --out-dir=<base>/<placement_name>) and writes ``report.html``
alongside it. Plotly is loaded from a CDN; the file is self-contained otherwise.

Sections:
  - Header (placement metadata, acceptance stats).
  - Interactive 3D Plotly: subject + every accepted ellipse + start/end markers.
  - 2D radius scatter (r_start vs r_end), colored by pair.
  - 2D azimuth polar (start + end angles around subject).
  - Per-pair table with ellipse params + pose summaries.
  - Rejection-reason breakdown.

Usage:
    python scripts/make_v7_pair_smoke_report.py outputs/v7_pair_smoke/<placement>/
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#a6cee3", "#b2df8a",
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _r(x: Any, n: int = 3) -> Any:
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def _ellipse_xyz(
    O: list[float],
    u: list[float],
    v: list[float],
    a: float,
    b: float,
    theta_start: float,
    theta_end: float,
    n: int = 200,
) -> tuple[list[float], list[float], list[float]]:
    xs, ys, zs = [], [], []
    if n <= 1:
        return xs, ys, zs
    span = theta_end - theta_start
    for i in range(n):
        t = theta_start + span * (i / (n - 1))
        c, s = math.cos(t), math.sin(t)
        r = a * b / math.sqrt((b * c) ** 2 + (a * s) ** 2)
        xs.append(O[0] + r * (c * u[0] + s * v[0]))
        ys.append(O[1] + r * (c * u[1] + s * v[1]))
        zs.append(O[2] + r * (c * u[2] + s * v[2]))
    return xs, ys, zs


def _full_ellipse_xyz(O, u, v, a, b, n: int = 360):
    return _ellipse_xyz(O, u, v, a, b, 0.0, 2.0 * math.pi, n=n)


def _data_uri_for(rel_path: str, base_dir: Path) -> str:
    """Encode a relative JPEG path as a data: URI. Returns '' if missing."""
    p = (base_dir / rel_path).resolve()
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    b = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b}"


def build_report(smoke_dir: Path, *, embed: bool = True) -> Path:
    smoke_dir = smoke_dir.resolve()
    data_path = smoke_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    data = json.loads(data_path.read_text())

    placement_name: str = data.get("placement", smoke_dir.name)
    scene_file: str = data.get("scene_file", "?")
    object_file: str = data.get("object_file", "?")
    O: list[float] = data.get("subject_center") or data.get("subject_position") or [0, 0, 0]
    foot: list[float] = data.get("subject_foot", O)
    h_subject: float = float(data.get("subject_height", 0.0))
    seed = data.get("seed", 0)
    K_target = int(data.get("K_target", 0))
    K_accepted = int(data.get("K_accepted", 0))
    attempts = int(data.get("attempts_used", 0))
    rejections = data.get("rejections_by_reason", {}) or {}
    sub_reasons = data.get("sub_reasons", {}) or {}
    accepted = data.get("accepted_pairs", []) or []
    render_records = data.get("render_records") or []

    # Build two maps:
    #   path_map  (pair, frame) -> short relative path (always visible in label)
    #   src_map   (pair, frame) -> data URI (embed) or same rel path (no-embed)
    # Customdata only carries the short rel path so it never displays as a giant
    # base64 string; src_map is referenced by JS lookup at click time.
    path_map: dict[tuple[int, int], str] = {}
    src_map: dict[tuple[int, int], str] = {}
    embed_bytes = 0
    embed_count = 0
    for i, pair_recs in enumerate(render_records):
        for rec in pair_recs:
            rel = str(rec["path_rel"])
            j = int(rec["frame_idx"])
            path_map[(i, j)] = rel
            if embed:
                uri = _data_uri_for(rel, smoke_dir)
                src_map[(i, j)] = uri
                if uri:
                    embed_bytes += len(uri)
                    embed_count += 1
            else:
                src_map[(i, j)] = rel
    if embed and embed_count:
        print(f"[report] embedded {embed_count} images "
              f"(~{embed_bytes / (1024*1024):.1f} MB base64 in HTML)")

    # ----- Plotly 3D traces -----
    traces = []
    traces.append({
        "x": [O[0]], "y": [O[1]], "z": [O[2]],
        "mode": "markers", "type": "scatter3d",
        "marker": {"size": 6, "color": "#000", "symbol": "diamond"},
        "name": "subject center O",
    })
    if foot and foot != O:
        traces.append({
            "x": [foot[0]], "y": [foot[1]], "z": [foot[2]],
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 4, "color": "#888", "symbol": "circle"},
            "name": "subject foot",
        })
    for i, pair in enumerate(accepted):
        col = _color(i)
        E = pair["ellipse"]
        full_xs, full_ys, full_zs = _full_ellipse_xyz(
            E["O"], E["u"], E["v"], E["a"], E["b"], n=200,
        )
        traces.append({
            "x": full_xs, "y": full_ys, "z": full_zs,
            "mode": "lines", "type": "scatter3d",
            "line": {"color": col, "width": 1.5, "dash": "dot"},
            "opacity": 0.35,
            "name": f"pair {i:02d} (full ellipse)",
            "legendgroup": f"pair{i}",
            "showlegend": False,
            "hoverinfo": "skip",
        })
        traj = pair.get("trajectory_32f") or []
        if traj:
            tx = [p["pos"][0] for p in traj]
            ty = [p["pos"][1] for p in traj]
            tz = [p["pos"][2] for p in traj]
            custom = [
                [i, j, path_map.get((i, j), "")]
                for j in range(len(traj))
            ]
            hovertemplate = (
                f"pair {i:02d} frame %{{customdata[1]}}<br>"
                "(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            )
            traces.append({
                "x": tx, "y": ty, "z": tz,
                "mode": "lines+markers", "type": "scatter3d",
                "line": {"color": col, "width": 4},
                "marker": {"size": 4, "color": col,
                           "line": {"color": "#fff", "width": 0.5}},
                "customdata": custom,
                "hovertemplate": hovertemplate,
                "name": (f"pair {i:02d}: r=[{pair['start']['r']:.2f},"
                         f"{pair['end']['r']:.2f}]m θ_n="
                         f"{E['theta_near_deg']:.0f}°"),
                "legendgroup": f"pair{i}",
            })
        s = pair["start"]; e = pair["end"]
        # trajectory_frames lerps from C_far (frame 0) → C_near (frame N-1).
        s_frame = 0 if pair["c_far_is_start"] else (len(traj) - 1 if traj else 0)
        e_frame = (len(traj) - 1 if traj else 0) if pair["c_far_is_start"] else 0
        s_path = path_map.get((i, s_frame), "")
        e_path = path_map.get((i, e_frame), "")
        traces.append({
            "x": [s["pos"][0]], "y": [s["pos"][1]], "z": [s["pos"][2]],
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 6, "color": col,
                       "line": {"color": "#222", "width": 1}},
            "customdata": [[i, s_frame, s_path]],
            "name": f"pair {i:02d} start",
            "legendgroup": f"pair{i}",
            "showlegend": False,
            "hovertemplate": f"pair {i:02d} START<br>r={s['r']:.2f} "
                             f"az={s['az_deg']:.1f}° elev={s['elev_deg']:.1f}°"
                             "<extra></extra>",
        })
        traces.append({
            "x": [e["pos"][0]], "y": [e["pos"][1]], "z": [e["pos"][2]],
            "mode": "markers", "type": "scatter3d",
            "marker": {"size": 6, "color": col, "symbol": "x",
                       "line": {"color": "#222", "width": 1}},
            "customdata": [[i, e_frame, e_path]],
            "name": f"pair {i:02d} end",
            "legendgroup": f"pair{i}",
            "showlegend": False,
            "hovertemplate": f"pair {i:02d} END<br>r={e['r']:.2f} "
                             f"az={e['az_deg']:.1f}° elev={e['elev_deg']:.1f}°"
                             "<extra></extra>",
        })
    layout3d = {
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "zaxis": {"title": "z"},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.2}},
        },
        "margin": {"l": 0, "r": 0, "t": 24, "b": 0},
        "legend": {"itemsizing": "constant", "font": {"size": 10}},
        "height": 640,
    }

    # ----- r_start vs r_end scatter -----
    r_traces = []
    for i, p in enumerate(accepted):
        r_traces.append({
            "x": [p["start"]["r"]], "y": [p["end"]["r"]],
            "mode": "markers", "type": "scatter",
            "marker": {"size": 12, "color": _color(i)},
            "name": f"pair {i:02d}",
            "hovertemplate": (f"pair {i:02d}<br>r_start=%{{x:.2f}}<br>"
                              "r_end=%{y:.2f}<extra></extra>"),
        })
    r_max_axis = max(
        [p["start"]["r"] for p in accepted]
        + [p["end"]["r"] for p in accepted]
        + [1.0]
    ) * 1.05
    r_traces.append({
        "x": [0, r_max_axis], "y": [0, r_max_axis],
        "mode": "lines", "type": "scatter",
        "line": {"color": "#aaa", "dash": "dash", "width": 1},
        "name": "r_start = r_end",
        "showlegend": False,
        "hoverinfo": "skip",
    })
    r_layout = {
        "xaxis": {"title": "r_start (m)", "range": [0, r_max_axis]},
        "yaxis": {"title": "r_end (m)", "range": [0, r_max_axis],
                  "scaleanchor": "x", "scaleratio": 1},
        "margin": {"l": 60, "r": 20, "t": 24, "b": 50},
        "height": 360,
        "legend": {"font": {"size": 10}},
        "title": "Radius distribution per pair (close/mid/far mix)",
    }

    # ----- Pitch + yaw envelope scatters -----
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.policy.data.sampling import (  # type: ignore
            PITCH_LERP_NEAR, PITCH_LERP_FAR,
            YAW_LERP_NEAR, YAW_LERP_FAR,
        )
        pitch_envelope = (PITCH_LERP_NEAR, PITCH_LERP_FAR)
        yaw_envelope = (YAW_LERP_NEAR, YAW_LERP_FAR)
    except Exception:
        pitch_envelope = None
        yaw_envelope = None

    def _envelope_scatter(field: str, envelope):
        traces = []
        for i, p in enumerate(accepted):
            col = _color(i)
            traces.append({
                "x": [p["start"]["r"], p["end"]["r"]],
                "y": [p["start"][field], p["end"][field]],
                "mode": "markers", "type": "scatter",
                "marker": {"size": 9, "color": col, "opacity": 0.85},
                "name": f"pair {i:02d}",
                "hovertemplate": (f"pair {i:02d}<br>r=%{{x:.2f}}m "
                                  + field + "=%{y:.1f}°<extra></extra>"),
            })
        if envelope is not None:
            (r_n, lo_n, hi_n), (r_f, lo_f, hi_f) = envelope
            traces.append({
                "x": [r_n, r_f], "y": [lo_n, lo_f],
                "mode": "lines", "type": "scatter",
                "line": {"color": "#666", "dash": "dash", "width": 1.5},
                "name": "lower bound",
            })
            traces.append({
                "x": [r_n, r_f], "y": [hi_n, hi_f],
                "mode": "lines", "type": "scatter",
                "line": {"color": "#666", "dash": "dash", "width": 1.5},
                "name": "upper bound",
            })
        return traces

    pitch_traces = _envelope_scatter("pitch_jitter_deg", pitch_envelope)
    yaw_traces = _envelope_scatter("yaw_jitter_deg", yaw_envelope)
    pitch_layout = {
        "xaxis": {"title": "r (m)  (cam ↔ subject)"},
        "yaxis": {"title": "pitch jitter δ (deg, +=look up)"},
        "margin": {"l": 60, "r": 20, "t": 24, "b": 50},
        "height": 320,
        "legend": {"font": {"size": 10}},
        "title": "Pitch envelope (close r → wider up-tilt)",
    }
    yaw_layout = {
        "xaxis": {"title": "r (m)  (cam ↔ subject)"},
        "yaxis": {"title": "yaw jitter δ (deg, +=pan left)"},
        "margin": {"l": 60, "r": 20, "t": 24, "b": 50},
        "height": 320,
        "legend": {"font": {"size": 10}},
        "title": "Yaw envelope (close r → wider horizontal pan)",
    }

    # ----- Azimuth polar -----
    polar_traces = []
    for i, p in enumerate(accepted):
        col = _color(i)
        polar_traces.append({
            "r": [p["start"]["r"], p["end"]["r"]],
            "theta": [p["start"]["az_deg"], p["end"]["az_deg"]],
            "mode": "markers+lines", "type": "scatterpolar",
            "marker": {"size": 8, "color": col},
            "line": {"color": col, "width": 1},
            "name": f"pair {i:02d}",
            "hovertemplate": (f"pair {i:02d}<br>r=%{{r:.2f}}m "
                              "az=%{theta:.1f}°<extra></extra>"),
        })
    polar_layout = {
        "polar": {
            "radialaxis": {"title": "r (m)"},
            "angularaxis": {"direction": "counterclockwise"},
        },
        "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
        "height": 420,
        "legend": {"font": {"size": 10}},
        "title": "Azimuth distribution (top-down view; start ●─x end)",
    }

    # ----- HTML -----
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>v7 pair smoke — {escape(placement_name)}</title>")
    parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
    parts.append("<style>")
    parts.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 24px; background: #fafafa; color: #222; }
        h1 { font-size: 20px; margin: 0 0 4px; }
        h2 { font-size: 16px; margin: 28px 0 8px; padding-top: 8px;
             border-top: 1px solid #ddd; }
        .meta { font-size: 12px; color: #555; margin-bottom: 16px; line-height: 1.4; }
        .meta code { background: #eee; padding: 1px 4px; border-radius: 3px; }
        .meta .ok  { color: #2a7a2a; font-weight: 600; }
        .meta .bad { color: #b04040; font-weight: 600; }
        .plot { background: white; border: 1px solid #ddd; border-radius: 4px;
                margin: 8px 0 18px; }
        table.pairs { font-family: ui-monospace, monospace; font-size: 12px;
                      border-collapse: collapse; width: 100%; background: white;
                      border: 1px solid #ddd; }
        table.pairs th, table.pairs td { padding: 4px 8px; border-bottom: 1px solid #eee;
                                          text-align: right; }
        table.pairs th { background: #f0f0f0; font-weight: 600; }
        table.pairs td.pair-id { text-align: left; font-weight: 600; }
        table.pairs td.pair-color { width: 12px; padding: 4px 0; }
        table.pairs td.pair-color div { width: 12px; height: 12px; border-radius: 2px; }
        .rejtable { font-family: ui-monospace, monospace; font-size: 12px;
                    background: white; border: 1px solid #ddd; padding: 12px;
                    border-radius: 4px; max-width: 600px; }
        .rejtable .k { color: #888; min-width: 240px; display: inline-block; }
        details { margin: 8px 0; }
        summary { cursor: pointer; font-size: 13px; color: #444; }
    """)
    parts.append("""
        .three-d-row { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px;
                       align-items: start; }
        .preview { background: white; border: 1px solid #ddd; border-radius: 4px;
                   padding: 12px; position: sticky; top: 12px; }
        .preview img { width: 100%; height: auto; display: block; border-radius: 3px;
                       border: 1px solid #ccc; background: #f5f5f5; }
        .preview .label { font-family: ui-monospace, monospace; font-size: 11px;
                          color: #444; margin-top: 8px; line-height: 1.4;
                          word-break: break-all; }
        .preview .empty { color: #999; font-style: italic; padding: 28px 8px;
                          text-align: center; border: 1px dashed #ccc; border-radius: 3px; }
        @media (max-width: 1100px) {
            .three-d-row { grid-template-columns: 1fr; }
            .preview { position: static; }
        }
    """)
    parts.append("</style></head><body>")

    parts.append(f"<h1>v7 pair smoke — {escape(placement_name)}</h1>")
    parts.append("<div class='meta'>")
    parts.append(f"Scene: <code>{escape(scene_file)}</code><br>")
    parts.append(f"Object: <code>{escape(object_file)}</code><br>")
    parts.append(
        f"Subject center O = <code>{[_r(v) for v in O]}</code>; "
        f"foot = <code>{[_r(v) for v in foot]}</code>; "
        f"height = <code>{_r(h_subject, 2)} m</code><br>"
    )
    rate = (K_accepted / max(1, attempts))
    badge = "ok" if K_accepted >= max(1, K_target // 2) else "bad"
    parts.append(
        f"Seed = <code>{seed}</code> · K = "
        f"<span class='{badge}'>{K_accepted}</span> / {K_target} · "
        f"attempts = {attempts} · acceptance rate = {rate:.2%}"
    )
    parts.append("</div>")

    # ---- 3D plot + preview panel side by side ----
    rendered_count = sum(len(r) for r in render_records)
    parts.append("<h2>Trajectories — interactive 3D"
                 f"{' (click a frame point for the render)' if rendered_count else ''}</h2>")
    parts.append("<div class='three-d-row'>")
    parts.append("<div id='plot3d' class='plot' style='margin:0'></div>")
    parts.append("<div class='preview'>")
    if rendered_count:
        parts.append("<div id='preview-img-wrap'>"
                     "<div class='empty' id='preview-empty'>"
                     "click any trajectory point to preview the render here"
                     "</div>"
                     "<img id='preview-img' style='display:none' alt='render'>"
                     "</div>")
        parts.append("<div class='label' id='preview-label'>"
                     f"(no point selected · {rendered_count} renders available)</div>")
    else:
        parts.append("<div class='empty'>renders not produced "
                     "(run with --render to enable)</div>")
    parts.append("</div></div>")
    # Nested {pair: {frame: src}} JS map. Kept out of customdata so the long
    # base64 URI never shows up as visible text in the click label.
    src_map_js: dict[int, dict[int, str]] = {}
    for (pi, fi), src in src_map.items():
        src_map_js.setdefault(pi, {})[fi] = src
    parts.append("<script>")
    parts.append("var traces3d = " + json.dumps(traces) + ";")
    parts.append("var layout3d = " + json.dumps(layout3d) + ";")
    parts.append("var SRC_MAP = " + json.dumps(src_map_js) + ";")
    parts.append("Plotly.newPlot('plot3d', traces3d, layout3d, "
                 "{responsive: true, displaylogo: false}).then(function(gd){"
                 "  gd.on('plotly_click', function(ev){"
                 "    if (!ev || !ev.points || !ev.points.length) return;"
                 "    var pt = ev.points[0];"
                 "    var cd = pt.customdata;"
                 "    if (!cd || cd.length < 2) return;"
                 "    var pair = cd[0], frame = cd[1];"
                 "    var path = (cd.length >= 3 ? cd[2] : '');"
                 "    var src = (SRC_MAP[pair] && SRC_MAP[pair][frame]) || '';"
                 "    var img = document.getElementById('preview-img');"
                 "    var empty = document.getElementById('preview-empty');"
                 "    var lbl = document.getElementById('preview-label');"
                 "    if (!img) return;"
                 "    if (src) { img.src = src; img.style.display = 'block';"
                 "               if (empty) empty.style.display = 'none';"
                 "               if (lbl) lbl.textContent = 'pair ' + pair +"
                 "                  ', frame ' + frame + (path ? ' · ' + path : ''); }"
                 "    else { img.style.display = 'none';"
                 "           if (empty) empty.style.display = 'block';"
                 "           if (lbl) lbl.textContent = 'pair ' + pair +"
                 "              ', frame ' + frame + ' — no render available'; }"
                 "  });"
                 "});")
    parts.append("</script>")

    # ---- Side-by-side 2D plots ----
    parts.append("<h2>Distribution checks</h2>")
    parts.append("<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px;'>")
    parts.append("  <div id='plot_radius' class='plot'></div>")
    parts.append("  <div id='plot_polar' class='plot'></div>")
    parts.append("  <div id='plot_pitch' class='plot'></div>")
    parts.append("  <div id='plot_yaw' class='plot'></div>")
    parts.append("</div>")
    parts.append("<script>")
    parts.append("var rTraces = " + json.dumps(r_traces) + ";")
    parts.append("var rLayout = " + json.dumps(r_layout) + ";")
    parts.append("Plotly.newPlot('plot_radius', rTraces, rLayout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("var polarTraces = " + json.dumps(polar_traces) + ";")
    parts.append("var polarLayout = " + json.dumps(polar_layout) + ";")
    parts.append("Plotly.newPlot('plot_polar', polarTraces, polarLayout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("var pitchTraces = " + json.dumps(pitch_traces) + ";")
    parts.append("var pitchLayout = " + json.dumps(pitch_layout) + ";")
    parts.append("Plotly.newPlot('plot_pitch', pitchTraces, pitchLayout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("var yawTraces = " + json.dumps(yaw_traces) + ";")
    parts.append("var yawLayout = " + json.dumps(yaw_layout) + ";")
    parts.append("Plotly.newPlot('plot_yaw', yawTraces, yawLayout, "
                 "{responsive: true, displaylogo: false});")
    parts.append("</script>")

    # ---- Per-pair table ----
    parts.append("<h2>Accepted pairs</h2>")
    parts.append("<table class='pairs'>")
    parts.append(
        "<tr>"
        "<th></th><th style='text-align:left'>pair</th>"
        "<th>r_start</th><th>r_end</th>"
        "<th>az_s</th><th>az_e</th>"
        "<th>elev_s</th><th>elev_e</th>"
        "<th>pitch_s</th><th>pitch_e</th>"
        "<th>yaw_s</th><th>yaw_e</th>"
        "<th>occ_s</th><th>occ_e</th>"
        "<th>a</th><th>b</th>"
        "<th>θ_near (deg)</th><th>arc-length<br>(approx)</th>"
        "</tr>"
    )
    for i, p in enumerate(accepted):
        E = p["ellipse"]
        col = _color(i)
        a, b = float(E["a"]), float(E["b"])
        # 32-frame arc length (sum segment distances)
        traj = p.get("trajectory_32f") or []
        if len(traj) > 1:
            arc = sum(
                math.dist(traj[k]["pos"], traj[k + 1]["pos"])
                for k in range(len(traj) - 1)
            )
        else:
            arc = 0.0
        parts.append(
            "<tr>"
            f"<td class='pair-color'><div style='background:{col}'></div></td>"
            f"<td class='pair-id'>{i:02d}</td>"
            f"<td>{p['start']['r']:.2f}</td>"
            f"<td>{p['end']['r']:.2f}</td>"
            f"<td>{p['start']['az_deg']:.1f}°</td>"
            f"<td>{p['end']['az_deg']:.1f}°</td>"
            f"<td>{p['start']['elev_deg']:.1f}°</td>"
            f"<td>{p['end']['elev_deg']:.1f}°</td>"
            f"<td>{p['start'].get('pitch_jitter_deg', 0.0):+.1f}°</td>"
            f"<td>{p['end'].get('pitch_jitter_deg', 0.0):+.1f}°</td>"
            f"<td>{p['start'].get('yaw_jitter_deg', 0.0):+.1f}°</td>"
            f"<td>{p['end'].get('yaw_jitter_deg', 0.0):+.1f}°</td>"
            f"<td>{p['start'].get('bbox_occupancy', 0.0)*100:.1f}%</td>"
            f"<td>{p['end'].get('bbox_occupancy', 0.0)*100:.1f}%</td>"
            f"<td>{a:.2f}</td>"
            f"<td>{b:.2f}</td>"
            f"<td>{float(E['theta_near_deg']):.1f}</td>"
            f"<td>{arc:.2f}m</td>"
            "</tr>"
        )
    parts.append("</table>")

    # ---- Rejection breakdown ----
    parts.append("<h2>Rejection breakdown</h2>")
    parts.append("<div class='rejtable'>")
    if rejections:
        for k, v in sorted(rejections.items(), key=lambda kv: -kv[1]):
            parts.append(f"<div><span class='k'>{escape(str(k))}</span> {v}</div>")
    else:
        parts.append("<div><em>(none)</em></div>")
    parts.append("</div>")
    if sub_reasons:
        parts.append("<details><summary>is_camera_valid sub-reasons</summary>")
        parts.append("<div class='rejtable' style='margin-top:8px'>")
        for k, v in sorted(sub_reasons.items(), key=lambda kv: -kv[1]):
            parts.append(f"<div><span class='k'>{escape(str(k))}</span> {v}</div>")
        parts.append("</div></details>")

    parts.append("</body></html>")

    out = smoke_dir / f"{placement_name}.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("smoke_dir", help="directory containing data.json from v7 smoke")
    p.add_argument("--no-embed", action="store_true",
                   help="Reference renders by relative URL instead of base64 embedding.")
    args = p.parse_args()
    out = build_report(Path(args.smoke_dir), embed=not args.no_embed)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
