#!/usr/bin/env python3
"""Build an HTML report summarizing a Stage 1 sampling sweep.

Reads:
  <stage1_dir>/summary.json                — aggregate stats
  <stage1_dir>/<placement>/data.json       — per-placement details

Writes:
  <stage1_dir>/report.html                 — self-contained (Plotly via CDN)

Visualizations:
  - top-line stat cards (counts, K_mean, wall time, acceptance rate)
  - K distribution (bar)
  - rejection-reason breakdown (bar)
  - sub-reason top-20 (bar)
  - per-placement attempts histogram
  - per-placement sample time histogram
  - per-placement setup time histogram
  - r_min / r_max distribution (overlaid hist)
  - per-scene K_mean (top 30 scenes by count, bar)
  - K=0 placements list (with top rejection reason)
  - sortable per-placement table

Usage:
    python scripts/v7_stage1_report.py outputs/v7_stage1_sample/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from html import escape
from pathlib import Path


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def collect(stage1_dir: Path) -> dict:
    """Walk every per-placement data.json and aggregate stats."""
    rows: list[dict] = []
    attempts: list[int] = []
    samples_s: list[float] = []
    setups_s: list[float] = []
    r_mins: list[float] = []
    r_maxs: list[float] = []
    per_scene_k: dict[str, list[int]] = defaultdict(list)
    k0_rows: list[dict] = []
    # occupancy filter analysis
    start_occ: list[float] = []
    end_occ: list[float] = []
    start_r: list[float] = []
    end_r: list[float] = []
    c_far_is_start_count: int = 0
    pairs_with_occ: int = 0

    t0 = time.time()
    dirs = [p for p in stage1_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    n = len(dirs)
    for i, pdir in enumerate(sorted(dirs)):
        dj = pdir / "data.json"
        if not dj.exists():
            continue
        try:
            d = json.loads(dj.read_text())
        except Exception:
            continue
        K = int(d.get("K_accepted", 0))
        att = int(d.get("attempts_used", 0))
        setup = float(d.get("time_setup_s", 0.0))
        sample = float(d.get("time_sample_s", 0.0))
        scene = (d.get("scene_file") or "").split("/")[-1].replace(".blend", "")
        obj = (d.get("object_file") or "").split("/")[-1].replace(".blend", "")
        rej = d.get("rejections_by_reason") or {}
        top_rej = max(rej.items(), key=lambda kv: kv[1])[0] if rej else ""

        attempts.append(att)
        samples_s.append(sample)
        setups_s.append(setup)
        per_scene_k[scene].append(K)

        for pair in d.get("accepted_pairs", []) or []:
            try:
                rs = float(pair["start"]["r"])
                re_ = float(pair["end"]["r"])
            except (KeyError, TypeError, ValueError):
                continue
            r_mins.append(min(rs, re_))
            r_maxs.append(max(rs, re_))
            try:
                so = float(pair["start"]["bbox_occupancy"])
                eo = float(pair["end"]["bbox_occupancy"])
            except (KeyError, TypeError, ValueError):
                continue
            start_occ.append(so)
            end_occ.append(eo)
            start_r.append(rs)
            end_r.append(re_)
            if bool(pair.get("c_far_is_start", False)):
                c_far_is_start_count += 1
            pairs_with_occ += 1

        row = {
            "name": d.get("placement", pdir.name),
            "scene": scene,
            "object": obj,
            "K": K,
            "att": att,
            "rate": (K / att) if att else 0.0,
            "setup": setup,
            "sample": sample,
            "top_rej": top_rej,
        }
        rows.append(row)
        if K == 0:
            k0_rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{n} ({time.time() - t0:.1f}s)")

    rows.sort(key=lambda r: (-r["K"], r["name"]))
    k0_rows.sort(key=lambda r: r["name"])

    per_scene_summary = []
    for scene, ks in per_scene_k.items():
        per_scene_summary.append({
            "scene": scene,
            "n": len(ks),
            "K_mean": sum(ks) / len(ks),
            "K_min": min(ks),
            "K_max": max(ks),
            "full_pct": 100.0 * sum(1 for k in ks if k == 12) / len(ks),
        })
    per_scene_summary.sort(key=lambda r: (-r["n"], r["scene"]))

    return {
        "rows": rows,
        "k0_rows": k0_rows,
        "per_scene": per_scene_summary,
        "attempts": attempts,
        "setups_s": setups_s,
        "samples_s": samples_s,
        "r_mins": r_mins,
        "r_maxs": r_maxs,
        "start_occ": start_occ,
        "end_occ": end_occ,
        "start_r": start_r,
        "end_r": end_r,
        "c_far_is_start_count": c_far_is_start_count,
        "pairs_with_occ": pairs_with_occ,
    }


def _histogram_bins(xs: list[float], nbins: int) -> tuple[list[float], list[int]]:
    if not xs:
        return [], []
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return [lo], [len(xs)]
    width = (hi - lo) / nbins
    edges = [lo + i * width for i in range(nbins + 1)]
    counts = [0] * nbins
    for x in xs:
        idx = int((x - lo) / width)
        if idx >= nbins:
            idx = nbins - 1
        counts[idx] += 1
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(nbins)]
    return centers, counts


def _append_occupancy_section(parts: list[str], agg: dict) -> None:
    s_occ = agg["start_occ"]
    e_occ = agg["end_occ"]
    s_r = agg["start_r"]
    e_r = agg["end_r"]
    n_pairs = agg["pairs_with_occ"]
    c_far_start = agg["c_far_is_start_count"]
    if not n_pairs:
        return

    parts.append("<h2>Occupancy filter analysis</h2>")
    parts.append(
        "<div class='note' style='font-size:12px;color:#555;"
        "background:#fff8e1;border-left:3px solid #d4a300;"
        "padding:6px 10px;margin:6px 0 10px;border-radius:4px'>"
        "<b>Filter rule</b>: <code>start.bbox_occupancy &ge; 0.01</code> "
        "<b>AND</b> <code>end.bbox_occupancy &ge; 0.01</code>. "
        "Midpoints (t = 0.25, 0.5, 0.75) <b>only</b> pass through "
        "<code>is_camera_valid</code> ray tests — <b>no</b> occupancy check. "
        "bbox_occupancy = (clipped 2D AABB area of projected subject mesh) "
        "&divide; (image area), measured in [0, 1]."
        "</div>"
    )

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs_s = sorted(xs)
        k = (len(xs_s) - 1) * (p / 100.0)
        lo = int(k); hi = min(lo + 1, len(xs_s) - 1)
        return xs_s[lo] + (xs_s[hi] - xs_s[lo]) * (k - lo)

    s_mean = sum(s_occ) / n_pairs
    e_mean = sum(e_occ) / n_pairs
    s_below_2 = sum(1 for x in s_occ if x < 0.02)
    e_below_2 = sum(1 for x in e_occ if x < 0.02)
    s_above_50 = sum(1 for x in s_occ if x >= 0.5)
    e_above_50 = sum(1 for x in e_occ if x >= 0.5)

    parts.append("<div class='cards'>")
    parts.append(
        f"<div class='card'><div class='label'>filter cutoff</div>"
        f"<div class='value'>1%</div>"
        f"<div class='sub'>min_occupancy = 0.01</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>pairs analyzed</div>"
        f"<div class='value'>{n_pairs:,}</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>start mean / p50</div>"
        f"<div class='value'>{100 * s_mean:.1f}%</div>"
        f"<div class='sub'>p50 = {100 * _pct(s_occ, 50):.1f}%</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>end mean / p50</div>"
        f"<div class='value'>{100 * e_mean:.1f}%</div>"
        f"<div class='sub'>p50 = {100 * _pct(e_occ, 50):.1f}%</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>c_far_is_start</div>"
        f"<div class='value'>{100 * c_far_start / n_pairs:.1f}%</div>"
        f"<div class='sub'>i.i.d. → expect 50%</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>near floor (occ &lt; 2%)</div>"
        f"<div class='value'>{s_below_2:,} / {e_below_2:,}</div>"
        f"<div class='sub'>start / end</div></div>"
    )
    parts.append(
        f"<div class='card'><div class='label'>large (occ &ge; 50%)</div>"
        f"<div class='value'>{s_above_50:,} / {e_above_50:,}</div>"
        f"<div class='sub'>start / end</div></div>"
    )
    parts.append("</div>")

    # ---- Row 1: PDF histogram + CDF ----
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_occ_pdf' class='plot' style='height:340px'></div>")
    parts.append("<div id='plot_occ_cdf' class='plot' style='height:340px'></div>")
    parts.append("</div>")

    nbins = 50
    edges = [i / nbins for i in range(nbins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(nbins)]

    def _hist_fixed(xs: list[float]) -> list[int]:
        counts = [0] * nbins
        for x in xs:
            idx = int(x * nbins)
            if idx >= nbins: idx = nbins - 1
            if idx < 0: idx = 0
            counts[idx] += 1
        return counts

    s_pdf = _hist_fixed(s_occ)
    e_pdf = _hist_fixed(e_occ)
    parts.append(
        f"<script>Plotly.newPlot('plot_occ_pdf', ["
        f"{{x:{centers},y:{s_pdf},type:'bar',name:'start',"
        f"marker:{{color:'#3a7bd5',opacity:0.6}}}}, "
        f"{{x:{centers},y:{e_pdf},type:'bar',name:'end',"
        f"marker:{{color:'#e15c5c',opacity:0.6}}}}], "
        f"{{margin:{{t:40,b:40,l:55,r:10}}, barmode:'overlay', "
        f"title:{{text:'Occupancy PDF (start vs end)',font:{{size:13}}}}, "
        f"xaxis:{{title:'bbox_occupancy',range:[0,1.0]}}, "
        f"yaxis:{{title:'#pairs (log)',type:'log'}}, "
        f"legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center'}}, "
        f"shapes:[{{type:'line',x0:0.01,x1:0.01,y0:0,y1:1,yref:'paper',"
        f"line:{{color:'#d4a300',width:2,dash:'dash'}}}}], "
        f"annotations:[{{x:0.012,y:1,yref:'paper',xref:'x',text:'cutoff 1%',"
        f"showarrow:false,xanchor:'left',font:{{color:'#a07000',size:11}}}}]"
        f"}}, {{displayModeBar:false}});</script>"
    )

    def _cdf(xs: list[float]) -> tuple[list[float], list[float]]:
        if not xs:
            return [], []
        xs_s = sorted(xs)
        n = len(xs_s)
        # downsample for plot
        step = max(1, n // 500)
        xv = xs_s[::step]
        yv = [(i * step + 1) / n for i in range(len(xv))]
        if xv[-1] != xs_s[-1]:
            xv.append(xs_s[-1]); yv.append(1.0)
        return xv, yv

    sxc, syc = _cdf(s_occ)
    exc, eyc = _cdf(e_occ)
    parts.append(
        f"<script>Plotly.newPlot('plot_occ_cdf', ["
        f"{{x:{sxc},y:{syc},type:'scatter',mode:'lines',name:'start',"
        f"line:{{color:'#3a7bd5',width:2}}}}, "
        f"{{x:{exc},y:{eyc},type:'scatter',mode:'lines',name:'end',"
        f"line:{{color:'#e15c5c',width:2}}}}], "
        f"{{margin:{{t:40,b:40,l:55,r:10}}, "
        f"title:{{text:'Occupancy CDF',font:{{size:13}}}}, "
        f"xaxis:{{title:'bbox_occupancy',range:[0,1]}}, "
        f"yaxis:{{title:'fraction of pairs &le; x',range:[0,1.02]}}, "
        f"legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center'}}, "
        f"shapes:[{{type:'line',x0:0.01,x1:0.01,y0:0,y1:1,line:{{color:'#d4a300',width:2,dash:'dash'}}}}, "
        f"{{type:'line',x0:0,x1:1,y0:0.5,y1:0.5,line:{{color:'#888',width:1,dash:'dot'}}}}]"
        f"}}, {{displayModeBar:false}});</script>"
    )

    # ---- Row 2: bucket bar + r distribution overlay ----
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_occ_buckets' class='plot' style='height:340px'></div>")
    parts.append("<div id='plot_r_se' class='plot' style='height:340px'></div>")
    parts.append("</div>")

    buckets = [
        ("1-2%", 0.01, 0.02),
        ("2-5%", 0.02, 0.05),
        ("5-10%", 0.05, 0.10),
        ("10-20%", 0.10, 0.20),
        ("20-30%", 0.20, 0.30),
        ("30-50%", 0.30, 0.50),
        ("50-100%", 0.50, 1.001),
    ]
    bkt_labels = [b[0] for b in buckets]
    s_bkt = [sum(1 for x in s_occ if lo <= x < hi) for _, lo, hi in buckets]
    e_bkt = [sum(1 for x in e_occ if lo <= x < hi) for _, lo, hi in buckets]
    parts.append(
        f"<script>Plotly.newPlot('plot_occ_buckets', ["
        f"{{x:{bkt_labels!r},y:{s_bkt},type:'bar',name:'start',"
        f"marker:{{color:'#3a7bd5'}}, text:{s_bkt}, textposition:'outside'}}, "
        f"{{x:{bkt_labels!r},y:{e_bkt},type:'bar',name:'end',"
        f"marker:{{color:'#e15c5c'}}, text:{e_bkt}, textposition:'outside'}}], "
        f"{{margin:{{t:40,b:40,l:55,r:10}}, barmode:'group', "
        f"title:{{text:'Occupancy by bucket (count)',font:{{size:13}}}}, "
        f"xaxis:{{title:'bbox_occupancy bucket'}}, "
        f"yaxis:{{title:'#pairs'}}, "
        f"legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center'}}"
        f"}}, {{displayModeBar:false}});</script>"
    )

    # r distribution: start_r vs end_r overlay (using log bins fits log-uniform sampling)
    log_edges = []
    n_lb = 40
    import math
    lo_r, hi_r = 0.8, 8.0
    for i in range(n_lb + 1):
        log_edges.append(math.exp(math.log(lo_r) + (math.log(hi_r) - math.log(lo_r)) * i / n_lb))
    log_centers = [(log_edges[i] + log_edges[i + 1]) / 2 for i in range(n_lb)]

    def _bin_log(xs: list[float]) -> list[int]:
        counts = [0] * n_lb
        for x in xs:
            if x < log_edges[0] or x > log_edges[-1]:
                continue
            t = (math.log(x) - math.log(lo_r)) / (math.log(hi_r) - math.log(lo_r))
            idx = int(t * n_lb)
            if idx >= n_lb: idx = n_lb - 1
            if idx < 0: idx = 0
            counts[idx] += 1
        return counts

    s_rbin = _bin_log(s_r)
    e_rbin = _bin_log(e_r)
    parts.append(
        f"<script>Plotly.newPlot('plot_r_se', ["
        f"{{x:{log_centers},y:{s_rbin},type:'bar',name:'start.r',"
        f"marker:{{color:'#3a7bd5',opacity:0.6}}}}, "
        f"{{x:{log_centers},y:{e_rbin},type:'bar',name:'end.r',"
        f"marker:{{color:'#e15c5c',opacity:0.6}}}}], "
        f"{{margin:{{t:40,b:40,l:55,r:10}}, barmode:'overlay', "
        f"title:{{text:'r distribution (start vs end, log-binned)',font:{{size:13}}}}, "
        f"xaxis:{{title:'r (m)',type:'log',range:[Math.log10(0.8),Math.log10(8.0)]}}, "
        f"yaxis:{{title:'#pairs'}}, "
        f"legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center'}}"
        f"}}, {{displayModeBar:false}});</script>"
    )

    # ---- Row 3: 2D histograms (r, occ) for start and end — pre-binned heatmaps ----
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_2d_start' class='plot' style='height:380px'></div>")
    parts.append("<div id='plot_2d_end' class='plot' style='height:380px'></div>")
    parts.append("</div>")

    NX, NY = 40, 40
    log_lo, log_hi = math.log(0.8), math.log(8.0)
    x_centers = [
        math.exp(log_lo + (log_hi - log_lo) * (i + 0.5) / NX) for i in range(NX)
    ]
    y_centers = [(i + 0.5) / NY for i in range(NY)]

    def _hist2d(rs: list[float], occs: list[float]) -> list[list[int]]:
        # rows = Y (occupancy), cols = X (r) — plotly heatmap expects z[y][x]
        z = [[0] * NX for _ in range(NY)]
        for r_, o_ in zip(rs, occs):
            if r_ < 0.8 or r_ > 8.0:
                continue
            tx = (math.log(r_) - log_lo) / (log_hi - log_lo)
            ty = max(0.0, min(0.999, o_))
            ix = min(NX - 1, max(0, int(tx * NX)))
            iy = min(NY - 1, max(0, int(ty * NY)))
            z[iy][ix] += 1
        return z

    z_s = _hist2d(s_r, s_occ)
    z_e = _hist2d(e_r, e_occ)
    z_max = max(max(max(row) for row in z_s), max(max(row) for row in z_e))

    parts.append(
        f"<script>Plotly.newPlot('plot_2d_start', [{{"
        f"z:{z_s},x:{x_centers},y:{y_centers},"
        f"type:'heatmap',colorscale:'Viridis',zmin:0,zmax:{z_max}, "
        f"colorbar:{{title:'#pairs'}}, "
        f"hovertemplate:'r=%{{x:.2f}}m<br>occ=%{{y:.2f}}<br>#=%{{z}}<extra></extra>'}}], "
        f"{{margin:{{t:40,b:50,l:55,r:10}}, "
        f"title:{{text:'start: r vs occupancy',font:{{size:13}}}}, "
        f"xaxis:{{title:'r (m)',type:'log'}}, "
        f"yaxis:{{title:'bbox_occupancy',range:[0,1]}}, "
        f"shapes:[{{type:'line',x0:0.8,x1:8.0,y0:0.01,y1:0.01,line:{{color:'#d4a300',width:2,dash:'dash'}}}}]"
        f"}}, {{displayModeBar:false}});</script>"
    )
    parts.append(
        f"<script>Plotly.newPlot('plot_2d_end', [{{"
        f"z:{z_e},x:{x_centers},y:{y_centers},"
        f"type:'heatmap',colorscale:'Viridis',zmin:0,zmax:{z_max}, "
        f"colorbar:{{title:'#pairs'}}, "
        f"hovertemplate:'r=%{{x:.2f}}m<br>occ=%{{y:.2f}}<br>#=%{{z}}<extra></extra>'}}], "
        f"{{margin:{{t:40,b:50,l:55,r:10}}, "
        f"title:{{text:'end: r vs occupancy',font:{{size:13}}}}, "
        f"xaxis:{{title:'r (m)',type:'log'}}, "
        f"yaxis:{{title:'bbox_occupancy',range:[0,1]}}, "
        f"shapes:[{{type:'line',x0:0.8,x1:8.0,y0:0.01,y1:0.01,line:{{color:'#d4a300',width:2,dash:'dash'}}}}]"
        f"}}, {{displayModeBar:false}});</script>"
    )

    # ---- Asymmetry note ----
    parts.append(
        "<details style='margin-top:10px'>"
        "<summary>Why is start systematically smaller than end? (selection bias)</summary>"
        "<div style='font-size:12px;color:#444;padding:8px 4px;line-height:1.5'>"
        "<code>start</code> &amp; <code>end</code> are i.i.d. log-uniform draws on r &isin; [0.8, 8] m. "
        "Naive expectation: identical marginals and <code>c_far_is_start</code> &asymp; 50%. "
        "Empirically <b>start mean r &lt; end mean r</b> and <code>c_far_is_start</code> &asymp; 32%. "
        "Mechanism: <code>start</code> is validated <b>first</b>; "
        "rejection rate is higher for large r (camera far &rArr; subject smaller &rArr; "
        "more likely off-frame, harder to clear occlusion ray tests). "
        "So large-r <code>start</code> draws get filtered out before <code>end</code> is even sampled. "
        "<code>end</code> is then evaluated in scenes where <code>start</code> already passed (selection on 'easy' scene), "
        "so its r distribution is closer to the unfiltered log-uniform. "
        "<b>Fix candidate</b>: randomly swap (start, end) in <code>try_pair</code> before validation, "
        "or randomize which is checked first. This is a sampling-order artifact, not a data-quality issue: "
        "both endpoints still pass the same filters."
        "</div></details>"
    )


def build_report(stage1_dir: Path, out_path: Path) -> None:
    summary_path = stage1_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    summary = json.loads(summary_path.read_text())

    print(f"[report] walking per-placement data.json under {stage1_dir} ...")
    agg = collect(stage1_dir)
    n_rows = len(agg["rows"])
    print(f"[report] aggregated {n_rows} placements with data.json")

    run = summary.get("run", {})
    K_dist = summary.get("K_distribution", {})
    rej = summary.get("rejections_by_reason", {})
    sub = summary.get("sub_reasons_top20", {})

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>v7 Stage 1 sweep — {escape(stage1_dir.name)}</title>")
    parts.append("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>")
    parts.append("""<style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 24px; background: #fafafa; color: #1a1a1a; }
        h1 { font-size: 24px; margin: 0 0 4px; }
        h2 { font-size: 16px; margin: 24px 0 8px; border-bottom: 1px solid #ddd;
             padding-bottom: 4px; }
        .meta { font-size: 12px; color: #666; margin-bottom: 16px;
                font-family: ui-monospace, monospace; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                 gap: 10px; margin: 12px 0 16px; }
        .card { background: white; border: 1px solid #d8d8d8; border-radius: 6px;
                padding: 10px 12px; }
        .card .label { font-size: 11px; color: #777; text-transform: uppercase;
                       letter-spacing: 0.5px; }
        .card .value { font-size: 20px; font-family: ui-monospace, monospace;
                       font-weight: 600; margin-top: 2px; color: #234; }
        .card .sub { font-size: 11px; color: #999; margin-top: 2px;
                     font-family: ui-monospace, monospace; }
        .plot { background: white; border: 1px solid #d8d8d8; border-radius: 6px;
                padding: 8px; }
        .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        table { border-collapse: collapse; width: 100%; font-size: 12px;
                font-family: ui-monospace, monospace; background: white;
                border: 1px solid #d8d8d8; border-radius: 6px; overflow: hidden; }
        th, td { padding: 6px 8px; border-bottom: 1px solid #eee; text-align: left; }
        th { background: #f1f3f6; cursor: pointer; user-select: none;
             position: sticky; top: 0; }
        th:hover { background: #e6eaf0; }
        tr:hover td { background: #fafbfc; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .badge { padding: 1px 6px; border-radius: 10px; font-size: 11px; }
        .badge.ok { background: #dff2dd; color: #2a7a2a; }
        .badge.mid { background: #fff3cc; color: #8a6d00; }
        .badge.bad { background: #f7dddd; color: #b04040; }
        details { background: white; border: 1px solid #d8d8d8; border-radius: 6px;
                  padding: 8px 12px; margin: 12px 0; }
        summary { font-size: 14px; font-weight: 600; cursor: pointer;
                  color: #234; }
        .table-wrap { max-height: 600px; overflow: auto; border-radius: 6px; }
        .filter { margin: 8px 0; padding: 6px 10px; font-size: 13px;
                  border: 1px solid #ccc; border-radius: 4px; width: 280px; }
    </style></head><body>""")

    # ----- header -----
    parts.append(f"<h1>v7 Stage 1 sampling sweep</h1>")
    parts.append(
        f"<div class='meta'>"
        f"path: <code>{escape(str(stage1_dir))}</code> · "
        f"wall time: {_fmt_dur(run.get('wall_time_s', 0))} on "
        f"{run.get('workers', '?')} workers"
        f"</div>"
    )

    # ----- stat cards -----
    n_total = run.get("n_total", summary.get("n_total_valid", 0))
    n_ok = run.get("n_ok", 0)
    n_fail = run.get("n_fail", 0)
    accepted = summary.get("accepted_total", 0)
    attempts = summary.get("attempts_total", 0)
    K_mean = summary.get("K_mean", 0.0)
    full_count = K_dist.get("12", K_dist.get(12, 0))
    full_pct = 100.0 * full_count / n_rows if n_rows else 0.0

    def card(label: str, value: str, sub: str = "") -> str:
        sub_html = f"<div class='sub'>{escape(sub)}</div>" if sub else ""
        return (f"<div class='card'><div class='label'>{escape(label)}</div>"
                f"<div class='value'>{escape(value)}</div>{sub_html}</div>")

    parts.append("<div class='cards'>")
    parts.append(card("placements", f"{n_total:,}",
                      f"ok={n_ok:,} fail={n_fail}"))
    parts.append(card("with data", f"{n_rows:,}", ""))
    parts.append(card("K_mean", f"{K_mean:.2f}", "of 12"))
    parts.append(card("K=12 (full)", f"{full_count:,}",
                      f"{full_pct:.1f}% of placed"))
    parts.append(card("accepted total", f"{accepted:,}",
                      f"= {accepted * 32:,} frames"))
    parts.append(card("acceptance rate",
                      f"{100 * summary.get('acceptance_rate', 0):.1f}%",
                      f"{attempts:,} attempts"))
    parts.append(card("setup time",
                      _fmt_dur(summary.get('time_setup_total_s', 0)),
                      f"avg {summary.get('time_setup_total_s', 0) / max(1, n_rows):.1f}s/p"))
    parts.append(card("sample time",
                      _fmt_dur(summary.get('time_sample_total_s', 0)),
                      f"avg {summary.get('time_sample_total_s', 0) / max(1, n_rows):.1f}s/p"))
    parts.append("</div>")

    # ----- K distribution -----
    parts.append("<h2>K distribution (accepted pairs per placement)</h2>")
    parts.append("<div id='plot_K' class='plot' style='height:320px'></div>")
    K_keys = sorted(int(k) for k in K_dist.keys())
    K_x = K_keys
    K_y = [K_dist.get(str(k), K_dist.get(k, 0)) for k in K_keys]
    colors = ["#b04040" if k == 0 else "#8a6d00" if k < 6 else "#2a7a2a" if k == 12 else "#7888a0"
              for k in K_keys]
    parts.append(f"<script>Plotly.newPlot('plot_K', "
                 f"[{{x: {K_x}, y: {K_y}, type: 'bar', "
                 f"marker: {{color: {colors!r}}}, "
                 f"text: {K_y}, textposition: 'outside'}}], "
                 f"{{margin: {{t:20,b:40,l:50,r:10}}, "
                 f"xaxis:{{title:'K_accepted',dtick:1}}, "
                 f"yaxis:{{title:'#placements'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    # ----- rejection reasons + sub-reasons (side by side) -----
    parts.append("<h2>Rejection breakdown</h2>")
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_rej' class='plot' style='height:340px'></div>")
    parts.append("<div id='plot_sub' class='plot' style='height:340px'></div>")
    parts.append("</div>")
    rej_items = sorted(rej.items(), key=lambda kv: -kv[1])
    rej_x = [k for k, _ in rej_items]
    rej_y = [v for _, v in rej_items]
    parts.append(f"<script>Plotly.newPlot('plot_rej', "
                 f"[{{x: {rej_y}, y: {rej_x!r}, type: 'bar', orientation: 'h', "
                 f"marker:{{color:'#b04040'}}, text: {rej_y}, textposition: 'outside'}}], "
                 f"{{margin:{{t:30,b:40,l:140,r:60}}, "
                 f"title:{{text:'Top reasons',font:{{size:13}}}}, "
                 f"xaxis:{{title:'# rejections'}}, "
                 f"yaxis:{{autorange:'reversed'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    sub_items = sorted(sub.items(), key=lambda kv: -kv[1])
    sub_x = [k for k, _ in sub_items]
    sub_y = [v for _, v in sub_items]
    parts.append(f"<script>Plotly.newPlot('plot_sub', "
                 f"[{{x: {sub_y}, y: {sub_x!r}, type: 'bar', orientation: 'h', "
                 f"marker:{{color:'#7080a0'}}, text: {sub_y}, textposition: 'outside'}}], "
                 f"{{margin:{{t:30,b:40,l:200,r:60}}, "
                 f"title:{{text:'Sub-reasons (top 20)',font:{{size:13}}}}, "
                 f"xaxis:{{title:'# rejections'}}, "
                 f"yaxis:{{autorange:'reversed'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    # ----- timing histograms + attempts histogram (3-up) -----
    parts.append("<h2>Per-placement timing & attempts</h2>")
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_setup' class='plot' style='height:280px'></div>")
    parts.append("<div id='plot_sample' class='plot' style='height:280px'></div>")
    parts.append("</div>")
    parts.append("<div class='row2'>")
    parts.append("<div id='plot_attempts' class='plot' style='height:280px'></div>")
    parts.append("<div id='plot_r' class='plot' style='height:280px'></div>")
    parts.append("</div>")

    sx, sy = _histogram_bins(agg["setups_s"], 40)
    parts.append(f"<script>Plotly.newPlot('plot_setup', "
                 f"[{{x:{sx},y:{sy},type:'bar',marker:{{color:'#5b8def'}}}}], "
                 f"{{margin:{{t:30,b:40,l:50,r:10}}, "
                 f"title:{{text:'setup_s per placement',font:{{size:13}}}}, "
                 f"xaxis:{{title:'seconds'}}, yaxis:{{title:'#placements'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    sx2, sy2 = _histogram_bins(agg["samples_s"], 40)
    parts.append(f"<script>Plotly.newPlot('plot_sample', "
                 f"[{{x:{sx2},y:{sy2},type:'bar',marker:{{color:'#e89b5a'}}}}], "
                 f"{{margin:{{t:30,b:40,l:50,r:10}}, "
                 f"title:{{text:'sample_s per placement',font:{{size:13}}}}, "
                 f"xaxis:{{title:'seconds'}}, yaxis:{{title:'#placements'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    ax, ay = _histogram_bins([float(a) for a in agg["attempts"]], 40)
    parts.append(f"<script>Plotly.newPlot('plot_attempts', "
                 f"[{{x:{ax},y:{ay},type:'bar',marker:{{color:'#8e75c8'}}}}], "
                 f"{{margin:{{t:30,b:40,l:50,r:10}}, "
                 f"title:{{text:'attempts_used per placement',font:{{size:13}}}}, "
                 f"xaxis:{{title:'attempts'}}, yaxis:{{title:'#placements'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    # r distribution (mins + maxs overlaid)
    rmin_x, rmin_y = _histogram_bins(agg["r_mins"], 40)
    rmax_x, rmax_y = _histogram_bins(agg["r_maxs"], 40)
    parts.append(f"<script>Plotly.newPlot('plot_r', ["
                 f"{{x:{rmin_x},y:{rmin_y},type:'bar',name:'r_min',"
                 f"marker:{{color:'#5b8def',opacity:0.7}}}}, "
                 f"{{x:{rmax_x},y:{rmax_y},type:'bar',name:'r_max',"
                 f"marker:{{color:'#e15c5c',opacity:0.7}}}}], "
                 f"{{margin:{{t:30,b:40,l:50,r:10}}, barmode:'overlay', "
                 f"title:{{text:'r range across all clips',font:{{size:13}}}}, "
                 f"xaxis:{{title:'r (m)'}}, yaxis:{{title:'#clips'}}, "
                 f"legend:{{orientation:'h',y:1.1}}}}, "
                 f"{{displayModeBar:false}});</script>")

    # ----- occupancy filter analysis -----
    _append_occupancy_section(parts, agg)

    # ----- per-scene K_mean -----
    parts.append("<h2>Per-scene aggregate (top 30 by # placements)</h2>")
    top_scenes = agg["per_scene"][:30]
    sc_labels = [s["scene"] for s in top_scenes]
    sc_kmean = [round(s["K_mean"], 2) for s in top_scenes]
    sc_n = [s["n"] for s in top_scenes]
    sc_full = [round(s["full_pct"], 1) for s in top_scenes]
    parts.append("<div id='plot_scene' class='plot' style='height:520px'></div>")
    parts.append(f"<script>Plotly.newPlot('plot_scene', [{{"
                 f"x: {sc_kmean}, y: {sc_labels!r}, type:'bar',orientation:'h',"
                 f"marker:{{color:{sc_kmean!r},colorscale:'RdYlGn',cmin:0,cmax:12,"
                 f"colorbar:{{title:'K_mean'}}}}, "
                 f"text: {sc_kmean}, textposition:'outside', "
                 f"customdata: {[[n, f] for n, f in zip(sc_n, sc_full)]!r}, "
                 f"hovertemplate:'%{{y}}<br>K_mean=%{{x}}<br>"
                 f"n=%{{customdata[0]}} placements<br>full=%{{customdata[1]}}%<extra></extra>'}}], "
                 f"{{margin:{{t:30,b:40,l:280,r:80}}, "
                 f"title:{{text:'K_mean by scene',font:{{size:13}}}}, "
                 f"xaxis:{{range:[0,12.5]}}, yaxis:{{autorange:'reversed'}}}}, "
                 f"{{displayModeBar:false}});</script>")

    # ----- K=0 placements list -----
    parts.append(f"<h2>K=0 placements ({len(agg['k0_rows'])})</h2>")
    if agg["k0_rows"]:
        parts.append("<details><summary>show list</summary>")
        parts.append("<div class='table-wrap' style='max-height:400px'><table>")
        parts.append("<thead><tr><th>placement</th><th>scene</th>"
                     "<th class='num'>attempts</th><th>top rejection</th></tr></thead><tbody>")
        for r in agg["k0_rows"]:
            parts.append(
                f"<tr><td>{escape(r['name'])}</td>"
                f"<td>{escape(r['scene'])}</td>"
                f"<td class='num'>{r['att']}</td>"
                f"<td>{escape(r['top_rej'])}</td></tr>"
            )
        parts.append("</tbody></table></div></details>")

    # ----- per-placement sortable table -----
    parts.append(f"<h2>Per-placement table ({n_rows} rows)</h2>")
    parts.append("<input id='filter' class='filter' "
                 "placeholder='filter by name / scene / object / reason...'>")
    parts.append("<div class='table-wrap'><table id='tbl'>")
    parts.append("<thead><tr>"
                 "<th data-col='name'>placement</th>"
                 "<th data-col='scene'>scene</th>"
                 "<th data-col='K' data-num='1'>K</th>"
                 "<th data-col='att' data-num='1'>att</th>"
                 "<th data-col='rate' data-num='1'>rate</th>"
                 "<th data-col='setup' data-num='1'>setup s</th>"
                 "<th data-col='sample' data-num='1'>sample s</th>"
                 "<th data-col='top_rej'>top rejection</th>"
                 "</tr></thead><tbody>")
    for r in agg["rows"]:
        if r["K"] == 12:
            cls = "ok"
        elif r["K"] >= 6:
            cls = "mid"
        else:
            cls = "bad"
        parts.append(
            f"<tr>"
            f"<td>{escape(r['name'])}</td>"
            f"<td>{escape(r['scene'])}</td>"
            f"<td class='num'><span class='badge {cls}'>{r['K']}</span></td>"
            f"<td class='num'>{r['att']}</td>"
            f"<td class='num'>{r['rate']:.0%}</td>"
            f"<td class='num'>{r['setup']:.1f}</td>"
            f"<td class='num'>{r['sample']:.1f}</td>"
            f"<td>{escape(r['top_rej'])}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table></div>")

    # client-side sort + filter
    parts.append("""<script>
    (function(){
        const tbl = document.getElementById('tbl');
        const filter = document.getElementById('filter');
        const tbody = tbl.tBodies[0];
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const headers = tbl.tHead.rows[0].cells;
        let sortIdx = -1, sortDir = 1;

        function applyFilter(){
            const q = filter.value.toLowerCase().trim();
            rows.forEach(r => {
                if (!q) { r.style.display = ''; return; }
                const text = r.textContent.toLowerCase();
                r.style.display = text.includes(q) ? '' : 'none';
            });
        }
        filter.addEventListener('input', applyFilter);

        Array.from(headers).forEach((h, i) => {
            h.addEventListener('click', () => {
                sortDir = (sortIdx === i) ? -sortDir : 1;
                sortIdx = i;
                const isNum = !!h.dataset.num;
                rows.sort((a, b) => {
                    const av = a.cells[i].textContent.trim();
                    const bv = b.cells[i].textContent.trim();
                    if (isNum) {
                        const af = parseFloat(av.replace('%','')) || 0;
                        const bf = parseFloat(bv.replace('%','')) || 0;
                        return (af - bf) * sortDir;
                    }
                    return av.localeCompare(bv) * sortDir;
                });
                tbody.innerHTML = '';
                rows.forEach(r => tbody.appendChild(r));
            });
        });
    })();
    </script>""")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage1_dir",
                    help="Stage 1 output directory (contains summary.json + per-placement subdirs).")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <stage1_dir>/report.html).")
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_dir).resolve()
    if not stage1_dir.is_dir():
        print(f"error: {stage1_dir} is not a directory", file=sys.stderr)
        return 1

    out_path = Path(args.out).resolve() if args.out else (stage1_dir / "report.html")
    build_report(stage1_dir, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[report] wrote {out_path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
