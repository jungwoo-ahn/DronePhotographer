"""HTML report: is V(frame, fixed goal) monotonically increasing along trajectories?

For each sampled trajectory (placement, accepted_pair) the report computes the
fixed-goal value curve  V(i) = -geometric_profile_distance(frame_i, goal) with
goal = the trajectory's last scored frame, then shows:

  - a grid of sparkline cards, sortable worst-first (least monotone) so the
    interesting failures are on top; each card shows corr(V, i) and the fraction
    of rising steps,
  - click a card -> large curve + frame scrubber with the current frame and the
    goal frame side by side (images lazy-load via the local HTTP server).

Images are NOT embedded — the report references `/v7_renders/...` URLs, so view
it through the http.server that serves `outputs/` (with the `v7_renders` symlink
in place). Not a standalone file.

Usage:
  PYTHONPATH=. python scripts/visualize_monotonicity_html.py \
      --src outputs/v7_renders \
      --out outputs/monotonicity.html \
      [--max-placements 40] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from src.policy.common.reward import _great_circle, pose_to_geometry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=Path("outputs/v7_renders"), type=Path,
                   help="directory of scored placements; also the URL prefix used in the page")
    p.add_argument("--out", default=Path("outputs/monotonicity.html"), type=Path)
    p.add_argument("--max-placements", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_payload(args: argparse.Namespace) -> list[dict]:
    placements = sorted(
        d.name for d in args.src.iterdir()
        if (d / "scored.flag").exists() and (d / "data.json").exists()
    )
    rng = random.Random(args.seed)
    if len(placements) > args.max_placements:
        placements = rng.sample(placements, args.max_placements)

    url_prefix = "/" + args.src.name  # served relative to outputs/
    out = []
    for pl in sorted(placements):
        doc = json.loads((args.src / pl / "data.json").read_text())
        center = doc.get("subject_center") or doc.get("subject_foot") or [0, 0, 0]
        height = float(doc.get("subject_height") or 1.7)
        pairs = doc.get("accepted_pairs") or []
        for pair_idx, recs in enumerate(doc.get("render_records") or []):
            traj = (pairs[pair_idx].get("trajectory_32f") or []) if pair_idx < len(pairs) else []
            by = {r["frame_idx"]: r for r in recs if "scores" in r}
            idxs = [i for i in sorted(by) if i < len(traj)]
            if len(idxs) < 4:
                continue
            # Geometry from POSES (same computation training uses for the value) —
            # exact at every frame, immune to the scorer's off-screen clamp.
            geo = {i: pose_to_geometry(traj[i]["pos"], traj[i]["forward"], traj[i]["up"], center, height)
                   for i in idxs}
            g_geo = geo[idxs[-1]]
            curve, t_view, t_size, t_aim, zero = [], [], [], [], []
            for i in idxs:
                a = geo[i]
                dv = _great_circle(a["az"], a["el"], g_geo["az"], g_geo["el"])
                ds = a["size"] - g_geo["size"]
                da = math.hypot(a["aim_x"] - g_geo["aim_x"], a["aim_y"] - g_geo["aim_y"])
                t_view.append(round(-dv, 4))
                t_size.append(round(-abs(ds), 4))
                t_aim.append(round(-da, 4))
                curve.append(round(-math.sqrt(dv * dv + ds * ds + da * da), 4))
                # mark frames the scorer clamped (their PROFILES are sentinel-zeroed;
                # training filters windows whose goal lands here)
                s = by[i]["scores"]
                zero.append(1 if (s["occupancy"] == 0 and s["bbox_y_offset"] == 0) else 0)
            if np.std(curve) < 1e-9:
                continue
            diffs = np.diff(curve)
            corr = float(np.corrcoef(curve, range(len(curve)))[0, 1])
            rising = float(np.mean(diffs >= -1e-9))
            out.append({
                "placement": pl,
                "pair_idx": pair_idx,
                "frame_idxs": idxs,
                "curve": curve,
                "t_view": t_view,
                "t_size": t_size,
                "t_aim": t_aim,
                "zero": zero,
                "corr": round(corr, 3),
                "rising": round(rising, 3),
                "imgs": [f"{url_prefix}/{pl}/{by[i]['path_rel']}" for i in idxs],
            })
    return out


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Fixed-goal value monotonicity</title>
<style>
  body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; background: #14161a; color: #e6e6e6; }}
  h1 {{ font-size: 19px; }} h2 {{ font-size: 15px; color: #9fd3ff; margin: 6px 0; }}
  .stats {{ color:#8a93a3; margin-bottom: 10px; }}
  .grid {{ display:flex; flex-wrap: wrap; gap: 10px; }}
  .cardS {{ background:#1d2026; border:1px solid #2c313a; border-radius:8px; padding:8px; cursor:pointer; }}
  .cardS:hover {{ border-color:#5aa9e6; }}
  .cardS.sel {{ border-color:#ffb86b; }}
  .cardS .meta {{ font-size:11px; color:#8a93a3; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .good {{ color:#7ee29a; }} .bad {{ color:#e6675a; }} .mid {{ color:#ffb86b; }}
  #detail {{ background:#1d2026; border:1px solid #2c313a; border-radius:10px; padding:14px; margin:16px 0; }}
  #detail img {{ border-radius:6px; max-width: 420px; display:block; }}
  .row {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .cap {{ color:#8a93a3; font-size:12px; margin-top:4px; }}
  select {{ background:#222; color:#eee; border:1px solid #444; padding:3px 8px; border-radius:6px; }}
  .bigval {{ font-size:22px; font-weight:700; color:#7ee29a; }}
</style></head><body>
<h1>Fixed-goal value monotonicity — V(frame, goal = last frame) per trajectory</h1>
<div class="stats" id="stats"></div>
<div>sort: <select id="sortSel">
  <option value="worst">least monotone first</option>
  <option value="best">most monotone first</option>
</select></div>
<div id="detail" style="display:none">
  <h2 id="dTitle"></h2>
  <div class="row">
    <div><svg id="dCurve" width="640" height="260"></svg>
      <div class="cap">
        <span style="color:#7ee29a">●</span> total V &nbsp;
        <span style="color:#5aa9e6">—</span> view (great-circle) &nbsp;
        <span style="color:#c792ea">—</span> size &nbsp;
        <span style="color:#e6c07b">—</span> aim &nbsp;
        <span style="color:#e6675a">✕</span> off-screen-clamped frame (scores zeroed) &nbsp;|&nbsp;
        <span style="color:#ffb86b">orange</span> = selected
      </div>
      <input type="range" id="dSlider" min="0" value="0" style="width:640px"></div>
    <div><h2>frame <span id="dIdx"></span></h2><img id="dImg"><div class="cap" id="dVal"></div></div>
    <div><h2>goal (last frame)</h2><img id="dGoal"><div class="cap">V = 0 here by construction</div></div>
  </div>
</div>
<div class="grid" id="grid"></div>
<script>
const DATA = {data_json};
const stats = document.getElementById('stats');
const corrs = DATA.map(t => t.corr);
const med = corrs.slice().sort((a,b)=>a-b)[Math.floor(corrs.length/2)];
stats.textContent = `${{DATA.length}} trajectories | median corr(V, frame) = ${{med.toFixed(3)}} | corr>0.8: ${{(100*corrs.filter(c=>c>0.8).length/corrs.length).toFixed(0)}}% | corr<0: ${{(100*corrs.filter(c=>c<0).length/corrs.length).toFixed(0)}}%`;
let order = DATA.map((_, i) => i);
let selIdx = null;
function cls(c) {{ return c > 0.8 ? 'good' : (c > 0.3 ? 'mid' : 'bad'); }}
function spark(t, w, h) {{
  const vmin = Math.min(...t.curve), vmax = Math.max(...t.curve);
  const pts = t.curve.map((v, i) =>
    `${{(w-4) * i / (t.curve.length-1) + 2}},${{2 + (h-4) * (vmax - v) / Math.max(1e-9, vmax - vmin)}}`).join(' ');
  return `<svg width="${{w}}" height="${{h}}"><polyline fill="none" stroke="#7ee29a" stroke-width="1.2" points="${{pts}}"/></svg>`;
}}
function renderGrid() {{
  const sortWorst = document.getElementById('sortSel').value === 'worst';
  order.sort((a, b) => sortWorst ? DATA[a].corr - DATA[b].corr : DATA[b].corr - DATA[a].corr);
  const g = document.getElementById('grid');
  g.innerHTML = '';
  order.forEach(i => {{
    const t = DATA[i];
    const d = document.createElement('div');
    d.className = 'cardS' + (i === selIdx ? ' sel' : '');
    d.innerHTML = spark(t, 150, 44) +
      `<div class="meta">${{t.placement}}</div>` +
      `<div class="meta">pair ${{t.pair_idx}} · corr <span class="${{cls(t.corr)}}">${{t.corr.toFixed(2)}}</span> · rising ${{(100*t.rising).toFixed(0)}}%</div>`;
    d.onclick = () => {{ selIdx = i; renderDetail(); renderGrid(); window.scrollTo({{top:0, behavior:'smooth'}}); }};
    g.appendChild(d);
  }});
}}
function renderDetail() {{
  const t = DATA[selIdx];
  const det = document.getElementById('detail');
  det.style.display = 'block';
  document.getElementById('dTitle').textContent = `${{t.placement}} · pair ${{t.pair_idx}} · corr ${{t.corr.toFixed(3)}}`;
  const sl = document.getElementById('dSlider');
  sl.max = t.curve.length - 1;
  if (+sl.value > +sl.max) sl.value = 0;
  sl.oninput = () => updateFrame(t);
  document.getElementById('dGoal').src = t.imgs[t.imgs.length - 1];
  updateFrame(t);
}}
function updateFrame(t) {{
  const k = +document.getElementById('dSlider').value;
  document.getElementById('dImg').src = t.imgs[k];
  document.getElementById('dIdx').textContent = t.frame_idxs[k];
  document.getElementById('dVal').innerHTML = `<span class="bigval">${{t.curve[k].toFixed(3)}}</span> &nbsp;V(frame ${{t.frame_idxs[k]}}, goal)`;
  drawCurve(t, k);
}}
function drawCurve(t, sel) {{
  const svg = document.getElementById('dCurve');
  const W = 640, H = 260, padL = 46, padB = 22, padT = 8, padR = 8;
  const all = t.curve.concat(t.t_view, t.t_size, t.t_aim);
  const vmin = Math.min(...all), vmax = Math.max(...all);
  const x = i => padL + (W-padL-padR) * i / (t.curve.length-1);
  const y = v => padT + (H-padT-padB) * (vmax - v) / Math.max(1e-9, vmax - vmin);
  let s = `<line x1="${{padL}}" y1="${{H-padB}}" x2="${{W-padR}}" y2="${{H-padB}}" stroke="#3a4150"/>`;
  s += `<text x="4" y="${{y(vmax)+4}}" fill="#8a93a3" font-size="11">${{vmax.toFixed(2)}}</text>`;
  s += `<text x="4" y="${{y(vmin)+4}}" fill="#8a93a3" font-size="11">${{vmin.toFixed(2)}}</text>`;
  const term = (arr, col) => '<polyline fill="none" stroke="' + col +
      '" stroke-width="1" opacity="0.85" points="' + arr.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ') + '"/>';
  s += term(t.t_view, '#5aa9e6') + term(t.t_size, '#c792ea') + term(t.t_aim, '#e6c07b');
  s += '<polyline fill="none" stroke="#7ee29a" stroke-width="1.8" points="' +
       t.curve.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ') + '"/>';
  t.curve.forEach((v,i) => {{
    s += `<circle cx="${{x(i)}}" cy="${{y(v)}}" r="${{i===sel?5:3}}" fill="${{i===sel?'#ffb86b':'#7ee29a'}}" data-i="${{i}}"/>`;
    if (t.zero[i]) {{
      s += `<text x="${{x(i)}}" y="${{y(v)-8}}" fill="#e6675a" font-size="13" text-anchor="middle" pointer-events="none">✕</text>`;
    }}
  }});
  svg.innerHTML = s;
  svg.onclick = e => {{
    const c = e.target.closest('circle');
    if (c) {{ document.getElementById('dSlider').value = c.dataset.i; updateFrame(t); }}
  }};
}}
document.getElementById('sortSel').onchange = renderGrid;
renderGrid();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    if not payload:
        raise SystemExit(f"no scored trajectories under {args.src}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_PAGE.format(data_json=json.dumps(payload)), encoding="utf-8")
    corrs = [t["corr"] for t in payload]
    print(f"wrote {args.out}: {len(payload)} trajectories, "
          f"median corr={sorted(corrs)[len(corrs)//2]:.3f}, "
          f"size={args.out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
