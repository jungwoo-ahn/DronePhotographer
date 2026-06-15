"""Render an interactive HTML report of dataset samples (images, goal, value, actions).

For each (placement, accepted_pair) the report shows:
  - the value curve  V(start_idx) = -geometric_profile_distance(start, end-of-window)
    over all windows of that trajectory (click a point to jump to that window),
  - a slider to scrub windows: start frame vs goal (end) frame side by side,
  - the 8 V5 scores of both frames,
  - the normalized action chunk as signed bars (chunk_size x 5).

Self-contained: thumbnails are base64-embedded, no server needed — scp the file
and open it in a browser.

Usage:
  python scripts/visualize_value_html.py \
      --roots data/trajectories \
      --out outputs/value_viz.html \
      [--chunk-size 8] [--stride 2] [--max-pairs 12] [--thumb-width 320]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from src.policy.common.dataset_base import BasePolicyDataset
from src.policy.common.goal_space import goal_keys

V5_KEYS = goal_keys(None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=["data/trajectories"], type=Path)
    p.add_argument("--out", default=Path("outputs/value_viz.html"), type=Path)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=12, help="cap on (placement, pair) groups")
    p.add_argument("--thumb-width", type=int, default=320)
    return p.parse_args()


def _thumb_b64(image_path: str, width: int, cache: dict[str, str]) -> str:
    if image_path in cache:
        return cache[image_path]
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    h = round(img.height * width / img.width)
    img = img.resize((width, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    cache[image_path] = b64
    return b64


def build_payload(args: argparse.Namespace) -> list[dict]:
    # Pin goal_sampling="end" — this viz plots the per-window value against the
    # window's own end frame, so the stochastic HER-future goal would scramble it.
    ds = BasePolicyDataset(args.roots, chunk_size=args.chunk_size, stride=args.stride, goal_sampling="end")
    groups: dict[tuple, list] = defaultdict(list)
    for i in range(len(ds)):
        s = ds[i]
        groups[(str(s.start.annotation_path), s.start.pair_idx)].append(s)

    from src.policy.common.reward import CameraIntrinsics, profile_distance_value

    cache: dict[str, str] = {}
    payload = []
    for (ann, pair_idx), samples in sorted(groups.items())[: args.max_pairs]:
        samples.sort(key=lambda s: s.start.frame_idx)
        placement = Path(ann).parent.name
        # Second slice for the curve: value against a FIXED goal (the last window's
        # end profile). This one should rise toward 0 along the trajectory; the
        # per-window (training-target) value need not — its goal moves with the
        # window, so it measures local geometric speed instead.
        fixed_goal = samples[-1].end.raw
        intr = CameraIntrinsics.from_render(samples[0].start.render_width, samples[0].start.render_height)
        windows = []
        for s in samples:
            windows.append({
                "start_idx": s.start.frame_idx,
                "end_idx": s.end.frame_idx,
                "value": round(float(s.value), 4),
                "value_fixed": round(float(profile_distance_value(s.start.raw, fixed_goal, intr)), 4),
                "start_img": _thumb_b64(s.start.image, args.thumb_width, cache),
                "end_img": _thumb_b64(s.end.image, args.thumb_width, cache),
                "start_scores": {k: s.start.raw.get(k) for k in V5_KEYS},
                "end_scores": {k: s.end.raw.get(k) for k in V5_KEYS},
                # raw (un-normalized) action chunk; the viewer normalizes for bars
                "action_chunk": [[round(float(v), 4) for v in row] for row in s.action_chunk],
            })
        payload.append({
            "placement": placement,
            "pair_idx": pair_idx,
            "scene": samples[0].start.scene,
            "object": samples[0].start.object,
            "windows": windows,
        })
    return payload


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Cosmos policy — sample values</title>
<style>
  body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; background: #14161a; color: #e6e6e6; }}
  h1 {{ font-size: 19px; }} h2 {{ font-size: 15px; margin: 6px 0; color: #9fd3ff; }}
  select, input[type=range] {{ font: inherit; }}
  select {{ background:#222; color:#eee; border:1px solid #444; padding:4px 8px; border-radius:6px; max-width: 95vw; }}
  .row {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 14px; }}
  .card {{ background: #1d2026; border: 1px solid #2c313a; border-radius: 10px; padding: 14px; }}
  .frame img {{ border-radius: 6px; display: block; }}
  .frame .cap {{ color: #8a93a3; font-size: 12px; margin-top: 4px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td, th {{ padding: 2px 10px; text-align: right; border-bottom: 1px solid #2c313a; }}
  th {{ color: #8a93a3; font-weight: 500; text-align: left; }}
  .delta {{ color: #ffb86b; }}
  #curve {{ cursor: crosshair; }}
  .bigval {{ font-size: 26px; font-weight: 700; color: #7ee29a; }}
  .bars td div {{ height: 10px; border-radius: 3px; }}
  .pos {{ background:#5aa9e6; margin-left:50%; }}
  .neg {{ background:#e6675a; margin-right:50%; float:right; }}
  .barcell {{ width: 120px; position: relative; background:#262b33; border-radius:3px; }}
</style></head><body>
<h1>Cosmos policy training samples — value &amp; action explorer</h1>
<div>
  trajectory:&nbsp;<select id="pairSel"></select>
  &nbsp;&nbsp;window:&nbsp;<input type="range" id="winSel" min="0" value="0" style="width:340px">
  <span id="winLabel"></span>
</div>
<div class="row">
  <div class="card"><h2>value curves over this trajectory</h2><svg id="curve" width="560" height="180"></svg>
    <div class="cap" style="color:#8a93a3;font-size:12px">
      <span style="color:#7ee29a">●</span> training target: goal = each window's own end → measures local geometric speed, need not rise.<br>
      <span style="color:#5aa9e6">●</span> fixed goal (trajectory end) → rises toward 0 as the camera approaches. Click to jump.</div></div>
  <div class="card"><h2>value of selected window</h2><div class="bigval" id="valBig"></div>
    <div id="valNote" style="color:#8a93a3;font-size:12px;max-width:260px"></div></div>
</div>
<div class="row">
  <div class="card frame"><h2>start frame</h2><img id="imgA"><div class="cap" id="capA"></div></div>
  <div class="card frame"><h2>goal frame (window end)</h2><img id="imgB"><div class="cap" id="capB"></div></div>
  <div class="card"><h2>V5 profile</h2><table id="scoreTbl"></table></div>
  <div class="card"><h2>action chunk (raw, m / rad)</h2><table id="actTbl" class="bars"></table></div>
</div>
<script>
const DATA = {data_json};
const pairSel = document.getElementById('pairSel');
const winSel  = document.getElementById('winSel');
DATA.forEach((g, i) => {{
  const o = document.createElement('option');
  o.value = i; o.textContent = `${{g.placement}}  ·  pair ${{g.pair_idx}}  (${{g.windows.length}} windows)`;
  pairSel.appendChild(o);
}});
function fmt(v) {{ return (v === null || v === undefined) ? '—' : v; }}
function render() {{
  const g = DATA[+pairSel.value];
  winSel.max = g.windows.length - 1;
  if (+winSel.value > +winSel.max) winSel.value = 0;
  const w = g.windows[+winSel.value];
  document.getElementById('winLabel').textContent = ` frames ${{w.start_idx}} → ${{w.end_idx}}`;
  document.getElementById('imgA').src = w.start_img;
  document.getElementById('imgB').src = w.end_img;
  document.getElementById('capA').textContent = `frame ${{w.start_idx}}`;
  document.getElementById('capB').textContent = `frame ${{w.end_idx}} — its profile is the goal`;
  document.getElementById('valBig').textContent = w.value.toFixed(3);
  document.getElementById('valNote').textContent = '0 = start already matches the goal framing; more negative = larger camera-subject move for the chunk to close.';
  // scores table
  let t = '<tr><th>key</th><td>start</td><td>goal</td><td class="delta">Δ</td></tr>';
  for (const k of Object.keys(w.start_scores)) {{
    const a = w.start_scores[k], b = w.end_scores[k];
    const d = (a !== null && b !== null) ? (b - a) : null;
    t += `<tr><th>${{k}}</th><td>${{fmt(a)}}</td><td>${{fmt(b)}}</td><td class="delta">${{d===null?'—':(d>0?'+':'')+d}}</td></tr>`;
  }}
  document.getElementById('scoreTbl').innerHTML = t;
  // action chunk bars (scaled per-dim by max |v| in this chunk)
  const dims = ['dx','dy','dz','dyaw','dpitch'];
  const maxAbs = dims.map((_,j) => Math.max(1e-9, ...w.action_chunk.map(r => Math.abs(r[j]))));
  let a = '<tr><th>step</th>' + dims.map(d=>`<th>${{d}}</th>`).join('') + '</tr>';
  w.action_chunk.forEach((r, s) => {{
    a += `<tr><th>${{s}}</th>` + r.map((v,j) => {{
      const pct = Math.min(50, 50*Math.abs(v)/maxAbs[j]);
      const bar = v >= 0 ? `<div class="pos" style="width:${{pct}}%"></div>` : `<div class="neg" style="width:${{pct}}%"></div>`;
      return `<td title="${{v}}"><div class="barcell">${{bar}}</div><span style="font-size:10.5px;color:#8a93a3">${{v.toFixed(3)}}</span></td>`;
    }}).join('') + '</tr>';
  }});
  document.getElementById('actTbl').innerHTML = a;
  drawCurve(g, +winSel.value);
}}
function drawCurve(g, sel) {{
  const svg = document.getElementById('curve');
  const W = 560, H = 180, padL = 42, padB = 24, padT = 8, padR = 8;
  const vals = g.windows.map(w => w.value);
  const fixed = g.windows.map(w => w.value_fixed);
  const xs = g.windows.map(w => w.start_idx);
  const all = vals.concat(fixed);
  const vmin = Math.min(...all), vmax = Math.max(...all);
  const x = i => padL + (W-padL-padR) * (xs[i]-xs[0]) / Math.max(1e-9, xs[xs.length-1]-xs[0]);
  const y = v => padT + (H-padT-padB) * (vmax - v) / Math.max(1e-9, vmax - vmin);
  let s = `<line x1="${{padL}}" y1="${{H-padB}}" x2="${{W-padR}}" y2="${{H-padB}}" stroke="#3a4150"/>`;
  s += `<text x="4" y="${{y(vmax)+4}}" fill="#8a93a3" font-size="11">${{vmax.toFixed(2)}}</text>`;
  s += `<text x="4" y="${{y(vmin)+4}}" fill="#8a93a3" font-size="11">${{vmin.toFixed(2)}}</text>`;
  s += '<polyline fill="none" stroke="#5aa9e6" stroke-width="1.5" stroke-dasharray="4 3" points="' +
       fixed.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ') + '"/>';
  s += '<polyline fill="none" stroke="#7ee29a" stroke-width="1.5" points="' +
       vals.map((v,i)=>`${{x(i)}},${{y(v)}}`).join(' ') + '"/>';
  fixed.forEach((v,i) => {{
    s += `<circle cx="${{x(i)}}" cy="${{y(v)}}" r="2.5" fill="#5aa9e6" data-i="${{i}}"/>`;
  }});
  vals.forEach((v,i) => {{
    s += `<circle cx="${{x(i)}}" cy="${{y(v)}}" r="${{i===sel?5:3}}" fill="${{i===sel?'#ffb86b':'#7ee29a'}}" data-i="${{i}}"/>`;
    s += `<text x="${{x(i)}}" y="${{H-8}}" fill="#8a93a3" font-size="10" text-anchor="middle">${{xs[i]}}</text>`;
  }});
  svg.innerHTML = s;
  svg.onclick = e => {{
    const t = e.target.closest('circle');
    if (t) {{ winSel.value = t.dataset.i; render(); }}
  }};
}}
pairSel.onchange = () => {{ winSel.value = 0; render(); }};
winSel.oninput = render;
render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    if not payload:
        raise SystemExit("no samples found under the given roots")
    page = _PAGE.format(data_json=json.dumps(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    n_win = sum(len(g["windows"]) for g in payload)
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(payload)} trajectories, {n_win} windows, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
