"""Interactive HTML rollout viz: goal -> (action, predicted next-frame, value) vs GT.

For a FIXED state (a placement's start frame) the policy is sampled under several
goals — the multiscale endpoints, each labelled by what it asks for (e.g. "dolly-in,
occupancy 40%->86%"). Per goal we show, next to the ground truth:

    current frame  ·  goal-conditioned action  ·  predicted next world-frame  ·  value

Everything is baked into a single self-contained interactive HTML (base64 images,
inline JS — no external assets), so it opens anywhere / publishes as an Artifact.

  PYTHONPATH=. python scripts/viz_rollout_html.py \
      --checkpoint runs/<ts>_cosmos_2b_v8/ckpt_last.pt \
      --data-root data/trajectories/<placement> --out /tmp/rollout.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_eval import load_policy

from src.policy.common.annotations import iter_multiscale_windows
from src.policy.common.dataset_base import _compute_action_chunk, _compute_value_sequence
from src.policy.common.goal_space import goal_vector, normalize_goal
from src.policy.cosmos.dataset import _load_image_as_tensor

PROFILE_FMT = [
    ("occupancy", "%", "{:.0f}"), ("body_in_frame", "%", "{:.0f}"),
    ("azimuth", "°", "{:.0f}"), ("elevation", "°", "{:+.0f}"),
    ("center_x", "px", "{:.0f}"), ("center_y", "px", "{:.0f}"),
    ("bbox_x", "px", "{:.0f}"), ("bbox_y", "px", "{:.0f}"),
]
ADIMS = ["Δright", "Δup", "Δfwd", "Δyaw", "Δpitch"]


def _b64(t: torch.Tensor) -> str:
    """(3,H,W) in [-1,1] -> base64 PNG data URI."""
    from PIL import Image
    arr = ((t.detach().float().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path, help="placement dir (holds data.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--offsets", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--seeds", type=int, default=4, help="noise draws averaged for the action (goal effect)")
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--negative-mode", choices=["flip", "null"], default="flip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    res = tuple(args.resolution)
    dj = args.data_root / "data.json"

    # forward (dolly-in) + reverse (dolly-out) windows for this exact (pair, start)
    wins = [w for w in iter_multiscale_windows(dj, chunk_size=chunk_size, offsets=tuple(args.offsets))
            if w.pair_idx == args.pair and w.start_frame_idx == args.start_frame]
    wins.sort(key=lambda w: w.end_frame_idx - w.start_frame_idx)
    if not wins:
        raise SystemExit(f"no windows for pair={args.pair} start={args.start_frame}")

    state_rec = wins[0].keyframes[0]
    state = _load_image_as_tensor(Path(state_rec.image), res).unsqueeze(0).to(device, dtype=dtype)
    state_occ = float(goal_vector(state_rec.raw, keys)[0])
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        image_latent = vae.encode_pair_frames(state, state)
    t_img = image_latent.shape[2]

    goals = []
    for w in wins:
        offset = w.end_frame_idx - w.start_frame_idx
        goal_raw = goal_vector(w.end.raw, keys)
        gt = torch.from_numpy(normalize_goal(goal_raw, keys)).unsqueeze(0).to(device, dtype=dtype)
        gt_action = _compute_action_chunk(w)
        gt_value = _compute_value_sequence(w, w.end, "cost_to_go", keys)
        gt_world = _load_image_as_tensor(Path(w.end.image), res)

        pred_actions, world0, val0 = [], None, None
        for s in range(args.seeds):
            torch.manual_seed(3000 + s)
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = policy.sample(image_latent=image_latent, goal_vec=gt, n_steps=args.n_steps,
                                    guidance_scale=args.guidance_scale, negative_mode=args.negative_mode)
            pred_actions.append(out.pred_action_chunk.squeeze(0).float().cpu().numpy())
            if s == 0:
                val0 = out.pred_value.squeeze(0).float().cpu().numpy() if out.pred_value is not None else None
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    world0 = vae.decode(out.pred_latents[:, :, t_img - 1:t_img])[:, :, 0][0]
        pred_action = np.mean(pred_actions, axis=0)

        goal_occ = float(goal_raw[0])
        direction = "dolly-in" if offset > 0 else "dolly-out"
        goals.append(dict(
            offset=int(offset), end=int(w.end_frame_idx), direction=direction,
            occ_state=round(state_occ), occ_goal=round(goal_occ),
            profile=[fmt.format(v) + u for (lbl, u, fmt), v in zip(PROFILE_FMT, goal_raw)],
            profile_labels=[lbl for lbl, _, _ in PROFILE_FMT],
            action_pred=[[round(float(x), 3) for x in step] for step in pred_action],
            action_gt=[[round(float(x), 3) for x in step] for step in gt_action],
            value_pred=[round(float(x), 3) for x in np.ravel(val0)] if val0 is not None else None,
            value_gt=[round(float(x), 3) for x in np.ravel(gt_value)],
            world_pred=_b64(world0), world_gt=_b64(gt_world),
        ))

    payload = dict(
        iteration=int(iteration), placement=args.data_root.name, pair=args.pair,
        start_frame=args.start_frame, adims=ADIMS,
        guidance=args.guidance_scale, negative=args.negative_mode,
        state_img=_b64(state[0]), goals=goals,
    )
    args.out.write_text(_render_html(payload))
    print(f"wrote {args.out}  ({len(goals)} goals, iter {iteration})")


def _render_html(p: dict) -> str:
    data = json.dumps(p)
    return _HTML.replace("/*__DATA__*/", data)


_HTML = r"""
<style>
  :root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f6f7f9;--line:#e3e6ea;--acc:#2563eb;--pred:#2563eb;--gt:#16a34a}
  @media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--mut:#9aa0a6;--card:#181b20;--line:#2a2e35;--acc:#60a5fa;--pred:#60a5fa;--gt:#4ade80}}
  :root[data-theme=dark]{--bg:#0f1115;--fg:#e8eaed;--mut:#9aa0a6;--card:#181b20;--line:#2a2e35;--acc:#60a5fa;--pred:#60a5fa;--gt:#4ade80}
  :root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f6f7f9;--line:#e3e6ea;--acc:#2563eb;--pred:#2563eb;--gt:#16a34a}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1200px;margin:0 auto;padding:20px}
  h1{font-size:19px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
  .state{display:flex;gap:16px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:16px}
  .state img{width:220px;border-radius:8px;border:1px solid var(--line)}
  .layout{display:grid;grid-template-columns:230px 1fr;gap:16px}
  .goals{display:flex;flex-direction:column;gap:8px}
  .gbtn{text-align:left;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer;color:var(--fg);font:inherit}
  .gbtn:hover{border-color:var(--acc)}.gbtn.on{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
  .gbtn .t{font-weight:600}.gbtn .d{color:var(--mut);font-size:12px;margin-top:2px}
  .pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;background:var(--line);color:var(--mut)}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
  .frames{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px}
  .frames figure{margin:0}.frames img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
  .frames figcaption{font-size:12px;color:var(--mut);margin-top:5px;text-align:center}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  h3{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12.5px}
  th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left;color:var(--mut)}
  .pred{color:var(--pred)}.gt{color:var(--gt)}
  .bars{margin-top:10px}.bar{display:grid;grid-template-columns:52px 1fr;gap:8px;align-items:center;margin:5px 0}
  .bar .lab{font-size:12px;color:var(--mut)}.track{position:relative;height:16px;background:var(--line);border-radius:8px}
  .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--mut);opacity:.5}
  .fill{position:absolute;top:2px;height:12px;border-radius:6px;opacity:.85}
  .prof{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
  .prof span{font-size:11.5px;background:var(--line);border-radius:6px;padding:2px 7px;color:var(--fg)}
  .legend{font-size:12px;color:var(--mut);margin:2px 0 10px}.legend b.pred{color:var(--pred)}.legend b.gt{color:var(--gt)}
</style>
<div class="wrap">
  <h1>Goal &rarr; action rollout <span class="pill" id="iter"></span></h1>
  <div class="sub" id="meta"></div>
  <div class="state">
    <img id="stateImg" alt="state">
    <div><h3>Current state (fixed)</h3><div id="stateInfo" class="sub" style="margin:0"></div>
    <div class="legend">Pick a goal &rarr; see the action it produces, the predicted next frame, and value &mdash; each vs <b class="gt">ground truth</b>. Action shown is the mean over noise seeds (the goal effect). <b class="pred">pred</b> / <b class="gt">GT</b>.</div></div>
  </div>
  <div class="layout">
    <div class="goals" id="goals"></div>
    <div class="panel" id="panel"></div>
  </div>
</div>
<script>
const P = /*__DATA__*/;
document.getElementById('iter').textContent = 'iter ' + P.iteration;
document.getElementById('meta').textContent = P.placement + '  ·  pair ' + P.pair + ', start frame ' + P.start_frame +
   (P.guidance!==1 ? '  ·  CFG s='+P.guidance+' ('+P.negative+')' : '');
document.getElementById('stateImg').src = P.state_img;
document.getElementById('stateInfo').textContent = 'occupancy ' + (P.goals[0]?P.goals[0].occ_state:'?') + '%  ·  ' + P.goals.length + ' goals';

function num(x){return (x>=0?'+':'')+x.toFixed(3);}
function bar(lab, pred, gt, scale){
  const w=v=>Math.max(2,Math.min(50,Math.abs(v)/scale*50)); const off=v=>v>=0?50:50-w(v);
  return `<div class="bar"><div class="lab">${lab}</div><div class="track"><div class="mid"></div>`+
    `<div class="fill gt"  style="left:${off(gt)}%;width:${w(gt)}%;background:var(--gt)"   title="GT ${num(gt)}"></div>`+
    `<div class="fill pred" style="left:${off(pred)}%;width:${w(pred)}%;background:var(--pred);opacity:.7" title="pred ${num(pred)}"></div>`+
    `</div></div>`;
}
function chunkTable(pred, gt){
  let h='<table><tr><th>step</th>'+P.adims.map(d=>`<th>${d}</th>`).join('')+'</tr>';
  for(let i=0;i<pred.length;i++){
    h+=`<tr><td>${i} pred</td>`+pred[i].map(x=>`<td class="pred">${num(x)}</td>`).join('')+'</tr>';
    h+=`<tr><td>${i} gt</td>`+gt[i].map(x=>`<td class="gt">${num(x)}</td>`).join('')+'</tr>';
  }
  return h+'</table>';
}
function valTable(pred, gt){
  if(!pred) return '<div class="sub">no value head</div>';
  let h='<table><tr><th>step</th>';for(let i=0;i<gt.length;i++)h+=`<th>${i}</th>`;h+='</tr>';
  h+='<tr><td>pred</td>'+pred.map(x=>`<td class="pred">${num(x)}</td>`).join('')+'</tr>';
  h+='<tr><td>gt</td>'+gt.map(x=>`<td class="gt">${num(x)}</td>`).join('')+'</tr>';
  return h+'</table>';
}
function render(i){
  const g=P.goals[i];
  document.querySelectorAll('.gbtn').forEach((b,j)=>b.classList.toggle('on',j===i));
  const amax=Math.max(0.2,...g.action_gt[0].map(Math.abs),...g.action_pred[0].map(Math.abs));
  const prof=g.profile.map((v,k)=>`<span>${g.profile_labels[k]} ${v}</span>`).join('');
  document.getElementById('panel').innerHTML =
    `<div class="frames">
       <figure><img src="${P.state_img}"><figcaption>current</figcaption></figure>
       <figure><img src="${g.world_pred}"><figcaption class="pred">predicted next</figcaption></figure>
       <figure><img src="${g.world_gt}"><figcaption class="gt">GT next (frame ${g.end})</figcaption></figure>
     </div>
     <div class="prof">${prof}</div>
     <div class="cols" style="margin-top:14px">
       <div><h3>Action[0] &mdash; the immediate move</h3><div class="bars">${g.adims_bars=P.adims.map((d,k)=>bar(d,g.action_pred[0][k],g.action_gt[0][k],amax)).join('')}</div>
         <h3 style="margin-top:14px">Full ${g.action_pred.length}-step chunk</h3>${chunkTable(g.action_pred,g.action_gt)}</div>
       <div><h3>Value (cost-to-go)</h3>${valTable(g.value_pred,g.value_gt)}</div>
     </div>`;
}
const gc=document.getElementById('goals');
P.goals.forEach((g,i)=>{
  const arrow=g.occ_goal>g.occ_state?'&uarr;':(g.occ_goal<g.occ_state?'&darr;':'&rarr;');
  const b=document.createElement('button');b.className='gbtn';
  b.innerHTML=`<div class="t">${g.direction} ${g.offset>0?'+':''}${g.offset}</div>`+
    `<div class="d">occupancy ${g.occ_state}% ${arrow} ${g.occ_goal}%</div>`;
  b.onclick=()=>render(i);gc.appendChild(b);
});
render(0);
</script>
"""


if __name__ == "__main__":
    main()
