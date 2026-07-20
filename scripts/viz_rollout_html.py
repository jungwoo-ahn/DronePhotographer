"""Interactive HTML rollout viz: goal -> (action, predicted next-frame, value) vs GT.

For a FIXED state (a placement's start frame) the policy is sampled under several
goals — the multiscale endpoints, each labelled by what it asks for (e.g. "dolly-in,
occupancy 40%->86%"). Per goal we show, next to the ground truth:

    current frame  ·  goal-conditioned action  ·  predicted next world-frame  ·  value

Everything is baked into a single self-contained interactive HTML (base64 images,
inline JS — no external assets), with a "how to read this" panel and hover tooltips
on every label, so it opens anywhere / publishes as an Artifact.

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
# Blender action-executed render (optional --blender path)
from src.policy.common.blender_env import BlenderRolloutEnv, PersistentBlenderRenderer
from src.policy.common.validation_sample import load_validation_sample

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


def _b64_pil(img, size_wh) -> str:
    """PIL RGB image -> base64 PNG data URI, resized to (W,H) to match the other columns."""
    buf = io.BytesIO()
    img.resize(size_wh).save(buf, format="PNG")
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
    # Blender: render what the predicted action ACTUALLY produces (4th column).
    ap.add_argument("--blender", action="store_true", help="render the predicted-action-executed frame in Blender")
    ap.add_argument("--blender-bin", default="blender/blender")
    ap.add_argument("--render-device", default="CPU", choices=["CPU", "GPU"],
                    help="CPU avoids the ~10-15min sm_100 Cycles kernel compile; GPU is faster once warm")
    ap.add_argument("--render-samples", type=int, default=16)
    ap.add_argument("--render-timeout", type=float, default=1200.0)
    ap.add_argument("--vlm-v6-dir", default="data/vlm_object_placing_v6_260428_061326")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    res = tuple(args.resolution)
    dj = args.data_root / "data.json"

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

    env = None
    if args.blender:
        # Blender needs libGL / libXfixes / libxkbcommon, which login + bare workers
        # lack; ship them from blender/syslibs/lib (same as sbatch_rollout_eval.sh) so
        # the spawned Blender subprocess inherits them — no external env var needed.
        import os
        syslibs = str(Path(args.blender_bin).resolve().parent / "syslibs" / "lib")
        os.environ["LD_LIBRARY_PATH"] = syslibs + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        sample = load_validation_sample(dj, args.vlm_v6_dir, require_vlm=True)
        sample.render_samples = args.render_samples
        sample.render_device = args.render_device
        renderer = PersistentBlenderRenderer(blender_bin=args.blender_bin, render_timeout_s=args.render_timeout)
        env = BlenderRolloutEnv.from_validation_sample(sample, renderer)
        print(f"[blender] env ready ({args.render_device}, {args.render_samples} samples) — "
              "first render compiles Cycles kernels, be patient", flush=True)

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

        exec_render = None
        if env is not None:
            # execute the predicted chunk from the current pose, render the endpoint
            env.reset(state_rec.camera_position, state_rec.camera_forward, state_rec.camera_up, render=False)
            obs = None
            for i, a in enumerate(pred_action):
                obs, _ = env.step(a, render=(i == len(pred_action) - 1))
            exec_render = _b64_pil(obs["image"], (res[1], res[0]))
            print(f"[blender] rendered action-executed endpoint for goal +{int(offset)}", flush=True)

        goal_occ = float(goal_raw[0])
        goals.append(dict(
            offset=int(offset), end=int(w.end_frame_idx),
            direction="dolly-in" if offset > 0 else "dolly-out",
            exec_render=exec_render,
            occ_state=round(state_occ), occ_goal=round(goal_occ),
            profile=[fmt.format(v) + u for (lbl, u, fmt), v in zip(PROFILE_FMT, goal_raw)],
            profile_labels=[lbl for lbl, _, _ in PROFILE_FMT],
            action_pred=[[round(float(x), 3) for x in step] for step in pred_action],
            action_gt=[[round(float(x), 3) for x in step] for step in gt_action],
            value_pred=[round(float(x), 3) for x in np.ravel(val0)] if val0 is not None else None,
            value_gt=[round(float(x), 3) for x in np.ravel(gt_value)],
            world_pred=_b64(world0), world_gt=_b64(gt_world),
        ))

    if env is not None:
        env.close()

    payload = dict(
        iteration=int(iteration), placement=args.data_root.name, pair=args.pair,
        start_frame=args.start_frame, adims=ADIMS, seeds=args.seeds, n_steps=args.n_steps,
        guidance=args.guidance_scale, negative=args.negative_mode, has_exec=bool(args.blender),
        state_img=_b64(state[0]), goals=goals,
    )
    args.out.write_text(_HTML.replace("/*__DATA__*/", json.dumps(payload)))
    print(f"wrote {args.out}  ({len(goals)} goals, iter {iteration})")


_HTML = r"""
<style>
  :root{--bg:#fbfbfc;--fg:#171a1f;--mut:#5c6570;--card:#f2f4f7;--line:#e0e4ea;--acc:#3b6fe0;--pred:#3b6fe0;--gt:#12a150}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1116;--fg:#e6e9ef;--mut:#8b95a3;--card:#171b22;--line:#272d37;--acc:#6f9bff;--pred:#6f9bff;--gt:#48d17e}}
  :root[data-theme=dark]{--bg:#0e1116;--fg:#e6e9ef;--mut:#8b95a3;--card:#171b22;--line:#272d37;--acc:#6f9bff;--pred:#6f9bff;--gt:#48d17e}
  :root[data-theme=light]{--bg:#fbfbfc;--fg:#171a1f;--mut:#5c6570;--card:#f2f4f7;--line:#e0e4ea;--acc:#3b6fe0;--pred:#3b6fe0;--gt:#12a150}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:22px}
  h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
  details{margin-bottom:14px;border:1px solid var(--line);border-radius:10px;background:var(--card);padding:0 14px}
  summary{cursor:pointer;padding:11px 0;font-weight:600;font-size:13px;list-style:none;user-select:none}
  summary::-webkit-details-marker{display:none}summary::before{content:"\24d8   ";color:var(--acc)}
  details[open] summary{border-bottom:1px solid var(--line)}
  details p{margin:10px 0;color:var(--mut);font-size:13px;max-width:72ch}details b{color:var(--fg);font-weight:600}
  .state{display:flex;gap:16px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:16px}
  .state img{width:220px;border-radius:8px;border:1px solid var(--line)}
  .layout{display:grid;grid-template-columns:236px 1fr;gap:16px}
  .goals{display:flex;flex-direction:column;gap:8px}
  .gbtn{text-align:left;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer;color:var(--fg);font:inherit}
  .gbtn:hover{border-color:var(--acc)}.gbtn.on{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
  .gbtn .t{font-weight:600}.gbtn .d{color:var(--mut);font-size:12px;margin-top:2px}
  .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;background:var(--line);color:var(--mut);font-weight:500}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
  .frames{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px}
  .frames figure{margin:0}.frames img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
  .frames figcaption{font-size:12px;color:var(--mut);margin-top:5px;text-align:center}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  h3{font-size:12px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12.5px;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left;color:var(--mut);font-family:-apple-system,sans-serif}
  th{color:var(--mut);font-weight:600}.pred{color:var(--pred)}.gt{color:var(--gt)}
  .bars{margin-top:8px}.bar{display:grid;grid-template-columns:58px 1fr;gap:8px;align-items:center;margin:5px 0}
  .bar .lab{font-size:12px;color:var(--mut)}.track{position:relative;height:16px;background:var(--line);border-radius:8px}
  .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--mut);opacity:.5}
  .fill{position:absolute;top:2px;height:12px;border-radius:6px}
  .prof{display:flex;flex-wrap:wrap;gap:6px;margin-top:2px}
  .prof span{font-size:11.5px;background:var(--line);border-radius:6px;padding:2px 8px;color:var(--fg)}
  .legend{font-size:12.5px;color:var(--mut);margin:2px 0 0;line-height:1.6}.legend b.pred{color:var(--pred)}.legend b.gt{color:var(--gt)}
  [title]{cursor:help}
</style>
<div class="wrap">
  <h1>Goal &rarr; action rollout <span class="pill" id="iter"></span></h1>
  <div class="sub" id="meta"></div>
  <details>
    <summary>How to read this</summary>
    <p>A <b>fixed current frame</b> is shown to the policy under several <b>goals</b> &mdash; target shot-profiles taken from points further along the camera path. For each goal the policy predicts, in a single forward pass: the <b>camera action</b> to reach it, the <b>next frame</b> it imagines after that action, and the <b>value</b> &mdash; each shown beside its <b class="gt">ground truth</b>.</p>
    <p>The <b class="pred">predicted next frame</b> is <b>diffusion-sampled</b>, not looked up: the policy denoises the next-frame / action / value latents from pure noise over 16 flow-matching steps, conditioned on the goal and the clean pinned current frame, then VAE-decodes the frame latent to pixels (single noise seed). The <b>action</b> shown is the mean over 4 seeds, which cancels sampler noise and isolates the goal's effect. When the <b>action executed</b> column is present it is a Blender render of the pose reached by <i>actually running</i> the predicted action &mdash; the real consequence: predicted-next &asymp; action-executed means the world head imagined correctly, and action-executed &asymp; GT-next means the action itself is good.</p>
    <p><b>Action</b> &mdash; per-step camera move in the camera's own axes: &Delta;right / &Delta;up / &Delta;fwd in metres, &Delta;yaw / &Delta;pitch in radians; <b>action[0]</b> is the immediate move. <b>Value</b> &mdash; cost-to-go, the normalized negative pose-distance from each step to the goal (higher = closer). Early checkpoints predict frames worst (the world head is downweighted), so read action / value first.</p>
  </details>
  <div class="state">
    <img id="stateImg" alt="current state">
    <div><h3>Current state (fixed)</h3><div id="stateInfo" class="sub" style="margin:0"></div>
    <div class="legend">Pick a goal on the left &rarr; its action, predicted next frame, and value appear on the right. <b class="pred">blue = predicted</b>, <b class="gt">green = ground truth</b>. Hover any label for what it means.</div></div>
  </div>
  <div class="layout">
    <div class="goals" id="goals"></div>
    <div class="panel" id="panel"></div>
  </div>
</div>
<script>
const P = /*__DATA__*/;
const ADESC={"Δright":"camera-local left(-)/right(+) shift, metres","Δup":"camera-local down(-)/up(+) shift, metres","Δfwd":"back(-)/toward-subject(+), metres","Δyaw":"turn left/right, radians","Δpitch":"tilt down/up, radians"};
const KDESC={"occupancy":"% of the frame the subject fills","body_in_frame":"% of the subject's bbox inside the frame (100 = fully framed)","azimuth":"camera->subject horizontal angle, 0-360°","elevation":"camera->subject vertical angle (- = camera above)","center_x":"subject centre X in px (frame 1024 wide; 512 = centred)","center_y":"subject centre Y in px (frame 768 tall; 384 = centred)","bbox_x":"half the subject bbox width, px - apparent-size cue","bbox_y":"half the subject bbox height, px - apparent-size cue"};
document.getElementById('iter').textContent = 'iter ' + P.iteration;
document.getElementById('meta').textContent = P.placement + '  ·  pair ' + P.pair + ', start frame ' + P.start_frame +
   '  ·  ' + P.seeds + ' seeds, ' + P.n_steps + ' sampler steps' + (P.guidance!==1 ? '  ·  CFG s='+P.guidance+' ('+P.negative+')' : '');
document.getElementById('stateImg').src = P.state_img;
document.getElementById('stateImg').title = 'The fixed current frame fed to the policy (a val / unseen subject).';
document.getElementById('stateInfo').textContent = 'occupancy ' + (P.goals[0]?P.goals[0].occ_state:'?') + '%  ·  ' + P.goals.length + ' goals';
function num(x){return (x>=0?'+':'')+x.toFixed(3);}
function bar(lab,pred,gt,scale){
  const w=v=>Math.max(2,Math.min(50,Math.abs(v)/scale*50)); const off=v=>v>=0?50:50-w(v);
  return `<div class="bar" title="${ADESC[lab]||''}"><div class="lab">${lab}</div><div class="track"><div class="mid"></div>`+
    `<div class="fill" style="left:${off(gt)}%;width:${w(gt)}%;background:var(--gt)" title="GT ${num(gt)}"></div>`+
    `<div class="fill" style="left:${off(pred)}%;width:${w(pred)}%;background:var(--pred);opacity:.72" title="pred ${num(pred)}"></div>`+
    `</div></div>`;
}
function chunkTable(pred,gt){
  let h='<table><tr><th>step</th>'+P.adims.map(d=>`<th title="${ADESC[d]||''}">${d}</th>`).join('')+'</tr>';
  for(let i=0;i<pred.length;i++){
    h+=`<tr><td>${i} pred</td>`+pred[i].map(x=>`<td class="pred">${num(x)}</td>`).join('')+'</tr>';
    h+=`<tr><td>${i} gt</td>`+gt[i].map(x=>`<td class="gt">${num(x)}</td>`).join('')+'</tr>';
  }
  return h+'</table>';
}
function valTable(pred,gt){
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
  const prof=g.profile.map((v,k)=>`<span title="${KDESC[g.profile_labels[k]]||''}">${g.profile_labels[k]} ${v}</span>`).join('');
  const barsHtml=P.adims.map((d,k)=>bar(d,g.action_pred[0][k],g.action_gt[0][k],amax)).join('');
  const fr=[`<figure><img src="${P.state_img}" title="Current frame (the policy input)."><figcaption>current</figcaption></figure>`,
    `<figure><img src="${g.world_pred}" title="Diffusion-sampled: denoised from noise over ${P.n_steps} flow-matching steps, conditioned on this goal + the pinned current frame, then VAE-decoded (1 seed). The model's IMAGINED frame after the action."><figcaption class="pred">predicted next &mdash; diffusion</figcaption></figure>`];
  if(g.exec_render) fr.push(`<figure><img src="${g.exec_render}" title="Blender render of the pose reached by actually EXECUTING the predicted 8-step action from the current frame — the real consequence of the action. ≈ GT next ⇒ the action is good; ≈ predicted-next ⇒ the world head imagined it correctly."><figcaption>action executed &mdash; Blender</figcaption></figure>`);
  fr.push(`<figure><img src="${g.world_gt}" title="The actual rendered frame at the goal endpoint (what the GT action achieves)."><figcaption class="gt">GT next (frame ${g.end})</figcaption></figure>`);
  document.getElementById('panel').innerHTML =
    `<div class="frames" style="grid-template-columns:repeat(${fr.length},1fr)">${fr.join('')}</div>
     <div class="prof" title="The goal: the target shot-profile the policy is conditioned on. Hover each key.">${prof}</div>
     <div class="cols" style="margin-top:14px">
       <div><h3 title="The immediate next camera move (step 0), camera-local. Bar: solid green = GT, translucent blue = pred; centre line = 0; length grows with magnitude.">Action[0] &mdash; the immediate move</h3><div class="bars">${barsHtml}</div>
         <h3 style="margin-top:14px" title="All ${g.action_pred.length} predicted steps vs GT (metres / radians, camera-local).">Full ${g.action_pred.length}-step chunk</h3>${chunkTable(g.action_pred,g.action_gt)}</div>
       <div><h3 title="Predicted cost-to-go per step: normalized -(pose distance) from that step to the goal. Higher (less negative) = closer to the goal.">Value (cost-to-go)</h3>${valTable(g.value_pred,g.value_gt)}</div>
     </div>`;
}
const gc=document.getElementById('goals');
P.goals.forEach((g,i)=>{
  const arrow=g.occ_goal>g.occ_state?'↑':(g.occ_goal<g.occ_state?'↓':'→');
  const b=document.createElement('button');b.className='gbtn';
  b.title='Goal = the shot-profile achieved '+Math.abs(g.offset)+' keyframes '+(g.offset>0?'ahead (dolly-in)':'back (dolly-out)')+' on the camera path.';
  b.innerHTML=`<div class="t">${g.direction} ${g.offset>0?'+':''}${g.offset}</div><div class="d">occupancy ${g.occ_state}% ${arrow} ${g.occ_goal}%</div>`;
  b.onclick=()=>render(i);gc.appendChild(b);
});
render(0);
</script>
"""


if __name__ == "__main__":
    main()
