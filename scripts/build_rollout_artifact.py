"""Assemble the DP-rollout viewer HTML: embed each GIF as base64 + goal-dist sparkline data."""
import base64, json
from pathlib import Path

V = Path("/tmp/dp_valsim")
OUT = Path("/home/nas5/jooyeolyun/repos/DronePhotographer/outputs/dp_rollout_gifs/rollouts.html")

CARDS = [
    dict(id="nature", scene="Nature-Snowy Mountain Retreat",
         gif="dp_nature_snowy_reached.gif",
         dir="Nature-Snowy-Mountain-Village-Retreat_f884ef3e__All-People-Are-Sisters_1795d425"),
    dict(id="basement", scene="Basement",
         gif="dp_basement_reached.gif",
         dir="basement_8f9ffd5b-654b-4efe-9f09-7df7e49d2ab8__All-People-Are-Sisters_1795d425"),
    dict(id="parking", scene="Parking",
         gif="dp_parking.gif",
         dir="Parking_70ae1f27__All-People-Are-Sisters_1795d425"),
]
THRESH = 0.12
GDIR = OUT.parent

data = {}
for c in CARDS:
    rj = json.loads((V / c["dir"] / "rollout.json").read_text())
    traj = [round(t["goal_dist"], 3) for t in rj["trajectory"]]
    b64 = base64.b64encode((GDIR / c["gif"]).read_bytes()).decode()
    data[c["id"]] = dict(scene=c["scene"], gif=b64, traj=traj,
                         reason=rj["reason"], steps=rj["n_steps"],
                         start=traj[0], final=traj[-1], mn=min(traj),
                         reached=(rj["reason"] == "goal_reached"))

def card_html(cid, d):
    reached = d["reached"]
    pill = ("reached", "s-good") if reached else ("shoot · partial", "s-warn")
    return f"""
    <article class="card">
      <div class="film"><img alt="DP closed-loop rollout on {d['scene']}"
           src="data:image/gif;base64,{d['gif']}"></div>
      <div class="meta">
        <div class="row1">
          <h2>{d['scene']}</h2>
          <span class="pill {pill[1]}">{pill[0]}</span>
        </div>
        <canvas class="spark" id="sp-{cid}" width="640" height="120"></canvas>
        <dl class="stats">
          <div><dt>start</dt><dd>{d['start']:.3f}</dd></div>
          <div><dt>final</dt><dd class="{'good' if reached else ''}">{d['final']:.3f}</dd></div>
          <div><dt>best</dt><dd>{d['mn']:.3f}</dd></div>
          <div><dt>steps</dt><dd>{d['steps']}</dd></div>
        </dl>
      </div>
    </article>"""

cards = "\n".join(card_html(c["id"], data[c["id"]]) for c in CARDS)
payload = json.dumps({k: {"traj": v["traj"], "reached": v["reached"]} for k, v in data.items()})

HTML = f"""<title>DP Rollout Telemetry</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{
  --bg:#eef1f4; --surface:#ffffff; --ink:#182029; --muted:#5c6773;
  --line:#dbe1e8; --accent:#0f8f8c; --good:#1f9d57; --warn:#c47f1d;
  --thresh:#9aa6b2;
}}
@media (prefers-color-scheme:dark){{ :root:not([data-theme="light"]){{
  --bg:#0e1319; --surface:#161d26; --ink:#e7edf3; --muted:#8a97a6;
  --line:#26303b; --accent:#19b8b4; --good:#2fae66; --warn:#e0952a; --thresh:#4a5766;
}} }}
:root[data-theme="dark"]{{
  --bg:#0e1319; --surface:#161d26; --ink:#e7edf3; --muted:#8a97a6;
  --line:#26303b; --accent:#19b8b4; --good:#2fae66; --warn:#e0952a; --thresh:#4a5766;
}}
*{{box-sizing:border-box}}
body{{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.5; -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:1080px; margin:0 auto; padding:clamp(24px,5vw,56px) clamp(16px,4vw,32px)}}
.eyebrow{{
  font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent); margin:0 0 14px;
}}
h1{{font-size:clamp(28px,5vw,42px); line-height:1.05; margin:0 0 14px; letter-spacing:-.02em; text-wrap:balance}}
.thesis{{font-size:clamp(16px,2.2vw,19px); color:var(--muted); max-width:62ch; margin:0 0 28px}}
.thesis b{{color:var(--ink); font-weight:600}}
.bench{{
  display:flex; flex-wrap:wrap; gap:10px; margin:0 0 40px;
  padding-top:22px; border-top:1px solid var(--line);
}}
.stat{{
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; min-width:150px; flex:1;
}}
.stat .k{{font:600 11px/1 ui-monospace,monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted)}}
.stat .v{{font:600 22px/1.2 ui-monospace,monospace; font-variant-numeric:tabular-nums; margin-top:7px}}
.stat .v small{{font-size:13px; color:var(--muted); font-weight:500}}
.stat.win{{border-color:color-mix(in srgb,var(--accent) 55%,var(--line))}}
.stat.win .v{{color:var(--accent)}}
.grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:22px}}
.card{{
  background:var(--surface); border:1px solid var(--line); border-radius:14px;
  overflow:hidden; display:flex; flex-direction:column;
}}
.film{{background:#0b0f14; border-bottom:1px solid var(--line); line-height:0}}
.film img{{width:100%; height:auto; display:block}}
.meta{{padding:16px 18px 18px; display:flex; flex-direction:column; gap:12px}}
.row1{{display:flex; align-items:center; justify-content:space-between; gap:10px}}
.row1 h2{{font-size:16px; margin:0; letter-spacing:-.01em}}
.pill{{
  font:600 11px/1 ui-monospace,monospace; letter-spacing:.06em; text-transform:uppercase;
  padding:5px 9px; border-radius:999px; white-space:nowrap;
}}
.s-good{{color:var(--good); background:color-mix(in srgb,var(--good) 14%,transparent)}}
.s-warn{{color:var(--warn); background:color-mix(in srgb,var(--warn) 15%,transparent)}}
.spark{{width:100%; height:auto; display:block}}
.stats{{display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin:0}}
.stats div{{display:flex; flex-direction:column; gap:3px}}
.stats dt{{font:600 10px/1 ui-monospace,monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--muted)}}
.stats dd{{margin:0; font:600 17px/1 ui-monospace,monospace; font-variant-numeric:tabular-nums}}
.stats dd.good{{color:var(--good)}}
.note{{
  margin:38px 0 0; padding-top:22px; border-top:1px solid var(--line);
  font-size:14px; color:var(--muted); max-width:70ch;
}}
.note b{{color:var(--ink); font-weight:600}}
.legend{{display:flex; gap:18px; flex-wrap:wrap; font:500 12px/1 ui-monospace,monospace; color:var(--muted); margin-top:8px}}
.legend i{{font-style:normal; display:inline-flex; align-items:center; gap:6px}}
.swatch{{width:14px; height:3px; border-radius:2px; display:inline-block}}
</style>

<div class="wrap">
  <p class="eyebrow">DronePhotographer · closed-loop simulation</p>
  <h1>Diffusion Policy navigates to the shot</h1>
  <p class="thesis">From a random start pose, the policy renders what it sees, predicts a camera
     move, and repeats — driving the achieved shot profile toward the target. On held-out
     scenes it <b>reaches the goal on 2 of 3 non-trivial cases</b> and halves the distance on the third.</p>

  <div class="bench">
    <div class="stat win"><div class="k">DP · recon</div><div class="v">50<small>cm</small> · 8.7<small>°</small></div></div>
    <div class="stat"><div class="k">pi0.5 · recon</div><div class="v">75<small>cm</small> · 57<small>°</small></div></div>
    <div class="stat"><div class="k">GR00T · recon</div><div class="v">260<small>cm</small> · 17<small>°</small></div></div>
  </div>

  <div class="grid">
    {cards}
  </div>

  <div class="legend">
    <i><span class="swatch" style="background:var(--accent)"></span>goal-distance</i>
    <i><span class="swatch" style="background:var(--thresh)"></span>reached threshold (0.12)</i>
  </div>

  <p class="note"><b>Reading the runs.</b> Goal-distance is mean normalized error between the achieved
     and target shot profile; below 0.12 counts as reached. Targets are each scene's own reachable
     shot (own-goal), on the 8 held-out validation scenes. Caveat: the VLA numbers above are from an
     unfair budget (0.59 epoch) — fair retrains are in progress, so treat DP's lead as provisional.</p>
</div>

<script>
const DATA = {payload};
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
function draw(id, d){{
  const cv = document.getElementById("sp-"+id); if(!cv) return;
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  const W = cv.width, H = cv.height; cv.width = W*dpr; cv.height = H*dpr;
  const g = cv.getContext("2d"); g.scale(dpr,dpr);
  const pad = {{l:6, r:6, t:10, b:6}};
  const t = d.traj, n = t.length;
  const ymax = Math.max(0.7, ...t)*1.05;
  const X = i => pad.l + (W-pad.l-pad.r) * (n<2?0:i/(n-1));
  const Y = v => pad.t + (H-pad.t-pad.b) * (1 - v/ymax);
  // threshold line
  g.strokeStyle = css("--thresh"); g.setLineDash([4,4]); g.lineWidth=1;
  g.beginPath(); g.moveTo(pad.l, Y(0.12)); g.lineTo(W-pad.r, Y(0.12)); g.stroke();
  g.setLineDash([]);
  // area fill
  const acc = css("--accent");
  const grad = g.createLinearGradient(0,pad.t,0,H);
  grad.addColorStop(0, acc+"38"); grad.addColorStop(1, acc+"00");
  g.beginPath(); g.moveTo(X(0),H-pad.b);
  t.forEach((v,i)=>g.lineTo(X(i),Y(v))); g.lineTo(X(n-1),H-pad.b); g.closePath();
  g.fillStyle=grad; g.fill();
  // line
  g.beginPath(); t.forEach((v,i)=> i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)));
  g.strokeStyle=acc; g.lineWidth=2; g.lineJoin="round"; g.stroke();
  // endpoints
  const end = d.reached ? css("--good") : css("--warn");
  g.fillStyle=css("--muted"); g.beginPath(); g.arc(X(0),Y(t[0]),3,0,7); g.fill();
  g.fillStyle=end; g.beginPath(); g.arc(X(n-1),Y(t[n-1]),4,0,7); g.fill();
}}
function drawAll(){{ for(const k in DATA) draw(k, DATA[k]); }}
drawAll(); window.addEventListener("resize", drawAll);
new MutationObserver(drawAll).observe(document.documentElement,{{attributes:true,attributeFilter:["data-theme"]}});
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", drawAll);
</script>
"""

OUT.write_text(HTML)
kb = OUT.stat().st_size // 1024
print(f"wrote {OUT} ({kb} KB)")
