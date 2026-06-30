"""Build a rich, self-contained HTML report from a rollout_eval output dir.

Reads summary.json + per-rollout JSONs + gifs/, and renders:
  - training val-loss context (from the run's slurm log)
  - overall metric cards + aggregate plots (distance-vs-step, per-goal improvement,
    start-vs-final scatter)
  - per-rollout cards: the drone GIF, a distance-to-goal curve, and a goal-vs-achieved
    8-key table (per-key error + within-tolerance highlight)

Everything (plots + GIFs) is base64-embedded, so the output is a single portable file.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/build_eval_report.py \
      --eval-dir runs/<ts>_cosmos_2b/rollout_eval [--slurm-log runs/slurm-...out] [--out report.html]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.scoring.bbox_control import V5_SCORE_KEYS  # noqa: E402

KEY_SHORT = {
    "occupancy": "occ", "body_in_frame_ratio": "in-frame%",
    "cam_to_obj_azimuth_deg": "azimuth", "cam_to_obj_elevation_deg": "elevation",
    "object_center_x": "center-x", "object_center_y": "center-y",
    "bbox_x_offset": "half-w", "bbox_y_offset": "half-h",
}


def b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    return b64(buf.getvalue(), "image/png")


def parse_val_log(path: Path):
    """(iters, total_flow_loss, action_mse, value_mae) from 'iter=N VAL ...' lines.

    New runs log `total=`; older runs logged `flow_loss_mean=` — accept either.
    """
    if not path or not Path(path).exists():
        return None
    pat = re.compile(r"iter=(\d+) VAL .*?(?:flow_loss_mean|total)=([\d.]+).*?action_mse=([\d.]+).*?value_mae=([\d.]+)")
    rows = []
    for ln in Path(path).read_text(errors="ignore").replace("\r", "\n").splitlines():
        m = pat.search(ln)
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    if not rows:
        return None
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def val_plot(val, eval_iter):
    it, flow, act, val_mae = val
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(it, flow, "-o", ms=2, lw=1.4, color="#2563eb", label="val total flow loss")
    ax.plot(it, act, "-o", ms=2, lw=1.0, color="#16a34a", label="val action_mse", alpha=0.8)
    if eval_iter and eval_iter > 0:
        ax.axvline(eval_iter, color="#dc2626", ls="--", lw=1.2, label=f"this ckpt (iter {eval_iter})")
    ax.set_xlabel("training iteration"); ax.set_ylabel("val loss")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
    ax.set_title("Training validation loss (context)", fontsize=9)
    return fig_b64(fig)


def dist_curve_plot(rollouts):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    maxlen = 0
    for r in rollouts:
        d = [s["distance"] for s in r["steps"]]
        ax.plot(range(len(d)), d, color="#94a3b8", lw=0.8, alpha=0.5)
        maxlen = max(maxlen, len(d))
    # mean curve (pad-forward to align lengths)
    grid = []
    for r in rollouts:
        d = [s["distance"] for s in r["steps"]]
        d = d + [d[-1]] * (maxlen - len(d))
        grid.append(d)
    if grid:
        mean = np.mean(grid, axis=0)
        ax.plot(range(maxlen), mean, color="#dc2626", lw=2.5, label="mean")
        ax.legend(fontsize=8)
    ax.set_xlabel("rollout step"); ax.set_ylabel("distance to goal (norm L2)")
    ax.set_title("Goal distance over the rollout (every run + mean)", fontsize=9)
    ax.grid(alpha=0.25)
    return fig_b64(fig)


def per_goal_plot(rollouts):
    by = {}
    for r in rollouts:
        by.setdefault(r["goal"], []).append(r["improvement"])
    items = sorted(((g, float(np.mean(v))) for g, v in by.items()), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.26 * len(items))))
    names = [g for g, _ in items]; vals = [v for _, v in items]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in vals]
    ax.barh(range(len(items)), vals, color=colors)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(names, fontsize=6)
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("mean improvement over no-op  (start dist − final dist)")
    ax.set_title("Per-goal goal-reaching improvement", fontsize=9); ax.grid(alpha=0.2, axis="x")
    return fig_b64(fig)


def scatter_plot(rollouts):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ds = [r["d_start"] for r in rollouts]; df = [r["d_final"] for r in rollouts]
    cols = ["#16a34a" if a > b else "#dc2626" for a, b in zip(ds, df)]
    ax.scatter(ds, df, c=cols, s=18, alpha=0.8)
    lo, hi = 0, max(max(ds + df), 0.1) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="#334155", lw=1, label="no change")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("start distance"); ax.set_ylabel("final distance")
    ax.set_title("Start vs final (below line = improved)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.25); ax.set_aspect("equal")
    return fig_b64(fig)


def mini_curve(r) -> str:
    d = [s["distance"] for s in r["steps"]]
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.plot(range(len(d)), d, "-o", ms=3, color="#2563eb")
    ax.set_xlabel("step", fontsize=6); ax.set_ylabel("dist", fontsize=6)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
    return fig_b64(fig)


def key_table(goal_profile, final_profile, tol) -> str:
    rows = []
    for k in V5_SCORE_KEYS:
        g, a = goal_profile.get(k), final_profile.get(k)
        if g is None or a is None:
            continue
        err = abs(a - g)
        ok = err <= tol.get(k, 1e9)
        cls = "ok" if ok else "no"
        rows.append(
            f"<tr><td>{KEY_SHORT.get(k, k)}</td><td>{g}</td><td>{a}</td>"
            f"<td class='{cls}'>{err} {'✓' if ok else '✗'}</td></tr>")
    return ("<table class='kt'><tr><th>key</th><th>goal</th><th>achieved</th><th>|err| (tol)</th></tr>"
            + "".join(rows) + "</table>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--slurm-log", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summ = json.loads((args.eval_dir / "summary.json").read_text())
    tol = summ.get("config", {}).get("success_tol", {})
    rollouts = []
    for f in sorted(args.eval_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        rollouts.append(json.loads(f.read_text()))
    rollouts.sort(key=lambda r: r["improvement"], reverse=True)

    eval_iter = summ.get("iteration", -1)
    slurm = args.slurm_log
    if slurm is None:
        cand = Path("runs/slurm-cosmos_policy_2b-26462.out")
        slurm = cand if cand.exists() else None
    val = parse_val_log(Path(slurm)) if slurm else None

    # plots
    plots = {
        "dist": dist_curve_plot(rollouts) if rollouts else "",
        "per_goal": per_goal_plot(rollouts) if rollouts else "",
        "scatter": scatter_plot(rollouts) if rollouts else "",
    }
    if val:
        plots["val"] = val_plot(val, eval_iter)

    def card(num, label, sub=""):
        return f"<div class='kpi'><div class='num'>{num}</div><div class='lbl'>{label}</div><div class='sub'>{sub}</div></div>"

    n = summ.get("n_rollouts", len(rollouts))
    kpis = "".join([
        card(n, "rollouts", f"{summ.get('n_placements','?')} placements × goals"),
        card(f"{summ.get('success_rate', 0) * 100:.0f}%", "success @ tol", "all 8 keys within tol"),
        card(f"{summ.get('mean_improvement_over_noop', 0):+.3f}", "mean improvement", "start − final distance"),
        card(f"{summ.get('mean_d_final', 0):.3f}", "mean final distance", "normalized L2"),
    ])

    # per-rollout cards
    cards = []
    for r in rollouts:
        tag = f"{r['placement'][:40]}__{r['goal']}"
        gif_path = args.eval_dir / "gifs" / f"{tag}.gif"
        gif = b64(gif_path.read_bytes(), "image/gif") if gif_path.exists() else None
        final = r["steps"][-1]["profile"]
        badge = "ok" if r["reached"] else "no"
        gif_html = f"<img class='gif' src='{gif}'>" if gif else "<div class='nogif'>(no gif)</div>"
        cards.append(f"""
        <div class='card'>
          <div class='chead'><b>{r['goal']}</b> <span class='pl'>{r['placement'][:46]}</span>
            <span class='badge {badge}'>{'REACHED' if r['reached'] else 'not reached'}</span></div>
          <div class='crow'>
            {gif_html}
            <div class='cmid'>
              <div class='dline'>dist {r['d_start']:.3f} → <b>{r['d_final']:.3f}</b>
                 <span class='{ "ok" if r["improvement"]>0 else "no" }'>({r['improvement']:+.3f})</span>
                 · {r['n_steps']} steps</div>
              <img class='mini' src='{mini_curve(r)}'>
            </div>
            <div class='ckt'>{key_table(r['goal_profile'], final, tol)}</div>
          </div>
        </div>""")

    val_html = f"<img class='wide' src='{plots['val']}'>" if "val" in plots else "<p class='muted'>(no training log found)</p>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Cosmos policy — rollout eval</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}
 .wrap{{max-width:1180px;margin:0 auto;padding:24px}}
 h1{{font-size:22px;margin:0 0 4px}} .meta{{color:#475569;font-size:13px;margin-bottom:18px}}
 .kpis{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
 .kpi{{background:#fff;border-radius:12px;padding:16px 20px;box-shadow:0 1px 3px #0001;flex:1;min-width:160px}}
 .kpi .num{{font-size:28px;font-weight:700}} .kpi .lbl{{font-size:13px;color:#334155}} .kpi .sub{{font-size:11px;color:#94a3b8}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px}}
 .panel{{background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 3px #0001}}
 .panel img,.wide{{width:100%;border-radius:6px}} h2{{font-size:15px;margin:22px 0 10px}}
 .card{{background:#fff;border-radius:12px;padding:14px;box-shadow:0 1px 3px #0001;margin-bottom:14px}}
 .chead{{font-size:14px;margin-bottom:8px}} .pl{{color:#64748b;font-size:12px}}
 .badge{{float:right;font-size:11px;padding:2px 8px;border-radius:20px;color:#fff}}
 .badge.ok{{background:#16a34a}} .badge.no{{background:#64748b}}
 .crow{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}}
 .gif{{width:360px;border-radius:8px;border:1px solid #e2e8f0}} .nogif{{width:360px;color:#94a3b8}}
 .cmid{{flex:1;min-width:200px}} .dline{{font-size:13px;margin-bottom:6px}} .mini{{width:240px}}
 .kt{{border-collapse:collapse;font-size:11px}} .kt td,.kt th{{border:1px solid #e2e8f0;padding:2px 6px;text-align:center}}
 td.ok,span.ok{{color:#16a34a;font-weight:600}} td.no,span.no{{color:#dc2626;font-weight:600}}
 .muted{{color:#94a3b8}}
</style></head><body><div class='wrap'>
 <h1>Cosmos goal-conditioned policy — closed-loop eval</h1>
 <div class='meta'>checkpoint: <code>{summ.get('checkpoint','?')}</code> (iter <b>{eval_iter}</b>) ·
   held-out val scenes · receding-horizon (execute_k={summ.get('config',{}).get('execute_k','?')},
   n_steps={summ.get('config',{}).get('n_steps','?')}, max_steps={summ.get('config',{}).get('max_steps','?')})</div>
 <div class='kpis'>{kpis}</div>
 <div class='panel' style='margin-bottom:16px'><h2 style='margin-top:0'>Training context</h2>{val_html}</div>
 <div class='grid2'>
   <div class='panel'><img src='{plots['dist']}'></div>
   <div class='panel'><img src='{plots['scatter']}'></div>
 </div>
 <div class='panel' style='margin-bottom:8px'><img src='{plots['per_goal']}'></div>
 <h2>Per-rollout detail ({len(rollouts)} rollouts, best improvement first)</h2>
 {''.join(cards)}
 <p class='muted' style='margin-top:24px'>Achieved profile per frame = geometric mesh-tight projection (validated
   IoU 1.0 / scores 5724/5724 vs the dataset). Distance = normalized weighted L2 over the 8 V5 keys.
   "Improvement" = start distance − final distance (vs the no-op baseline of staying at the start pose).</p>
</div></body></html>"""

    out = Path(args.out) if args.out else args.eval_dir / "report.html"
    out.write_text(html)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB) — {len(rollouts)} rollouts, "
          f"success {summ.get('success_rate',0)*100:.0f}%, mean improve {summ.get('mean_improvement_over_noop',0):+.3f}")


if __name__ == "__main__":
    main()
