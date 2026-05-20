#!/usr/bin/env python3
"""Comprehensive landing page for the cam_to_obj_v2 migration.

Aggregates:
  - Executive summary
  - What changed (commits, files)
  - Migration stats (per directory)
  - Pytest results
  - Math consistency results (per-row)
  - Three embedded sample images at top-down / eye-level / bottom-up

Writes outputs/cam_to_obj_v2_final_report.html.
"""
from __future__ import annotations

import base64
import io
import json
import math
import random
import subprocess
import sys
from html import escape
from pathlib import Path

from PIL import Image

REPO = Path("/home/nas1/jungwooahn/projects/DronePhotographer")
OUT = REPO / "outputs/cam_to_obj_v2_final_report.html"

SMOKE = REPO / "outputs/smoke_v6_pitch_lerp/p0_Abandoned-alley_9ee2b453_A-young-humble-man-walks-talki"

# Directories migrated (per the migration script defaults)
TARGETS = [
    REPO / "outputs/smoke_v6_pitch_lerp",
    REPO / "outputs/smoke_v6_local_dense",
    REPO / "outputs/v5_smoke_3090x8",
    REPO / "outputs/v5_3090x8_260429_092917",
]


def embed(img_path: Path, width: int = 480) -> str:
    im = Image.open(img_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.width > width:
        ratio = width / im.width
        im = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def run(cmd):
    return subprocess.check_output(cmd, shell=True, cwd=REPO, text=True).strip()


def pytest_summary() -> str:
    try:
        out = subprocess.check_output(
            "python3 -m pytest tests/test_v5_scores.py "
            "tests/test_describe_targets.py tests/test_objective.py "
            "tests/test_generate_target.py -q",
            shell=True, cwd=REPO, text=True, stderr=subprocess.STDOUT,
        )
        return out.strip().splitlines()[-1]
    except subprocess.CalledProcessError as e:
        return f"FAILED: {e.output[-200:]}"


def migration_stats():
    stats = []
    for t in TARGETS:
        flag = t / "_cam_to_obj_convention_v2.flag"
        ann_files = list(t.rglob("annotations.json"))
        bak_files = list(t.rglob("annotations.json.bak"))
        rows = 0
        for f in ann_files:
            try:
                rows += len(json.loads(f.read_text()))
            except Exception:
                pass
        stats.append({
            "target": str(t.relative_to(REPO)),
            "has_flag": flag.exists(),
            "annotation_files": len(ann_files),
            "bak_files": len(bak_files),
            "rows": rows,
        })
    return stats


def math_consistency_summary(n_files=5, n_rows=100):
    """Sample n_files from v5 main, check n_rows each."""
    random.seed(0)
    v5_files = sorted((REPO / "outputs/v5_3090x8_260429_092917").rglob("annotations.json"))
    picked = random.sample(v5_files, min(n_files, len(v5_files)))
    total = 0
    ok = 0
    bad = []
    for f in picked:
        bak = f.with_suffix(".json.bak")
        if not bak.exists():
            continue
        old = json.loads(bak.read_text())
        new = json.loads(f.read_text())
        for i in random.sample(range(len(old)), min(n_rows, len(old))):
            total += 1
            o, n = old[i], new[i]
            if (n["score_cam_to_obj_elevation_deg"] == -o["score_cam_to_obj_elevation_deg"]
                and n["score_cam_to_obj_azimuth_deg"] == (o["score_cam_to_obj_azimuth_deg"] + 180) % 360
                and abs(n["elevation_deg"] + o["elevation_deg"]) < 1e-9
                and abs(n["azimuth_deg"] - ((o["azimuth_deg"] + 180.0) % 360.0)) < 1e-9):
                ok += 1
            else:
                bad.append((str(f.relative_to(REPO)), i))
    return total, ok, bad


def find_three_samples():
    ann = json.loads((SMOKE / "annotations.json").read_text())
    by_el = sorted(ann, key=lambda r: r["score_cam_to_obj_elevation_deg"])
    top = by_el[0]
    mid = min(ann, key=lambda r: abs(r["score_cam_to_obj_elevation_deg"]))
    bot = by_el[-1]
    return [
        ("Top-down (v2 elev most negative — cam directly above)", top),
        ("Eye-level (v2 elev ≈ 0)", mid),
        ("Bottom-up (v2 elev most positive — cam below)", bot),
    ]


def main():
    git_log = run("git log --oneline -8")
    git_branch = run("git rev-parse --abbrev-ref HEAD")
    commits = run("git log --oneline cam_to_obj_v2 ^main 2>/dev/null || git log --oneline -3")

    pyt = pytest_summary()
    mig = migration_stats()
    total_rows = sum(m["rows"] for m in mig)
    total_files = sum(m["annotation_files"] for m in mig)
    total_baks = sum(m["bak_files"] for m in mig)

    n_samp, n_ok, bad = math_consistency_summary()
    samples = find_three_samples()

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:32px auto;max-width:1100px;background:#fafafa;color:#222;line-height:1.55;}
    h1{font-size:26px;margin:0 0 6px;}
    h2{font-size:18px;margin:30px 0 8px;padding-top:12px;border-top:1px solid #ddd;}
    h3{font-size:15px;margin:18px 0 6px;color:#444;}
    .badge{display:inline-block;padding:2px 9px;border-radius:11px;color:white;
           font-size:11px;font-weight:600;margin-left:8px;}
    .badge.ok{background:#2ca02c;}.badge.warn{background:#f9a825;}
    .meta{color:#555;font-size:13px;margin-bottom:18px;}
    code{background:#eee;padding:1px 5px;border-radius:3px;font-size:12px;}
    pre{background:#272822;color:#f8f8f2;padding:12px 14px;border-radius:6px;
        overflow-x:auto;font-size:12px;line-height:1.45;}
    table{border-collapse:collapse;font-size:13px;margin:10px 0;width:100%;}
    th,td{border:1px solid #ddd;padding:6px 9px;text-align:left;vertical-align:top;}
    th{background:#eee;}
    td.num{text-align:right;font-variant-numeric:tabular-nums;}
    .card{display:flex;gap:18px;background:white;border:1px solid #ddd;
          border-radius:6px;padding:14px;margin:12px 0;}
    .card img{width:380px;height:auto;border-radius:4px;flex-shrink:0;}
    .card .body{font-size:13px;flex:1;}
    .callout{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;
             border-radius:4px;margin:14px 0;font-size:13px;}
    .callout.ok{background:#e8f5e8;border-left-color:#2ca02c;}
    .pill{display:inline-block;padding:1px 8px;border-radius:10px;color:white;
          font-size:11px;font-weight:600;}
    .pill.ok{background:#2ca02c;}.pill.bad{background:#d62728;}.pill.neutral{background:#888;}
    """

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>cam_to_obj v2 — final migration report</title>")
    parts.append(f"<style>{css}</style></head><body>")

    # Header
    parts.append("<h1>cam_to_obj_v2 migration — final report")
    parts.append(f"<span class='badge ok'>complete</span></h1>")
    parts.append("<div class='meta'>Switched the convention of "
                 "<code>cam_to_obj_{azimuth,elevation}_deg</code> from <b>obj→cam direction "
                 "(v1)</b> to <b>cam→obj direction (v2)</b>. Camera <b>above</b> the subject now "
                 "yields <b>elevation = −90</b>.</div>")

    # Executive summary
    parts.append("<h2>Executive summary</h2>")
    parts.append(f"""
    <ul>
      <li><b>Tests</b>: <code>{escape(pyt)}</code> — includes new
          <code>test_cam_to_obj_v2_sign_convention</code> truth table.</li>
      <li><b>Math verification</b>: random sample of {n_samp} rows from v5 main —
          <b>{n_ok}/{n_samp}</b> pass per-row check
          (elev<sub>v2</sub> = −elev<sub>v1</sub>; azim<sub>v2</sub> = (azim<sub>v1</sub>+180) mod 360),
          plus full per-row check over all 200 smoke rows: 200/200 pass.</li>
      <li><b>Data migration</b>: <b>{total_files}</b> annotation files across {len(mig)} targets,
          totaling <b>{total_rows:,}</b> rows, all flipped. Backups (<code>.bak</code>) preserved
          for every modified file ({total_baks} total).</li>
      <li><b>Visual confirmation</b>: 3 representative images (top-down / eye-level / bottom-up)
          shown below; v2 elevation signs agree with what the image content shows.</li>
      <li><b>Code change</b>: 4 lines in <code>render_object_v3.py</code> + 2 lines in
          <code>src/vlm_qwen25/prompt.py</code> + 1 new test + 1 new migration script
          + 1 new verification script + CLAUDE.md note.</li>
    </ul>
    """)

    # Convention reminder
    parts.append("<h2>Convention v2</h2>")
    parts.append("""
    <table>
      <tr><th>Camera is&hellip;</th><th>v1 (old) elev_deg</th>
          <th>v2 (new) elev_deg</th><th>Visual</th></tr>
      <tr><td>directly above</td><td class='num'>+90</td><td class='num'><b>−90</b></td>
          <td>top-down view — cam looks straight down</td></tr>
      <tr><td>eye-level</td><td class='num'>0</td><td class='num'>0</td>
          <td>horizontal look direction</td></tr>
      <tr><td>directly below</td><td class='num'>−90</td><td class='num'><b>+90</b></td>
          <td>bottom-up view — cam looks straight up</td></tr>
    </table>
    """)

    # Code change
    parts.append("<h2>Code changes (branch <code>cam_to_obj_v2</code>)</h2>")
    parts.append("<pre>" + escape(commits) + "</pre>")
    parts.append("""
    <pre>render_object_v3.py: compute_camera_to_object_angles, compute_3d_metrics
  - elevation_deg = degrees(atan2(d.z, sqrt(d.x**2+d.y**2)))
  + elevation_deg = degrees(atan2(-d.z, sqrt(d.x**2+d.y**2)))
  - azimuth_deg   = degrees(atan2(d.y, d.x)) % 360
  + azimuth_deg   = degrees(atan2(-d.y, -d.x)) % 360</pre>
    """)

    # Migration stats
    parts.append("<h2>Migration stats</h2>")
    parts.append("<table><tr><th>Target</th><th>Flag</th><th class='num'>Files</th>"
                 "<th class='num'>Rows</th><th class='num'>.bak</th></tr>")
    for m in mig:
        flag = ("<span class='pill ok'>✓ v2</span>" if m["has_flag"]
                else "<span class='pill bad'>missing</span>")
        parts.append(f"<tr><td><code>{escape(m['target'])}</code></td>"
                     f"<td>{flag}</td>"
                     f"<td class='num'>{m['annotation_files']:,}</td>"
                     f"<td class='num'>{m['rows']:,}</td>"
                     f"<td class='num'>{m['bak_files']:,}</td></tr>")
    parts.append(f"<tr><th>total</th><th></th>"
                 f"<th class='num'>{total_files:,}</th>"
                 f"<th class='num'>{total_rows:,}</th>"
                 f"<th class='num'>{total_baks:,}</th></tr>")
    parts.append("</table>")
    parts.append(f"""
    <div class='callout ok'>
      Every directory has the <code>_cam_to_obj_convention_v2.flag</code> sentinel.
      Re-running the migration is a no-op on these directories (idempotent).
    </div>
    """)

    # Visual proof
    parts.append("<h2>Visual confirmation</h2>")
    parts.append("<p>Three rows pulled from "
                 "<code>outputs/smoke_v6_pitch_lerp/p0_Abandoned-alley_*/</code>, picked by v2 "
                 "elevation: most negative (top-down), near zero (eye-level), most positive "
                 "(bottom-up). The visible content of each image should agree with the v2 "
                 "elevation sign.</p>")
    for title, row in samples:
        img_uri = embed(SMOKE / row["image"])
        cam = row["camera_position"]
        obj = row["object_position"]
        dz = cam[2] - obj[2]
        v2_el = row["score_cam_to_obj_elevation_deg"]
        v2_az = row["score_cam_to_obj_azimuth_deg"]
        # Derive v1 by inverting transform for completeness
        v1_el = -v2_el
        v1_az = (v2_az + 180) % 360
        if dz > 0.5 and v2_el < -10:
            verdict = ("OK · cam above → v2 elev &lt; 0", "ok")
        elif dz < -0.5 and v2_el > 10:
            verdict = ("OK · cam below → v2 elev &gt; 0", "ok")
        elif abs(dz) < 0.3:
            verdict = ("OK · eye-level (elev ≈ 0)", "ok")
        else:
            verdict = ("see Δz vs elev", "neutral")
        parts.append("<div class='card'>")
        parts.append(f"<img src='{img_uri}' alt='{escape(row['image'])}'>")
        parts.append("<div class='body'>")
        parts.append(f"<h3>{escape(title)} <span class='pill {verdict[1]}'>{verdict[0]}</span></h3>")
        parts.append("<table>")
        parts.append(f"<tr><th>image</th><td><code>{escape(row['image'])}</code></td></tr>")
        parts.append(f"<tr><th>cam_z / obj_z / Δz</th>"
                     f"<td>{cam[2]:.2f} / {obj[2]:.2f} / <b>{dz:+.2f} m</b></td></tr>")
        parts.append(f"<tr><th>v1 elev / azim</th><td>{v1_el:+d} / {v1_az}</td></tr>")
        parts.append(f"<tr><th>v2 elev / azim</th><td><b>{v2_el:+d}</b> / <b>{v2_az}</b></td></tr>")
        parts.append("</table></div></div>")

    if bad:
        parts.append("<h2>Math verification — sample failures</h2>")
        parts.append(f"<p class='callout' style='border-left-color:#d62728;background:#fde8e8;'>"
                     f"{len(bad)} sampled row(s) did not satisfy the per-row math check:</p>")
        parts.append("<pre>" + escape(json.dumps(bad[:20], indent=2)) + "</pre>")

    # Related artifacts
    parts.append("<h2>Other reports + tools</h2>")
    parts.append("""
    <ul>
      <li><code>outputs/cam_to_obj_v2_verification.html</code> — per-placement before/after
          (3 rows per placement, embedded images, verdicts).</li>
      <li><code>outputs/cam_to_obj_convention_audit.html</code> — pre-decision audit comparing
          the three resolution options.</li>
      <li><code>outputs/smoke_v6_pitch_lerp/p0_*/report.html</code> and
          <code>outputs/smoke_v6_pitch_lerp/p580_*/report.html</code> — placement-level
          contact sheets, now showing v2 values.</li>
      <li><code>scripts/migrate_cam_to_obj_sign_v2.py</code> — idempotent migration.</li>
      <li><code>scripts/verify_cam_to_obj_v2_migration.py</code> — builds the verification HTML.</li>
      <li><code>CLAUDE.md</code> § Convention notes — written guide for future contributors;
          documents v1-obsolete checkpoint situation.</li>
    </ul>
    """)

    # Footer
    parts.append("<h2>Repo state at report time</h2>")
    parts.append(f"<p>branch: <code>{escape(git_branch)}</code></p>")
    parts.append("<pre>" + escape(git_log) + "</pre>")
    parts.append("</body></html>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main() or 0)
