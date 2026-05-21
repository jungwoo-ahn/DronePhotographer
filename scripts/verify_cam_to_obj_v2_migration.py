#!/usr/bin/env python3
"""Build a 'before vs after' HTML report verifying the v2 sign-convention
migration. Pulls a handful of rows from a smoke run dir (one that has
annotations.json AND annotations.json.bak side by side) and shows:

  - the embedded thumbnail
  - cam_z, obj_z, delta_z
  - v1 (pre-migration) score_cam_to_obj_elevation_deg, _azimuth_deg
  - v2 (post-migration) score_cam_to_obj_elevation_deg, _azimuth_deg
  - a sanity flag: v2 should agree with the convention (cam above => elev<0).

Writes outputs/cam_to_obj_v2_verification.html.
"""
from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path

from PIL import Image

REPO = Path("/home/nas1/jungwooahn/projects/DronePhotographer")
SMOKE = REPO / "outputs/smoke_v6_pitch_lerp"
OUT = REPO / "outputs/cam_to_obj_v2_verification.html"


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


def collect_samples():
    """For each placement dir under SMOKE that has both annotations.json
    and annotations.json.bak, pick 3 representative rows by elevation:
    most negative (camera highest), near zero, most positive (camera lowest).
    """
    samples = []
    for run in sorted(SMOKE.glob("p*_*")):
        ann_now = run / "annotations.json"
        ann_bak = run / "annotations.json.bak"
        if not (ann_now.exists() and ann_bak.exists()):
            continue
        try:
            old = json.loads(ann_bak.read_text())
            new = json.loads(ann_now.read_text())
        except Exception:
            continue
        if len(old) != len(new):
            continue
        # Sort by v2 elevation (now in the live file)
        order = sorted(range(len(new)), key=lambda i: new[i].get("score_cam_to_obj_elevation_deg", 0))
        picks = [order[0], order[len(order) // 2], order[-1]]
        for idx in picks:
            samples.append({
                "run": run.name,
                "row_index": idx,
                "old": old[idx],
                "new": new[idx],
            })
    return samples


def main():
    samples = collect_samples()
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:32px auto;max-width:1080px;background:#fafafa;color:#222;line-height:1.55;}
    h1{font-size:22px;margin:0 0 6px;}
    h2{font-size:16px;margin:24px 0 6px;padding-top:8px;border-top:1px solid #ddd;color:#555;}
    .meta{font-size:13px;color:#555;margin:8px 0 16px;}
    .card{display:flex;gap:16px;background:white;border:1px solid #ddd;
          border-radius:6px;padding:14px;margin:10px 0;}
    .card img{width:320px;height:auto;border-radius:4px;flex-shrink:0;}
    .card .body{font-size:13px;}
    table{border-collapse:collapse;font-size:13px;margin-top:8px;}
    th,td{border:1px solid #ccc;padding:5px 9px;text-align:right;}
    th{background:#eee;text-align:left;}
    td.lbl{text-align:left;font-weight:600;color:#444;}
    .ok{background:#e8f5e8;color:#2ca02c;font-weight:600;}
    .bad{background:#fde8e8;color:#d62728;font-weight:600;}
    .neutral{background:#eee;color:#666;font-weight:600;}
    code{background:#eee;padding:1px 5px;border-radius:3px;font-size:12px;}
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>v2 migration verification</title>",
             f"<style>{css}</style></head><body>",
             "<h1>Migration verification: cam_to_obj_v2 signs</h1>",
             "<div class='meta'>For each placement run, three sampled rows (lowest v2 elevation, "
             "middle, highest v2 elevation). Old values from <code>annotations.json.bak</code>; new "
             "from current <code>annotations.json</code>. The 'verdict' column says whether the v2 "
             "value agrees with the visible image content.</div>"]

    by_run: dict[str, list[dict]] = {}
    for s in samples:
        by_run.setdefault(s["run"], []).append(s)

    for run_name in sorted(by_run.keys()):
        parts.append(f"<h2>{escape(run_name)}</h2>")
        for s in by_run[run_name]:
            old = s["old"]
            new = s["new"]
            cam = old.get("camera_position") or new.get("camera_position")
            obj = old.get("object_position") or new.get("object_position")
            dz = cam[2] - obj[2] if cam and obj else None
            v1_el = old.get("score_cam_to_obj_elevation_deg")
            v2_el = new.get("score_cam_to_obj_elevation_deg")
            v1_az = old.get("score_cam_to_obj_azimuth_deg")
            v2_az = new.get("score_cam_to_obj_azimuth_deg")
            # Verdict logic: in v2, cam above (dz>0) should give elev<0
            if dz is None or v2_el is None:
                verdict = ("?", "neutral")
            elif dz > 0.1 and v2_el < 0:
                verdict = ("OK · cam above → elev &lt; 0", "ok")
            elif dz < -0.1 and v2_el > 0:
                verdict = ("OK · cam below → elev &gt; 0", "ok")
            elif abs(dz) < 0.5 and abs(v2_el) < 15:
                verdict = ("OK · eye-level", "ok")
            elif abs(dz) < 1.0 and abs(v2_el) < 30:
                verdict = ("near eye-level", "neutral")
            else:
                verdict = ("FAIL · unexpected sign", "bad")

            # Azimuth verdict: v1 + 180 (mod 360) should == v2
            az_expected = (v1_az + 180) % 360 if v1_az is not None else None
            az_verdict = ("OK", "ok") if az_expected == v2_az else ("MISMATCH", "bad")

            img_path = (SMOKE / run_name / new.get("image", ""))
            img_uri = embed(img_path) if img_path.exists() else ""

            parts.append("<div class='card'>")
            if img_uri:
                parts.append(f"<img src='{img_uri}' alt=''>")
            parts.append("<div class='body'>")
            parts.append(f"<div><code>{escape(new.get('image',''))}</code></div>")
            parts.append("<table>")
            parts.append(f"<tr><td class='lbl'>cam_z</td><td>{cam[2]:.2f}</td>"
                         f"<td class='lbl'>obj_z</td><td>{obj[2]:.2f}</td>"
                         f"<td class='lbl'>Δz</td><td><b>{dz:+.2f} m</b></td></tr>")
            parts.append(f"<tr><td class='lbl'>elevation</td>"
                         f"<td>v1: <b>{v1_el:+d}</b></td>"
                         f"<td>v2: <b>{v2_el:+d}</b></td>"
                         f"<td class='lbl'>verdict</td>"
                         f"<td class='{verdict[1]}' colspan='2'>{verdict[0]}</td></tr>")
            parts.append(f"<tr><td class='lbl'>azimuth</td>"
                         f"<td>v1: <b>{v1_az}</b></td>"
                         f"<td>v2: <b>{v2_az}</b></td>"
                         f"<td class='lbl'>(v1+180)%360</td>"
                         f"<td>{az_expected}</td>"
                         f"<td class='{az_verdict[1]}'>{az_verdict[0]}</td></tr>")
            parts.append("</table></div></div>")

    parts.append("</body></html>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
