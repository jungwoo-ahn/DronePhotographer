#!/usr/bin/env python3
"""Build an HTML audit report on the cam_to_obj_{elevation,azimuth} sign
convention question. Self-contained (base64 thumbnails) so it can be shared.

Writes outputs/cam_to_obj_convention_audit.html.
"""
from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path

from PIL import Image

REPO = Path("/home/nas1/jungwooahn/projects/DronePhotographer")
SMOKE_DIR = REPO / "outputs/smoke_v6_pitch_lerp/p0_Abandoned-alley_9ee2b453_A-young-humble-man-walks-talki"
OUT = REPO / "outputs/cam_to_obj_convention_audit.html"


def embed(img_path: Path, width: int = 640) -> str:
    im = Image.open(img_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.width > width:
        ratio = width / im.width
        im = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def main():
    ann = json.loads((SMOKE_DIR / "annotations.json").read_text())
    by_el = sorted(ann, key=lambda r: r["elevation_deg"])

    LOW = by_el[0]    # most-negative elevation (camera below)
    MID = by_el[len(by_el) // 2]
    HIGH = by_el[-1]  # most-positive elevation (camera above)

    samples = [
        ("Looking UP at the object", LOW,
         "Camera is BELOW the object (cam_z &lt; obj_z). "
         "Current code stores this as <b>negative</b> score_cam_to_obj_elevation_deg."),
        ("Eye-level / slight above", MID,
         "Camera is roughly at subject eye-level."),
        ("Looking DOWN at the object", HIGH,
         "Camera is ABOVE the object (cam_z &gt; obj_z). "
         "Current code stores this as <b>positive</b> score_cam_to_obj_elevation_deg."),
    ]

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:32px auto;max-width:980px;background:#fafafa;color:#222;line-height:1.55;}
    h1{font-size:24px;margin:0 0 4px;}
    h2{font-size:18px;margin:32px 0 8px;padding-top:12px;border-top:1px solid #ddd;}
    h3{font-size:15px;margin:18px 0 6px;}
    code{background:#eee;padding:1px 5px;border-radius:3px;font-size:13px;}
    pre{background:#272822;color:#f8f8f2;padding:12px 14px;border-radius:6px;
        overflow-x:auto;font-size:12px;line-height:1.45;}
    table{border-collapse:collapse;font-size:13px;margin:10px 0;}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top;}
    th{background:#eee;}
    .meta{color:#555;font-size:12px;margin-bottom:18px;}
    .card{display:flex;gap:16px;background:white;border:1px solid #ddd;
          border-radius:6px;padding:14px;margin:12px 0;}
    .card img{width:320px;height:auto;border-radius:4px;display:block;flex-shrink:0;}
    .card .body{font-size:13px;}
    .pill{display:inline-block;padding:1px 8px;border-radius:10px;color:white;
          font-size:11px;font-weight:600;}
    .ok{background:#2ca02c;}.bad{background:#d62728;}.neutral{background:#888;}
    .callout{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;
             border-radius:4px;margin:14px 0;font-size:13px;}
    .danger{background:#fde8e8;border-left:4px solid #d62728;}
    .good{background:#e8f5e8;border-left:4px solid #2ca02c;}
    """

    parts = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<title>cam_to_obj_* sign convention audit</title>")
    parts.append(f"<style>{css}</style></head><body>")

    parts.append("<h1>Audit: <code>cam_to_obj_{azimuth,elevation}_deg</code> sign convention</h1>")
    parts.append("<div class='meta'>Question raised: should <code>cam_to_obj_elevation_deg = -90</code> "
                 "mean &ldquo;top-down view&rdquo; (i.e., interpret <code>cam_to_obj</code> literally as the cam&rarr;obj vector, "
                 "which points <i>down</i> when the camera is above)?</div>")

    # ---------- TL;DR ----------
    parts.append("<h2>TL;DR</h2>")
    parts.append("""
    <ol>
      <li>The <b>code, every downstream consumer, and all real annotation data</b> use the convention
          <b>+90 = camera above (top-down view)</b>, <b>-90 = camera below (bottom-up view)</b>.
          Internally consistent — no bug.</li>
      <li>The recent commit on <code>main</code> (<code>ef7d0e5</code>) just brought the prompt text
          <i>into agreement</i> with what the math and data have always said. That part is correct.</li>
      <li>The user&rsquo;s separate concern is a <b>naming-convention</b> issue: literally,
          &ldquo;cam_to_obj&rdquo; means the vector pointing from cam to obj, and the elevation
          of <i>that</i> vector would be <b>-90 when the camera is above</b>. The current code uses
          the opposite vector (obj&rarr;cam) under a misleading name.</li>
      <li>Changing the data sign to match the literal naming would <b>invalidate the integer scores
          in 17K+ rendered v5 placements + the in-progress v6 dataset</b>, and any model already trained
          on these. The cheapest fix preserving semantics is to <b>rename the field</b>, not flip the math.</li>
    </ol>
    """)

    # ---------- The math ----------
    parts.append("<h2>What the code computes</h2>")
    parts.append("<p><b>Source of truth</b>: "
                 "<code>render_object_v3.py:102</code> (and duplicate at <code>:141</code>)</p>")
    parts.append("""
    <pre>d_world = cam_pos - obj_pos          # vector from object TO camera (obj&rarr;cam)
d_local = inv_rot @ d_world          # expressed in object-local frame
elevation_deg = degrees(atan2(d_local.z, sqrt(d_local.x**2 + d_local.y**2)))
azimuth_deg   = degrees(atan2(d_local.y, d_local.x)) % 360</pre>
    """)
    parts.append("""
    <p>So elevation is the angle of the <b>object&rarr;camera</b> vector above the horizon, even
    though the field is named <code>cam_to_obj_*</code>. This is the heart of the naming/math mismatch.</p>
    """)

    # truth tables
    parts.append("<h3>Elevation truth table</h3>")
    parts.append("""
    <table>
      <tr><th>Camera is&hellip;</th><th>(cam-obj).z</th>
          <th>Current code score</th><th>Literal &ldquo;cam&rarr;obj&rdquo; elevation</th></tr>
      <tr><td>directly above</td><td>+</td>
          <td>+90 (top-down)</td><td>-90 (vector points down)</td></tr>
      <tr><td>eye-level</td><td>0</td><td>0</td><td>0</td></tr>
      <tr><td>directly below</td><td>-</td>
          <td>-90 (bottom-up)</td><td>+90 (vector points up)</td></tr>
    </table>
    <p>The two conventions are <b>perfect mirrors</b> &mdash; both are valid; both lead to a
    well-defined model. The codebase happens to use the first.</p>
    """)

    # ---------- Empirical verification ----------
    parts.append("<h2>Empirical verification (real data)</h2>")
    parts.append("<p>Sampled rows from "
                 "<code>outputs/smoke_v6_pitch_lerp/p0_*/annotations.json</code> (100 rows from the "
                 "approved pitch-lerp smoke). Three representative cases, with the actual rendered "
                 "images embedded:</p>")
    for title, row, blurb in samples:
        img_uri = embed(SMOKE_DIR / row["image"])
        cam = row["camera_position"]
        obj = row["object_position"]
        dz = cam[2] - obj[2]
        score = row["score_cam_to_obj_elevation_deg"]
        raw = row["elevation_deg"]
        # Color the score pill
        if score > 30:
            pill = "<span class='pill ok'>cam ABOVE</span>"
        elif score < -30:
            pill = "<span class='pill bad'>cam BELOW</span>"
        else:
            pill = "<span class='pill neutral'>eye-level&ish</span>"
        parts.append(
            f"<div class='card'>"
            f"<img src='{img_uri}' alt='{escape(row['image'])}'>"
            f"<div class='body'>"
            f"<h3>{escape(title)} {pill}</h3>"
            f"<p>{blurb}</p>"
            f"<table>"
            f"<tr><th>image</th><td><code>{escape(row['image'])}</code></td></tr>"
            f"<tr><th>cam_pos.z</th><td>{cam[2]:.2f}</td></tr>"
            f"<tr><th>obj_pos.z</th><td>{obj[2]:.2f}</td></tr>"
            f"<tr><th>&Delta;z = cam.z &minus; obj.z</th><td><b>{dz:+.2f} m</b></td></tr>"
            f"<tr><th>raw elevation_deg</th><td>{raw:.2f}&deg;</td></tr>"
            f"<tr><th>score_cam_to_obj_elevation_deg</th><td><b>{score:+d}</b></td></tr>"
            f"</table>"
            f"</div></div>"
        )
    parts.append("""
    <p><b>Pattern</b>: every row where the camera is above the object has a <i>positive</i> score;
    every row where the camera is below has a <i>negative</i> score. The visual content of each
    image agrees with the score (looking down vs looking up at the subject).
    Cross-checked against 15 rows in older v5 data &mdash; same pattern, 15/15 agreement.</p>
    """)

    # ---------- Code path audit ----------
    parts.append("<h2>Code path audit &mdash; does anything depend on the current sign?</h2>")
    parts.append("""
    <table>
      <tr><th>File</th><th>What it does with these scores</th><th>Sign-sensitive?</th></tr>
      <tr><td><code>render_object_v3.py:102, 141</code></td>
          <td>Computes <code>elevation_deg = atan2((cam&minus;obj).z, horiz)</code>.</td>
          <td>Defines the convention.</td></tr>
      <tr><td><code>src/scoring/bbox_control.py:160</code></td>
          <td>Clamps + rounds the value: <code>max(-90, min(90, int(round(elevation_deg))))</code>.</td>
          <td>No transform.</td></tr>
      <tr><td><code>src/scoring/evaluator.py:36-43</code></td>
          <td><code>normalize_score_value</code> for V5 keys is <code>int(round(value))</code>.</td>
          <td>No transform.</td></tr>
      <tr><td><code>src/vlm_qwen25/dataset.py:215-220</code></td>
          <td>Loads <code>score_cam_to_obj_*</code> from annotation as raw int into the target dict.</td>
          <td>No transform.</td></tr>
      <tr><td><code>src/vlm_qwen25/schema.py:18-31</code> + <code>collator.py</code></td>
          <td>Formats integer verbatim into the JSON label the VLM learns to emit.</td>
          <td>No transform.</td></tr>
      <tr><td><code>src/vlm_qwen25/prompt.py:13</code></td>
          <td>Just the description text the model reads. Recently corrected on <code>main</code>.</td>
          <td>No math; only meaning.</td></tr>
      <tr><td><code>src/vlm_qwen25/mpc.py</code>, <code>objective.py</code></td>
          <td>Treats these as observation scores (read predicted, weight in objective). Not used as
              control targets.</td>
          <td>No transform.</td></tr>
      <tr><td><b><code>src/blender/camera.py:21</code></b></td>
          <td><b><code>cam_z = target.z + ... + radius * sin(elevation_deg)</code></b> &mdash;
              the <i>inverse</i> mapping for placing a camera given a target elevation.</td>
          <td><b>Already agrees with current convention</b>: <code>sin(+90)=+1</code> &rarr; cam
              placed ABOVE. <code>sin(&minus;90)=&minus;1</code> &rarr; cam placed BELOW.</td></tr>
      <tr><td><code>tests/test_v5_scores.py:87-91</code></td>
          <td>Tests pin <code>+95 &rarr; 90</code> and <code>-120 &rarr; -90</code> after clamping.</td>
          <td>No sign manipulation.</td></tr>
    </table>
    <div class="callout">
      <b>Most important finding</b>: <code>src/blender/camera.py:21</code> uses
      <code>radius * sin(elev)</code> to place the camera height when given a target elevation.
      Because <code>sin(+90)=+1</code>, <b>target +90 places the camera above</b> &mdash;
      i.e., the inverse mapping <b>also</b> uses the &ldquo;+90 = cam above&rdquo; convention.
      The convention is consistent in both directions across the codebase.
    </div>
    """)

    # ---------- Options ----------
    parts.append("<h2>Three options</h2>")
    parts.append("""
    <h3>Option A &mdash; keep current data, keep current name, just trust the description (status quo after <code>ef7d0e5</code>)</h3>
    <table>
      <tr><th>Cost</th><td>&euro;0 &mdash; already done. The prompt now matches what the data and code have always said.</td></tr>
      <tr><th>Pro</th><td>Zero rework. All existing v5/v6 annotations and trained models stay valid.</td></tr>
      <tr><th>Con</th><td>The field name <code>cam_to_obj_elevation_deg</code> reads as if the
          elevation is of the cam&rarr;obj vector, but the value is actually the opposite vector&rsquo;s
          elevation. Whoever reads the schema cold could misinterpret.</td></tr>
    </table>

    <h3>Option B &mdash; rename the field (e.g. <code>camera_elevation_above_obj_deg</code>)</h3>
    <table>
      <tr><th>Cost</th><td>Medium. Touches <code>src/scoring/bbox_control.py</code> (V5_SCORE_KEYS),
          <code>render_object_v3.py</code> (annotation dict key), all <code>configs/*.yaml</code> that
          reference the key, all <code>src/vlm_qwen25/*</code> prompt/schema/dataset code, eval code,
          and downstream consumers. <b>No data change</b>: just the dict key string changes.</td></tr>
      <tr><th>Pro</th><td>Eliminates the confusing name. Future readers correctly interpret the value.</td></tr>
      <tr><th>Con</th><td>Breaks compatibility with existing annotation JSON files that use the old key
          name (would need a backwards-compat alias or a one-time migration). Existing trained models
          would still produce correct outputs but in a JSON schema with the old key name &mdash; may
          need fine-tuning step to adopt the new key.</td></tr>
    </table>

    <h3>Option C &mdash; flip the math to match the literal &ldquo;cam_to_obj&rdquo; reading</h3>
    <table>
      <tr><th>Cost</th><td><b>High.</b> Must (a) flip the sign in
          <code>render_object_v3.py:102,141</code>, (b) flip
          <code>src/blender/camera.py:21</code> from <code>+sin(elev)</code> to <code>&minus;sin(elev)</code>
          to keep the inverse consistent, (c) <b>re-render or post-process all existing annotations</b>
          (multiply <code>score_cam_to_obj_elevation_deg</code> by &minus;1, ditto for azimuth), and
          (d) <b>retrain any model that learned the current sign</b> (otherwise its predictions are
          now inverted relative to ground truth).</td></tr>
      <tr><th>Pro</th><td>The name finally matches the math.</td></tr>
      <tr><th>Con</th><td>Highest blast radius. Risks introducing new bugs in any consumer that wasn&rsquo;t
          fully audited. Invalidates existing checkpoints.</td></tr>
    </table>
    """)

    # ---------- Recommendation ----------
    parts.append("<h2>Recommendation</h2>")
    parts.append("""
    <div class="callout good">
      Stay with <b>Option A</b> for now (current state on <code>main</code> already), and
      consider <b>Option B (rename)</b> as a cleanup PR when convenient. The rename is the only
      change that actually addresses the user&rsquo;s concern (semantic clarity) without invalidating
      existing data or models.
    </div>
    <p><b>Do NOT take Option C</b> unless retraining the v5/v6 models from scratch is already on
    the roadmap. The cost/benefit is heavily negative: we&rsquo;d invalidate every annotation in
    <code>data/vlm_object_placing*</code> plus the smoke renders, and any partially-trained
    checkpoint, just to make the name read literally. Better to fix the name.</p>
    """)

    parts.append("<h2>Summary</h2>")
    parts.append("""
    <ul>
      <li>The recent main commit (<code>ef7d0e5</code>) is correct in saying "+90 = camera above".</li>
      <li>The colleague&rsquo;s original complaint that "+90 doesn&rsquo;t match the cam_to_obj name"
          is also legitimate &mdash; but it&rsquo;s a naming bug, not a math bug.</li>
      <li>Internal consistency is preserved (forward calc, inverse placement, all consumers agree).</li>
      <li>Best forward path: <b>keep the math, optionally rename the field</b>.</li>
    </ul>
    """)

    parts.append("</body></html>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
