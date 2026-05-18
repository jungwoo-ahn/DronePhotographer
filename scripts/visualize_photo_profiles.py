#!/usr/bin/env python3
"""Visualize photo profile labels as an HTML gallery with radar charts.

Filters out images where the subject is barely visible (low subject_visible/saliency).

Usage:
    python scripts/visualize_photo_profiles.py \
        --annotations_path outputs/.../annotations.json \
        --image_root outputs/.../ \
        --output gallery_photo_profiles.html \
        --min_visibility 3
"""

import argparse
import base64
import json
from pathlib import Path

PROFILE_KEYS = [
    "subject_visible", "saliency",
    "rule_of_thirds", "centeredness", "symmetry", "leading_lines", "negative_space",
    "shot_close_up", "shot_medium", "shot_full", "shot_wide", "shot_overhead",
    "angle_eye_level", "angle_high", "angle_low",
    "lighting_front", "lighting_back", "lighting_left", "lighting_right",
    "lighting_top", "lighting_bottom", "lighting_ambient",
]

GROUPS = {
    "Visibility": ["subject_visible", "saliency"],
    "Composition": ["rule_of_thirds", "centeredness", "symmetry", "leading_lines", "negative_space"],
    "Shot Size": ["shot_close_up", "shot_medium", "shot_full", "shot_wide", "shot_overhead"],
    "Camera Angle": ["angle_eye_level", "angle_high", "angle_low"],
    "Lighting": ["lighting_front", "lighting_back", "lighting_left", "lighting_right",
                  "lighting_top", "lighting_bottom", "lighting_ambient"],
}

GROUP_COLORS = {
    "Visibility": "#ba68c8",
    "Composition": "#4fc3f7",
    "Shot Size": "#81c784",
    "Camera Angle": "#ffb74d",
    "Lighting": "#e57373",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output", default="gallery_photo_profiles.html")
    p.add_argument("--min_visibility", type=int, default=3, help="minimum subject_visible score to include")
    p.add_argument("--min_saliency", type=int, default=3, help="minimum saliency score to include")
    p.add_argument("--sort_by", default="saliency", help="sort images by this photo_profile key (descending)")
    return p.parse_args()


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_profile(entry):
    return {k: entry.get(f"photo_profile_{k}", 0) for k in PROFILE_KEYS}


def make_bar_html(profile):
    """Generate grouped horizontal bar chart HTML."""
    html = ""
    for group_name, keys in GROUPS.items():
        color = GROUP_COLORS[group_name]
        html += f'<div class="group-label" style="color:{color}">{group_name}</div>'
        for k in keys:
            v = profile.get(k, 0)
            label = k.replace("_", " ")
            pct = v * 10
            html += f'''<div class="bar-row">
  <span class="bar-label">{label}</span>
  <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
  <span class="bar-val">{v}</span>
</div>'''
    return html


def build_html(entries, image_root, default_sort="subject_visible"):
    sort_options = "\n    ".join(
        f'<option value="{k}" {"selected" if k == default_sort else ""}>{k.replace("_"," ")}</option>'
        for k in PROFILE_KEYS
    )
    cards_html = ""
    for entry in entries:
        profile = get_profile(entry)
        img_path = Path(image_root) / entry["image"]
        if not img_path.exists():
            continue
        b64 = img_to_base64(img_path)
        bars = make_bar_html(profile)

        # Top tags
        shot = max(["shot_close_up", "shot_medium", "shot_full", "shot_wide", "shot_overhead"], key=lambda k: profile[k])
        angle = max(["angle_eye_level", "angle_high", "angle_low"], key=lambda k: profile[k])
        shot_label = shot.replace("shot_", "").replace("_", " ").title()
        angle_label = angle.replace("angle_", "").replace("_", " ").title()

        data_attrs = " ".join(f'data-pp-{k}="{profile[k]}"' for k in PROFILE_KEYS)
        cards_html += f'''<div class="card" {data_attrs}>
  <div class="img-wrap"><img src="data:image/png;base64,{b64}"></div>
  <div class="card-body">
    <div class="tags">
      <span class="tag shot">{shot_label}</span>
      <span class="tag angle">{angle_label}</span>
      <span class="tag vis">vis:{profile["subject_visible"]}</span>
      <span class="tag sal">sal:{profile["saliency"]}</span>
    </div>
    <div class="bars">{bars}</div>
    <div class="img-name">{entry["image"]}</div>
  </div>
</div>'''

    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo Profiles ({len(entries)} images)</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:20px}}
h1{{text-align:center;padding:15px 0;font-size:1.5em;color:#fff}}
.stats{{text-align:center;color:#888;margin-bottom:20px;font-size:.85em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;max-width:1800px;margin:0 auto}}
.card{{background:#1a1a2e;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.img-wrap{{width:100%;background:#000}}
.img-wrap img{{width:100%;height:auto;display:block}}
.card-body{{padding:10px 12px;flex:1;min-width:0}}
.tags{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}}
.tag{{font-size:.7em;padding:2px 6px;border-radius:4px;font-weight:600}}
.tag.shot{{background:#81c784;color:#000}}
.tag.angle{{background:#ffb74d;color:#000}}
.tag.vis{{background:#ba68c8;color:#fff}}
.tag.sal{{background:#4fc3f7;color:#000}}
.group-label{{font-size:.65em;font-weight:700;margin:6px 0 2px;text-transform:uppercase;letter-spacing:.5px}}
.bar-row{{display:flex;align-items:center;gap:4px;margin:1px 0}}
.bar-label{{font-size:.6em;width:70px;text-align:right;color:#aaa;flex-shrink:0}}
.bar-bg{{flex:1;height:8px;background:#222;border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width .3s}}
.bar-val{{font-size:.6em;width:16px;color:#888}}
.img-name{{font-size:.6em;color:#555;margin-top:6px}}
.sort-bar{{text-align:center;margin-bottom:16px}}
.sort-bar select{{background:#222;color:#ddd;border:1px solid #444;padding:6px 12px;border-radius:6px;font-size:.85em;cursor:pointer}}
.sort-bar label{{color:#888;font-size:.85em;margin-right:6px}}
</style></head><body>
<h1>Photo Profile Labels</h1>
<div class="stats">{len(entries)} images (filtered by visibility>={entries[0].get("_min_vis",3)}, saliency>={entries[0].get("_min_sal",3)})</div>
<div class="sort-bar">
  <label>Sort by:</label>
  <select id="sortKey" onchange="sortCards()">
    {sort_options}
  </select>
</div>
<div class="grid" id="grid">{cards_html}</div>
<script>
function sortCards() {{
  const key = document.getElementById('sortKey').value;
  const grid = document.getElementById('grid');
  const cards = Array.from(grid.children);
  cards.sort((a, b) => (parseInt(b.getAttribute('data-pp-'+key)) || 0) - (parseInt(a.getAttribute('data-pp-'+key)) || 0));
  cards.forEach(c => grid.appendChild(c));
}}
</script>
</body></html>'''


def main():
    args = parse_args()
    annotations = json.loads(Path(args.annotations_path).read_text())

    # Filter: must have labels + minimum visibility/saliency
    filtered = []
    for e in annotations:
        if not any(k.startswith("photo_profile_") for k in e):
            continue
        vis = e.get("photo_profile_subject_visible", e.get("photo_profile_visibility", 0))
        sal = e.get("photo_profile_saliency", 0)
        if vis >= args.min_visibility and sal >= args.min_saliency:
            e["_min_vis"] = args.min_visibility
            e["_min_sal"] = args.min_saliency
            filtered.append(e)

    # Sort
    sort_key = f"photo_profile_{args.sort_by}"
    filtered.sort(key=lambda e: e.get(sort_key, 0), reverse=True)

    print(f"Total: {len(annotations)}, after filter: {len(filtered)}")

    if not filtered:
        print("No images passed the filter.")
        return

    html = build_html(filtered, args.image_root, default_sort=args.sort_by)
    out = Path(args.output)
    out.write_text(html)
    print(f"Saved: {out} ({len(filtered)} images)")


if __name__ == "__main__":
    main()


"""
python scripts/visualize_photo_profiles.py \
    --annotations_path outputs/multi_260325_125640/p23_wood-grid-fence-with-ivy_rp_posedplus_00068_18_100k/annotations.json \
    --image_root outputs/multi_260325_125640/p23_wood-grid-fence-with-ivy_rp_posedplus_00068_18_100k/ \
    --output gallery_photo_profiles.html \
    --min_visibility 3 \
    --min_saliency 3
"""
