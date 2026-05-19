#!/usr/bin/env python3
"""Visualize fewshot_scores.json as a self-contained HTML gallery (inline base64 images).

Usage:
    python scripts/visualize_fewshot_scores.py \
        --scores_path outputs/Namaqualand_namaqualand_v3_260401_024633/fewshot_scores.json \
        --image_root outputs/Namaqualand_namaqualand_v3_260401_024633/ \
        --output notes/fewshot_scores_viz.html
"""

import argparse
import base64
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

FEWSHOT_KEYS = ["rule_of_thirds", "centeredness", "breathing_space", "symmetry"]

KEY_COLORS = {
    "rule_of_thirds": "#4fc3f7",
    "centeredness": "#ba68c8",
    "breathing_space": "#81c784",
    "symmetry": "#ffb74d",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output", default="notes/fewshot_scores_viz.html")
    p.add_argument("--sort_by", default="rule_of_thirds",
                   help=f"default sort key, one of: {', '.join(FEWSHOT_KEYS)}")
    p.add_argument("--thumb_width", type=int, default=512,
                   help="downscale images to this width (px) before inlining")
    p.add_argument("--jpeg_quality", type=int, default=80)
    return p.parse_args()


def img_to_base64(path: Path, max_width: int, quality: int) -> tuple[str, str]:
    """Return (mime_type, base64_string) — JPEG if downscaling, PNG passthrough otherwise."""
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        new_h = int(img.height * max_width / img.width)
        img = img.resize((max_width, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8")


def make_bars_html(scores: dict) -> str:
    html = ""
    for k in FEWSHOT_KEYS:
        v = scores.get(k, 0)
        label = k.replace("_", " ")
        pct = v * 10
        color = KEY_COLORS[k]
        html += (
            '<div class="bar-row">'
            f'<span class="bar-label">{label}</span>'
            '<div class="bar-bg">'
            f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div>'
            '</div>'
            f'<span class="bar-val">{v}</span>'
            '</div>'
        )
    return html


def make_card(rel_path: str, scores: dict, image_root: Path, is_fewshot: bool,
              thumb_width: int, jpeg_quality: int) -> str:
    img_path = image_root / rel_path
    if not img_path.exists():
        return ""
    mime, b64 = img_to_base64(img_path, thumb_width, jpeg_quality)
    bars = make_bars_html(scores)
    data_attrs = " ".join(f'data-{k}="{scores.get(k, 0)}"' for k in FEWSHOT_KEYS)
    fewshot_class = " fewshot" if is_fewshot else ""
    fewshot_badge = '<span class="fewshot-badge">REFERENCE</span>' if is_fewshot else ""
    return (
        f'<div class="card{fewshot_class}" {data_attrs}>'
        f'<div class="img-wrap">{fewshot_badge}<img src="data:{mime};base64,{b64}"></div>'
        '<div class="card-body">'
        f'<div class="bars">{bars}</div>'
        f'<div class="img-name">{rel_path}</div>'
        '</div>'
        '</div>'
    )


def compute_stats(scores_by_image: dict) -> dict:
    stats = {}
    for k in FEWSHOT_KEYS:
        vals = [s[k] for s in scores_by_image.values()]
        if not vals:
            continue
        stats[k] = {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "dist": dict(Counter(vals)),
        }
    return stats


def make_stats_html(stats: dict, n_total: int) -> str:
    rows = ""
    for k in FEWSHOT_KEYS:
        s = stats.get(k, {})
        if not s:
            continue
        color = KEY_COLORS[k]
        dist_bars = ""
        max_count = max(s["dist"].values())
        for score_val in range(1, 11):
            count = s["dist"].get(score_val, 0)
            h = int((count / max_count) * 30) if max_count else 0
            dist_bars += (
                f'<div class="hist-col">'
                f'<div class="hist-bar" style="height:{h}px;background:{color}"></div>'
                f'<div class="hist-label">{score_val}</div>'
                f'<div class="hist-count">{count}</div>'
                '</div>'
            )
        rows += (
            '<div class="stat-block">'
            f'<div class="stat-title" style="color:{color}">{k.replace("_", " ")}</div>'
            f'<div class="stat-meta">mean {s["mean"]:.2f}  ·  range {s["min"]}–{s["max"]}</div>'
            f'<div class="hist">{dist_bars}</div>'
            '</div>'
        )
    return (
        f'<div class="stats-box"><div class="stats-title">Score distribution across {n_total} images</div>'
        f'<div class="stats-grid">{rows}</div></div>'
    )


def build_html(data: dict, image_root: Path, default_sort: str,
               thumb_width: int, jpeg_quality: int) -> str:
    fewshot_examples = data.get("fewshot_examples", [])
    scores = data.get("scores", {})
    model = data.get("model", "unknown")

    fewshot_cards = []
    fewshot_paths = set()
    for ex in fewshot_examples:
        rel = ex["image"]
        fewshot_paths.add(rel)
        card = make_card(rel, ex["scores"], image_root, True, thumb_width, jpeg_quality)
        if card:
            fewshot_cards.append((rel, ex["scores"], card))

    target_items = [(rel, s) for rel, s in scores.items() if rel not in fewshot_paths]
    target_items.sort(key=lambda x: x[1].get(default_sort, 0), reverse=True)
    target_cards = [make_card(rel, s, image_root, False, thumb_width, jpeg_quality)
                    for rel, s in target_items]
    target_cards = [c for c in target_cards if c]

    stats = compute_stats({rel: s for rel, s in scores.items() if rel not in fewshot_paths})
    stats_html = make_stats_html(stats, len(target_items))

    sort_options = "".join(
        f'<option value="{k}"{" selected" if k == default_sort else ""}>{k.replace("_", " ")}</option>'
        for k in FEWSHOT_KEYS
    )

    fewshot_grid = "".join(c for _, _, c in fewshot_cards)
    target_grid = "".join(target_cards)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Few-shot scores ({len(target_items)} images)</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0f0f14;color:#ddd;padding:20px}}
h1{{text-align:center;padding:15px 0;font-size:1.6em;color:#fff;font-weight:600}}
h2{{font-size:.95em;font-weight:600;color:#bbb;margin:24px auto 12px;max-width:1800px;padding-left:4px;text-transform:uppercase;letter-spacing:1px}}
.meta{{text-align:center;color:#888;margin-bottom:8px;font-size:.85em}}
.stats-box{{max-width:1800px;margin:0 auto 24px;background:#151520;border-radius:10px;padding:16px}}
.stats-title{{font-size:.75em;font-weight:700;color:#aaa;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
.stat-block{{background:#1a1a28;border-radius:8px;padding:12px}}
.stat-title{{font-size:.85em;font-weight:700;text-transform:capitalize;margin-bottom:2px}}
.stat-meta{{font-size:.7em;color:#888;margin-bottom:8px}}
.hist{{display:flex;align-items:flex-end;gap:2px;height:60px}}
.hist-col{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;min-width:0}}
.hist-bar{{width:100%;border-radius:2px 2px 0 0;min-height:1px;opacity:.85}}
.hist-label{{font-size:.6em;color:#666;margin-top:2px}}
.hist-count{{font-size:.55em;color:#555}}
.sort-bar{{text-align:center;margin-bottom:16px}}
.sort-bar select{{background:#1a1a28;color:#ddd;border:1px solid #333;padding:6px 12px;border-radius:6px;font-size:.85em;cursor:pointer}}
.sort-bar label{{color:#888;font-size:.85em;margin-right:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;max-width:1800px;margin:0 auto}}
.card{{background:#1a1a28;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;border:1px solid #232336}}
.card.fewshot{{border:2px solid #ffd54f;box-shadow:0 0 0 1px rgba(255,213,79,.15)}}
.img-wrap{{position:relative;width:100%;background:#000}}
.img-wrap img{{width:100%;height:auto;display:block}}
.fewshot-badge{{position:absolute;top:8px;left:8px;background:#ffd54f;color:#000;font-size:.6em;font-weight:800;padding:3px 7px;border-radius:4px;letter-spacing:.5px}}
.card-body{{padding:10px 12px}}
.bar-row{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.bar-label{{font-size:.7em;width:110px;text-align:right;color:#aaa;flex-shrink:0;text-transform:capitalize}}
.bar-bg{{flex:1;height:10px;background:#0f0f17;border-radius:5px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:5px;transition:width .3s}}
.bar-val{{font-size:.72em;width:18px;color:#ccc;text-align:right;font-weight:600}}
.img-name{{font-size:.6em;color:#555;margin-top:6px;font-family:monospace}}
</style></head><body>
<h1>Few-shot image scoring</h1>
<div class="meta">model: {model}  ·  {len(target_items)} scored images + {len(fewshot_cards)} reference examples</div>

{stats_html}

<h2>Reference examples (user-provided scores)</h2>
<div class="grid">{fewshot_grid}</div>

<h2>Scored images (sort to re-order)</h2>
<div class="sort-bar">
  <label>Sort by (descending):</label>
  <select id="sortKey" onchange="sortCards()">{sort_options}</select>
</div>
<div class="grid" id="grid">{target_grid}</div>

<script>
function sortCards() {{
  const key = document.getElementById('sortKey').value;
  const grid = document.getElementById('grid');
  const cards = Array.from(grid.children);
  cards.sort((a, b) => (parseInt(b.getAttribute('data-'+key)) || 0) - (parseInt(a.getAttribute('data-'+key)) || 0));
  cards.forEach(c => grid.appendChild(c));
}}
</script>
</body></html>"""


def main():
    args = parse_args()
    data = json.loads(Path(args.scores_path).read_text())
    image_root = Path(args.image_root)

    html = build_html(data, image_root, args.sort_by, args.thumb_width, args.jpeg_quality)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_mb = out.stat().st_size / (1024 * 1024)
    n_fewshot = len(data.get("fewshot_examples", []))
    n_target = sum(1 for rel in data.get("scores", {}) if rel not in {e["image"] for e in data.get("fewshot_examples", [])})
    print(f"Saved: {out} ({size_mb:.1f} MB, {n_target} scored + {n_fewshot} references, thumb_width={args.thumb_width})")


if __name__ == "__main__":
    main()
