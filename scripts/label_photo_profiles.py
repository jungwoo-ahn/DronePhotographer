#!/usr/bin/env python3
"""Label rendered images with photo profile attributes using Gemini 2.5 Flash via Staix Gateway.

Uses OpenAI SDK with Staix base_url. Async batch processing with concurrency control.
Results are merged into annotations.json with photo_profile_ prefix.

Usage:
    export LETSUR_API_KEY="your_key"
    python scripts/label_photo_profiles.py \
        --annotations_path outputs/.../annotations.json \
        --image_root outputs/.../ \
        --concurrency 10 \
        --limit 5
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError

PHOTO_PROFILE_KEYS = [
    "rule_of_thirds", "centeredness", "symmetry", "leading_lines", "negative_space",
    "shot_close_up", "shot_medium", "shot_full", "shot_wide",
    "angle_low", "angle_high", "angle_eye_level",
    "lighting_front", "lighting_back", "lighting_side", "lighting_top", "lighting_ambient",
    "visibility", "saliency",
]

PROMPT = """You are a photography composition analyst. Analyze this drone photograph of {SUBJECT}.

Rate ALL of the following on a scale of 1 to 10 (integer only).

Composition:
- rule_of_thirds: Subject alignment with rule-of-thirds grid (1=far off, 10=perfect intersection)
- centeredness: How centered the subject is (1=edge, 10=dead center)
- symmetry: Vertical axis symmetry of overall composition (1=asymmetric, 10=perfect)
- leading_lines: Environmental lines guiding eye to subject (1=none, 10=strong convergence)
- negative_space: Effective use of empty space (1=cluttered, 10=clean purposeful)

Shot size (rate how much this image matches each type):
- shot_close_up: Face/upper body fills frame (1=not at all, 10=clearly close-up)
- shot_medium: Waist-up framing (1=not at all, 10=clearly medium)
- shot_full: Full body visible head to toe (1=not at all, 10=clearly full shot)
- shot_wide: Subject small in environment (1=not at all, 10=clearly wide)

Camera angle (rate how much this image matches each angle):
- angle_low: Camera below subject looking up (1=not at all, 10=clearly low angle)
- angle_high: Camera above subject looking down (1=not at all, 10=clearly high angle)
- angle_eye_level: Camera at subject's eye height (1=not at all, 10=clearly eye level)

Lighting (rate intensity from each direction):
- lighting_front: Light hitting subject from camera direction (1=none, 10=strong)
- lighting_back: Backlight/rim light behind subject (1=none, 10=strong)
- lighting_side: Light from left or right side (1=none, 10=strong)
- lighting_top: Overhead/downward light (1=none, 10=strong)
- lighting_ambient: Soft diffused light with no clear direction (1=none, 10=strong)

Perceptual:
- visibility: Subject clarity and visibility (1=mostly hidden/tiny, 10=fully visible)
- saliency: How strongly subject draws attention vs background (1=blends in, 10=pops out)

Return a single JSON object with exactly these 19 keys, all integer values 1-10."""


def parse_args():
    p = argparse.ArgumentParser(description="Label images with photo profile attributes via Gemini.")
    p.add_argument("--annotations_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="process only first N unlabeled images (0=all)")
    p.add_argument("--subject", default="a person", help="subject description for the prompt")
    p.add_argument("--overwrite", action="store_true", help="re-label images that already have photo_profile_* fields")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--api_key_env", default="LETSUR_API_KEY")
    p.add_argument("--base_url", default="https://gateway.letsur.ai/v1")
    return p.parse_args()


def has_photo_profile(entry):
    return any(k.startswith("photo_profile_") for k in entry)


def image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_response(text):
    """Extract and validate JSON from model response."""
    # Try to find JSON in the response
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)

    result = {}
    for key in PHOTO_PROFILE_KEYS:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing key: {key}")
        val = int(val)
        if not 1 <= val <= 10:
            raise ValueError(f"Value out of range for {key}: {val}")
        result[key] = val

    return result


async def label_one(client, entry, image_root, model, semaphore, idx, total, prompt):
    """Label a single image. Returns (index, result_dict) or (index, None) on failure."""
    image_path = Path(image_root) / entry["image"]
    if not image_path.exists():
        print(f"  [{idx+1}/{total}] SKIP (file missing): {entry['image']}")
        return idx, None

    data_url = image_to_data_url(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]

    async with semaphore:
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                text = resp.choices[0].message.content
                try:
                    result = parse_response(text)
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt < 4:
                        print(f"  [{idx+1}/{total}] Parse error (attempt {attempt+1}): {e}")
                        print(f"    Raw response: {text[:200]}")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print(f"  [{idx+1}/{total}] FAIL after 5 attempts: {e}")
                        print(f"    Last response: {text[:300]}")
                        return idx, None

                print(f"  [{idx+1}/{total}] OK: {entry['image']}")
                return idx, result

            except RateLimitError:
                wait = 2 ** attempt
                print(f"  [{idx+1}/{total}] Rate limited, retry in {wait}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  [{idx+1}/{total}] Error: {type(e).__name__}: {e}")
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return idx, None

    return idx, None


def save_annotations(annotations, path):
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


async def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Error: {args.api_key_env} environment variable not set.")
        sys.exit(1)

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)

    annotations_path = Path(args.annotations_path)
    annotations = json.loads(annotations_path.read_text())
    image_root = Path(args.image_root)

    # Build prompt with subject
    prompt = PROMPT.replace("{SUBJECT}", args.subject)
    print(f"Subject: {args.subject}")

    # Find entries to label
    if args.overwrite:
        to_label = [(i, e) for i, e in enumerate(annotations)]
    else:
        to_label = [(i, e) for i, e in enumerate(annotations) if not has_photo_profile(e)]
    if args.limit > 0:
        to_label = to_label[:args.limit]

    print(f"Total: {len(annotations)}, unlabeled: {len(to_label)}, model: {args.model}")
    if not to_label:
        print("Nothing to label.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.time()
    done_count = 0
    fail_count = 0

    # Process in batches for periodic saving
    batch_size = args.save_every
    for batch_start in range(0, len(to_label), batch_size):
        batch = to_label[batch_start:batch_start + batch_size]

        tasks = [
            label_one(client, annotations[ann_idx], image_root, args.model, semaphore, batch_start + j, len(to_label), prompt)
            for j, (ann_idx, _) in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)

        for j, (_, result) in enumerate(results):
            ann_idx = batch[j][0]
            if result is not None:
                for key, val in result.items():
                    annotations[ann_idx][f"photo_profile_{key}"] = val
                done_count += 1
            else:
                fail_count += 1

        # Save after each batch
        save_annotations(annotations, annotations_path)
        elapsed = time.time() - start
        print(f"  Saved. {done_count + fail_count}/{len(to_label)} processed, {fail_count} failed. [{elapsed:.0f}s]")

    elapsed = time.time() - start
    print(f"\nDone. {done_count} labeled, {fail_count} failed. [{elapsed:.0f}s]")
    print(f"Saved to: {annotations_path}")


if __name__ == "__main__":
    asyncio.run(main())


"""
export LETSUR_API_KEY="your_key"
python scripts/label_photo_profiles.py \
    --annotations_path outputs/multi_260325_125640/p23_wood-grid-fence-with-ivy_rp_posedplus_00068_18_100k/annotations.json \
    --image_root outputs/multi_260325_125640/p23_wood-grid-fence-with-ivy_rp_posedplus_00068_18_100k/ \
    --concurrency 10 \
    --overwrite \
    --subject "a person" \
    --limit 5
"""
