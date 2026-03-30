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
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError

PHOTO_PROFILE_KEYS = [
    "subject_visible", "saliency",
    "rule_of_thirds", "centeredness", "symmetry", "leading_lines", "negative_space",
    "shot_close_up", "shot_medium", "shot_full", "shot_wide", "shot_overhead",
    "angle_eye_level", "angle_high", "angle_low",
    "lighting_front", "lighting_back", "lighting_left", "lighting_right",
    "lighting_top", "lighting_bottom", "lighting_ambient",
]

PROMPT = """You are a drone photography analyst. You will analyze a rendered image that may or may not contain {SUBJECT}.

IMPORTANT: You must FIRST describe what you actually see, THEN score. Do not assume the subject is present — look carefully.

=== STEP 1: OBSERVATION (required) ===
Write 2-3 sentences describing:
1. Is {SUBJECT} actually visible in this image? If so, how much of them is visible (full body, partial, tiny speck, etc.)?
2. Where in the frame is the subject (center, edge, corner, not present)?
3. What is the apparent camera viewing angle relative to the subject (looking up at them, eye level, looking slightly down, looking steeply down, directly overhead)?

=== STEP 2: SCORING ===
Rate ALL keys below on a scale of 1 to 10 (integer only).
CRITICAL RULE: If the subject is NOT visible or barely visible (just a tiny edge/speck), then ALL subject-dependent scores (everything except negative_space and symmetry) MUST be 1-2.

--- Visibility & Saliency ---
VERIFICATION: Before rating subject_visible above 4, verify that you can describe what the subject is wearing AND identify at least one body part (head, arm, torso, leg). If you cannot answer both, set subject_visible to 1-3.
- subject_visible: Is the subject actually present and identifiable? (1=not visible at all / only a tiny sliver or edge, 3=visible but very small or mostly occluded, 5=visible but small in frame, 7=clearly visible at moderate size, 10=large and fully visible filling significant portion of frame)
- saliency: How strongly the subject draws visual attention vs the background (1=subject absent or blends completely, 5=noticeable but does not dominate, 10=immediately draws the eye)

--- Composition ---
- rule_of_thirds: A great photograph breathes — the subject occupies one region while the remaining space carries its own weight. (10=placement feels inevitable, moving the subject would upset the whole image, 5=some awareness of frame regions but tension unresolved, 1=placement feels purely accidental)
- centeredness: Centering done well is a deliberate declaration, not a default. (10=subject at exact heart of frame, world arranged around it, 5=roughly central but choice feels incidental, 1=subject drifted toward edge with no intention)
- symmetry: Symmetry is about equilibrium, not mathematics — the two halves in quiet conversation. (10=near-perfect mirror in tone, shape, and spatial weight, 5=rough balance but diverging in unresolved ways, 1=one side dominates entirely). Score based on whole image.
- leading_lines: The best images have a current that carries your eye to the subject before you realize it. (10=environment pointing — roads, shadows, rivers converging at subject, 5=lines exist but disperse before arriving, 1=eye has nowhere to go)
- negative_space: Empty space is not absence — it is pressure that makes the subject louder. (10=restraint intentional, emptiness clearly load-bearing, 5=open space exists but not purposefully working, 1=every corner competes and nothing wins). Can be scored even without a visible subject.

--- Shot Size (MUTUALLY EXCLUSIVE — rate ONLY ONE type as 7-10, others must be proportionally lower. Your highest score must be at least 3 points above any other shot type.) ---
- shot_close_up: Face or upper body fills most of the frame at large scale (1=subject small or absent, 10=face/upper body fills >50% of frame)
- shot_medium: Subject shown roughly waist-up, with environment context visible (1=does not match, 10=classic waist-up framing)
- shot_full: Subject's entire body visible head to toe with some space around (1=does not match, 10=textbook full-body head-to-toe framing)
- shot_wide: Subject is small relative to the environment, landscape dominates (1=subject fills frame, 10=subject is a small figure in a vast scene)
- shot_overhead: Camera is nearly directly above, looking straight down. Subject appears foreshortened/flattened (1=camera is level or slightly elevated, 5=clearly oblique ~45 degrees, 10=near-vertical bird's-eye directly above)

--- Camera Angle (based on GEOMETRIC camera-subject relationship, NOT visual impression of seeing ground/sky) ---
IMPORTANT: Seeing the ground in the background does NOT mean high angle. angle_high should be high ONLY if the subject appears vertically compressed/foreshortened (head looks larger relative to feet, body appears squished top-to-bottom). If the subject has normal human proportions (not vertically compressed), the camera is near eye level even if ground is visible behind them.
- angle_eye_level: Camera at approximately the same height as the subject (1=clearly above or below, 5=slightly off eye level, 10=exact eye level, horizon bisects subject)
- angle_high: Camera positioned ABOVE the subject, looking DOWN. You can see the top of subject's head. (1=at or below subject level, 3=slightly above 5-15 degrees, 5=moderately above 20-40 degrees, 8=steeply above 50-70 degrees, 10=nearly directly overhead)
- angle_low: Camera positioned BELOW the subject, looking UP. You can see under the subject's chin. (1=at or above subject level, 5=moderately below, 10=extreme low angle looking straight up)

--- Lighting (relative to the SUBJECT's body orientation, NOT the camera) ---
"Front" = subject's face/chest side. "Back" = subject's spine side. Left/right = subject's own left/right.
- lighting_front: Light illuminating the FRONT of the subject (face/chest side appears bright) (1=face side in shadow or subject not visible, 5=soft partial front light, 10=strong direct light on face)
- lighting_back: Light from BEHIND the subject creating rim light or silhouette on edges (1=no backlight or subject not visible, 5=subtle rim, 10=strong backlight with clear rim/silhouette)
- lighting_left: Light hitting the subject's LEFT side (may be viewer's right). That body side appears brighter. (1=none or subject not visible, 5=moderate, 10=strong)
- lighting_right: Light hitting the subject's RIGHT side. That body side appears brighter. (1=none or subject not visible, 5=moderate, 10=strong)
- lighting_top: Light from directly above illuminating top of head and shoulders. Rate 7+ ONLY if the top surfaces (head crown, shoulders) are noticeably brighter than side surfaces. Normal outdoor daylight is NOT strong top light — rate it 4-5. (1=no overhead light or subject not visible, 5=moderate, 10=strong overhead light with bright top surfaces clearly contrasting with sides)
- lighting_bottom: Light from below the subject (uplighting), illuminating under chin and body (1=no uplight or subject not visible, 5=moderate, 10=strong uplight)
- lighting_ambient: Soft diffused light with no single dominant direction, shadows are soft or absent (1=harsh directional light with strong shadows, 5=mix of directional and diffused, 10=completely even diffused light, no shadows)

=== OUTPUT FORMAT ===
First write your observation (2-3 sentences starting with "Observation:"), then output a JSON object with exactly these 22 keys, all integer values 1-10.

Example:
Observation: The subject is clearly visible standing in the center-right of the frame, shown from approximately waist up. The camera is at roughly eye level. Lighting comes primarily from the upper left.
```json
{"subject_visible": 8, "saliency": 7, "rule_of_thirds": 6, "centeredness": 5, "symmetry": 3, "leading_lines": 2, "negative_space": 6, "shot_close_up": 2, "shot_medium": 8, "shot_full": 3, "shot_wide": 1, "shot_overhead": 1, "angle_eye_level": 8, "angle_high": 2, "angle_low": 1, "lighting_front": 6, "lighting_back": 2, "lighting_left": 7, "lighting_right": 2, "lighting_top": 4, "lighting_bottom": 1, "lighting_ambient": 4}
```"""


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
    """Extract observation text and validated JSON scores from model response."""
    text = text.strip()

    # 1. Extract observation text
    observation = ""
    obs_match = re.search(r'Observation:\s*(.*?)(?=```|\{)', text, re.DOTALL)
    if obs_match:
        observation = obs_match.group(1).strip()

    # 2. Extract JSON (inside code fence or raw braces)
    json_match = re.search(r'\{[^{}]*\}', text)
    if not json_match:
        raise ValueError("No JSON found in response")

    data = json.loads(json_match.group())

    # 3. Validate all keys
    result = {}
    for key in PHOTO_PROFILE_KEYS:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing key: {key}")
        val = int(val)
        if not 1 <= val <= 10:
            raise ValueError(f"Value out of range for {key}: {val}")
        result[key] = val

    return result, observation


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
                    result, observation = parse_response(text)
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
                return idx, (result, observation)

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

        for j, (_, result_pair) in enumerate(results):
            ann_idx = batch[j][0]
            if result_pair is not None:
                result, observation = result_pair
                for key, val in result.items():
                    annotations[ann_idx][f"photo_profile_{key}"] = val
                if observation:
                    annotations[ann_idx]["photo_profile_observation"] = observation
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
