#!/usr/bin/env python3
"""Few-shot image scoring with Gemini 2.5 Flash via Letsur gateway.

Each API call contains 3 user-provided example images with their scores (as
alternating user/assistant turns) followed by the target image. Scores cover
four keys: rule_of_thirds, centeredness, breathing_space, symmetry.

Usage:
    export LETSUR_API_KEY="your_key"
    python scripts/score_fewshot.py \
        --annotations_path outputs/Namaqualand_namaqualand_v3_260401_024633/annotations.json \
        --image_root outputs/Namaqualand_namaqualand_v3_260401_024633/ \
        --output_path outputs/Namaqualand_namaqualand_v3_260401_024633/fewshot_scores.json \
        --limit 100 \
        --concurrency 10
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

FEWSHOT_KEYS = ["rule_of_thirds", "centeredness", "breathing_space", "symmetry"]

FEWSHOT_EXAMPLES = [
    {"image": "images/img_0000.png", "scores": {"rule_of_thirds": 3, "centeredness": 2, "breathing_space": 9, "symmetry": 2}},
    {"image": "images/img_0022.png", "scores": {"rule_of_thirds": 5, "centeredness": 9, "breathing_space": 4, "symmetry": 7}},
    {"image": "images/img_0026.png", "scores": {"rule_of_thirds": 8, "centeredness": 6, "breathing_space": 7, "symmetry": 4}},
]

SYSTEM_PROMPT = """You are a drone photography analyst. Score each image on 4 criteria, each as an integer 1-10.

- rule_of_thirds: Subject aligned with thirds-grid intersections or lines. 1 = placement ignores the grid entirely; 5 = some awareness of frame regions but tension unresolved; 10 = subject locked to a thirds line, placement feels inevitable.
- centeredness: How deliberately the subject sits at the exact center of the frame. 1 = subject drifted toward an edge with no intention; 5 = roughly central but incidental; 10 = subject at the exact heart of the frame with the world arranged around it.
- breathing_space: Lead room / looking room — empty space in the direction the subject is facing or moving toward (NOT general negative space). 1 = subject pushed against the edge they face, no room ahead of them; 5 = modest lead room; 10 = ample open space in the direction of gaze/movement.
- symmetry: Equilibrium between the two halves of the frame in tone, shape, and spatial weight. 1 = one side dominates entirely; 5 = rough balance but diverging in unresolved ways; 10 = near-perfect mirror.

You will be shown 3 example images with their reference scores, then asked to score a new image using the same rubric and reference scale. Output ONLY a JSON object with exactly these 4 keys. No explanation, no code fence, no extra text."""

USER_TEXT = "Score this image."


def image_to_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_fewshot_messages(image_root: Path):
    """Build the static prefix of messages (system + 3 few-shot turns)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOT_EXAMPLES:
        img_path = image_root / ex["image"]
        if not img_path.exists():
            raise FileNotFoundError(f"Few-shot image not found: {img_path}")
        data_url = image_to_data_url(img_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["scores"]),
        })
    return messages


def parse_response(text: str) -> dict:
    text = text.strip()
    json_match = re.search(r'\{[^{}]*\}', text)
    if not json_match:
        raise ValueError("No JSON found in response")
    data = json.loads(json_match.group())
    result = {}
    for key in FEWSHOT_KEYS:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing key: {key}")
        val = int(val)
        if not 1 <= val <= 10:
            raise ValueError(f"Value out of range for {key}: {val}")
        result[key] = val
    return result


async def score_one(client, image_rel_path, image_root, model, semaphore, prefix_messages, idx, total):
    image_path = image_root / image_rel_path
    if not image_path.exists():
        print(f"  [{idx+1}/{total}] SKIP (file missing): {image_rel_path}")
        return image_rel_path, None

    data_url = image_to_data_url(image_path)
    target_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": USER_TEXT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    messages = prefix_messages + [target_turn]

    async with semaphore:
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content
                try:
                    scores = parse_response(text)
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt < 4:
                        print(f"  [{idx+1}/{total}] Parse error (attempt {attempt+1}): {e}")
                        print(f"    Raw response: {text[:200]}")
                        await asyncio.sleep(1)
                        continue
                    print(f"  [{idx+1}/{total}] FAIL after 5 attempts: {e}")
                    print(f"    Last response: {text[:300]}")
                    return image_rel_path, None

                print(f"  [{idx+1}/{total}] OK: {image_rel_path} -> {scores}")
                return image_rel_path, scores

            except RateLimitError:
                wait = 2 ** attempt
                print(f"  [{idx+1}/{total}] Rate limited, retry in {wait}s...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  [{idx+1}/{total}] Error: {type(e).__name__}: {e}")
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return image_rel_path, None

    return image_rel_path, None


def load_existing_output(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if "scores" in data and "fewshot_examples" in data:
                return data
        except Exception as e:
            print(f"Warning: could not parse existing {path}: {e}. Starting fresh.")
    return {
        "model": None,
        "fewshot_examples": FEWSHOT_EXAMPLES,
        "scores": {},
        "failed": [],
    }


def save_output(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot image scoring via Gemini.")
    p.add_argument("--annotations_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="process only first N entries from annotations (0=all)")
    p.add_argument("--overwrite", action="store_true", help="re-score images already present in output")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--api_key_env", default="LETSUR_API_KEY")
    p.add_argument("--base_url", default="https://gateway.letsur.ai/v1")
    return p.parse_args()


async def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Error: {args.api_key_env} environment variable not set.")
        sys.exit(1)

    annotations_path = Path(args.annotations_path)
    image_root = Path(args.image_root)
    output_path = Path(args.output_path)

    annotations = json.loads(annotations_path.read_text())
    if args.limit > 0:
        annotations = annotations[:args.limit]

    output = load_existing_output(output_path)
    output["model"] = args.model
    output["fewshot_examples"] = FEWSHOT_EXAMPLES

    existing_scores = output["scores"]
    fewshot_image_set = {ex["image"] for ex in FEWSHOT_EXAMPLES}

    to_score = []
    for entry in annotations:
        image_rel = entry["image"]
        if image_rel in fewshot_image_set:
            continue
        if not args.overwrite and image_rel in existing_scores:
            continue
        to_score.append(image_rel)

    print(f"Annotations considered: {len(annotations)}")
    print(f"Few-shot examples excluded: {len(fewshot_image_set)}")
    print(f"Already scored (skipped): {sum(1 for e in annotations if e['image'] in existing_scores and e['image'] not in fewshot_image_set)}")
    print(f"To score: {len(to_score)} (model: {args.model}, concurrency: {args.concurrency})")
    if not to_score:
        print("Nothing to score.")
        save_output(output, output_path)
        return

    prefix_messages = build_fewshot_messages(image_root)
    print(f"Few-shot prefix built: {len(prefix_messages)} messages (system + 3 examples)")

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    start = time.time()
    done_count = 0
    fail_count = 0

    batch_size = args.save_every
    for batch_start in range(0, len(to_score), batch_size):
        batch = to_score[batch_start:batch_start + batch_size]
        tasks = [
            score_one(client, rel, image_root, args.model, semaphore, prefix_messages,
                      batch_start + j, len(to_score))
            for j, rel in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)

        for rel, scores in results:
            if scores is not None:
                output["scores"][rel] = scores
                if rel in output["failed"]:
                    output["failed"].remove(rel)
                done_count += 1
            else:
                if rel not in output["failed"]:
                    output["failed"].append(rel)
                fail_count += 1

        save_output(output, output_path)
        elapsed = time.time() - start
        print(f"  Saved. {done_count + fail_count}/{len(to_score)} processed, {fail_count} failed. [{elapsed:.0f}s]")

    elapsed = time.time() - start
    print(f"\nDone. {done_count} scored, {fail_count} failed. [{elapsed:.0f}s]")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
