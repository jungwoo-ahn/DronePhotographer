#!/usr/bin/env python3
"""Generate MPC target_json from natural language shot descriptions using an LLM.

Targets the v6 (V5_SCORE_KEYS) trained model: occupancy / body_in_frame_ratio /
cam_to_obj_azimuth_deg / cam_to_obj_elevation_deg / object_center_x / object_center_y
/ bbox_x_offset / bbox_y_offset.

Adapted from the earlier `generate_target.py` (which targeted the older
with_c2o 6D-vector schema) by replacing the camera 6D vectors with the simpler
2-angle representation that v6 uses, and dropping aspect_ratio (no v5 equivalent).

Usage:
    python scripts/generate_target.py "left face from above, wide shot"
    python scripts/generate_target.py --backend claude-cli "front centered close-up"
    python scripts/generate_target.py --generate-script "over shoulder follow shot"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SYSTEM_PROMPT = textwrap.dedent("""\
You are a drone cinematography expert that converts natural language shot
descriptions into precise numerical target parameters for a Model Predictive
Control (MPC) camera system.

## Project context

A drone camera moves through a 3D scene to photograph a human subject.
The trained model predicts 8 scores per frame; the MPC optimizer iteratively
adjusts the camera pose to minimize error between predicted and target values.

## Output format

Return EXACTLY one JSON object. Keys you may emit (all optional except occupancy):

- "center_x" (float 0-1, default 0.5): horizontal position of subject bbox center
  in the frame (0 = left edge, 1 = right edge). The MPC compiler converts this
  to pixel coordinates using the actual render resolution.
- "center_y" (float 0-1, default 0.5): vertical position (0 = top, 1 = bottom).
- "occupancy" (float 0-1, REQUIRED): fraction of frame area the subject's bbox
  should occupy. The compiler converts this to 0-100 percent for the model.
- "body_in_frame_ratio" (float 0-1, default 1.0): fraction of the subject's
  full body inside the frame. 1.0 = fully visible; less = intentionally
  cropped. Compiler scales to 0-100.
- "cam_to_obj_azimuth_deg" (float 0-360): angle of the CAMERA AROUND the
  subject in the horizontal plane. The training convention encodes this as
  atan2(cam_y - obj_y, cam_x - obj_x), so the absolute direction depends on
  the scene's coordinate frame. Most scenes are noisy on this axis; OMIT this
  key unless the user explicitly asks for a specific compass-style direction.
- "cam_to_obj_elevation_deg" (float -90 to +90): camera height angle.
    - +90 → camera DIRECTLY ABOVE the subject (top-down / bird's eye)
    - +45 → high angle (drone hovering above)
    -   0 → eye-level
    - -45 → low angle (hero shot / looking up at subject)
    - -90 → camera below the subject (worm's eye)

## Shot size (occupancy) guide

- Extreme wide / establishing: 0.05 - 0.15
- Wide / full shot: 0.15 - 0.25
- Medium shot: 0.25 - 0.40
- Close-up: 0.40 - 0.55
- Extreme close-up: 0.55 - 0.70

## Framing (center_x, center_y)

- Centered: center_x=0.5, center_y=0.5
- Rule of thirds: center_x ∈ {0.333, 0.667}, center_y ∈ {0.333, 0.667}
- Lead room: offset center AWAY from the subject's gaze / motion direction.
  Typical low-angle hero: center_y ≈ 0.55-0.65 (subject sits in lower-frame).
  Typical high-angle aerial: center_y ≈ 0.4-0.5.

## Examples

User: "front eye-level, centered, medium shot"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.35,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":0}
```

User: "aerial top-down, centered, wide"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.2,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":85}
```

User: "low-angle hero shot, looking up at subject, medium"
```json
{"center_x":0.5,"center_y":0.6,"occupancy":0.4,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":-30}
```

User: "rule of thirds top-right, eye-level, wide shot"
```json
{"center_x":0.667,"center_y":0.333,"occupancy":0.2,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":0}
```

User: "tight portrait, slightly high angle, centered"
```json
{"center_x":0.5,"center_y":0.45,"occupancy":0.55,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":15}
```

User: "subject cropped (only torso visible), close-up"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.5,"body_in_frame_ratio":0.6,"cam_to_obj_elevation_deg":0}
```

User: "over shoulder, subject lead-room to the right"
```json
{"center_x":0.35,"center_y":0.5,"occupancy":0.35,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":-5}
```

User: "bird's eye view, subject small in lower-third"
```json
{"center_x":0.5,"center_y":0.667,"occupancy":0.15,"body_in_frame_ratio":1.0,"cam_to_obj_elevation_deg":80}
```

## Instructions

Given the user's shot description, output ONLY a single JSON object (no markdown,
no explanation). Omit keys you have no clear intent to constrain. Default
body_in_frame_ratio to 1.0 unless the user explicitly asks for cropping.
""")

# Score keys the v6 model emits and the LLM may target.
V6_TARGET_KEYS = {
    "center_x",
    "center_y",
    "occupancy",
    "body_in_frame_ratio",
    "cam_to_obj_azimuth_deg",
    "cam_to_obj_elevation_deg",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MPC target_json from natural language.")
    p.add_argument("description", nargs="?", help="Shot description (interactive if omitted)")
    p.add_argument("--backend", default="gateway", choices=["gateway", "claude-cli"],
                   help="LLM backend: 'gateway' (Letsur/OpenAI-compatible) or 'claude-cli'")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Model name for gateway backend")
    p.add_argument("--api_key_env", default="LETSUR_API_KEY")
    p.add_argument("--base_url", default="https://gateway.letsur.ai/v1")
    p.add_argument("--generate-script", action="store_true",
                   help="Also generate a shell script for infer_mpc_blender.py")
    p.add_argument("--run_dir", default="outputs/Namaqualand_namaqualand_v3_260331_054741")
    p.add_argument(
        "--model_path",
        default="/home/nas5/jungwooahn/projects/DronePhotographer/runs/"
                "20260514_122526_qwen35_vl_2b_1xh200_v5/checkpoints/checkpoint-26000",
    )
    p.add_argument("--config", default="configs/qwen35_vl_2b_1xh200_v5.yaml")
    return p.parse_args()


def call_gateway(description: str, args: argparse.Namespace) -> str:
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Error: {args.api_key_env} environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def call_claude_cli(description: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nUser request: {description}\n\nOutput ONLY the JSON object."
    result = subprocess.run(
        ["claude", "--print", "--model", "sonnet", "-p", prompt],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"claude CLI error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def extract_json(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"No JSON found in response:\n{text}")
    return json.loads(text[start:end + 1])


def validate_target(target: dict) -> dict:
    if "occupancy" not in target:
        raise ValueError("target must specify 'occupancy'")

    if not (0.0 < float(target["occupancy"]) <= 1.0):
        raise ValueError(f"occupancy out of range (0..1]: {target['occupancy']}")
    for k in ("center_x", "center_y"):
        if k in target and not (0.0 <= float(target[k]) <= 1.0):
            raise ValueError(f"{k} out of range [0..1]: {target[k]}")
    if "body_in_frame_ratio" in target:
        v = float(target["body_in_frame_ratio"])
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"body_in_frame_ratio out of range [0..1]: {v}")
    if "cam_to_obj_azimuth_deg" in target:
        target["cam_to_obj_azimuth_deg"] = float(target["cam_to_obj_azimuth_deg"]) % 360.0
    if "cam_to_obj_elevation_deg" in target:
        v = float(target["cam_to_obj_elevation_deg"])
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"cam_to_obj_elevation_deg out of range [-90, +90]: {v}")

    unknown = [k for k in target if k not in V6_TARGET_KEYS]
    if unknown:
        raise ValueError(
            f"unrecognized target keys for v6 schema: {unknown}. "
            f"Allowed: {sorted(V6_TARGET_KEYS)}"
        )
    return target


def generate_shell_script(description: str, target: dict, args: argparse.Namespace) -> tuple[str, str]:
    safe_name = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")[:40]
    target_json_str = json.dumps(target, separators=(",", ":"))

    script = textwrap.dedent(f"""\
    #!/usr/bin/env bash
    set -euo pipefail
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # {description}
    RUN_DIR="{args.run_dir}"
    MODEL_PATH="{args.model_path}"
    BLENDER_BIN="blender/blender"
    BLENDER_THREADS="${{BLENDER_THREADS:-4}}"
    CANDIDATE_BATCH_SIZE="${{CANDIDATE_BATCH_SIZE:-96}}"

    export OMP_NUM_THREADS="${{BLENDER_THREADS}}"
    export OPENBLAS_NUM_THREADS="${{BLENDER_THREADS}}"
    export MKL_NUM_THREADS="${{BLENDER_THREADS}}"

    CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}" python scripts/infer_mpc_blender.py \\
      --run_dir "${{RUN_DIR}}" \\
      --model_path "${{MODEL_PATH}}" \\
      --config {args.config} \\
      --blender_bin "${{BLENDER_BIN}}" \\
      --initial_seed 721 \\
      --num_steps 50 \\
      --translation_values_m=-0.2,-0.1,0,0.1,0.2 \\
      --rotation_values_deg=-5,0,5 \\
      --max_translation_norm_m 0.3 \\
      --max_rotation_norm_deg 7.5 \\
      --max_candidates 720 \\
      --candidate_batch_size "${{CANDIDATE_BATCH_SIZE}}" \\
      --max_new_tokens 256 \\
      --target_json '{target_json_str}'
    """)
    return script, safe_name


def main():
    args = parse_args()
    description = args.description or input("Shot description: ").strip()
    if not description:
        print("Error: empty description.", file=sys.stderr)
        sys.exit(1)

    if args.backend == "gateway":
        raw = call_gateway(description, args)
    else:
        raw = call_claude_cli(description)

    target = validate_target(extract_json(raw))
    print(json.dumps(target, indent=2))

    if args.generate_script:
        script, name = generate_shell_script(description, target, args)
        out_path = Path("scripts") / f"infer_mpc_{name}.sh"
        out_path.write_text(script)
        out_path.chmod(0o755)
        print(f"\nShell script written: {out_path}")


if __name__ == "__main__":
    main()
