#!/usr/bin/env python3
"""Generate MPC target_json from natural language shot descriptions using an LLM.

Usage:
    python scripts/generate_target.py "left face from above, wide shot"
    python scripts/generate_target.py --backend claude-cli "정면 클로즈업"
    python scripts/generate_target.py --generate-script "over shoulder follow shot"
"""
from __future__ import annotations

import argparse
import json
import math
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
You are a drone cinematography expert that converts natural language shot descriptions
into precise numerical target parameters for a Model Predictive Control (MPC) camera system.

## Project context

A drone camera moves through a 3D scene to photograph a human subject.
The MPC optimizer iteratively adjusts the camera pose to minimize error between
the model's predicted scores and the target values you produce.

## Output format

Return EXACTLY one JSON object with these fields:
- Bbox composition (at least occupancy):
  - center_x (float 0-1, default 0.5): horizontal position of subject in frame
  - center_y (float 0-1, default 0.5): vertical position (0=top, 1=bottom)
  - occupancy (float 0.1-0.6): fraction of frame area occupied by subject
  - aspect_ratio (float 0.5-2.0): subject bbox width/height ratio
- Camera-to-object 6D orientation (all required):
  - camera_to_object_fx, fy, fz: object forward vector in camera frame
  - camera_to_object_ux, uy, uz: object up vector in camera frame

## camera_to_object_6d: what each value means

The 6D values describe how the subject appears relative to the camera.

### Forward vector (fx, fy, fz) — "where is the camera relative to the subject?"

- **fy** (most important): front vs back
  - fy ≈ +1.0 → camera sees the subject's FACE (front view)
  - fy ≈  0.0 → side profile (90° side view)
  - fy ≈ -1.0 → camera sees the subject's BACK
- **fx**: which side profile is visible
  - fx ≈ +1.0 → camera is on subject's LEFT → left profile visible
  - fx ≈  0.0 → straight front or back (no lateral offset)
  - fx ≈ -1.0 → camera is on subject's RIGHT → right profile visible
- **fz**: camera height
  - fz > 0 → camera is BELOW subject (low angle / hero shot)
  - fz ≈ 0 → eye level
  - fz < 0 → camera is ABOVE subject (high angle / aerial)

### Up vector (ux, uy, uz) — "how is the camera tilted?"

- **ux**: subject tilt (roll) — usually ≈ 0 (subject appears upright)
- **uy**: camera pitch indicator
  - uy > 0 → camera looking UP (low angle)
  - uy ≈ 0 → camera horizontal (eye level)
  - uy < 0 → camera looking DOWN (high angle)
  - uy ≈ -1.0 → complete top-down bird's eye
- **uz**: how level the camera is — usually ≈ 1.0 (upright camera)

### Constraints
- Forward and up vectors MUST be orthogonal: fx*ux + fy*uy + fz*uz ≈ 0
- Each vector should be approximately unit length
- Round values to 1 decimal place

## Bbox composition guide

- **Shot size** (occupancy):
  - Extreme wide / establishing: 0.10 - 0.15
  - Wide / full shot: 0.15 - 0.25
  - Medium shot: 0.30 - 0.40
  - Close-up: 0.45 - 0.55
  - Extreme close-up: 0.55 - 0.65
- **Framing** (center_x, center_y):
  - Centered: center_x=0.5, center_y=0.5
  - Rule of thirds: center_x ∈ {0.333, 0.667}, center_y ∈ {0.333, 0.667}
  - Lead room: offset center AWAY from subject's gaze direction
- **Aspect ratio**:
  - Portrait: 0.6 - 0.8
  - Square: 1.0
  - Landscape / cinematic: 1.3 - 2.0

## Examples

User: "front eye-level, centered, medium shot"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.4,"aspect_ratio":1.0,"camera_to_object_fx":0.0,"camera_to_object_fy":1.0,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}
```

User: "behind and above, aerial follow shot, wide"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.3,"aspect_ratio":1.2,"camera_to_object_fx":0.0,"camera_to_object_fy":-0.7,"camera_to_object_fz":-0.5,"camera_to_object_ux":0.0,"camera_to_object_uy":-0.5,"camera_to_object_uz":0.8}
```

User: "left profile portrait, close-up"
```json
{"center_x":0.45,"center_y":0.5,"occupancy":0.45,"aspect_ratio":0.75,"camera_to_object_fx":0.7,"camera_to_object_fy":0.5,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}
```

User: "rule of thirds top-right, front, wide shot"
```json
{"center_x":0.667,"center_y":0.333,"occupancy":0.25,"aspect_ratio":1.0,"camera_to_object_fx":0.0,"camera_to_object_fy":1.0,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}
```

User: "hero shot from low right, looking up at subject"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.4,"aspect_ratio":1.0,"camera_to_object_fx":-0.5,"camera_to_object_fy":0.7,"camera_to_object_fz":0.4,"camera_to_object_ux":-0.3,"camera_to_object_uy":0.4,"camera_to_object_uz":0.8}
```

User: "over-the-shoulder, subject on right with lead room left"
```json
{"center_x":0.6,"center_y":0.45,"occupancy":0.35,"aspect_ratio":1.3,"camera_to_object_fx":0.6,"camera_to_object_fy":-0.5,"camera_to_object_fz":-0.2,"camera_to_object_ux":0.0,"camera_to_object_uy":-0.2,"camera_to_object_uz":0.95}
```

User: "top-down bird's eye, centered"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.35,"aspect_ratio":1.0,"camera_to_object_fx":0.0,"camera_to_object_fy":0.0,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":-1.0,"camera_to_object_uz":0.0}
```

User: "right side, eye level, medium"
```json
{"center_x":0.55,"center_y":0.5,"occupancy":0.35,"aspect_ratio":1.0,"camera_to_object_fx":-1.0,"camera_to_object_fy":0.0,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}
```

User: "front, slightly from above, close-up, centered"
```json
{"center_x":0.5,"center_y":0.5,"occupancy":0.5,"aspect_ratio":0.8,"camera_to_object_fx":0.0,"camera_to_object_fy":0.8,"camera_to_object_fz":-0.3,"camera_to_object_ux":0.0,"camera_to_object_uy":-0.4,"camera_to_object_uz":0.9}
```

## Instructions

Given the user's shot description, output ONLY a single JSON object (no markdown, no explanation).
Ensure the forward and up vectors are orthogonal and approximately unit length.
""")


REQUIRED_C2O_KEYS = [
    "camera_to_object_fx", "camera_to_object_fy", "camera_to_object_fz",
    "camera_to_object_ux", "camera_to_object_uy", "camera_to_object_uz",
]
COMPOSITION_KEYS = ["center_x", "center_y", "occupancy", "aspect_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MPC target_json from natural language.")
    p.add_argument("description", nargs="?", help="Shot description (interactive if omitted)")
    p.add_argument("--backend", default="gateway", choices=["gateway", "claude-cli"],
                   help="LLM backend: 'gateway' (Letsur/OpenAI-compatible) or 'claude-cli'")
    p.add_argument("--model", default="gemini-2.5-flash", help="Model name for gateway backend")
    p.add_argument("--api_key_env", default="LETSUR_API_KEY")
    p.add_argument("--base_url", default="https://gateway.letsur.ai/v1")
    p.add_argument("--generate-script", action="store_true",
                   help="Also generate a shell script for infer_mpc_blender.py")
    p.add_argument("--run_dir", default="outputs/Namaqualand_namaqualand_v3_260331_054741")
    p.add_argument("--model_path", default="runs/20260403_151944_qwen35_vl_2b_1xh200_with_c2o_5k/final")
    p.add_argument("--config", default="configs/qwen35_vl_2b_1xh200_with_c2o_5k.yaml")
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
    return response.choices[0].message.content.strip()


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
    for key in REQUIRED_C2O_KEYS:
        if key not in target:
            raise ValueError(f"Missing required key: {key}")

    if "occupancy" not in target:
        raise ValueError("Missing required key: occupancy")

    fx = target["camera_to_object_fx"]
    fy = target["camera_to_object_fy"]
    fz = target["camera_to_object_fz"]
    ux = target["camera_to_object_ux"]
    uy = target["camera_to_object_uy"]
    uz = target["camera_to_object_uz"]

    dot = fx * ux + fy * uy + fz * uz
    if abs(dot) > 0.1:
        print(f"Warning: f·u = {dot:.3f} (should be ~0). Orthogonalizing...", file=sys.stderr)
        f_norm_sq = fx * fx + fy * fy + fz * fz
        if f_norm_sq > 1e-6:
            proj = dot / f_norm_sq
            ux -= proj * fx
            uy -= proj * fy
            uz -= proj * fz
        u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if u_norm > 1e-6:
            ux /= u_norm
            uy /= u_norm
            uz /= u_norm
        target["camera_to_object_ux"] = round(ux, 2)
        target["camera_to_object_uy"] = round(uy, 2)
        target["camera_to_object_uz"] = round(uz, 2)

    return target


def describe_target(target: dict) -> str:
    from src.vlm_qwen25.mpc import _describe_targets
    weights = {k: 1.0 for k in target if k.startswith(("bbox_", "camera_to_object_"))}
    weights.update({k: 1.0 for k in COMPOSITION_KEYS if k in target})
    return _describe_targets(target, weights)


def generate_shell_script(description: str, target: dict, args: argparse.Namespace) -> str:
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

    DEFAULT_SCORE_WEIGHTS='{{"bbox_occupancy_ratio":2.0,"bbox_margin_top":1.0,"bbox_margin_bottom":1.0,"bbox_margin_left":1.0,"bbox_margin_right":1.0,"bbox_aspect_ratio":1.0,"bbox_centroid_offset":2.0,"camera_to_object_fx":1.0,"camera_to_object_fy":1.0,"camera_to_object_fz":1.0,"camera_to_object_ux":1.0,"camera_to_object_uy":1.0,"camera_to_object_uz":1.0}}'
    SCORE_WEIGHTS_JSON="${{SCORE_WEIGHTS_JSON:-$DEFAULT_SCORE_WEIGHTS}}"

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
      --score_weights_json "${{SCORE_WEIGHTS_JSON}}" \\
      --translation_penalty_weight 0.0 \\
      --rotation_penalty_weight 0.0 \\
      --blender_threads "${{BLENDER_THREADS}}" \\
      --disable_roll \\
      --target_json '{target_json_str}'
    """)

    script_path = REPO_ROOT / "scripts" / f"infer_mpc_namaqualand_{safe_name}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return str(script_path)


def main() -> None:
    args = parse_args()

    description = args.description
    if not description:
        description = input("Describe the shot: ").strip()
        if not description:
            print("No description provided.", file=sys.stderr)
            sys.exit(1)

    print(f"Generating target for: \"{description}\"", file=sys.stderr)

    if args.backend == "gateway":
        raw = call_gateway(description, args)
    else:
        raw = call_claude_cli(description)

    target = extract_json(raw)
    target = validate_target(target)
    human = describe_target(target)

    result = {
        "description": description,
        "target_json": target,
        "human_readable": human,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.generate_script:
        path = generate_shell_script(description, target, args)
        print(f"\nScript saved: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
