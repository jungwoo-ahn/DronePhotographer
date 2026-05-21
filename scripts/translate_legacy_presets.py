#!/usr/bin/env python3
"""Translate jungwooahn's legacy `infer_mpc_*.sh` scripts into v6 YAML presets.

Reads every `scripts/infer_mpc_*.sh` shell script in this repo (most are leftovers
from the c2o-6D era), extracts the embedded `--target_json` and key MPC flags,
converts the schema to v5_SCORE_KEYS namespace, and writes the result to
`configs/inference/<name>.yaml`.

Conversion rules:
- bbox_occupancy_ratio  → occupancy (fraction, same value)
- center_x/y, occupancy → kept as-is (compiler handles fractions → pixels/percent)
- aspect_ratio          → dropped (no v5 equivalent)
- bbox_margin_*, bbox_centroid_offset → dropped (legacy bbox-margin schema not in v5)
- camera_to_object_fx/fy/fz/ux/uy/uz:
    elevation_deg = -degrees(atan2(fz, sqrt(fx**2 + fy**2)))   # jungwooahn fz>0 (below) → v6 elev<0
    azimuth        = DROPPED (jungwooahn 6D is subject-local, v6 azimuth is world-frame)

Usage:
    python scripts/translate_legacy_presets.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO / "scripts"
OUT_DIR = REPO / "configs" / "inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_shell_script(path: Path) -> dict:
    """Extract --target_json (and a few MPC flags) from an infer_mpc_*.sh file."""
    text = path.read_text(encoding="utf-8")
    out: dict = {}

    m = re.search(r"--target_json\s+'(\{[^']+\})'", text)
    if m:
        out["raw_target"] = json.loads(m.group(1))

    def grab(flag: str, conv):
        rx = re.search(rf"--{flag}\s+([\d.\-,]+)", text)
        if rx:
            try:
                out[flag] = conv(rx.group(1))
            except Exception:
                pass
        rx = re.search(rf"--{flag}=([\d.\-,]+)", text)
        if rx:
            try:
                out[flag] = conv(rx.group(1))
            except Exception:
                pass

    grab("num_steps", int)
    grab("max_translation_norm_m", float)
    grab("max_rotation_norm_deg", float)
    grab("max_new_tokens", int)
    grab("max_candidates", int)

    for flag in ("translation_values_m", "rotation_values_deg"):
        rx = re.search(rf"--{flag}=([\d.\-,]+)", text)
        if rx:
            out[flag] = [float(x) for x in rx.group(1).split(",") if x]
        else:
            rx = re.search(rf"--{flag}\s+([\d.\-,]+)", text)
            if rx:
                out[flag] = [float(x) for x in rx.group(1).split(",") if x]

    if "--disable_roll" in text:
        out["disable_roll"] = True
    return out


def translate_target(raw: dict) -> dict:
    """Return a v6-namespace target dict from a raw c2o-era target JSON."""
    out: dict = {}

    # composition keys
    if "center_x" in raw:
        out["center_x"] = float(raw["center_x"])
    if "center_y" in raw:
        out["center_y"] = float(raw["center_y"])
    if "occupancy" in raw:
        out["occupancy"] = float(raw["occupancy"])
    elif "bbox_occupancy_ratio" in raw:
        out["occupancy"] = float(raw["bbox_occupancy_ratio"])

    # If the legacy preset used bbox_margin_* (off-center derivation),
    # convert margins back to a center_x/y fraction (subject sits in the
    # rectangle defined by margins).
    if "bbox_margin_left" in raw and "bbox_margin_right" in raw:
        ml, mr = float(raw["bbox_margin_left"]), float(raw["bbox_margin_right"])
        out.setdefault("center_x", round((ml + (1.0 - mr)) / 2.0, 3))
    if "bbox_margin_top" in raw and "bbox_margin_bottom" in raw:
        mt, mb = float(raw["bbox_margin_top"]), float(raw["bbox_margin_bottom"])
        out.setdefault("center_y", round((mt + (1.0 - mb)) / 2.0, 3))

    # camera elevation (skip azimuth — see module docstring)
    fx = float(raw.get("camera_to_object_fx", 0.0))
    fy = float(raw.get("camera_to_object_fy", 0.0))
    fz = float(raw.get("camera_to_object_fz", 0.0))
    uy = float(raw.get("camera_to_object_uy", 1.0))
    has_c2o = any(k in raw for k in (
        "camera_to_object_fx", "camera_to_object_fy", "camera_to_object_fz",
        "camera_to_object_ux", "camera_to_object_uy", "camera_to_object_uz",
    ))
    if has_c2o:
        horiz = math.sqrt(fx * fx + fy * fy)
        if horiz > 1e-6 or abs(fz) > 1e-6:
            # jungwooahn: fz>0 means "camera below subject" (low angle).
            # v6: elev<0 means "camera below subject" (low angle, looking up).
            elev = -math.degrees(math.atan2(fz, horiz))
            out["cam_to_obj_elevation_deg"] = int(round(elev))
        else:
            # Degenerate position (top-down / bottom-up: f-vector is zero).
            # Derive pitch from the up-vector: uy=-1 → top-down (+90); uy=+1 → bottom-up (-90).
            uy_c = max(-1.0, min(1.0, uy))
            elev = -math.degrees(math.asin(uy_c))
            if abs(elev) > 1.0:  # ignore eye-level default
                out["cam_to_obj_elevation_deg"] = int(round(elev))

    out.setdefault("body_in_frame_ratio", 1.0)
    return out


def script_to_yaml(path: Path) -> str | None:
    parsed = parse_shell_script(path)
    if "raw_target" not in parsed:
        return None
    target = translate_target(parsed["raw_target"])
    if not target.get("occupancy"):
        return None

    mpc = {
        k: parsed[k]
        for k in (
            "num_steps", "translation_values_m", "rotation_values_deg",
            "max_translation_norm_m", "max_rotation_norm_deg",
            "max_candidates", "max_new_tokens", "disable_roll",
        )
        if k in parsed
    }

    name = path.stem.removeprefix("infer_mpc_")
    description = name.replace("_", " ")

    lines = [
        f"# Translated from {path.name}",
        f"name: {name}",
        f'description: "{description}"',
        "",
        "target:",
    ]
    for k in ("center_x", "center_y", "occupancy", "body_in_frame_ratio",
              "cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"):
        if k in target:
            lines.append(f"  {k}: {target[k]}")
    if mpc:
        lines.append("")
        lines.append("mpc:")
        for k, v in mpc.items():
            if isinstance(v, list):
                lines.append(f"  {k}: [{', '.join(str(x) for x in v)}]")
            elif isinstance(v, bool):
                lines.append(f"  {k}: {str(v).lower()}")
            else:
                lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    sh_files = sorted(SCRIPTS_DIR.glob("infer_mpc_*.sh"))
    written = 0
    skipped = 0
    for sh in sh_files:
        yaml_text = script_to_yaml(sh)
        if yaml_text is None:
            print(f"  skip  {sh.name}  (no target_json found)")
            skipped += 1
            continue
        name = sh.stem.removeprefix("infer_mpc_")
        out_path = OUT_DIR / f"{name}.yaml"
        out_path.write_text(yaml_text, encoding="utf-8")
        print(f"  wrote {out_path.relative_to(REPO)}")
        written += 1
    print(f"\nDone: {written} YAML presets in {OUT_DIR.relative_to(REPO)}, {skipped} skipped.")


if __name__ == "__main__":
    main()
