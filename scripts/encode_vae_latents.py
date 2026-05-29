"""Precompute Cosmos VAE latents for every rendered v7 frame.

Iterates each placement's `data.json` → `render_records[i][j].path_rel`, encodes
the JPEG once, and saves a `.pt` cache keyed by `(data_json_path, pair_idx,
frame_idx)`. Each frame is encoded as a single-frame VAE latent (the image
repeated across the 4-frame temporal chunk the VAE expects).

Usage:
  python scripts/encode_vae_latents.py \
      --annotation_roots outputs/v7_stage2_renders \
      --output runs/vae_cache/v7.pt \
      [--max_samples 1000]

NOTE: the trainer currently encodes a 2-frame (start, end) clip per sample on the
fly and does not yet read this cache. Wiring a cache-load path into
`CosmosPolicyTrainer` (keyed by the window's start+end frame identity) is a TODO;
this script exists so that precompute is ready when that lands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.policy.common.annotations import list_annotation_files
from src.policy.cosmos.dataset import _load_image_as_tensor
from src.policy.cosmos.vae import CosmosVAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_roots", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--repo_id", default="nvidia/Cosmos-Predict2.5-2B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    vae = CosmosVAEWrapper.from_pretrained(args.repo_id, dtype=dtype, device_map=args.device)
    vae.to(device).eval()

    files = list_annotation_files(args.annotation_roots)
    cache: dict[tuple[str, int, int], torch.Tensor] = {}
    n = 0
    for data_json in tqdm(files, desc="placements"):
        doc = json.loads(Path(data_json).read_text())
        placement_dir = Path(data_json).parent
        for pair_idx, pair_recs in enumerate(doc.get("render_records") or []):
            for rec in pair_recs:
                if args.max_samples and n >= args.max_samples:
                    break
                img_path = placement_dir / rec["path_rel"]
                if not img_path.exists():
                    continue
                img = _load_image_as_tensor(img_path, tuple(args.resolution)).unsqueeze(0).to(device, dtype=dtype)
                # Single-frame latent: repeat the image across the VAE's 4-frame chunk.
                clip = vae.assemble_clip(img, img)
                latent = vae.encode(clip).squeeze(0).cpu()
                cache[(str(data_json), pair_idx, int(rec.get("frame_idx", -1)))] = latent
                n += 1
            if args.max_samples and n >= args.max_samples:
                break
        if args.max_samples and n >= args.max_samples:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache": cache, "resolution": list(args.resolution)}, args.output)
    print(f"saved {len(cache)} latents to {args.output}")


if __name__ == "__main__":
    main()
