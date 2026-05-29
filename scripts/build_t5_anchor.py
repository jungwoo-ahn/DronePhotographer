"""Build the fixed T5-11B anchor used by `ShotProfileVectorConditioner`.

One-shot script. Loads T5-11B (`google-t5/t5-11b`), tokenizes a fixed prompt,
encodes it, and saves the resulting `(max_len, hidden) bfloat16` tensor plus
metadata (real_len, padding mask, prompt string) to `assets/t5_anchor.pt`.

The output file is then loaded by the conditioner at every training run; T5 is
never touched again.

Resource note: T5-11B is ~22 GB in bfloat16. You need an H100 / A100-80GB
(or larger CPU RAM if running with `--device cpu`). Wall clock: ~3 minutes
including model download (cached afterward).

Usage:
  python scripts/build_t5_anchor.py \
      --prompt "A drone cinematography" \
      --output assets/t5_anchor.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A drone cinematography",
                   help="Fixed anchor prompt. Pick something that biases the backbone "
                        "toward the high-level domain (camera, framing, video).")
    p.add_argument("--output", default="assets/t5_anchor.pt", type=Path)
    p.add_argument("--model_name", default="google-t5/t5-11b")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cache_dir", default=None, help="HF cache override")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import T5EncoderModel, T5TokenizerFast

    print(f"loading {args.model_name} on {args.device}...")
    tok = T5TokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    enc = T5EncoderModel.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=dtype,
    ).to(args.device).eval()

    print(f"encoding prompt: {args.prompt!r}")
    tokens = tok(
        [args.prompt],
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = tokens.input_ids.to(args.device)
    attention_mask = tokens.attention_mask.to(args.device)
    real_len = int(attention_mask[0].sum().item())

    with torch.inference_mode():
        out = enc(input_ids=input_ids, attention_mask=attention_mask)
    embedding = out.last_hidden_state[0].to(dtype).cpu()         # (max_len, hidden)
    padding_mask = attention_mask[0].to(torch.bool).cpu()         # (max_len,)

    blob = {
        "embedding": embedding,
        "real_len": real_len,
        "padding_mask": padding_mask,
        "prompt": args.prompt,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "hidden_size": embedding.shape[-1],
        "dtype": str(dtype),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(
        f"saved {args.output}: shape={tuple(embedding.shape)} "
        f"real_len={real_len} hidden={embedding.shape[-1]}"
    )


if __name__ == "__main__":
    main()
