"""Build the fixed text anchor for `ShotProfileVectorConditioner` (Qwen2.5-VL).

Cosmos-Predict2.5's text encoder is Qwen2.5-VL (Cosmos-Reason1), NOT T5: the
transformer's `crossattn_proj` consumes the concatenation of ALL the VLM's
per-layer hidden states (28 layers x 3584 = 100352 per token). This script
replicates `Cosmos2_5_PredictBasePipeline._get_prompt_embeds` exactly:

  1. chat template with the pipeline's fixed system prompt, padded to 512,
  2. Qwen2.5-VL forward with `output_hidden_states=True`,
  3. per-layer standardization ((h - mean) / std over the feature dim),
  4. concat layers 1..N -> (512, 100352).

One-time cost: downloads the text encoder (~16 GB) the trainer deliberately
skips; one forward pass; saved anchor ~100 MB. Run on a free GPU.

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/build_text_anchor.py \
      --output assets/text_anchor.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_PROMPT = (
    "A smooth aerial drone shot. The camera moves continuously in one take, "
    "deliberately adjusting its viewpoint."
)

# Fixed system prompt from Cosmos2_5_PredictBasePipeline._get_prompt_embeds —
# part of the pretraining convention, must match verbatim.
PIPELINE_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default=Path("assets/text_anchor.pt"), type=Path)
    p.add_argument("--repo_id", default="nvidia/Cosmos-Predict2.5-2B")
    p.add_argument("--revision", default="diffusers/base/post-trained")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    print(f"loading text encoder from {args.repo_id} ({args.revision}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, subfolder="tokenizer", revision=args.revision)
    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.repo_id, subfolder="text_encoder", revision=args.revision, torch_dtype=dtype,
    ).to(args.device).eval()

    conversations = [
        {"role": "system", "content": [{"type": "text", "text": PIPELINE_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
    ]
    # Padded ids (the pipeline convention) + unpadded ids to learn the real length
    ids_padded = tokenizer.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=False, add_vision_id=False,
        max_length=args.max_length, truncation=True, padding="max_length",
    )
    ids_real = tokenizer.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=False, add_vision_id=False,
        max_length=args.max_length, truncation=True,
    )
    ids_padded = ids_padded["input_ids"] if not isinstance(ids_padded, list) and "input_ids" in ids_padded else ids_padded
    ids_real = ids_real["input_ids"] if not isinstance(ids_real, list) and "input_ids" in ids_real else ids_real
    real_len = len(ids_real)
    input_ids = torch.LongTensor(ids_padded).unsqueeze(0).to(args.device)

    print(f"prompt occupies {real_len}/{args.max_length} tokens (incl. chat template)")
    with torch.inference_mode():
        out = encoder(input_ids, output_hidden_states=True)
    hs = out.hidden_states
    normalized = [
        (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
        for h in hs[1:]
    ]
    embedding = torch.cat(normalized, dim=-1)[0].to(dtype).cpu()   # (max_length, n_layers*hidden)

    padding_mask = torch.zeros(args.max_length, dtype=torch.bool)
    padding_mask[:real_len] = True

    blob = {
        "embedding": embedding,
        "real_len": real_len,
        "padding_mask": padding_mask,
        "prompt": args.prompt,
        "system_prompt": PIPELINE_SYSTEM_PROMPT,
        "model_name": f"{args.repo_id}@{args.revision}:text_encoder",
        "max_length": args.max_length,
        "hidden_size": embedding.shape[-1],
        "dtype": str(dtype),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(f"saved {args.output}: shape={tuple(embedding.shape)} real_len={real_len} "
          f"({args.output.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
