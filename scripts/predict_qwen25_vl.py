from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.scoring import DEFAULT_TARGET_SCORE_KEYS
from src.vlm_qwen25.prompt import build_user_prompt
from src.vlm_qwen25.schema import parse_scores_from_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--action_text", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--score_keys",
        type=str,
        default=",".join(DEFAULT_TARGET_SCORE_KEYS),
        help="comma separated keys in target JSON order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_keys = [key.strip() for key in args.score_keys.split(",") if key.strip()]

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    user_prompt = build_user_prompt(args.action_text, target_score_keys=score_keys)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(args.max_new_tokens),
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    generated_text = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0]
    parsed_scores = parse_scores_from_text(generated_text, score_keys=score_keys)

    output = {
        "generated_text": generated_text,
        "score_keys": score_keys,
        "parsed_scores": parsed_scores,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
