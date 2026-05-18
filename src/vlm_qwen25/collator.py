from __future__ import annotations

from typing import Sequence

import torch

from .prompt import build_user_prompt
from .schema import SCORE_KEYS

_VALUE_CHARS = set("0123456789-")


class QwenVLScoreCollator:
    def __init__(
        self,
        processor,
        max_length: int = 2048,
        target_score_keys: Sequence[str] | None = None,
        action_frame: str = "camera_local",
        rotation_representation: str = "orientation_6d",
    ) -> None:
        self.processor = processor
        self.max_length = int(max_length)
        self.target_score_keys = list(SCORE_KEYS if target_score_keys is None else target_score_keys)
        self.action_frame = str(action_frame)
        self.rotation_representation = str(rotation_representation)
        if self.action_frame not in {"camera_local", "world"}:
            raise ValueError("action_frame must be 'camera_local' or 'world'")
        if self.rotation_representation not in {"orientation_6d", "rotvec"}:
            raise ValueError("rotation_representation must be 'orientation_6d' or 'rotvec'")

        # Pre-compute a bool mask over the vocabulary: True iff the token's
        # decoded text (stripped) is non-empty and consists only of characters
        # in {0-9, -}. Used in __call__ to restrict the loss to the integer
        # value tokens in the assistant JSON (keys, brackets, colons, commas
        # are all deterministic given the schema, so the model should not
        # waste CE budget memorising them).
        tokenizer = self.processor.tokenizer
        vocab_size = len(tokenizer)
        is_value_token = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in range(vocab_size):
            txt = tokenizer.decode([tid], skip_special_tokens=False).strip()
            if txt and all(c in _VALUE_CHARS for c in txt):
                is_value_token[tid] = True
        self.is_value_token = is_value_token

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        images = [sample["image"] for sample in batch]

        full_texts: list[str] = []
        prompt_only_texts: list[str] = []
        target_texts: list[str] = []

        for sample in batch:
            action_text = str(sample["action_text"])
            target_text = str(sample["target_text"])
            target_texts.append(target_text)
            user_prompt = build_user_prompt(
                action_text,
                target_score_keys=self.target_score_keys,
                action_frame=self.action_frame,
                rotation_representation=self.rotation_representation,
            )

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}],
                },
            ]
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_only_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_only_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        for row in range(labels.size(0)):
            prompt_len = int(prompt_inputs["attention_mask"][row].sum().item())
            labels[row, :prompt_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100

        # Digit-only mask: only tokens whose decoded text is in {0-9, -}
        # contribute to cross-entropy. Keys, brackets, colons, commas are
        # deterministic and excluded.
        non_value = ~self.is_value_token[full_inputs["input_ids"].cpu()]
        labels[non_value] = -100

        # Per-key id tensor: same shape as labels, -100 by default, integer
        # 0..K-1 on the value tokens of the k-th target_score_key. The target
        # JSON is canonical `{"k0":v0,"k1":v1,...,"kK-1":vK-1}` (no spaces),
        # so values appear strictly in the configured key order. We tag each
        # unmasked value token by its k position based on the cumulative
        # count of value runs seen on the row.
        key_id = torch.full_like(labels, -100)
        n_keys = len(self.target_score_keys)
        for row in range(labels.size(0)):
            valid = (labels[row] != -100).nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                continue
            # Split valid positions into contiguous runs (one run per JSON value).
            cur_k = 0
            prev = -2
            for p in valid.tolist():
                if p != prev + 1:
                    if prev >= 0:
                        cur_k += 1
                if cur_k >= n_keys:
                    break
                key_id[row, p] = cur_k
                prev = p

        # Per-sample distance bucket from the dataset (Step 3 plumbing).
        bucket = [int(s.get("bucket_idx", -1)) for s in batch]
        full_inputs["labels"] = labels
        full_inputs["key_id"] = key_id
        full_inputs["bucket_idx"] = torch.tensor(bucket, dtype=torch.long)
        return full_inputs
