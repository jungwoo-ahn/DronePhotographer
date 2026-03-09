from __future__ import annotations

from typing import Sequence

from .prompt import build_user_prompt
from .schema import SCORE_KEYS


class QwenVLScoreCollator:
    def __init__(
        self,
        processor,
        max_length: int = 2048,
        target_score_keys: Sequence[str] | None = None,
    ) -> None:
        self.processor = processor
        self.max_length = int(max_length)
        self.target_score_keys = list(SCORE_KEYS if target_score_keys is None else target_score_keys)

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        images = [sample["image"] for sample in batch]

        full_texts: list[str] = []
        prompt_only_texts: list[str] = []

        for sample in batch:
            action_text = str(sample["action_text"])
            target_text = str(sample["target_text"])
            user_prompt = build_user_prompt(
                action_text,
                target_score_keys=self.target_score_keys,
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
        full_inputs["labels"] = labels
        return full_inputs
