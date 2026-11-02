"""Caption goal frames with a VLM, so the caption can be a CONDITIONING CHANNEL.

The policy is conditioned on a goal today through one channel: a structured sentence built
from the goal vector (`src/goal_authoring/serialize.py`). That sentence is computed from the
subject's projected mesh -- it knows the geometry exactly and cannot be fooled by what the
photograph actually shows. Two other channels are possible and neither has been tried:

  * the goal FRAME itself, as a second image panel
  * a natural-language CAPTION of that frame, written by a VLM that sees only the picture

This script produces the third. It captions `ViewRecord.image` for sampled goal frames and
writes the VLM caption beside the training prompt, so the two can be read against each other.

WHAT THE PROMPT ASKS FOR, AND WHY IT STOPS WHERE IT DOES. The caption has to carry the axes the
policy acts on -- where the camera is, how big the subject is, where it sits, what is cut -- or
it cannot be a goal. It must NOT be handed the profile itself: a prompt that asks the VLM to
fill in our eight keys, or that quotes their vocabulary, would not be testing whether an image
yields a usable goal, it would be laundering the structured prompt through a language model and
measuring our own annotation twice. So the prompt names the QUESTIONS and leaves the ANSWERS
(and their words, and any number) to the model.

The prior here is not good, and is worth stating before the numbers arrive:
`src/goal_authoring/from_reference.py` already tried a VLM on exactly this task -- estimating
shot attributes from a reference photograph -- and it lost to YOLO-pose plus two small
regressors "by a wide margin: the VLM was at chance on elevation and 68 %/32 % on bearing".
That was a VLM asked for structured values. Whether free text does better is the open part.

The sample is stratified, because the interesting failures are not uniformly distributed:
~30 % of goal frames contain no detectable person (occlusion, small subjects, non-human assets)
while still carrying a perfectly ordinary shot profile. Those frames are where an image or
caption goal must diverge from a text goal, so they are sampled deliberately rather than left
to chance.

    .venv-vla/bin/python scripts/caption_goal_frames.py --n 60 --out runs/eval/goal_captions.json
venv: .venv-vla (transformers 5.5.4 + torch 2.9.1; both Qwen3.5 and Qwen3-VL load there).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))

from src.common.annotations import iter_goal_start_windows          # noqa: E402
from src.common.facing import sector8                                # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY  # noqa: E402
from src.goal_authoring.serialize import goal_prompt                 # noqa: E402

# Two registers of the same request. Both name the QUESTIONS and never the vocabulary or the
# values: a prompt that asked the VLM to fill in our eight keys, or that quoted their words,
# would not test whether a photograph yields a usable goal -- it would launder the structured
# prompt through a language model and measure our own annotation twice.
#
# `instruct` is the one that gets trained on. The policy's text channel has only ever carried
# imperatives of one shape ("Move the camera to achieve this shot: a wide shot of the subject
# from the subject's back-right, at eye level, ..."), and 44,560/44,560 training strings match
# it. A declarative caption differs from that in two ways at once -- what it says AND how it is
# phrased -- so a run on declarative captions cannot separate "natural language carries less
# goal" from "the model has never been addressed this way". Matching the register removes the
# second explanation and leaves the first, which is the one worth measuring.
#
# `describe` is kept because it produced the first gallery, and deleting the prompt behind a
# published measurement makes that measurement unreproducible.
PROMPTS = {
    "instruct": (
        "Write a single instruction that tells a camera operator, who cannot see this "
        "photograph, how to place and frame their camera so that they reproduce it. "
        "Begin with exactly: 'Move the camera to achieve this shot: '. Then, in one or two "
        "sentences of plain prose, say which side of the person the camera is on and whether "
        "it is above, level with, or below them; how much of the picture the person takes up; "
        "where in the frame they sit; and whether the frame cuts off any part of them. "
        "Do not use lists, headings, or numbers. If no person is visible, still begin with "
        "'Move the camera to achieve this shot: ' and then say in one sentence that no person "
        "is visible and what is there instead."
    ),
    "describe": (
        "You are describing a photograph to a camera operator who must reproduce it, and who "
        "cannot see it. In two or three sentences, say where the camera is placed relative to "
        "the person (which side of them it is on, and whether it is above, level with, or "
        "below them), how much of the picture the person takes up, where in the frame they "
        "sit, and whether any part of them is cut off by the edge. Write plain prose. Do not "
        "use lists, headings, or numbers. If no person is visible, say so in one sentence and "
        "describe what is there instead."
    ),
}

MODELS = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=[str(V12 / "data/trajectories/v7_stage2_renders_lookat075")])
ap.add_argument("--val-scenes", default=str(V12 / "configs" / "val_scenes.json"))
ap.add_argument("--split", default="val", choices=["val", "train", "any"],
                help="val by default: the gallery should be judged on scenes no policy trained on")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--frames-from", default="",
                help="JSON from `export_lerobot.py --dump-goal-frames`: caption THESE frames "
                     "instead of drawing a stratified sample. The draw and the export then "
                     "come from the same seeded fill(), so no caption can be keyed to a frame "
                     "the export never selected.")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1,
                help="split --frames-from across N independent jobs; shard i takes every Nth "
                     "frame, so any subset of shards is still a uniform sample of the data")
ap.add_argument("--resize", type=int, default=512,
                help="longest edge fed to the VLM. The renders are 1024x768, which the "
                     "processor turns into 768 vision tokens and then pays as prefill before "
                     "writing a single word: 17.5 s a caption, 688 GPU-hours for this dataset. "
                     "At 512 it is 192 tokens and 5.1 s (measured), and the captions say the "
                     "same things -- the policy that consumes them only ever sees 256x256.")
ap.add_argument("--batch", type=int, default=8,
                help="8 is where the curve flattens: batch 16 measured no faster (5.5 vs 5.1 "
                     "s/img), so this model is compute-bound, not memory-bound.")
ap.add_argument("--models", nargs="+", default=["qwen3.5-4b"], choices=sorted(MODELS),
                help="one or more. The frames are drawn ONCE, before any model loads, so every "
                     "model captions exactly the same images and the gallery compares models "
                     "rather than draws.")
ap.add_argument("--max-new-tokens", type=int, default=150)
ap.add_argument("--placement-scan", type=int, default=120, help="placements to scan for candidates")
ap.add_argument("--per-placement", type=int, default=2)
ap.add_argument("--style", default="instruct", choices=sorted(PROMPTS),
                help="instruct matches the register of every training prompt; "
                     "describe is the declarative wording the first gallery used")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(V12 / "runs" / "eval" / "goal_captions.json"))
args = ap.parse_args()


def sample_goal_frames() -> list[dict]:
    """Stratified draw of goal frames: every bearing sector, and both occupancy extremes."""
    placements: list[tuple[str, Path]] = []
    for root in args.roots:
        r = Path(root)
        if not r.is_dir():
            continue
        for d in sorted(os.listdir(r)):
            p = r / d / "data.json"
            if p.exists():
                placements.append((d, p))
    val = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"]) if args.val_scenes else frozenset()
    if args.split != "any":
        want_val = args.split == "val"
        placements = [(n, p) for n, p in placements if (n.split("__")[0] in val) == want_val]
    random.seed(args.seed)
    random.shuffle(placements)
    print(f"{len(placements)} placements in split '{args.split}'", flush=True)

    out: list[dict] = []
    sectors: Counter = Counter()
    cap = max(1, args.n // 8)
    for name, path in placements[: args.placement_scan]:
        if len(out) >= args.n:
            break
        try:
            windows = list(iter_goal_start_windows(path, chunk_size=8, max_per_pair=1))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name[:44]}: {type(exc).__name__}: {exc}")
            continue
        random.shuffle(windows)
        taken = 0
        for w in windows:
            if taken >= args.per_placement or len(out) >= args.n:
                break
            gf = w.goal_frame
            if gf is None or not Path(gf.image).exists():
                continue
            obj = name.split("__", 1)[1] if "__" in name else name
            from src.common.goal_space import goal_vector
            g = goal_vector(gf.raw, DEFAULT_GOAL_KEYS, object_key=obj)
            if not np.isfinite(g).all():
                continue
            sec = sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]))
            if sectors[sec] >= cap and len(out) < args.n - 4:   # leave slack to finish the draw
                continue
            out.append({
                "image": str(gf.image),
                "placement": name,
                "scene": name.split("__")[0],
                "frame_idx": int(gf.frame_idx),
                "sector": sec,
                "occupancy": float(gf.raw.get("occupancy", float("nan"))),
                "visible_frac": float(gf.raw.get("visible_frac", float("nan"))),
                "goal_prompt": goal_prompt(g, crop=gf.raw),
                "goal_vec": [float(x) for x in g],
            })
            sectors[sec] += 1
            taken += 1
    print(f"sampled {len(out)} goal frames; sectors {dict(sectors)}", flush=True)
    return out


def load_frames_from() -> list[dict]:
    """The exact frames the export will use, sharded. No goal vector or prompt is available
    here -- the dump carries paths only -- so the records are minimal; the export joins the
    caption back on by path."""
    doc = json.loads(Path(args.frames_from).read_text())
    frames = doc["frames"][args.shard :: args.nshards]
    print(f"{doc['distinct']} frames in {args.frames_from}; "
          f"shard {args.shard}/{args.nshards} takes {len(frames)}", flush=True)
    return [{"image": f, "placement": "", "scene": Path(f).parts[-3] if len(Path(f).parts) > 2 else "",
             "frame_idx": -1, "sector": "", "occupancy": float("nan"),
             "visible_frac": float("nan"), "goal_prompt": "", "goal_vec": []} for f in frames]


def main() -> int:
    samples = load_frames_from() if args.frames_from else sample_goal_frames()
    if not samples:
        print("no goal frames sampled", file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")
    for s in samples:
        s["captions"] = {}

    for key in args.models:
        # AutoModelForImageTextToText, NOT AutoModelForCausalLM. Both Qwen3.5 and Qwen3-VL
        # resolve under the causal-LM factory to a TEXT-ONLY shell that silently drops the
        # vision tower, and the first sign of it is `generate` rejecting pixel_values as an
        # unused kwarg -- after the weights load, so it reads as a call-signature problem
        # rather than the wrong model.
        repo = MODELS[key]
        print(f"\nloading {repo} ...", flush=True)
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(repo)
        model = AutoModelForImageTextToText.from_pretrained(
            repo, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

        proc.tokenizer.padding_side = "left"   # decoder-only batched generation
        t0 = time.time()
        for i0 in range(0, len(samples), args.batch):
            batch = samples[i0 : i0 + args.batch]
            imgs = []
            for s in batch:
                im = Image.open(s["image"]).convert("RGB")
                if args.resize:
                    im.thumbnail((args.resize, args.resize), Image.LANCZOS)
                imgs.append(im)
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                                  {"type": "text", "text": PROMPTS[args.style]}]}]
                    for im in imgs]
            # enable_thinking=False. Qwen3.5 is a hybrid reasoning model whose default is to
            # emit a scratchpad first ("The user wants a description... I need to focus on:
            # 1. Camera position..."), which spends the whole token budget before the caption
            # begins -- every sample in the first smoke was truncated mid-plan.
            try:
                inputs = proc.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, padding=True,
                    return_dict=True, return_tensors="pt", enable_thinking=False,
                ).to(model.device)
            except TypeError:       # a template without the switch
                inputs = proc.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, padding=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            for j, s in enumerate(batch):
                text = proc.decode(ids[j][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
                # Belt and braces: if a template ignores the flag, drop the scratchpad rather
                # than storing it -- a leaked plan reads like a caption and would quietly
                # become the conditioning string.
                if "</think>" in text:
                    text = text.rsplit("</think>", 1)[1].strip()
                s["captions"][key] = text
            done = i0 + len(batch)
            if done % (args.batch * 25) < args.batch or i0 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(samples)}] {el:.0f}s ({el/max(1,done):.2f}s/img, "
                      f"eta {(len(samples)-done)*el/max(1,done)/3600:.1f}h)  "
                      f"{s['captions'][key][:60]}...", flush=True)
        print(f"  {key}: {len(samples)} captions in {time.time()-t0:.0f}s "
              f"({(time.time()-t0)/max(1,len(samples)):.1f}s each)", flush=True)
        del model, proc
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_what": "VLM captions of goal frames, beside the structured training prompt for the "
                 "same frame. The caption is a candidate CONDITIONING channel, not an "
                 "annotation of the data.",
        "models": {k: MODELS[k] for k in args.models}, "style": args.style,
        "prompt": PROMPTS[args.style],
        "split": args.split, "n": len(samples), "samples": samples}, indent=1))
    print(f"\nwrote {out}  ({len(samples)} frames x {len(args.models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
