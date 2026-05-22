"""Benchmark inference speed: vanilla HF vs HF+flash+compile vs vLLM.

Generates 720 candidates from a fixed pose, scores them under each backend,
and reports wall-clock time per scoring call.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.mpc import (
    generate_local_candidate_actions,
    score_action_candidates,
    score_action_candidates_vllm,
)


def make_candidates(position, forward, up, action_frame, rotation_representation):
    translation_values = [-0.12, 0.0, 0.12]
    rotation_values_rad = [math.radians(d) for d in [-3.0, 0.0, 3.0]]
    return generate_local_candidate_actions(
        position=position,
        forward=forward,
        up=up,
        translation_values=translation_values,
        rotation_values_rad=rotation_values_rad,
        max_translation_norm=0.18,
        max_rotation_norm_rad=math.radians(4.5),
        action_frame=action_frame,
        rotation_representation=rotation_representation,
        disable_roll=False,  # keep roll to get ~720 candidates (realistic)
    )


def benchmark_hf(
    *,
    model_path: str,
    processor,
    image: Image.Image,
    candidates,
    score_keys: list[str],
    action_frame: str,
    rotation_representation: str,
    attn_impl: str | None,
    use_compile: bool,
    batch_size: int,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    label: str = "",
):
    import torch
    from transformers import AutoModelForImageTextToText

    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    print(f"\n{'='*60}")
    print(f"[{label}] Loading model (attn={attn_impl}, compile={use_compile}) ...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    model.eval()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile applied")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # Warmup
    for i in range(warmup_runs):
        print(f"  Warmup {i+1}/{warmup_runs} ...")
        _ = score_action_candidates(
            model=model,
            processor=processor,
            image=image,
            candidates=candidates,
            target_score_keys=score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            max_new_tokens=128,
            candidate_batch_size=batch_size,
        )

    # Timed runs
    times = []
    for i in range(timed_runs):
        torch.cuda.synchronize()
        t_start = time.time()
        results = score_action_candidates(
            model=model,
            processor=processor,
            image=image,
            candidates=candidates,
            target_score_keys=score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            max_new_tokens=128,
            candidate_batch_size=batch_size,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t_start
        times.append(elapsed)
        n_parsed = sum(1 for r in results if r.predicted_scores is not None)
        print(f"  Run {i+1}/{timed_runs}: {elapsed:.2f}s  (parsed {n_parsed}/{len(results)})")

    # Cleanup
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "load_time_s": round(load_time, 2),
        "n_candidates": len(candidates),
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "times_s": [round(t, 3) for t in times],
        "mean_s": round(np.mean(times), 3),
        "std_s": round(np.std(times), 3),
    }


def benchmark_vllm(
    *,
    model_path: str,
    processor,
    image: Image.Image,
    candidates,
    score_keys: list[str],
    action_frame: str,
    rotation_representation: str,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    label: str = "vLLM",
):
    from vllm import LLM

    print(f"\n{'='*60}")
    print(f"[{label}] Loading vLLM ...")
    t0 = time.time()
    model = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
    )
    load_time = time.time() - t0
    print(f"  vLLM loaded in {load_time:.1f}s")

    # Warmup
    for i in range(warmup_runs):
        print(f"  Warmup {i+1}/{warmup_runs} ...")
        _ = score_action_candidates_vllm(
            llm=model,
            processor=processor,
            image=image,
            candidates=candidates,
            target_score_keys=score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            max_new_tokens=128,
        )

    # Timed runs
    times = []
    for i in range(timed_runs):
        t_start = time.time()
        results = score_action_candidates_vllm(
            llm=model,
            processor=processor,
            image=image,
            candidates=candidates,
            target_score_keys=score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            max_new_tokens=128,
        )
        elapsed = time.time() - t_start
        times.append(elapsed)
        n_parsed = sum(1 for r in results if r.predicted_scores is not None)
        print(f"  Run {i+1}/{timed_runs}: {elapsed:.2f}s  (parsed {n_parsed}/{len(results)})")

    # Cleanup
    del model
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "label": label,
        "load_time_s": round(load_time, 2),
        "n_candidates": len(candidates),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "times_s": [round(t, 3) for t in times],
        "mean_s": round(np.mean(times), 3),
        "std_s": round(np.std(times), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="runs/20260403_151944_qwen35_vl_2b_1xh200_with_c2o_5k/final")
    parser.add_argument("--run_dir", type=str,
                        default="outputs/Namaqualand_namaqualand_v3_260401_024633")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip_vanilla", action="store_true", help="Skip vanilla HF (slowest)")
    args = parser.parse_args()

    # Load config
    from transformers import AutoProcessor
    import yaml

    config_path = Path(args.model_path).resolve().parent / "config.yaml"
    if not config_path.exists():
        for p in Path(args.model_path).resolve().parents[:4]:
            if (p / "config.yaml").exists():
                config_path = p / "config.yaml"
                break
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    action_frame = data_cfg.get("action_frame", "camera_local")
    rotation_representation = data_cfg.get("rotation_representation", "orientation_6d")
    score_keys = list(data_cfg.get("target_score_keys", []))

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Get a test image from the scene
    run_dir = Path(args.run_dir)
    annotations_path = run_dir / "annotations_5k.json"
    if not annotations_path.exists():
        annotations_path = run_dir / "annotations.json"
    with annotations_path.open() as f:
        annots = json.load(f)
    first = annots[0]
    image_path = run_dir / str(first["image"])
    image = Image.open(image_path).convert("RGB")

    # Fixed pose for candidate generation
    position = np.array(first["camera_position"], dtype=np.float32)
    forward = np.array(first.get("final_forward", first.get("base_forward")), dtype=np.float32)
    up = np.array(first.get("final_up", first.get("base_up")), dtype=np.float32)

    candidates = make_candidates(position, forward, up, action_frame, rotation_representation)
    print(f"Generated {len(candidates)} candidates")
    print(f"Image: {image_path} ({image.size})")
    print(f"Score keys: {len(score_keys)}")

    results = []

    # --- 1. Vanilla HF (no flash, no compile) ---
    if not args.skip_vanilla:
        r1 = benchmark_hf(
            model_path=args.model_path,
            processor=processor,
            image=image,
            candidates=candidates,
            score_keys=score_keys,
            action_frame=action_frame,
            rotation_representation=rotation_representation,
            attn_impl=None,
            use_compile=False,
            batch_size=args.batch_size,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
            label="1_vanilla_hf",
        )
        results.append(r1)

    # --- 2. HF + sdpa + torch.compile ---
    # (flash_attention_2 was removed due to vLLM conflict; sdpa is the best available)
    r2 = benchmark_hf(
        model_path=args.model_path,
        processor=processor,
        image=image,
        candidates=candidates,
        score_keys=score_keys,
        action_frame=action_frame,
        rotation_representation=rotation_representation,
        attn_impl="sdpa",
        use_compile=True,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        label="2_hf_sdpa_compile",
    )
    results.append(r2)

    # --- 3. vLLM ---
    r3 = benchmark_vllm(
        model_path=args.model_path,
        processor=processor,
        image=image,
        candidates=candidates,
        score_keys=score_keys,
        action_frame=action_frame,
        rotation_representation=rotation_representation,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        label="3_vllm",
    )
    results.append(r3)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Candidates: {len(candidates)}")
    print(f"GPU: NVIDIA H200 (1x)")
    print(f"Model: Qwen3.5-2B (bfloat16)")
    print()

    baseline_mean = results[0]["mean_s"] if results else None
    for r in results:
        speedup = ""
        if baseline_mean and r["mean_s"] > 0 and r is not results[0]:
            speedup = f"  ({baseline_mean / r['mean_s']:.1f}x faster vs baseline)"
        print(f"  {r['label']:30s}  {r['mean_s']:7.2f}s ± {r['std_s']:.2f}s{speedup}")

    # Save
    out_path = Path("notes/benchmark_inference_speed.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
