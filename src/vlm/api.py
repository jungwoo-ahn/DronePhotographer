"""VLM API client for placement evaluation."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time

log = logging.getLogger(__name__)


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_vlm_json(text: str) -> dict | None:
    """Extract a JSON object from VLM response text (handles markdown fences)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    log.error(f"Could not parse VLM response: {text[:200]}")
    return None


def call_vlm(
    image_path: str,
    object_name: str,
    scene_name: str,
    camera_radius: float,
    camera_elevation: float,
    vlm_config: dict,
    extra_context: str = "",
    scene_context_image: str | None = None,
) -> dict | None:
    """Call a VLM API to evaluate an object placement image.

    Args:
        vlm_config: dict with keys: api_base, api_key_env, model,
            max_tokens, temperature, retry_attempts, retry_delay_seconds.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        return None

    from src.vlm.prompts import VLM_SYSTEM_PROMPT

    api_key = os.environ.get(
        vlm_config.get("api_key_env", "LETSUR_API_KEY"),
        "sk-PloeoBXIjFj3xnflrpanUA",
    )

    client = OpenAI(
        base_url=vlm_config["api_base"],
        api_key=api_key,
    )

    img_b64 = encode_image_base64(image_path)

    # Build user message with optional scene context
    if scene_context_image:
        user_text = (
            f"First, here is the scene '{scene_name}' from its default camera view "
            f"(no object placed yet). Use this to understand the scene layout.\n"
            f"Now evaluate the object placement. The object is: {object_name}. "
            f"Camera is {camera_radius:.1f}m from the object at "
            f"{camera_elevation:.0f} degrees elevation."
        )
    else:
        user_text = (
            f"Evaluate this object placement. The object is: {object_name}. "
            f"The scene is: {scene_name}. "
            f"Camera is {camera_radius:.1f}m from the object at "
            f"{camera_elevation:.0f} degrees elevation."
        )
    if extra_context:
        user_text += extra_context

    # Build content list: text + optional scene image + placement image
    content = [{"type": "text", "text": user_text}]
    if scene_context_image:
        scene_b64 = encode_image_base64(scene_context_image)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{scene_b64}"},
        })
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
    })

    retries = vlm_config.get("retry_attempts", 3)
    delay = vlm_config.get("retry_delay_seconds", 2)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=vlm_config["model"],
                messages=[
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=vlm_config.get("max_tokens", 512),
                temperature=vlm_config.get("temperature", 0.2),
            )

            text = response.choices[0].message.content.strip()
            return parse_vlm_json(text)

        except Exception as e:
            log.warning(f"VLM API attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    log.error("VLM API failed after all retries")
    return None


def estimate_scene_scale(
    scene_preview_path: str,
    object_name: str,
    object_height_m: float,
    scene_name: str,
    vlm_config: dict,
) -> float:
    """Ask the VLM to estimate what scale factor the scene needs so that
    an object of known metric size looks correctly proportioned.

    Returns a float scale factor (e.g., 0.3 means shrink scene to 30%).
    Returns 1.0 if no scaling needed or on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return 1.0

    api_key = os.environ.get(
        vlm_config.get("api_key_env", "LETSUR_API_KEY"),
        "sk-PloeoBXIjFj3xnflrpanUA",
    )
    client = OpenAI(base_url=vlm_config["api_base"], api_key=api_key)

    img_b64 = encode_image_base64(scene_preview_path)
    prompt = (
        f"Look at this rendered scene. I will place a {object_name} that is "
        f"{object_height_m:.1f} meters tall in this scene.\n\n"
        f"The scene might be in wrong units (too large or too small). "
        f"Look at the furniture, doors, walls, and other objects in the scene. "
        f"Estimate what scale factor I should apply to the SCENE so that a "
        f"{object_height_m:.1f}m {object_name} would look correctly proportioned.\n\n"
        f"For example:\n"
        f"- If the scene looks like a normal room and a 1.7m person would fit naturally, "
        f"return scale_factor: 1.0\n"
        f"- If the scene looks like a giant room where a person would be tiny (counter tops "
        f"at head height), the scene is too big — return scale_factor: 0.5 or 0.3\n"
        f"- If the scene looks like a dollhouse, return scale_factor: 2.0 or 3.0\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"scale_factor": <float>, "reasoning": "<one sentence>"}}'
    )

    try:
        response = client.chat.completions.create(
            model=vlm_config["model"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    }},
                ],
            }],
            max_tokens=vlm_config.get("max_tokens", 2048),
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        result = parse_vlm_json(text)
        if result and "scale_factor" in result:
            sf = float(result["scale_factor"])
            reasoning = result.get("reasoning", "")
            log.info(f"VLM scene scale estimate: {sf:.2f} ({reasoning})")
            # Clamp to reasonable range
            return max(0.05, min(sf, 10.0))
    except Exception as e:
        log.warning(f"VLM scale estimation failed: {e}")

    return 1.0


def estimate_scale_from_placement(
    image_path: str,
    object_label: str,
    object_height_m: float,
    scene_name: str,
    vlm_config: dict,
) -> tuple[float, str]:
    """Ask the VLM to judge scene scale by comparing the object to scene furniture.

    The object has already been placed at its correct absolute metric size.
    The VLM compares it to nearby furniture (counters, doors, chairs) and
    returns a scale factor to apply to the SCENE.

    Args:
        image_path: Rendered image showing the object IN the scene.
        object_label: Human-readable category (e.g. "person", "cat").
        object_height_m: Real-world height of the object in meters.
        scene_name: Scene name for context.

    Returns:
        (scale_factor, reasoning). scale_factor=1.0 on failure or no change.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return 1.0, "openai not installed"

    api_key = os.environ.get(
        vlm_config.get("api_key_env", "LETSUR_API_KEY"),
        "sk-PloeoBXIjFj3xnflrpanUA",
    )
    client = OpenAI(base_url=vlm_config["api_base"], api_key=api_key)
    img_b64 = encode_image_base64(image_path)

    # Build object-type-aware reference hints.
    if object_height_m >= 1.4:
        # Person / horse / tall animal. Use waist/head as body landmarks.
        body_refs = (
            f"- Where does the nearest counter/table top reach on the {object_label}?\n"
            f"  (a) Below knee  (b) At knee  (c) At waist  (d) At chest  (e) At head  (f) Above head\n"
            f"- How tall are the doors compared to the {object_label}?\n"
            f"  (a) {object_label} fits through with headroom  "
            f"(b) {object_label}'s head nearly touches the top  "
            f"(c) {object_label} is much shorter than the door (head below door handle)  "
            f"(d) {object_label} is taller than the door"
        )
        mapping = (
            f"- Counter at waist (~0.9m for a 1.7m person) → scale_factor: 1.0 (correct)\n"
            f"- Counter at chest → 0.8 (scene slightly too big)\n"
            f"- Counter at head → 0.5 (scene ~2x too big)\n"
            f"- Counter above head / {object_label} looks like a child in a giant's house → 0.3 or smaller\n"
            f"- {object_label} towers over furniture, head above doorframe → 2.0 (scene too small)\n"
            f"- {object_label} dwarfs everything (dollhouse-like) → 3.0+"
        )
    elif object_height_m >= 0.3:
        # Small animal / vase sitting on surfaces. Judge by the surface it's on.
        body_refs = (
            f"- The {object_label} is {object_height_m:.2f}m tall in real life.\n"
            f"- How does its size compare to nearby furniture (chairs, tables, doors)?\n"
            f"  (a) Looks natural — like a real {object_label} sitting there\n"
            f"  (b) {object_label} looks tiny — could fit in a teacup\n"
            f"  (c) {object_label} looks giant — taller than a chair or table"
        )
        mapping = (
            f"- Looks natural → scale_factor: 1.0\n"
            f"- {object_label} looks like a speck / tiny bug next to furniture → 0.3 (scene too big)\n"
            f"- {object_label} is as tall as a chair → scene is too small, scale_factor: 2.0+"
        )
    else:
        body_refs = (
            f"- The {object_label} is only {object_height_m:.2f}m tall (very small).\n"
            f"- Compared to nearby furniture, does it look like a correctly-sized {object_label}?"
        )
        mapping = (
            f"- Natural size → 1.0\n"
            f"- Too small relative to furniture → scene is too big → < 1.0\n"
            f"- Too large → > 1.0"
        )

    prompt = (
        f"You are looking at a rendered image of a {object_label} placed in a scene called '{scene_name}'.\n"
        f"The {object_label} is {object_height_m:.2f}m tall in real life and has been sized correctly.\n"
        f"The {object_label} IS placed in the scene (not missing) — look carefully for it.\n"
        f"Your job is to judge whether the SCENE is at the right scale relative to the {object_label}.\n\n"
        f"Answer these specific visual questions:\n"
        f"{body_refs}\n\n"
        f"Based on your observations, what scale_factor should multiply the SCENE "
        f"so that furniture is realistically proportioned to the {object_label}?\n"
        f"{mapping}\n\n"
        f"IMPORTANT CASES:\n"
        f"- If you can find the {object_label} but it looks like a tiny speck / barely a few pixels "
        f"next to huge furniture → scene is FAR TOO BIG → scale_factor: 0.2 or 0.3\n"
        f"- If you cannot find the {object_label} at all AND the scene looks unnaturally large "
        f"(enormous walls, oversized furniture) → the {object_label} is probably too small to see → "
        f"scale_factor: 0.3\n"
        f"- If you cannot find the {object_label} and the scene looks normal-sized → the camera "
        f"may just be pointed away from it → scale_factor: 1.0, reasoning: 'object off-camera'\n"
        f"- If the {object_label} is huge — filling the frame or dwarfing all furniture → "
        f"scene is too small → scale_factor: 2.0 or 3.0\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"scale_factor": <float>, "reasoning": "<one concise sentence describing what you saw>"}}'
    )

    try:
        response = client.chat.completions.create(
            model=vlm_config["model"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    }},
                ],
            }],
            max_tokens=vlm_config.get("max_tokens", 2048),
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        result = parse_vlm_json(text)
        if result and "scale_factor" in result:
            sf = float(result["scale_factor"])
            reasoning = result.get("reasoning", "")
            log.info(f"VLM placement-scale: {sf:.2f} ({reasoning})")
            return max(0.05, min(sf, 10.0)), reasoning
    except Exception as e:
        log.warning(f"Placement-scale estimation failed: {e}")

    return 1.0, "estimation failed"


def estimate_scale_multi_view(
    image_paths: list[str],
    object_label: str,
    object_height_m: float,
    scene_name: str,
    vlm_config: dict,
) -> tuple[float, str]:
    """Multi-view scale estimate: median of per-view scale_factors.

    Calls estimate_scale_from_placement on each image and returns the
    median scale_factor along with a summary of per-view responses.
    Robust against outlier VLM responses (e.g. one view returning 10.0
    because the camera happened to frame only ground texture).

    Returns (scale_factor, reasoning). scale_factor=1.0 if image_paths empty.
    """
    import statistics as _stats
    if not image_paths:
        return 1.0, "no views"

    per_view = []
    for img in image_paths:
        sf, r = estimate_scale_from_placement(
            image_path=img,
            object_label=object_label,
            object_height_m=object_height_m,
            scene_name=scene_name,
            vlm_config=vlm_config,
        )
        per_view.append((sf, r))

    scales = [sf for sf, _ in per_view]
    median = _stats.median(scales)
    summary = (
        f"median of {len(scales)} views "
        f"(values: {', '.join(f'{s:.2f}' for s in scales)}) → {median:.2f}"
    )
    log.info(f"VLM multi-view scale: {summary}")
    return max(0.05, min(median, 10.0)), summary


def estimate_interest_region(
    scene_preview_path: str,
    vlm_config: dict,
) -> dict | None:
    """Ask the VLM where in the scene preview "interesting" surroundings live.

    Returns a dict with a normalized image-fraction bbox:
        {"bbox": [x_lo, y_lo, x_hi, y_hi], "reasoning": str}
    where all bbox values are in [0, 1] (y increases downward, standard
    image convention). Returns None on failure.

    Intended as a per-scene call — one VLM query, cached, reused across all
    object pairs for that scene.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai not installed; skipping interest-region estimation")
        return None

    from src.vlm.prompts import INTEREST_REGION_PROMPT

    api_key = os.environ.get(
        vlm_config.get("api_key_env", "LETSUR_API_KEY"),
        "sk-PloeoBXIjFj3xnflrpanUA",
    )
    client = OpenAI(base_url=vlm_config["api_base"], api_key=api_key)
    try:
        img_b64 = encode_image_base64(scene_preview_path)
    except Exception as e:
        log.warning(f"Could not read scene preview {scene_preview_path}: {e}")
        return None

    try:
        response = client.chat.completions.create(
            model=vlm_config["model"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": INTEREST_REGION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    }},
                ],
            }],
            max_tokens=vlm_config.get("max_tokens", 2048),
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        result = parse_vlm_json(text)
        if not result or "bbox" not in result:
            # Log full text so we can debug parse issues (markdown fences etc.)
            log.warning(
                f"Interest-region VLM response missing bbox. Full text "
                f"({len(text)} chars):\n{text}"
            )
            return None
        bbox = result["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            log.warning(f"Interest-region bbox wrong shape: {bbox}")
            return None
        # Clamp to [0,1] and enforce ordering
        try:
            x_lo, y_lo, x_hi, y_hi = [max(0.0, min(1.0, float(v))) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x_hi <= x_lo or y_hi <= y_lo:
            return None
        log.info(
            f"VLM interest region: bbox=[{x_lo:.2f},{y_lo:.2f},{x_hi:.2f},{y_hi:.2f}] "
            f"({result.get('reasoning', '')})"
        )
        return {
            "bbox": [x_lo, y_lo, x_hi, y_hi],
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        log.warning(f"Interest-region estimation failed: {e}")
        return None


def check_compatibility(
    scene_preview_path: str,
    object_name: str,
    scene_name: str,
    vlm_config: dict,
) -> dict | None:
    """Ask a VLM whether an object fits naturally in a scene.

    Args:
        scene_preview_path: Path to a rendered preview image of the scene.
        object_name: Human-readable name of the object.
        scene_name: Human-readable name of the scene.
        vlm_config: VLM config dict (same format as call_vlm).

    Returns:
        dict with keys: compatible (bool), confidence (float), reasoning (str).
        None on API failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        return None

    from src.vlm.prompts import COMPATIBILITY_SYSTEM_PROMPT

    api_key = os.environ.get(
        vlm_config.get("api_key_env", "LETSUR_API_KEY"),
        "sk-PloeoBXIjFj3xnflrpanUA",
    )

    client = OpenAI(
        base_url=vlm_config["api_base"],
        api_key=api_key,
    )

    img_b64 = encode_image_base64(scene_preview_path)
    user_text = (
        f"Would the following object look natural if placed in this scene?\n"
        f"Object: {object_name}\n"
        f"Scene: {scene_name}"
    )

    retries = vlm_config.get("retry_attempts", 3)
    delay = vlm_config.get("retry_delay_seconds", 2)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=vlm_config["model"],
                messages=[
                    {"role": "system", "content": COMPATIBILITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=vlm_config.get("max_tokens", 256),
                temperature=vlm_config.get("temperature", 0.2),
            )

            text = response.choices[0].message.content.strip()
            return parse_vlm_json(text)

        except Exception as e:
            log.warning(f"Compatibility check attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    log.error("Compatibility check failed after all retries")
    return None
