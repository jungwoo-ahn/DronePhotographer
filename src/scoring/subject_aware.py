from __future__ import annotations

SUBJECT_AWARE_SCORE_KEYS = [
    "subject_dominance",
    "rule_of_thirds",
    "lead_room",
    "figure_ground_separation",
    "leading_lines",
    "avoiding_merges",
    "center_of_gravity",
    "clear_margins",
]

SUBJECT_AWARE_DESCRIPTIONS = {
    "subject_dominance": "Subject is clearly the hero, large enough, not touching frame edges.",
    "rule_of_thirds": "Important subject feature is aligned with thirds lines or intersections.",
    "lead_room": "More room in front of gaze/motion direction than behind.",
    "figure_ground_separation": "Subject pops from background via contrast/color/blur separation.",
    "leading_lines": "Scene lines guide the viewer's eye toward the subject.",
    "avoiding_merges": "No distracting background overlaps that visually cut through the subject.",
    "center_of_gravity": "Visual weight is balanced and does not feel tilted to one side.",
    "clear_margins": "Safe margins keep limbs/details from accidental cropping.",
}


def build_subject_aware_scoring_prompt() -> str:
    lines = [
        "Score this image for subject-aware composition.",
        "Return only JSON with these keys and values in [0, 1].",
    ]
    for key in SUBJECT_AWARE_SCORE_KEYS:
        lines.append(f"- {key}: {SUBJECT_AWARE_DESCRIPTIONS[key]}")
    return "\n".join(lines)
