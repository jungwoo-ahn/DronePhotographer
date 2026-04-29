"""Expected metric sizes for known object categories (no bpy dependency)."""

from __future__ import annotations

# Known object dimensions in metric (height in meters)
_OBJECT_SIZE_RULES = [
    # (keywords, expected_height_meters, label)
    (("vase", "flower-vase"), 0.30, "vase"),
    (("kitten", "domestic-cat", "wooden-cat", "cat-i", "cat_v"), 0.35, "cat"),
    (("puppy",), 0.30, "puppy"),
    (("german-shepherd", "german-shepard", "sheltie", "shepherd"), 0.55, "dog"),
    (("dog",), 0.55, "dog"),
    (("snowman", "snow-man"), 1.50, "snowman"),
    (("arabian-horse", "horse-beige", "horse"), 1.60, "horse"),
    (("deer",), 1.40, "deer"),
    (("tiger",), 0.95, "tiger"),
    (("tree",), 6.00, "tree"),
    (("dahlia", "orchid", "flower"), 0.40, "flower"),
    # "cooler" alone — be specific. "standing-cool" is too generic and
    # matched assets like "standing-cool-bald-man" (a person, not a cooler).
    (("cooler", "ice-box", "ice-cooler"), 0.90, "cooler"),
    (("porsche", "pourche", "mclaren", "lamborghini", "cabriolet",
      "wagon-car", "cartoon-car"), 1.40, "car"),
    # Sitting / crouching / kneeling / squatting people — bbox height is
    # ~1.0m, not 1.7m. MUST come before the standing-person rule below
    # (first match wins). Note: the auto-scaler in normalize_to_metric now
    # also has a wide ±2× tolerance so even if a sitting-person isn't
    # caught here, it won't be wrongly rescaled — this label mostly affects
    # downstream VLM prompts.
    (("sitting", "seated", "on-chair", "on-stool", "on-bench",
      "kneeling", "kneel", "crouch", "crouching", "squat", "squatting",
      "perched", "buddy-sitting", "shoelaces", "tying-his-shoe"),
     1.05, "person-sitting"),
    (("andrew", "charles", "john", "koky", "luke", "wenceslavus", "utel",
      "spider-man", "explorer", "scout", "survivor", "girl", "guy", "man",
      "boy", "rigged", "character", "rp_posed", "standing-cute",
      "standing-girl", "young", "brazilian", "cyberpunk", "anime",
      "lowpoly-man", "low-poly-man", "boitumelo", "mia", "tomas", "victor",
      "oscar", "moma", "business", "farmer", "cute"), 1.70, "person"),
]


def expected_object_height_m(object_path) -> tuple[float, str]:
    """Look up the expected metric height for an object based on its filename.

    Returns (height_meters, category_label).
    Defaults to 1.70m (person) if no keyword matches.
    """
    name = str(object_path).lower()
    for keywords, height, label in _OBJECT_SIZE_RULES:
        if any(kw in name for kw in keywords):
            return height, label
    return 1.70, "unknown(default-person)"
