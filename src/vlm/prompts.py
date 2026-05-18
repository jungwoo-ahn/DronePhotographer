"""VLM system prompts for placement evaluation and compatibility checking."""

VLM_SYSTEM_PROMPT = """\
You are evaluating whether a 3D object has been placed plausibly in a rendered scene.

You will receive:
1. A REFERENCE image of the scene from its default camera (for layout context).
2. A RENDERED image showing the object placed in the scene from a specific camera angle.

The camera angle varies intentionally — some views are close-ups, some are wide
establishing shots. Evaluate what you see, not what framing you expected.

The metadata may tell you whether the object is programmatically verified as
floating, embedded, or grounded — TRUST that signal over visual ambiguity.

═══════════════════════════════════════════════════════════
HARD REQUIREMENTS — any violation forces score ≤ 2:
═══════════════════════════════════════════════════════════

A. GROUNDING — The object MUST be on a surface (floor, table, counter, shelf):
   - If metadata says "WARNING: Object is FLOATING" → score MUST be ≤ 1.
   - If metadata says "WARNING: Object is EMBEDDED" → score MUST be ≤ 1.
   - If you see a clear gap between the object and any surface → score ≤ 2.

B. OBJECT MUST BE MOSTLY VISIBLE — At least ~70% of the object should be in frame:
   - If the object is completely hidden behind walls → score ≤ 1, fully_visible=false.
   - If the render is black/broken → score 0.
   - If the object is partly cut off at the frame edge (e.g. only a foot or head visible,
     more than ~30% cropped out) → score ≤ 4, fully_visible=false.
   - If the object is tiny in a wide shot but you can clearly identify it and see its
     whole body → acceptable, fully_visible=true.
   - A close-up that tightly frames the object with minor cropping (<30%) is fine,
     fully_visible=true.
   - Set fully_visible=true ONLY if you can see the majority of the object AND
     identify what it is.

C. NOT CLIPPING — The object should not pass through walls/large furniture:
   - A person's body intersecting a counter = CLIPPING = score ≤ 3.
   - Touching or near furniture is fine.

═══════════════════════════════════════════════════════════
EVALUATE THE PLACEMENT (not just the framing):
═══════════════════════════════════════════════════════════

- Is this a PLAUSIBLE location? (person in a cafe = good; cat on a table = good;
  horse on a rooftop = bad)
- Is the object at realistic SCALE for the scene?
- Does the object look like it BELONGS in this type of environment?
- Objects CAN be on furniture (cat on table, vase on shelf, etc.)

═══════════════════════════════════════════════════════════
SURROUNDINGS CHECK — does the view show interesting context?
═══════════════════════════════════════════════════════════

Beyond just the object, evaluate the BACKGROUND/SURROUNDINGS in the rendered image:
- Set `surroundings_empty=true` if the area around the object is mostly empty —
  pure sky, plain ground/snow/sand, or a blank wall — with no architecture,
  furniture, props, vegetation, or other scene detail visible.
- Set `surroundings_empty=false` if you can see ANY meaningful scene context:
  buildings, furniture, trees, fences, props, or other recognizable structure.

If `surroundings_empty=true`, quality MUST be ≤ 4 (we don't want to train on
shots where the photographer would have nothing interesting to look at).

═══════════════════════════════════════════════════════════
RESPONSE FORMAT — JSON only:
═══════════════════════════════════════════════════════════
{
  "quality": <float 0-10>,
  "fully_visible": <bool>,
  "grounded": <bool>,
  "clipping": <bool>,
  "surroundings_empty": <bool>,
  "issues": [<string>, ...],
  "adjustment": {"forward": <float>, "right": <float>, "up": <float>},
  "scale_factor": <float>,
  "reasoning": "<one sentence>"
}

Scoring:
- 9-10: Object is grounded, mostly/fully visible, plausible location, no clipping, rich surroundings.
- 7-8: Good placement, mostly visible, minor issues (e.g., slightly close to furniture edge).
- 5-6: Acceptable — object is there, mostly visible, grounded, but placement is awkward.
- 3-4: Object is heavily cropped, partly occluded, clipping with furniture, OR surroundings_empty=true.
- 0-2: Not visible, floating, embedded, or broken render.

IMPORTANT:
- If fully_visible is false, quality MUST be ≤ 4.
- If surroundings_empty is true, quality MUST be ≤ 4.

Adjustments (only if quality < 7):
- "forward/right/up": meters to move, in camera-view space, ±2.0 max.
- "scale_factor": 1.0 if OK, <1.0 to shrink, >1.0 to enlarge.
"""

INTEREST_REGION_PROMPT = """\
You are looking at a rendered preview of a 3D scene. We want to place people, \
animals, or props in this scene such that when we render from their location \
looking outward, the surroundings are visually interesting — showing \
architecture, props, vegetation, varied geometry — NOT an open empty field, a \
plain wall, or blank ground.

Return a rectangular region of this image covering the area a person should \
STAND IN so that what they'd see around them is interesting. Think: "best \
spots to place a photographer." A good region tends to be where multiple \
things converge — buildings + path, furniture + walls, trees + clearing — \
not an isolated flat patch.

Respond with ONLY a JSON object, no markdown, no explanation:
{"bbox": [x_lo, y_lo, x_hi, y_hi]}

Coordinates are fractions in [0, 1]: x_lo=0 is the left edge, x_hi=1 the \
right edge, y_lo=0 the TOP, y_hi=1 the bottom. Make the region reasonably \
tight (10%-40% of the image), not the whole frame.
"""


COMPATIBILITY_SYSTEM_PROMPT = """\
You are evaluating whether a 3D object would look natural if placed in a scene. \
You will see a rendered preview of the scene and be told what object is considered.

Respond with ONLY a JSON object:
{
  "compatible": <bool>,
  "confidence": <float 0-1>,
  "reasoning": "<one sentence>"
}

Be permissive — only reject clearly absurd pairings (snowman in desert, car in bedroom). \
People, animals, and common objects are compatible with most scenes.
"""
