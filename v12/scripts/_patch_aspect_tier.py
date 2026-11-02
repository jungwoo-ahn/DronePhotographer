"""Add a 2:1 aspect ratio to the "256" resolution tier of the vendored checkout. Idempotent.

`find_closest_target_size` reads VIDEO_RES_SIZE_INFO, but utils.py defines TWO tables with an
identical "256" tier (IMAGE_ and VIDEO_). Both are patched: leaving one behind would make the
two disagree about what a 2:1 frame is, and which one a future code path reads is not something
this script should have to predict.
"""
import sys
from pathlib import Path

ANCHOR = '        "9,16": (192, 320),\n    },'
ADDED = '        "9,16": (192, 320),\n        "2,1": (512, 256),\n    },'

p = Path(sys.argv[1])
s = p.read_text()
if '"2,1": (512, 256)' in s:
    print("[setup] 2:1 aspect ratio already present")
    raise SystemExit(0)
n = s.count(ANCHOR)
if n != 2:
    print(f"[setup] ERROR: expected the '256' tier twice (IMAGE_ and VIDEO_ tables), found {n}"
          f" — the aspect table changed shape; re-read utils.py before patching", file=sys.stderr)
    raise SystemExit(1)
p.write_text(s.replace(ANCHOR, ADDED))
print(f"[setup] added the 2:1 aspect ratio to both '256' tiers")
