"""Emit the shared-manifest results table straight from runs/shared/*/summary.json.

Hand-copying ten numbers per row into docs/baselines.md is how transcription errors get
published, so the table is generated. Column order matches docs/baselines.md.

  python scripts/shared_manifest_table.py                      # every run under runs/shared
  python scripts/shared_manifest_table.py pi05 dp unic gpt5_text    # pick and order rows
"""

import argparse
import json
from pathlib import Path

SHARED = Path("runs/shared")
LABELS = {                       # pretty names for the rows we publish
    "pi05": "pi0.5 @300k", "pi05_270000": "pi0.5 @270k", "dp": "DP (converged)",
    "groot": "GR00T @130k", "groot_210000": "GR00T @210k",
    "gpt5_text": "GPT-5 agent (text goal)", "gpt5_image": "GPT-5 agent (goal image)",
    "gpt5_text_fb": "GPT-5 agent + measured profile", "unic": "UNIC composition",
    "photoflow": "PhotoFlow-style search (24 renders)",
    "photoflow_matched": "PhotoFlow-style search (render-matched)",
}
HEAD = ("policy", "mean imp", "median", "frac+", "d_end", "d_best",
        "@0.05", "@0.10", "@0.20", "shoot", "n")


def row(name, s):
    r = s.get("results") or []
    n = len(r) or s.get("n_rollouts") or 0
    dbest = (sum(x["d_best"] for x in r) / n) if r else float("nan")
    shoot = sum(1 for x in r if x.get("declared_stop") is not None or x.get("declared") is not None)
    at = s.get("success_at") or {}
    g = lambda k: (at.get(k) or {}).get("end", 0.0) * 100.0          # noqa: E731
    return [
        LABELS.get(name, name),
        f"{s.get('mean_improvement_over_noop', float('nan')):+.3f}",
        f"{s.get('median_improvement', float('nan')):+.3f}",
        f"{s.get('frac_positive', float('nan')):.2f}",
        f"{s.get('mean_d_end', float('nan')):.3f}",
        f"{dbest:.3f}",
        f"{g('0.05'):.0f}%", f"{g('0.1'):.0f}%", f"{g('0.2'):.0f}%",
        f"{shoot}/{n}", str(n),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run dir names under runs/shared (default: all)")
    ap.add_argument("--root", default=str(SHARED))
    a = ap.parse_args()
    root = Path(a.root)
    names = a.runs or sorted(p.name for p in root.iterdir()
                             if (p / "summary.json").exists())
    rows, notes = [], []
    for nm in names:
        f = root / nm / "summary.json"
        if not f.exists():
            notes.append(f"- `{nm}`: no summary.json (run unfinished or failed)")
            continue
        s = json.loads(f.read_text())
        rows.append(row(nm, s))
        extra = []
        for k in ("model", "goal_mode", "feedback", "render_budget", "api_calls", "api_failures"):
            if s.get(k) not in (None, False):
                extra.append(f"{k}={s[k]}")
        if extra:
            notes.append(f"- `{nm}`: " + ", ".join(extra))

    w = [max(len(HEAD[i]), *(len(r[i]) for r in rows)) if rows else len(HEAD[i])
         for i in range(len(HEAD))]
    line = lambda c: "| " + " | ".join(c[i].ljust(w[i]) for i in range(len(HEAD))) + " |"  # noqa: E731
    print(line(list(HEAD)))
    print("|" + "|".join("-" * (w[i] + 2) for i in range(len(HEAD))) + "|")
    for r in rows:
        print(line(r))
    if notes:
        print("\n" + "\n".join(notes))


if __name__ == "__main__":
    main()
