#!/usr/bin/env python3
"""Step 5 -- mine failure cases for the reports.

    python scripts/05_mine.py                  # -> results/failures/candidates.jsonl
    python scripts/05_mine.py --list           # list one candidate per class
    python scripts/05_mine.py --show-cases     # print the five report cases

With no explicit paths, smoke runs and dry-run stub rows are skipped when real
model results are present.

The reports typeset these rows as text cards (exact prompt, full response,
model, ground truth) in both HTML and LaTeX; no frame images are rendered here.
Failure classes are heuristics over the result rows so that selection is
reproducible rather than hand-picked. They are starting points for the
analysis, not conclusions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402
from vsibench_eval import failure_cases as F  # noqa: E402

# A model that declines to answer is not making a scale error, and it is not a
# formatting problem either: it is abstention, and it deserves its own class.
_REFUSAL = re.compile(
    r"cannot (?:determine|provide|estimate|tell)|can't (?:determine|provide|estimate)|"
    r"unable to|not (?:possible|able|enough information)|insufficient information|"
    r"don't know|do not know|no way to",
    re.IGNORECASE)


def classify(r: dict, task_median_pred: float | None) -> str | None:
    """Return a coarse failure class, or None if the item is not a failure."""
    gt = C.to_float(r.get("ground_truth"))
    strict = r.get("strict_score", 0.0) or 0.0
    lenient = r.get("lenient_score", 0.0) or 0.0
    pred = r.get("lenient_parsed")
    raw = (r.get("raw_output") or "").strip()

    if strict > 0.999:
        return None                                   # correct under official rules

    # 1. the answer was right in substance but the official scorer could not see it
    if lenient > 0.999 and strict < 0.5:
        return "format_or_unit"

    if _REFUSAL.search(raw):
        return "refusal"

    if r["task"] == "route_planning":
        if pred and pred != r["ground_truth"]:
            return "egocentric_direction"
        return "format_or_unit"

    # numeric tasks
    if pred is None:
        return "format_or_unit"
    if not (gt and gt > 0):
        return None
    rel = abs(pred - gt) / gt
    # MRA gives partial credit, so a 10%-error answer is not a failure case.
    # Treat anything within 30% relative error (roughly MRA >= 0.5) as a near
    # miss and keep the cards for answers that are actually wrong.
    if rel < 0.30:
        return None
    # A prediction pinned near the task's own median while the true value is far
    # away is the prior-anchoring signature; everything else is a plain scale
    # error.
    if task_median_pred and abs(pred - task_median_pred) / task_median_pred < 0.15:
        return "scale_prior_anchor"
    return "scale_confusion"


def load_all(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows += C.read_jsonl(p)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show-cases", action="store_true",
                    help="print the five failure cases used by the reports")
    ap.add_argument("--include-blind", action="store_true",
                    help="also mine blind-arm rows (control arm, no visual input)")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "failures"))
    args = ap.parse_args()

    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        # Same default as 04_score.py: exclude smoke/dry-run rows when real
        # model results are present, but allow a stub-only dry run.
        all_paths = sorted(C.RESULTS_DIR.glob("*.jsonl"))
        paths = [p for p in all_paths
                 if not p.name.startswith(("smoke_", "stub"))]
        if not paths:
            paths = [p for p in all_paths if not p.name.startswith("smoke_")]
    rows = [r for r in load_all(paths) if "error" not in r]
    if not rows:
        raise SystemExit("no results to mine")

    # Blind-arm rows are the no-image control, not examples of normal failure.
    # They are scored in 04_score.py; keep them out of the report gallery unless
    # explicitly requested.
    n_blind = sum(1 for r in rows if r.get("blind"))
    if n_blind and not args.include_blind:
        rows = [r for r in rows if not r.get("blind")]
        print(f"excluding {n_blind} blind-arm rows (control arm); "
              f"use --include-blind to mine them anyway")

    medians = {}
    for task in {r["task"] for r in rows}:
        vals = [r["lenient_parsed"] for r in rows
                if r["task"] == task and isinstance(r.get("lenient_parsed"), (int, float))]
        medians[task] = sorted(vals)[len(vals) // 2] if vals else None

    buckets = defaultdict(list)
    for r in rows:
        c = classify(r, medians.get(r["task"]))
        if c:
            buckets[c].append(r)

    print(f"{len(rows)} answered questions, "
          f"{sum(len(v) for v in buckets.values())} failures")
    for c, v in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:22s} {len(v):5d}  ({len(v)/len(rows)*100:4.1f}%)")

    cand_path = Path(args.out) / "candidates.jsonl"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cand_path, "w") as f:
        for c, v in buckets.items():
            for r in v:
                f.write(json.dumps({**r, "failure_class": c}, ensure_ascii=False) + "\n")
    print(f"\ncandidates -> {cand_path}")
    if args.list:
        for c, v in sorted(buckets.items()):
            r = v[0]
            print(f"\n[{c}] id={r['id']} {r['model']} {r['task']}")
            print(f"   Q: {r['question'][:150]}")
            print(f"   GT={r['ground_truth']} pred={r.get('raw_output')!r}")
        return 0

    if args.show_cases:
        mined = [{**r, "failure_class": c}
                 for c, v in buckets.items() for r in v]
        for i, r in enumerate(F.select_cases(5, rows=mined), 1):
            print(f"\nCase {i} -- {F.class_label(r)}")
            print(f"  {F.model_id_label(r)} | {F.task_label(r)} | id {r['id']}")
            print(f"  GT: {F.ground_truth_text(r)}")
            print(f"  PROMPT:\n{F.prompt_text(r)}")
            print(f"  RESPONSE:\n{F.response_text(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
