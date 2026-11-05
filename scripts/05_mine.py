#!/usr/bin/env python3
"""Step 5 -- mine failure cases and render the figure panels for the report.

    python scripts/05_mine.py --list                 # candidates only, no images
    python scripts/05_mine.py --per-category 2       # render 2 examples per class
    python scripts/05_mine.py --ids 41,233,900       # render specific question ids

With no explicit paths, smoke runs and dry-run stub rows are skipped when real
model results are present.

Each rendered PNG is a single self-contained panel: an 8-frame filmstrip of the
actual input (labelled with the source frame index), then the question, the
ground truth, and the model's raw answer. Those panels are what the report
needs -- "visual input, question, ground truth, and model prediction".

Failure classes are heuristics over the result rows so that selection is
reproducible rather than hand-picked. They are starting points for the
analysis, not conclusions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402

N_SHOWN = int(__import__("os").environ.get("VSI_PANEL_FRAMES", "6"))

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
    # miss and keep the gallery for answers that are actually wrong.
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


def render(r: dict, out_path: Path) -> None:
    """Print-sized panel: 960 px wide with ~30 px text.

    The size is not arbitrary. At 3.6 in printed width (half of a letter page in
    a two-column gallery) a 960 px image renders at 267 dpi, so a 30 px glyph
    lands at 30 * 72 / 267 = 8.1 pt -- legible. Screenshot-scale panels with
    9 px text become unreadable 2 pt smudges in the PDF.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(cand, sz)
            except OSError:
                pass
        return ImageFont.load_default()

    f_small, f_body, f_head = font(20), font(30), font(32)
    frames = C.frame_paths(r["dataset"], r["scene"])
    idx = (r.get("source_frame_indices") or list(range(len(frames))))
    step = max(1, len(frames) // N_SHOWN)
    pick = list(range(0, min(len(frames), step * N_SHOWN), step))[:N_SHOWN]

    W, pad, gap = 960, 10, 2
    thumb_w = (W - 2 * pad - gap * (len(pick) - 1)) // len(pick)
    thumb_h = int(thumb_w * 0.62)
    strip_h = thumb_h + 26
    strip_y = pad

    def wrap_ans():
        return (f"GT {r.get('ground_truth','')}   ·   "
                f"Pred {(r.get('raw_output') or '').strip()[:60]}")

    q_lines = textwrap.wrap(f"Q: {r.get('question','')}", width=64)[:4]
    ans_lines = textwrap.wrap(wrap_ans(), width=64)[:2]
    head = f"{r['model']}  ·  {r['task']}  ·  {r['scene']}"
    H = strip_y + strip_h + 10 + 38 + 36 * (len(q_lines) + len(ans_lines)) + pad

    img = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(img)

    for k, fi in enumerate(pick):
        cx = pad + k * (thumb_w + gap)
        im = Image.open(frames[fi]).convert("RGB")
        scale = max(thumb_w / im.width, thumb_h / im.height)
        im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1))
        left = (im.width - thumb_w) // 2
        top = (im.height - thumb_h) // 2
        img.paste(im.crop((left, top, left + thumb_w, top + thumb_h)), (cx, strip_y))
        label = f"{idx[fi]}" if fi < len(idx) else f"{fi}"
        d.text((cx + 2, strip_y + thumb_h + 4), label, fill=(90, 90, 90), font=f_small)

    y = strip_y + strip_h + 10
    d.text((pad, y), head, fill=(0, 0, 0), font=f_head)
    y += 38
    for line in q_lines:
        d.text((pad, y), line, fill=(25, 25, 25), font=f_body)
        y += 36
    for line in ans_lines:
        d.text((pad, y), line, fill=(20, 20, 20), font=f_body)
        y += 36

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  wrote {out_path.name}  ({img.width}x{img.height})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--per-category", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--include-blind", action="store_true",
                    help="also mine blind-arm rows (they have no visual input to show)")
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

    # Blind-arm rows were produced without frames, so a panel built from them
    # would show video the model never saw -- indefensible as "visual input" in
    # a failure-case figure. They are scored in 04_score.py; keep them out of
    # the gallery unless explicitly requested.
    n_blind = sum(1 for r in rows if r.get("blind"))
    if n_blind and not args.include_blind:
        rows = [r for r in rows if not r.get("blind")]
        print(f"excluding {n_blind} blind-arm rows (no visual input); "
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

    sel: list[dict] = []
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        sel = [r for r in rows if r["id"] in want]
    elif args.per_category:
        for c, v in sorted(buckets.items()):
            v = sorted(v, key=lambda r: (r["strict_score"], r["lenient_score"]))
            sel += v[:args.per_category]
    index = {}
    for r in sel:
        c = classify(r, medians.get(r["task"])) or "unclassified"
        name = f"{c}_{r['model']}_{r['task']}_{r['id']}.png"
        render(r, Path(args.out) / name)
        index[name] = {"class": c, "model": r["model"], "task": r["task"], "id": r["id"]}
    if index:
        idx_path = Path(args.out) / "index.json"
        existing = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        existing.update(index)
        idx_path.write_text(json.dumps(existing, indent=2))
    print(f"\nrendered {len(sel)} panels into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
