#!/usr/bin/env python3
"""Self-test -- no GPU, no model weights, no network.

    python scripts/00_selftest.py

Checks the three things that quietly destroy a benchmark number if they drift:
the official metric code, the prompt strings, and the lenient parser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402

fails: list[str] = []


def check(name: str, got, want, tol: float = 1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and isinstance(got, (int, float)) \
        else (got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        fails.append(name)


print("== official metric code (verbatim from thinking-in-space)")
# fuzzy_matching takes the FIRST whitespace token only -- this is the quirk
# that makes prose answers score zero, and it is what we report as a finding.
check("fuzzy_matching('About 75 cm')", C.fuzzy_matching("About 75 cm"), "About")
check("fuzzy_matching('75 cm.')", C.fuzzy_matching("75 cm."), "75")
check("exact_match case-insensitive", C.exact_match("c", "C"), 1.0)
check("abs_dist_norm", C.abs_dist_norm(2.9, 2.9), 0.0)
check("MRA exact hit", C.mean_relative_accuracy(2.9, 2.9, .5, .95, .05), 1.0)
check("MRA 10% high", C.mean_relative_accuracy(3.19, 2.9, .5, .95, .05), 9 / 10)
check("MRA 60% high", C.mean_relative_accuracy(4.64, 2.9, .5, .95, .05), 0.0)
# The official code yields 10 thresholds, matching the paper's 1/10 formula:
# int(0.45/0.05 + 2) truncates 10.999999999999998 -> 10, giving linspace(.5,.95,10).
_thr = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))
check("n_thresholds", len(_thr), 10)
check("threshold step", round(float(_thr[1] - _thr[0]), 3), 0.05)

print("== official_score dispatch")
check("MCA correct", C.official_score("route_planning", "C", "C"), 1.0)
check("MCA prose -> 0 (documented quirk)", C.official_score("route_planning", "The answer is C", "C"), 0.0)
check("NA prose -> 0 (documented quirk)", C.official_score("object_abs_distance", "About 2.9", "2.9"), 0.0)

print("== lenient parser (unit aware; GT in metres / m^2)")
for task, raw, gt, want in [
    ("object_abs_distance", "About 2.9", "2.9", 2.9),
    ("object_abs_distance", "290 cm", "2.9", 2.9),
    ("object_abs_distance", "2.9 m", "2.9", 2.9),
    ("object_abs_distance", "approximately 9.5 feet", "2.9", 2.8956),
    ("object_abs_distance", "no idea", "2.9", None),
    ("room_size_estimation", "26.4", "26.4", 26.4),
    ("room_size_estimation", "26.4 square meters", "26.4", 26.4),
    ("room_size_estimation", "about 284 square feet", "26.4", 26.384),
    ("route_planning", "<|channel>thought\nhmm\n<channel|>C", "C", "C"),
    ("route_planning", "The answer is C", "C", "C"),
    ("route_planning", "B. Turn Left, Turn Left", "B", "B"),
]:
    got = C.lenient_parse(raw, task, gt)
    if want is None:
        check(f"{task}: {raw!r} -> None", got is None, True)
    elif isinstance(want, str):
        check(f"{task}: {raw!r}", got, want)
    else:
        check(f"{task}: {raw!r}", None if got is None else round(got, 3), round(want, 3))

print("== the lenient scorer recovers what the official one misses")
check("strict 0 / lenient 1 on 'About 2.9'",
      (C.official_score("object_abs_distance", "About 2.9", "2.9"),
       C.lenient_score("object_abs_distance", "About 2.9", "2.9")), (0.0, 1.0))

print("== prompt construction (must match vsibench_doc_to_text verbatim)")
na_row = {"question_type": "object_abs_distance", "question": "How far is the sofa from the stove (in meters)?", "options": None}
mca_row = {"question_type": "route_planning", "question": "You are a robot ...", "options": ["A. Turn Left", "B. Turn Right"]}
check("NA prompt",
      C.build_prompt(na_row),
      "These are frames of a video.\nHow far is the sofa from the stove (in meters)?\n"
      "Please answer the question using a single word or phrase.")
check("MCA prompt",
      C.build_prompt(mca_row),
      "These are frames of a video.\nYou are a robot ...\nOptions:\nA. Turn Left\nB. Turn Right\n"
      "Answer with the option's letter from the given choices directly.")
check("blind prompt drops the frame prefix",
      C.build_prompt(na_row, with_video=False).startswith("How far"),
      True)

print("== clustered bootstrap")
pairs = [("s1", 1.0)] * 10 + [("s2", 0.0)] * 10
lo, hi = C.clustered_bootstrap_ci(pairs, n_boot=300)
check("CI brackets the mean", lo <= 0.5 <= hi, True)

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("all checks passed")
