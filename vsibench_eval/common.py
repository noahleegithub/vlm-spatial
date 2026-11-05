"""Shared config, IO, prompt construction and official metric code.

The metric functions below (fuzzy_matching, exact_match, abs_dist_norm,
mean_relative_accuracy) are copied VERBATIM from the official VSI-Bench
evaluation harness:

    https://github.com/vision-x-nyu/thinking-in-space
    lmms_eval/tasks/vsibench/utils.py   (lines 82-101)

They are reproduced bit-for-bit so that our numbers are comparable to the
published ones. Do not "improve" them; the lenient scorer lives in
lenient_parse()/lenient_score() and is reported as a SECONDARY metric.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

# ----------------------------------------------------------------------------
# Paths / config (all overridable by env var so the cluster job script owns it)
# ----------------------------------------------------------------------------

SCRATCH = Path(os.environ.get("SCRATCH", os.path.expanduser("~")))
ROOT = Path(os.environ.get("VSI_ROOT", SCRATCH / "vsi"))
HF_HOME = Path(os.environ.get("HF_HOME", ROOT / "hf"))
VIDEO_DIR = Path(os.environ.get("VSI_VIDEO_DIR", HF_HOME / "vsibench"))
CACHE_DIR = Path(os.environ.get("VSI_CACHE", ROOT / "cache"))
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULTS_DIR = Path(os.environ.get("VSI_RESULTS", ROOT / "results"))
FRAME_DIR = CACHE_DIR / "frames"

HF_DATASET = "nyu-visionx/VSI-Bench"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

# Fixed protocol constants. Changing these invalidates comparability.
N_FRAMES = 32
MAX_NEW_TOKENS = 48
FRAME_MAX_SIDE = 448          # cap per-frame resolution; ~256 vision tokens @28px patches
SEED = 1234

# The three tasks we evaluate. Category mapping is from the assignment brief.
TASKS = {
    "route_planning":        {"metric": "ACC", "category": 1, "unit": None},
    "object_abs_distance":   {"metric": "MRA", "category": 2, "unit": "m"},
    "room_size_estimation":  {"metric": "MRA", "category": 2, "unit": "m2"},
}

# Verbatim from the official harness.
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

# Verbatim default prompts from vsibench_doc_to_text().
PRE_PROMPT = "These are frames of a video."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."


# ----------------------------------------------------------------------------
# Official metric code -- VERBATIM (do not edit)
# ----------------------------------------------------------------------------

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def official_score(question_type: str, raw_pred: str, ground_truth: str) -> float:
    """Replicates vsibench_process_results() + the per-type metric dispatch."""
    if question_type in MCA_QUESTION_TYPES:
        return exact_match(fuzzy_matching(raw_pred), ground_truth)
    if question_type in NA_QUESTION_TYPES:
        pred = to_float(fuzzy_matching(raw_pred))
        gt = to_float(ground_truth)
        if pred is None or gt is None or gt == 0:
            return 0.0                       # WORST_CASE_FOR_METRICS
        return float(mean_relative_accuracy(pred, gt, start=.5, end=.95, interval=.05))
    raise ValueError(f"Unknown question type: {question_type}")


# ----------------------------------------------------------------------------
# Lenient, unit-aware secondary scorer
# ----------------------------------------------------------------------------

_MC_LETTER = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

_UNIT_FACTORS = {          # -> metres / m^2
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}
_SQ_HINT = re.compile(r"square\s*(meter|metre|foot|feet|inch|centimeter)s?|sq\.?\s*(m|ft|in|cm)|m2|m²|ft2|ft²", re.I)

# area factors -> m^2 (room_size_estimation's unit)
_AREA = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "m2": 1.0, "m²": 1.0,
    "ft": 0.09290304, "foot": 0.09290304, "feet": 0.09290304, "ft2": 0.09290304, "ft²": 0.09290304,
    "in": 0.00064516, "inch": 0.00064516, "inches": 0.00064516,
    "cm": 0.0001, "centimeter": 0.0001, "centimeters": 0.0001,
    "yd": 0.836127, "yard": 0.836127, "yards": 0.836127,
}
_AREA_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:square\s*|sq\.?\s*)?"
    r"(meters?|metres?|m2|m²|feet|foot|ft2|ft²|inches?|in|centimeters?|cm|yards?|yd)\b", re.I)


def _parse_area(text: str) -> float | None:
    m = _AREA_RE.search(text)
    if not m:
        return None
    tok = m.group(2).lower()
    if tok in _AREA:
        return float(m.group(1)) * _AREA[tok]
    return None



def _strip_channels(text: str) -> str:
    """Remove Gemma-4 reasoning channel markers: <|channel>thought ... <channel|>."""
    if "<channel|>" in text:
        text = text.rsplit("<channel|>", 1)[-1]
    text = re.sub(r"<\|?channel\|?>?", " ", text)
    return text.strip()


def lenient_parse(raw_pred: str, question_type: str, ground_truth: str):
    """Return a best-effort parsed numeric prediction (in the GT's unit) or None.

    This deliberately fixes what the official scorer cannot: prose answers,
    explicit unit tokens, and unit conversion. Reported as a SECONDARY metric.
    """
    text = _strip_channels(raw_pred or "")

    if question_type in MCA_QUESTION_TYPES:
        m = _MC_LETTER.search(text)
        return m.group(1).upper() if m else None

    gt_unit = TASKS.get(question_type, {}).get("unit", "m")
    clean = text.replace(",", "")
    if not _NUM.search(clean):
        return None

    if gt_unit == "m2":
        area = _parse_area(clean)
        if area is not None:
            return area
        return float(_NUM.search(clean).group(0))

    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z²]+)?", clean):
        tok = (m.group(2) or "").lower().strip(".")
        if tok in _UNIT_FACTORS:
            return float(m.group(1)) * _UNIT_FACTORS[tok]
    return float(_NUM.search(clean).group(0))


def lenient_score(question_type: str, raw_pred: str, ground_truth: str) -> float:
    if question_type in MCA_QUESTION_TYPES:
        p = lenient_parse(raw_pred, question_type, ground_truth)
        return 1.0 if p is not None and p.lower() == ground_truth.lower() else 0.0
    p = lenient_parse(raw_pred, question_type, ground_truth)
    gt = to_float(ground_truth)
    if p is None or gt is None or gt == 0:
        return 0.0
    return float(mean_relative_accuracy(p, gt, start=.5, end=.95, interval=.05))


# ----------------------------------------------------------------------------
# Prompt construction -- port of vsibench_doc_to_text()
# ----------------------------------------------------------------------------

def build_prompt(row: dict, with_video: bool = True) -> str:
    pre_prompt = PRE_PROMPT if with_video else ""
    qtype = row["question_type"]
    question = row["question"]

    if qtype in NA_QUESTION_TYPES:
        parts = [p for p in (pre_prompt, question, NA_POST_PROMPT) if p]
        return "\n".join(parts)
    if qtype in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(row["options"])
        parts = [p for p in (pre_prompt, question, options, MCA_POST_PROMPT) if p]
        return "\n".join(parts)
    raise ValueError(f"Unknown question type: {qtype}")


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_manifest(task: str) -> list[dict]:
    p = MANIFEST_DIR / f"{task}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"missing manifest {p} -- run scripts/01_prepare.py first")
    return read_jsonl(p)


def video_path(dataset: str, scene: str) -> Path:
    return VIDEO_DIR / dataset / f"{scene}.mp4"


def scene_frames_dir(dataset: str, scene: str) -> Path:
    return FRAME_DIR / dataset / scene


def frame_paths(dataset: str, scene: str) -> list[str]:
    d = scene_frames_dir(dataset, scene)
    files = sorted(d.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(
            f"no cached frames in {d} -- run scripts/02_frames.py first")
    return [str(f) for f in files]


def config_digest() -> str:
    blob = json.dumps({
        "n_frames": N_FRAMES, "max_new_tokens": MAX_NEW_TOKENS,
        "frame_max_side": FRAME_MAX_SIDE, "pre": PRE_PROMPT,
        "na_post": NA_POST_PROMPT, "mca_post": MCA_POST_PROMPT,
        "seed": SEED,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------
# Statistics: cluster bootstrap over scenes
# ----------------------------------------------------------------------------

def clustered_bootstrap_ci(pairs: list[tuple[str, float]], n_boot: int = 2000,
                           alpha: float = 0.05) -> tuple[float, float]:
    """95% CI resampling SCENES (not questions) -- questions cluster in scenes."""
    rng = np.random.default_rng(SEED)
    by_scene: dict[str, list[float]] = {}
    for scene, s in pairs:
        by_scene.setdefault(scene, []).append(s)
    scenes = list(by_scene)
    if len(scenes) < 2:
        m = float(np.mean([s for _, s in pairs])) if pairs else float("nan")
        return (m, m)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(scenes), size=len(scenes), replace=True)
        vals = [v for i in pick for v in by_scene[scenes[i]]]
        stats.append(float(np.mean(vals)))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def numeric_diagnostics(preds: list[float], gts: list[float]) -> dict:
    """Evidence for prior-anchoring / scale-collapse: correlation + spread ratio."""
    if len(preds) < 3:
        return {}
    p, g = np.array(preds, float), np.array(gts, float)
    ok = np.isfinite(p) & np.isfinite(g)
    p, g = p[ok], g[ok]
    if len(p) < 3 or g.std() == 0:
        return {}
    # A model answering with one number every time has no correlation to
    # measure; that is itself the strongest possible prior-anchoring signal.
    constant = bool(p.std() == 0)
    r = None if constant else float(np.corrcoef(p, g)[0, 1])
    return {
        "pearson_r": None if r is None else round(r, 4),
        "pred_is_constant": constant,
        "sigma_pred_over_gt": round(float(p.std() / g.std()), 4),
        "gt_sigma": round(float(g.std()), 4),
        "pred_sigma": round(float(p.std()), 4),
        "median_abs_rel_err": round(float(np.median(np.abs(p - g) / g)), 4),
        "n_valid_numeric": int(len(p)),
    }
