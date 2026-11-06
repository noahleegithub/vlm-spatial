"""Shared selection and formatting for the report's failure-case cards.

Both report builders (the printable HTML and the NeurIPS LaTeX) read the same
``candidates.jsonl`` produced by ``scripts/05_mine.py``.  Keeping the selection
and the prompt/response/ground-truth formatting here means the two reports
cannot disagree about which cases they show or what those cases say.

The five cases are curated from the rule-based candidates for legibility:
one per mechanism, spread across the three models and the three tasks.  If a
curated row is missing (for example after a partial rerun), the selector falls
back to the first candidate of the same class, preferring the same family of
model.
"""

from __future__ import annotations

from pathlib import Path

from . import common as C

MODEL_LABEL = {
    "qwen3vl": "Qwen3-VL-8B-Instruct",
    "gemma4": "Gemma-4-12B-it",
    "molmo2": "Molmo2-8B",
}

TASK_LABEL = {
    "route_planning": "Route planning",
    "object_abs_distance": "Absolute distance",
    "room_size_estimation": "Room size",
}

CLASS_LABEL = {
    "scale_confusion": "Scale confusion",
    "scale_prior_anchor": "Scale confusion (prior anchoring)",
    "egocentric_direction": "Egocentric direction",
    "refusal": "Refusal / abstention",
    "format_or_unit": "Answer format",
}

# (failure_class, model key, question id).  These are the cases reviewed for
# the report: every model and task is represented, and the format case is one
# where the value is right but the official first-token parser misses it.
CURATED_CASES = (
    ("scale_confusion", "molmo2", 533),        # room size: GT 30.9 m2 -> 100
    ("scale_prior_anchor", "qwen3vl", 825),    # distance: GT 3.6 m -> 1.7
    ("egocentric_direction", "qwen3vl", 4961),  # route: GT A -> B
    ("refusal", "gemma4", 530),                # room size: refuses
    ("format_or_unit", "gemma4", 682),          # distance: 2.15 m prose, GT 2.1
)

_PREFERRED_MODELS = {
    "scale_confusion": ("molmo2", "qwen3vl", "gemma4"),
    "scale_prior_anchor": ("qwen3vl", "molmo2", "gemma4"),
    "egocentric_direction": ("qwen3vl", "molmo2", "gemma4"),
    "refusal": ("gemma4",),
    "format_or_unit": ("gemma4", "molmo2"),
}


def candidates_path() -> Path:
    return C.RESULTS_DIR / "failures" / "candidates.jsonl"


def load_candidates(path: Path | str | None = None) -> list[dict]:
    """Read the mined failure rows (empty list when the file does not exist)."""
    return C.read_jsonl(Path(path) if path else candidates_path())


def _key(row: dict) -> tuple:
    return (row.get("failure_class"), row.get("model"), row.get("id"))


def _pick(rows: list[dict], failure_class: str) -> dict | None:
    if not rows:
        return None
    if failure_class == "format_or_unit":
        good = [r for r in rows
                if (r.get("lenient_score") or 0) >= 0.999
                and (r.get("strict_score") or 0) < 0.5]
        rows = good or rows
        return sorted(rows, key=lambda r: (r.get("id", 0)))[0]
    if failure_class == "refusal":
        return sorted(rows, key=lambda r: (len(response_text(r)), r.get("id", 0)))[0]
    if failure_class == "scale_confusion":
        rooms = [r for r in rows if r.get("task") == "room_size_estimation"]
        rows = rooms or rows
        return sorted(rows, key=lambda r: (r.get("id", 0)))[0]
    if failure_class == "scale_prior_anchor":
        distances = [r for r in rows if r.get("task") == "object_abs_distance"]
        rows = distances or rows
        return sorted(rows, key=lambda r: (r.get("id", 0)))[0]
    return sorted(rows, key=lambda r: (r.get("id", 0)))[0]


def _fallback(rows: list[dict], failure_class: str, used: set[tuple]) -> dict | None:
    available = [r for r in rows
                 if r.get("failure_class") == failure_class and _key(r) not in used]
    for model in _PREFERRED_MODELS.get(failure_class, ()):
        picked = _pick([r for r in available if r.get("model") == model], failure_class)
        if picked is not None:
            return picked
    return _pick(available, failure_class)


def select_cases(n: int = 5, rows: list[dict] | None = None) -> list[dict]:
    """Return up to ``n`` curated failure rows, with a rule-based fallback."""
    rows = load_candidates() if rows is None else rows
    by_key = {}
    for row in rows:
        by_key.setdefault(_key(row), row)

    chosen: list[dict] = []
    used: set[tuple] = set()
    for failure_class, model, qid in CURATED_CASES:
        row = by_key.get((failure_class, model, qid))
        if row is None:
            row = _fallback(rows, failure_class, used)
        if row is not None and _key(row) not in used:
            chosen.append(row)
            used.add(_key(row))
        if len(chosen) >= n:
            return chosen[:n]

    # A caller may ask for more than the five curated cases.  Fill from the
    # remaining candidates in file order rather than dropping the request.
    for row in rows:
        if len(chosen) >= n:
            break
        if _key(row) not in used:
            chosen.append(row)
            used.add(_key(row))
    return chosen[:n]


def class_label(row: dict) -> str:
    cls = row.get("failure_class", "case")
    return CLASS_LABEL.get(cls, str(cls).replace("_", " "))


def task_label(row: dict) -> str:
    task = row.get("task", "")
    return TASK_LABEL.get(task, str(task).replace("_", " "))


def model_label(row: dict) -> str:
    short = MODEL_LABEL.get(row.get("model"), str(row.get("model", "unknown")))
    model_id = row.get("model_id") or ""
    if model_id and model_id not in short:
        return f"{short} ({model_id})"
    return short


def model_id_label(row: dict) -> str:
    """Compact, unambiguous checkpoint name for narrow report columns."""
    model_id = row.get("model_id")
    if model_id:
        return str(model_id)
    return MODEL_LABEL.get(row.get("model"), str(row.get("model", "unknown")))


def prompt_text(row: dict) -> str:
    return str(row.get("prompt") or "").strip("\n")


def response_text(row: dict) -> str:
    raw = row.get("raw_output")
    return "(no response recorded)" if raw is None else str(raw)


def ground_truth_text(row: dict, latex: bool = False) -> str:
    """Ground truth with a readable unit or the full multiple-choice option."""
    gt = str(row.get("ground_truth", "")).strip()
    for option in row.get("options") or []:
        text = str(option).strip()
        if text == gt or text.startswith((f"{gt}.", f"{gt} ", f"{gt})")):
            return text
    unit = C.TASKS.get(row.get("task", ""), {}).get("unit")
    if unit == "m":
        return f"{gt} m"
    if unit == "m2":
        return f"{gt} m$^2$" if latex else f"{gt} m²"
    return gt


def score_text(row: dict) -> str:
    def fmt(value) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "--"

    return f"{fmt(row.get('strict_score'))} strict / {fmt(row.get('lenient_score'))} lenient"
