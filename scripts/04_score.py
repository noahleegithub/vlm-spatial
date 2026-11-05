#!/usr/bin/env python3
"""Step 4 -- score results, with cluster-bootstrap CIs and diagnostics.

    python scripts/04_score.py                        # every results/*.jsonl
    python scripts/04_score.py results/qwen3vl.jsonl

With no explicit paths, smoke runs and dry-run stub rows are skipped when real
model results are present, so the final table contains only the full/blind arms.

Emits
  results/scores.csv     one row per model x task (strict + lenient + CI)
  results/table.md       the markdown table for the report
  results/stats.json     numeric diagnostics, confusion matrices, parse failures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402


def analyse(rows: list[dict], task: str) -> dict:
    meta = C.TASKS[task]
    clean = [r for r in rows if "error" not in r]
    n_err = len(rows) - len(clean)

    strict = [(r["scene"], r["strict_score"]) for r in clean]
    lenient = [(r["scene"], r["lenient_score"]) for r in clean]
    s_mean = sum(s for _, s in strict) / len(strict) if strict else float("nan")
    l_mean = sum(s for _, s in lenient) / len(lenient) if lenient else float("nan")
    s_ci = C.clustered_bootstrap_ci(strict)
    l_ci = C.clustered_bootstrap_ci(lenient)

    per_ds = defaultdict(list)
    for r in clean:
        per_ds[r["dataset"]].append(r["strict_score"])

    out = {
        "task": task, "metric": meta["metric"], "category": meta["category"],
        "n": len(clean), "n_scenes": len({r["scene"] for r in clean}), "n_errors": n_err,
        f"{meta['metric']}_strict": round(s_mean, 4),
        f"{meta['metric']}_strict_ci": [round(x, 4) for x in s_ci],
        f"{meta['metric']}_lenient": round(l_mean, 4),
        f"{meta['metric']}_lenient_ci": [round(x, 4) for x in l_ci],
        "lenient_minus_strict": round(l_mean - s_mean, 4),
        "per_dataset": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_ds.items())},
    }

    if meta["metric"] == "MRA":
        preds = [r.get("lenient_parsed") for r in clean]
        gts = [C.to_float(r["ground_truth"]) for r in clean]
        pairs = [(p, g) for p, g in zip(preds, gts)
                 if isinstance(p, (int, float)) and g is not None]
        if pairs:
            out["diagnostics"] = C.numeric_diagnostics([p for p, _ in pairs],
                                                       [g for _, g in pairs])
        out["strict_parse_failures"] = sum(1 for r in clean
                                           if r.get("strict_parsed") is None)
        out["strict_parse_fail_rate"] = round(
            out["strict_parse_failures"] / len(clean), 4) if clean else None
    else:
        conf = Counter()
        for r in clean:
            p = r.get("lenient_parsed")
            conf[(r["ground_truth"], p if p else "?")] += 1
        out["confusion"] = {f"gt={g}->pred={p}": c for (g, p), c in sorted(conf.items())}
        out["pred_class_balance"] = {k: v for k, v in
                                     Counter(r.get("lenient_parsed") for r in clean).items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        # Exclude smoke runs (kept in results/smoke/ or legacy smoke_*.jsonl)
        # and dry-run stub rows whenever real model results exist.  If the only
        # thing present is a stub run, score it so the local dry run still works.
        all_paths = sorted(C.RESULTS_DIR.glob("*.jsonl"))
        paths = [p for p in all_paths
                 if not p.name.startswith(("smoke_", "stub"))]
        if not paths:
            paths = [p for p in all_paths if not p.name.startswith("smoke_")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no result files in {C.RESULTS_DIR}")

    by_model_task = defaultdict(list)
    for p in paths:
        for r in C.read_jsonl(p):
            by_model_task[(r["model"] + ("_blind" if r.get("blind") else ""), r["task"])].append(r)

    records, all_stats = [], {}
    for (model, task), rows in sorted(by_model_task.items()):
        if task not in C.TASKS:
            continue
        st = analyse(rows, task)
        st["model"] = model
        records.append(st)
        all_stats[f"{model}/{task}"] = st
        k = st["metric"]
        lo, hi = st[f"{k}_strict_ci"]
        print(f"{model:16s} {task:22s} {k} strict={st[f'{k}_strict']:.3f} "
              f"[{lo:.3f},{hi:.3f}] lenient={st[f'{k}_lenient']:.3f} n={st['n']}")

    if not records:
        raise SystemExit("nothing scored")

    # ---- csv -------------------------------------------------------------
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["model", "task", "metric", "category", "n", "n_scenes", "n_errors",
            "ACC_strict", "ACC_strict_ci", "ACC_lenient", "ACC_lenient_ci",
            "MRA_strict", "MRA_strict_ci", "MRA_lenient", "MRA_lenient_ci",
            "lenient_minus_strict", "strict_parse_fail_rate"]
    with open(C.RESULTS_DIR / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            for k in ("ACC_strict_ci", "ACC_lenient_ci", "MRA_strict_ci", "MRA_lenient_ci"):
                if isinstance(row.get(k), list):
                    row[k] = f"[{row[k][0]:.3f},{row[k][1]:.3f}]"
            w.writerow(row)

    # ---- markdown table --------------------------------------------------
    models = sorted({r["model"] for r in records})
    lines = ["| Model | Task | n | " + " | ".join(
        f"{t} ({C.TASKS[t]['metric']})" for t in C.TASKS) + " |",
        "|---|---|---|" + "---|" * len(C.TASKS)]
    for m in models:
        cells = []
        for t in C.TASKS:
            r = next((x for x in records if x["model"] == m and x["task"] == t), None)
            if not r:
                cells.append("–")
                continue
            k = r["metric"]
            lo, hi = r[f"{k}_strict_ci"]
            cells.append(f"**{r[f'{k}_strict']*100:.1f}** [{lo*100:.1f}, {hi*100:.1f}]")
        lines.append(f"| {m} | all | {sum(r['n'] for r in records if r['model']==m)} | "
                     + " | ".join(cells) + " |")
    (C.RESULTS_DIR / "table.md").write_text("\n".join(lines) + "\n")

    (C.RESULTS_DIR / "stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"\nwrote {C.RESULTS_DIR}/scores.csv, table.md, stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
