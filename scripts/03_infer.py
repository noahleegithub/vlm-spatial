#!/usr/bin/env python3
"""Step 3 -- run inference and log one JSON row per question.

    python scripts/03_infer.py --model qwen3vl
    python scripts/03_infer.py --model gemma4 --tasks route_planning
    python scripts/03_infer.py --model stub --limit 4        # no-GPU dry run
    python scripts/03_infer.py --model qwen3vl --blind        # control: no frames

Resume-safe: successful questions whose id is already in --out are skipped, so
a job that hits the wall clock can simply be resubmitted.  Rows that recorded an
error are dropped and retried on the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qwen3vl", "gemma4", "molmo2", "stub"])
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blind", action="store_true",
                    help="control condition: no frames, question text only")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tag = "_blind" if args.blind else ""
    out_path = Path(args.out or (C.RESULTS_DIR / f"{args.model}{tag}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = C.read_jsonl(out_path)
    done = {r["id"] for r in existing if "error" not in r}
    stale_errors = [r for r in existing if "error" in r]
    if stale_errors:
        # A row that failed because of a transient OOM or a missing frame must
        # be retried on resubmission.  Drop the old error rows so a successful
        # retry does not leave a duplicate (error + success) in the file.
        kept = [r for r in existing if "error" not in r]
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
        print(f"retrying {len(stale_errors)} previously failed rows")

    todo: list[dict] = []
    for t in tasks:
        rows = C.load_manifest(t)
        if args.limit:
            rows = rows[:args.limit]
        todo += [r for r in rows if r["id"] not in done]
    if not todo:
        print(f"nothing to do; {len(done)} rows already in {out_path}")
        return 0
    print(f"{args.model}{' [BLIND]' if args.blind else ''}: {len(todo)} questions "
          f"-> {out_path}  ({len(done)} already done)")

    from vsibench_eval.adapters import build_adapter, MODEL_IDS
    adapter = build_adapter(args.model).load()
    model_id = MODEL_IDS.get(args.model, "stub")
    print(f"loaded {model_id}; content key = {adapter.content_key_used}")

    t0 = time.time()
    rows_out, n_err = [], 0
    for i, row in enumerate(todo, 1):
        scene = row["scene"]
        try:
            frames = [] if args.blind else C.frame_paths(row["dataset"], scene)
            fmeta_p = C.scene_frames_dir(row["dataset"], scene) / "frames.json"
            fmeta = json.loads(fmeta_p.read_text()) if fmeta_p.exists() else {}
            prompt = C.build_prompt(row, with_video=not args.blind)

            t1 = time.time()
            raw = adapter.generate(frames, prompt)
            dt = time.time() - t1

            rec = {
                "model": args.model, "model_id": model_id,
                "task": row["task"], "id": row["id"],
                "dataset": row["dataset"], "scene": scene,
                "question": row["question"], "options": row.get("options"),
                "ground_truth": row["ground_truth"],
                "prompt": prompt, "raw_output": raw,
                "blind": args.blind,
                "n_frames": len(frames),
                "source_frame_indices": fmeta.get("source_frame_indices"),
                "source_frames": fmeta.get("source_frames"),
                "content_key": adapter.content_key_used,
                "strict_score": C.official_score(row["question_type"], raw, row["ground_truth"]),
                "lenient_score": C.lenient_score(row["question_type"], raw, row["ground_truth"]),
                "strict_parsed": C.to_float(C.fuzzy_matching(raw))
                                 if row["question_type"] in C.NA_QUESTION_TYPES
                                 else C.fuzzy_matching(raw),
                "lenient_parsed": C.lenient_parse(raw, row["question_type"], row["ground_truth"]),
                "seconds": round(dt, 3),
                "config_digest": C.config_digest(),
            }
        except Exception as e:                          # noqa: BLE001
            n_err += 1
            rec = {"model": args.model, "model_id": model_id, "task": row["task"],
                   "id": row["id"], "dataset": row["dataset"], "scene": scene,
                   "ground_truth": row["ground_truth"], "error": repr(e),
                   "blind": args.blind, "config_digest": C.config_digest()}
            print(f"  !! id={row['id']} {e}")
            if n_err >= 5:
                C.append_jsonl(out_path, rows_out)
                print("aborting after 5 errors")
                return 2

        rows_out.append(rec)
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(todo) - i) / rate if rate else 0
            print(f"  {i}/{len(todo)}  {rate:.2f} q/s  eta {eta/60:.1f} min  "
                  f"last='{(rec.get('raw_output') or '')[:40]}'")
            C.append_jsonl(out_path, rows_out)          # checkpoint
            rows_out = []

    if rows_out:
        C.append_jsonl(out_path, rows_out)
    print(f"done in {(time.time()-t0)/60:.1f} min, {n_err} errors -> {out_path}")
    return 2 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
