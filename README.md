# VSI-Bench spatial-reasoning evaluation

Coursework harness for "evaluate the 3D spatial reasoning of VLMs". Three
open-weight models, three VSI-Bench tasks, one frozen protocol.

| | |
|---|---|
| **Models** | `Qwen/Qwen3-VL-8B-Instruct`, `google/gemma-4-12B-it`, `allenai/Molmo2-8B` (all Apache-2.0) |
| **Tasks** | `route_planning` (ACC, n=194), `object_abs_distance` (MRA, n=834), `room_size_estimation` (MRA, n=288) |
| **Input** | 32 uniformly sampled frames as a multi-image input, 448 px long side |
| **Decoding** | greedy, `max_new_tokens=48` |
| **Metrics** | official VSI-Bench ACC / MRA (verbatim), plus a lenient unit-aware scorer as a secondary check |
| **Control** | blind arm: same questions, no frames |

See `PROTOCOL.md` for the frozen decisions and their justifications.

## Quick start

```bash
# 0) once, on the LOGIN node (needs network)
module purge
module load GCCcore/13.3.0 Python/3.12.3
bash envs/setup_envs.sh
source envs/activate.sh qwen3vl
export HF_HOME="$SCRATCH/hf"
mkdir -p "$HF_HOME"
hf download Qwen/Qwen3-VL-8B-Instruct
hf download google/gemma-4-12B-it
hf download allenai/Molmo2-8B

# 1) sanity check the metric code and prompts (no GPU, 1 s)
python scripts/00_selftest.py

# 2) data + frames (CPU job, ~5.7 GB download, ~40 min)
DATA_JOB=$(sbatch --parsable slurm/00_data.slurm)
echo "data job: $DATA_JOB"

# 3-5) for each model: smoke after the data job, then full run and blind arm.
# The dependencies prevent GPU jobs from starting before the frames exist.
for M in qwen3vl gemma4 molmo2; do
  SMOKE_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB \
      slurm/01_smoke.slurm "$M")
  sbatch --dependency=afterok:$SMOKE_JOB slurm/02_infer.slurm "$M"
  sbatch --dependency=afterok:$DATA_JOB slurm/03_blind.slurm "$M"
done

# 6) wait for every job to finish, then score/report
squeue -u "$USER"                               # should be empty
source envs/aces_modules.sh                     # if this is a new shell
source envs/activate.sh qwen3vl
bash envs/freeze.sh                              # -> envs/lock-<date>.txt (record these versions)
python scripts/04_score.py                       # -> results/scores.csv, table.md, stats.json
python scripts/05_mine.py                        # -> results/failures/candidates.jsonl
python scripts/05_mine.py --show-cases           # print the five report cases
python scripts/06_report.py                      # -> report/report.html (print to PDF)
```

Step 0 needs a new shell after logging in; `source envs/activate.sh qwen3vl`
is what puts `hf`, `python`, `numpy` and `Pillow` on PATH.  The job scripts load
the same module stack themselves, so they do not depend on the submitting shell.
If an inference job exits on a few failed questions, resubmit the same `sbatch`
command: successful rows are skipped and the error rows are retried.

`report/neurips/main.pdf` is the NeurIPS-2026-format version (same two pages, same numbers):
`python scripts/07_report_tex.py && ~/.local/bin/tectonic -X compile report/neurips/main.tex`.
It needs the style file `report/neurips_2026.sty` and a tectonic binary. Failure cases are typeset
as text cards, not images: each card shows the exact prompt, the full response, the model checkpoint,
and the ground truth.

`report/report.html` is a single self-contained file. Print it: open in a browser,
Ctrl-P, US Letter, no scaling, "print backgrounds" on. It is built for two pages. With the full three-model table, five
failure cards and a populated taxonomy, build it with `--compact` and check the
browser's print preview before submitting.

## Local dry run (no GPU, no torch)

The whole data path, prompt construction and scoring can be exercised without a
GPU. This is exactly what was validated before handing the project over. On
ACES, activate the venv first (`source envs/activate.sh qwen3vl`); on a laptop,
use a Python with `numpy`. If you run this on ACES, point
`VSI_ROOT` at a separate scratch directory
(`export VSI_ROOT=$SCRATCH/vsi_dryrun`) so the dry-run manifests do not replace
the full ones:

```bash
python scripts/01_prepare.py --datasets scannetpp
python scripts/02_frames.py --limit-scenes 50   # all 50 ScanNet++ scenes
python scripts/03_infer.py --model stub --limit 6
python scripts/04_score.py
python scripts/05_mine.py --show-cases
```

`--model stub` returns canned answers but still walks the real frame loader,
prompt builder and both scorers.

## Layout

```
envs/aces_modules.sh        ACES module stack (GCCcore + Python 3.12)
vsibench_eval/common.py     config, verbatim official metrics, lenient parser, bootstrap CI
vsibench_eval/failure_cases.py  curated failure selection + prompt/GT formatting
vsibench_eval/adapters.py   the three model adapters + a StubAdapter for dry runs
scripts/00_selftest.py      asserts the metric code, prompts and parser (no GPU)
scripts/01_prepare.py       fetch + verify + write task manifests
scripts/02_frames.py        deterministic 32-frame extraction (ffmpeg select)
scripts/03_infer.py         inference, one JSON row per question, resume-safe
scripts/04_score.py         ACC/MRA + cluster-bootstrap CI + diagnostics
scripts/05_mine.py          failure classification + candidates.jsonl
scripts/06_report.py        assembles the 2-page printable report
slurm/                      job scripts (ACES: --partition=gpu --gres=gpu:h100:1)
```

## Environment notes

* ACES loads **no modules by default**. `envs/setup_envs.sh` and the Slurm jobs
  source `envs/aces_modules.sh`, which loads `GCCcore/13.3.0` and
  `Python/3.12.3`. If you run login-side commands manually, load the same stack
  first (or run `source envs/aces_modules.sh`); otherwise the venv may not have
  the newer `libstdc++` that PyTorch needs. If HPRC retires those versions,
  check `module spider Python/3.12.3` and `module spider FFmpeg/7.0.2` and update
  the module names in `envs/aces_modules.sh` / `slurm/00_data.slurm`.
* Home on ACES is **10 GB / 10,000 files and not expandable**. `HF_HOME`, the
  venvs, and all data live on `$SCRATCH`, and `setup_envs.sh` redirects the
  pip/uv/HuggingFace caches there too. Do not sync the local `_local/` directory
  (about 2 GB) to ACES home. If the login node's one-hour CPU limit interrupts
  the environment build, rerun it in a short interactive CPU job with
  `WebProxy` loaded; completed downloads are cached on scratch.
* Weights are fetched on the login node and the GPU jobs run with
  `HF_HUB_OFFLINE=1`. The data job is the only job that needs compute-node
  egress; it loads `WebProxy` before downloading. `00_data.slurm` also loads
  `FFmpeg/7.0.2`, because `scripts/02_frames.py` shells out to `ffmpeg`/`ffprobe`.
* The Slurm scripts set `--account=158685321433` (the allocation used by
  `../trustfall`). Change that line if your allocation differs.
* Smoke outputs go to `results/smoke/`, and `04_score.py`/`05_mine.py` ignore
  smoke and stub rows when real model results are present. The full-run and
  blind jobs write only their own JSONL; scoring is run once at the end.
* Blind-arm rows are scored (model label `<model>_blind`) and shown in the
  report's control table, but `05_mine.py` excludes them from failure mining by
  default: they are the no-image control, not examples of normal failure. Use
  `--include-blind` only if you want prior-anchoring cases and will label them
  as such.
* `envs/setup_envs.sh` does not pin `transformers` for the `main` env, so a
  rebuild months later can resolve differently. Run `bash envs/freeze.sh` after
  the smoke test and quote those versions in the report.
* Molmo 2 pins `transformers==4.57.1` and needs `trust_remote_code=True`, so it
  gets its own interpreter; Qwen 3-VL and Gemma 4 share a recent one.

## Result row schema

Each line of `results/<model>.jsonl` is one question:

```json
{"model": "qwen3vl", "id": 4959, "task": "route_planning", "dataset": "arkitscenes",
 "scene": "42446167", "question": "...", "options": ["A. ...", "..."],
 "ground_truth": "C", "prompt": "...", "raw_output": "C",
 "n_frames": 32, "source_frame_indices": [0, 41, ...], "source_frames": 1337,
 "content_key": "image=path", "strict_score": 1.0, "lenient_score": 1.0,
 "strict_parsed": "C", "lenient_parsed": "C", "seconds": 0.87,
 "config_digest": "ab12cd34ef56"}
```

`config_digest` hashes the frozen protocol constants, so any accidental change
to frame count, prompts or decoding shows up as a different digest in the rows.
