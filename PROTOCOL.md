# Protocol (frozen)

Everything in this file is part of the experiment. Changing any of it changes
the numbers, so it is hashed into `config_digest` and recorded in every result
row. This is the text the reproducibility section of the report is written from.

## Benchmark

VSI-Bench (`nyu-visionx/VSI-Bench`, public, not gated), 5,130 questions over 288
egocentric videos from the ScanNet, ScanNet++, and ARKitScenes validation sets.
No ScanNet registration is needed: the videos ship inside the dataset repo.

Selected tasks — full sets, no sampling:

| task | category | metric | n | scenes |
|---|---|---|---|---|
| `route_planning` | 1 — navigation / egocentric | ACC | 194 | 118 |
| `object_abs_distance` | 2 — 3D geometry | MRA | 834 | 231 |
| `room_size_estimation` | 2 — 3D geometry | MRA | 288 | 288 |

`room_size_estimation` has exactly one question per scene, which makes it the
cleanest per-scene measure in the benchmark.

## Input construction

* Exactly **32 frames** per video, chosen at
  `round(linspace(0, n_frames-1, 32))` over the decoded frame count, extracted
  with ffmpeg's `select` filter (single pass, no fps resampling drift).
  The exact source-frame indices are stored per scene in `cache/frames/*/frames.json`
  and copied into every result row.
* Frames resized so the long side is ≤ **448 px** (~256 vision tokens per frame
  at 28 px patches, ≈8.2 k tokens for the stack).
* All three models receive the same **multi-image** input. None uses native
  video ingestion. Rationale: it holds the visual input constant across three
  different video pipelines, and it matches how the VSI-Bench paper evaluated
  its 15 baselines. Consequence, disclosed in the report: models with native
  temporal encoders may be understated relative to their native-video results.

## Prompts

Verbatim from the official `vsibench_doc_to_text()`:

* shared prefix: `These are frames of a video.`
* numeric-answer suffix: `Please answer the question using a single word or phrase.`
* multiple-choice suffix: `Answer with the option's letter from the given choices directly.`
* multiple-choice questions include `Options:\n` followed by the option list.

The blind control arm keeps everything except the frame prefix and the frames.

## Decoding

Greedy (`do_sample=False`), `max_new_tokens=48`. This deliberately overrides
Gemma 4's recommended sampling (T=1.0, top_p=0.95, top_k=64): greedy is what the
benchmark's own protocol uses, and a benchmark comparison should not contain
sampling noise. Disclose the deviation.

Gemma 4 runs with thinking **disabled** (no `<|think|>` token). Its output still
carries an empty `<|channel>thought<channel|>` wrapper, which the scorer strips.

## Metrics

**Primary — official, copied verbatim** from
`thinking-in-space/lmms_eval/tasks/vsibench/utils.py`:

* MCA: `exact_match(fuzzy_matching(pred), gt)` where
  `fuzzy_matching(p) = p.split(' ')[0].rstrip('.').strip()`
* NA: `mean_relative_accuracy(pred, gt, start=.5, end=.95, interval=.05)`
  = mean over `linspace(0.5, 0.95, 10)` of `1[|pred-gt|/gt ≤ 1-θ]`, i.e. 10
  thresholds θ = 0.50, 0.55, …, 0.95, so an answer scores 0.5 when it is within
  25% relative error and 1.0 only within 5%. The official code computes the
  count as `int((end-start)/interval + 2)`, which truncates to 10 because
  `0.45/0.05` evaluates to `8.999999999999998`; the paper's `1/10` is correct
  and the code agrees with it.
* unparseable numeric answers score 0 (`WORST_CASE_FOR_METRICS`)

**Secondary — lenient, unit-aware.** Extracts the first numeric token that
carries a unit, converts to the ground-truth unit (m, or m² for room size),
strips reasoning-channel markers, and accepts prose. Reported alongside the
official number, never instead of it.

The gap between the two is itself a result: the official `fuzzy_matching` takes
only the first whitespace-delimited token, so `"About 75 cm"` scores 0 and
`"1.5 m"` against a 150 cm ground truth scores 0. That is a measurement
property, not a spatial-reasoning failure, and the report separates the two.

**Aggregation.** Micro average (every question weighted equally) within each
task, matching the paper. Note that lmms-eval's own reported "overall" is a
macro average over question types; do not compare the two silently.

**Uncertainty.** 95% CI from a cluster bootstrap that resamples **scenes**, not
questions (questions cluster within scenes: 834 distance questions live in 231
videos). 2,000 resamples, seed 1234.

## Diagnostics used in the failure analysis

| diagnostic | what it evidences |
|---|---|
| Pearson r(pred, gt) per task | whether the model reads the scene at all |
| σ_pred / σ_gt per task | prior anchoring: a model answering from priors has near-zero spread while GT varies |
| strict-parse failure rate | how much of MRA is answer formatting |
| blind-arm MRA vs full-input MRA | whether vision contributes anything |
| GT→pred confusion matrix (MCA) | systematic left/right mirroring, not random error |

## Failure classes (heuristic selection, human-checked)

`format_or_unit` — correct in substance, invisible to the official scorer.
`scale_prior_anchor` — prediction sits near a canonical value while GT varies.
`egocentric_direction` — wrong turn sequence in route planning.
`tolerance_boundary` — within 30% relative error, still outside the MRA thresholds.

`scripts/05_mine.py` proposes candidates by rule so selection is reproducible;
the five cases that appear in the report are chosen from those for legibility.
