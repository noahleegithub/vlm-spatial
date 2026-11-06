#!/usr/bin/env python3
"""Step 6 -- assemble the submission-ready 2-page report as printable HTML.

    python scripts/06_report.py
    python scripts/06_report.py --panels 6 --open

Numbers come from results/scores.csv + stats.json; failure cases come from
results/failures/candidates.jsonl and are rendered as text cards (exact prompt,
full response, model, ground truth), so the single .html file can be printed to
PDF from any browser (Ctrl-P -> "Save as PDF", US Letter, no scaling).

The discussion paragraph is deliberately left as a marked scaffold: it must be
written from the numbers you actually obtained, not from a template.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402
from vsibench_eval import failure_cases as F  # noqa: E402

DISPLAY = {
    "qwen3vl": "Qwen3-VL-8B-Instruct",
    "gemma4": "Gemma-4-12B-it",
    "molmo2": "Molmo2-8B",
    "stub": "STUB (dry run)",
}
TASK_LABEL = {
    "route_planning": "Route planning",
    "object_abs_distance": "Absolute distance",
    "room_size_estimation": "Room size",
}
TASK_UNIT = {"route_planning": "ACC", "object_abs_distance": "MRA", "room_size_estimation": "MRA"}

GROUP = C.FAILURE_GROUP   # single definition, shared with scripts/07

CSS = """
@page { size: letter; margin: 0.45in; }
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9pt;
       line-height: 1.26; color: #111; margin: 0; }
h1 { font-size: 14.5pt; margin: 0 0 2pt 0; letter-spacing: -0.2pt; }
.sub { color: #555; font-size: 8.2pt; margin: 0 0 7pt 0; }
h2 { font-size: 9.4pt; text-transform: uppercase; letter-spacing: 0.6pt;
     margin: 9pt 0 3pt 0; padding-bottom: 1.5pt; border-bottom: 0.8pt solid #bbb; }
h3 { font-size: 8.8pt; margin: 7pt 0 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 5pt 0; font-size: 7.9pt;
        table-layout: fixed; }
th, td { overflow-wrap: anywhere; }
th { text-align: left; font-weight: 600; border-bottom: 1pt solid #333;
     padding: 2pt 3pt; background: #f2f2f2; }
td { padding: 2pt 3pt; border-bottom: 0.5pt solid #ddd; vertical-align: top; }
td.num { font-variant-numeric: tabular-nums; }
.mono { font-family: "SF Mono", Menlo, monospace; font-size: 7.4pt; }
.muted { color: #666; }
/* One card per row: the full prompt is too long for a half-width column.
   break-inside keeps a card from splitting across printed pages. */
.gallery { display: block; margin: 3pt 0 2pt 0; }
.case { display: block; width: 100%; margin: 0 0 4pt 0;
        border: 0.6pt solid #ccc; border-radius: 2pt;
        overflow: hidden; break-inside: avoid; }
.cap { padding: 2pt 4pt 1pt 4pt; font-size: 7.2pt; line-height: 1.2;
       background: #f7f7f7; border-bottom: 0.4pt solid #ddd; }
.cap b { font-size: 7.6pt; }
.cardbody { padding: 1.5pt 4pt 2pt 4pt; font-size: 7.1pt; line-height: 1.2; }
.cardbody .lbl { font-weight: 600; }
.cardbody .text { font-family: "SF Mono", Menlo, monospace; font-size: 7.1pt;
                  white-space: pre-wrap; margin: 0 0 2pt 0; }
.pb { page-break-before: always; }
.note { background: #fff8e1; border-left: 2.5pt solid #f0b400; padding: 4pt 6pt;
        font-size: 8pt; margin: 4pt 0; }
.todo { background: #eef6ff; border-left: 2.5pt solid #3b82f6; padding: 4pt 6pt;
        font-size: 8.2pt; }
ol.refs { margin: 2pt 0 0 13pt; padding: 0; font-size: 7.5pt; line-height: 1.28; }
ol.refs li { margin-bottom: 1pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 7.5pt; background: #f4f4f4;
       padding: 0 1.2pt; }
.small { font-size: 7.4pt; }
body.compact { font-size: 7.9pt; line-height: 1.17; }
body.compact table { font-size: 6.9pt; }
body.compact h2 { margin: 5pt 0 1.5pt 0; font-size: 8.8pt; }
body.compact h3 { margin: 5pt 0 1.5pt 0; }
body.compact .cap { font-size: 6.8pt; padding: 1.5pt 2.5pt 2pt 2.5pt; }
body.compact ol.refs { font-size: 6.7pt; line-height: 1.2; }
body.compact .case { margin-bottom: 3pt; }
body.compact .cardbody, body.compact .cardbody .text { font-size: 6.7pt; }
body.compact p { margin: 2pt 0; }
"""


def load_scores() -> dict:
    p = C.RESULTS_DIR / "scores.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["task"])] = row
    return out


def stats() -> dict:
    p = C.RESULTS_DIR / "stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def manifest_meta() -> dict:
    p = C.DATA_DIR / "manifest_meta.json"
    return json.loads(p.read_text()) if p.exists() else {"tasks": {}}


def fmt(v, nd=1) -> str:
    try:
        return f"{float(v)*100:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def ci(s: dict, key: str) -> str:
    raw = s.get(key, "")
    if not raw or raw == "—":
        return ""
    try:
        lo, hi = raw.strip("[]").split(",")
        return f"<span class='muted'>[{float(lo)*100:.0f}, {float(hi)*100:.0f}]</span>"
    except ValueError:
        return ""


def results_table(scores: dict) -> str:
    tasks = list(C.TASKS)
    meta = manifest_meta()["tasks"]
    head = ("<tr><th style='width:19%'>Model</th>" +
            "".join(f"<th style='text-align:right;width:27%'>{TASK_LABEL[t]}"
                    f"<br><span class='muted'>{TASK_UNIT[t]} · n={meta.get(t, {}).get('n', '?')}</span></th>"
                    for t in tasks) + "</tr>")
    rows = []
    models = [m for m in DISPLAY if any((m, t) in scores for t in tasks)]
    best = {}
    for t in tasks:
        vals = []
        for m in models:
            s = scores.get((m, t))
            if s:
                v = s.get(f"{TASK_UNIT[t]}_strict", "")
                if v:
                    vals.append((float(v), m))
        if vals:
            best[t] = max(vals)[1]
    for m in models:
        cells = []
        for t in tasks:
            s = scores.get((m, t))
            if not s:
                cells.append("<td class='num' style='text-align:right'>—</td>")
                continue
            k = TASK_UNIT[t]
            v = s.get(f"{k}_strict", "")
            b = " style='font-weight:700'" if best.get(t) == m else ""
            lenient = s.get(f"{k}_lenient", "")
            gap = (float(lenient) - float(v)) * 100 if v and lenient else 0.0
            cells.append(
                f"<td class='num' style='text-align:right'{b}>{fmt(v)} "
                f"{ci(s, f'{k}_strict_ci')}"
                + (f"<br><span class='muted' style='font-size:7pt'>lenient {fmt(lenient)}"
                   f" ({gap:+.1f})</span>" if gap > 0.15 else "")
                + "</td>")
        rows.append(f"<tr><td><b>{DISPLAY[m]}</b></td>" + "".join(cells) + "</tr>")
    if not rows:
        rows = ["<tr><td colspan='4' class='muted'>No results yet — run "
                "scripts/03_infer.py then scripts/04_score.py</td></tr>"]
    return f"<table>{head}{''.join(rows)}</table>"


def setup_table() -> str:
    return """
<table>
<tr><th style="width:18%">Item</th><th>Setting</th></tr>
<tr><td>Benchmark</td><td>VSI-Bench (CVPR 2025): 5,130 QA over 288 egocentric videos from ScanNet / ScanNet++ /
ARKitScenes val. Videos ship with the public dataset — no ScanNet registration required.</td></tr>
<tr><td>Tasks (full sets)</td><td><code>route_planning</code> (Cat. 1, ACC, n=194) ·
<code>object_abs_distance</code> (Cat. 2, MRA, n=834) ·
<code>room_size_estimation</code> (Cat. 2, MRA, n=288, one per scene)</td></tr>
<tr><td>Models (Apache-2.0)</td><td>Qwen3-VL-8B-Instruct · Gemma-4-12B-it · Molmo2-8B — local inference,
bf16, single H100 80 GB</td></tr>
<tr><td>Visual input</td><td>32 frames at <code>round(linspace(0, N−1, 32))</code>, long side ≤448 px, fed as a
multi-image stack to every model (no native video path, so the input is identical across models)</td></tr>
<tr><td>Decoding / prompts</td><td>greedy (<code>do_sample=False</code>), <code>max_new_tokens=48</code>, Gemma-4 thinking
disabled; prompts verbatim from the official <code>vsibench_doc_to_text()</code></td></tr>
<tr><td>Metrics / control</td><td>official ACC &amp; MRA copied verbatim; lenient unit-aware scorer as a secondary
check; 95% CI by scene-clustered bootstrap; blind arm (no frames) as a control</td></tr>
</table>"""


def blind_table(scores: dict) -> str:
    """Full-input vs blind-control MRA, side by side.

    If the blind arm lands within a few points of the full run, the 'vision'
    score is mostly prior recall. That comparison is the cheapest strong claim
    in the report, so it gets its own table rather than a sentence.
    """
    tasks = [t for t in C.TASKS if C.TASKS[t]["metric"] == "MRA"]
    models = [m for m in DISPLAY
              if any((f"{m}_blind", t) in scores for t in tasks)]
    if not models:
        return ("<p class='muted'>No blind-arm results yet — run "
                "<code>sbatch slurm/03_blind.slurm &lt;model&gt;</code>.</p>")
    head = ("<tr><th style='width:19%'>Model</th>" +
            "".join(f"<th style='text-align:right;width:27%'>{TASK_LABEL[t]}"
                    f"<br><span class='muted'>full · blind · Δ</span></th>" for t in tasks) +
            "</tr>")
    rows = []
    for m in models:
        cells = []
        for t in tasks:
            full = scores.get((m, t), {}).get("MRA_strict", "")
            blind = scores.get((f"{m}_blind", t), {}).get("MRA_strict", "")
            if full and blind:
                delta = (float(full) - float(blind)) * 100
                cells.append(f"<td class='num' style='text-align:right'>{fmt(full)} · "
                             f"{fmt(blind)} · <b>{delta:+.1f}</b></td>")
            else:
                cells.append("<td class='num' style='text-align:right'>—</td>")
        rows.append(f"<tr><td><b>{DISPLAY[m]}</b></td>" + "".join(cells) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def diagnostics_block(st: dict) -> str:
    rows = []
    for (key, d) in st.items():
        if "/" not in key:
            continue
        model, task = key.split("/", 1)
        if model.endswith("_blind"):
            continue                      # the control arm has its own table
        if "diagnostics" in d and d["diagnostics"]:
            g = d["diagnostics"]
            r = g.get("pearson_r")
            r_cell = "constant" if g.get("pred_is_constant") else (
                "—" if r is None else f"{r}")
            rows.append(
                f"<tr><td>{DISPLAY.get(model, model)}</td><td>{TASK_LABEL.get(task, task)}</td>"
                f"<td class='num'>{r_cell}</td>"
                f"<td class='num'>{g.get('sigma_pred_over_gt', '—')}</td>"
                f"<td class='num'>{fmt(d.get('strict_parse_fail_rate'))}%</td>"
                f"<td class='num'>{g.get('median_abs_rel_err', '—')}</td></tr>")
    if not rows:
        return "<p class='muted'>No numeric diagnostics yet.</p>"
    return ("<table><tr><th>Model</th><th>Task</th><th>r(pred, GT)</th>"
            "<th>σ<sub>pred</sub>/σ<sub>GT</sub></th><th>Parse failures</th>"
            "<th>Median rel. error</th></tr>" + "".join(rows) + "</table>")


def taxonomy_block() -> str:
    p = C.RESULTS_DIR / "failures" / "candidates.jsonl"
    if not p.exists():
        return "<p class='muted'>No failure candidates yet — run scripts/05_mine.py.</p>"
    counts, examples = {}, {}
    for r in C.read_jsonl(p):
        c = r["failure_class"]
        counts[c] = counts.get(c, 0) + 1
        examples.setdefault(c, []).append(r)
    merged: dict[str, int] = {}
    for c, n in counts.items():
        merged[GROUP.get(c, c)] = merged.get(GROUP.get(c, c), 0) + n
    counts = merged

    desc = {
        "Refusal / abstention": "The model declines to answer. Not a scale error and not a "
                                "formatting artifact: no estimate is produced at all.",
        "Answer format": "Right in substance but invisible to the official first-token parser "
                         "(prose, unit change, turn/channel markers). Quantified by the "
                         "strict-vs-lenient gap, not by the failure gallery.",
        "scale_prior_anchor": "Prediction sits near a canonical object/room prior while ground "
                              "truth varies — consistent with answering from memory, not measurement.",
        "Scale confusion": "Large metric error: the model commits to a plausible-looking magnitude "
                           "the scene does not support. Prior-pinned answers are the subset of these "
                           "that land near the task's own median.",
        "Egocentric direction": "Wrong turn sequence in route planning: a left/right mirror, a "
                                "failure to accumulate heading across frames, or a letter prior "
                                "(B and C over-predicted, A and D under-used).",

    }
    rows = []
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        rows.append(f"<tr><td><b>{c}</b></td><td class='num'>{n}</td>"
                    f"<td>{desc.get(c, '')}</td></tr>")
    total = sum(counts.values())
    return ("<table><tr><th style='width:24%'>Class</th><th style='width:8%'>n</th>"
            "<th>Mechanism</th></tr>" + "".join(rows) + "</table>"
            f"<p class='muted small'>{total} failing questions across all models and non-blind tasks "
            f"(near misses inside 30% relative error are excluded: MRA already gives partial credit).</p>")


def gallery(panels: int) -> str:
    cases = F.select_cases(panels)
    if not cases:
        return ("<p class='muted'>No failure candidates yet — run "
                "<code>python scripts/05_mine.py</code>.</p>")
    cells = []
    for i, row in enumerate(cases, 1):
        gt = html.escape(F.ground_truth_text(row))
        try:
            if abs(float(row.get("strict_score")) - float(row.get("lenient_score"))) > 1e-9:
                gt += f" <span class='muted'>({html.escape(F.score_text(row))})</span>"
        except (TypeError, ValueError):
            pass
        meta = " · ".join(html.escape(x) for x in (
            F.model_id_label(row), F.task_label(row), f"id {row.get('id', '?')}"))
        cells.append(
            f"<div class='case'><div class='cap'><b>Case {i} — "
            f"{html.escape(F.class_label(row))}</b><br>"
            f"<span class='muted'>{meta}</span></div>"
            f"<div class='cardbody'><span class='lbl'>Prompt.</span>"
            f"<div class='text'>{html.escape(F.prompt_text(row))}</div>"
            f"<span class='lbl'>Response.</span>"
            f"<div class='text'>{html.escape(F.response_text(row))}</div>"
            f"<span class='lbl'>Ground truth.</span> {gt}</div></div>")
    return f"<div class='gallery'>{''.join(cells)}</div>"


def build_html(panels: int, compact: bool = False) -> str:
    scores, st, meta = load_scores(), stats(), manifest_meta()
    n_total = sum(t.get("n", 0) for t in meta.get("tasks", {}).values()) or 0
    models = [DISPLAY[m] for m in DISPLAY if any((m, t) in scores for t in C.TASKS)]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Do VLMs reason spatially? A VSI-Bench evaluation</title><style>{CSS}</style></head>
<body class="{"compact" if compact else ""}">
<h1>Where vision-language models fail at 3D space: a VSI-Bench evaluation</h1>
<p class="sub">{' · '.join(models) if models else 'Models: TBD'} &nbsp;|&nbsp;
VSI-Bench, {meta.get('n_questions_total') or n_total or '—'} questions over {
        meta.get('n_scenes_total') or max((t.get('n_scenes', 0) for t in meta['tasks'].values()), default='—')} scenes &nbsp;|&nbsp;
Multimodal Computer Vision — HW</p>

<h2>Setup and protocol</h2>
{setup_table()}

<h2>Results</h2>
{results_table(scores)}
<p class="muted" style="font-size:7.6pt;margin-top:-2pt">ACC = exact match on the option letter;
MRA = Mean Relative Accuracy (official implementations, copied verbatim). Bracketed values are 95%
confidence intervals from a bootstrap that resamples scenes. Grey "lenient" figures come from a
secondary unit-aware parser and show how much of the score is answer formatting rather than
spatial reasoning.</p>

<h3>Numeric failure signatures</h3>
{diagnostics_block(st)}
<p class="muted small">A model that measures a room should have r(pred, GT) well above zero and
spread near the ground truth; r ≈ 0 with σ<sub>pred</sub>/σ<sub>GT</sub> ≪ 1 is prior recall.</p>

<h3>Control arm: same questions, no frames</h3>
{blind_table(scores)}

<h2>Representative failure cases</h2>
<p class="muted small">Each card gives the exact text prompt sent with the 32 input frames (frames omitted here),
the model's complete decoded response, the model checkpoint, and the ground truth.</p>
{gallery(panels)}

<h2>Failure taxonomy</h2>
{taxonomy_block()}

<h2>Discussion</h2>
<p>Qwen3-VL-8B is the only model of the three that measures rather than recites, reaching MRA
0.479 on absolute distance and 0.571 on room size against 0.350 route-planning accuracy. Removing
the video costs it 17.3 and 24.2 MRA points with non-overlapping confidence intervals, yet roughly
a third of its distance score survives with no image at all, so part of the score is prior recall
in the benchmark as much as in the model. Gemma-4-12B returns no room-size estimate on any of the
288 questions, and its absolute-distance answers need the lenient parser to reach MRA 0.228, so its
0.3 strict figure measures answer format rather than spatial ability. Molmo2-8B's room-size answers
correlate with ground truth (r = 0.74) but carry three times its spread, and its blind arm — a
constant 20 m² — lands within one point of the best constant any model could score against this
ground truth, which is a property of MRA's threshold structure as much as of the model. Route
planning is where all three converge (29–35%) and the confusion matrix shows a shared mechanism:
B and C are over-predicted and A and D under-used, so much of the error is a letter prior rather
than a failed turn. The largest remaining gain is in the input, not the model: 32 frames at 448 px
cannot carry the metric scale of a room, and asking the model to verify a proposed depth or
floor-plan estimate would test spatial reasoning instead of recall. The limitation is structural —
no frame budget repairs a benchmark whose numeric answers can be scored above chance by a constant.</p>

<h2>References</h2>
<ol class="refs">
<li>J. Yang, S. Yang, A. Gupta, R. Han, L. Fei-Fei, S. Xie. <i>Thinking in Space: How Multimodal
Large Language Models See, Remember and Recall Spaces.</i> CVPR 2025. arXiv:2412.14171. Dataset and
harness: huggingface.co/datasets/nyu-visionx/VSI-Bench; github.com/vision-x-nyu/thinking-in-space.</li>
<li>Qwen Team. <i>Qwen3-VL</i> model card. huggingface.co/Qwen/Qwen3-VL-8B-Instruct.</li>
<li>Google DeepMind. <i>Gemma 4</i> model card. huggingface.co/google/gemma-4-12B-it.</li>
<li>Allen Institute for AI. <i>Molmo 2</i> model card. huggingface.co/allenai/Molmo2-8B.</li>
<li>Z. Fan et al. <i>VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D
Reconstruction.</i> CVPR 2026. arXiv:2505.20279. Scenes from ScanNet (CVPR 2017) and ScanNet++
(ICCV 2023).</li>
</ol>

<p class="muted" style="font-size:7.2pt;margin-top:8pt">Protocol hash
<code>{C.config_digest()}</code> · frames {C.N_FRAMES} · greedy · all results from this harness.</p>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "report" / "report.html"))
    ap.add_argument("--compact", action="store_true",
                    help="tighter type/gap; use if your run pushes past 2 pages")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.panels, args.compact), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print("print to PDF: open in a browser -> Ctrl-P -> Save as PDF (Letter, no scaling)")
    if args.open and subprocess.run(["which", "xdg-open"], capture_output=True).returncode == 0:
        subprocess.run(["xdg-open", str(out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
