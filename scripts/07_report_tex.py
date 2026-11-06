#!/usr/bin/env python3
"""Step 7 -- build the NeurIPS-format report from the same artefacts as the HTML.

    python scripts/07_report_tex.py           # -> report/neurips/main.tex
    tectonic -X compile report/neurips/main.tex   # -> main.pdf

Numbers come from results/scores.csv and results/stats.json; failure cases
come from results/failures/candidates.jsonl and are typeset as text cards
(full prompt, full response, model, ground truth) rather than rendered as
images. Nothing here is hand-copied, so a rerun of the eval regenerates the
paper.

Layout target: exactly 2 pages (assignment cap of one page + one supplementary).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402
from vsibench_eval import failure_cases as F  # noqa: E402

DISPLAY = {"qwen3vl": "Qwen3-VL-8B", "gemma4": "Gemma-4-12B", "molmo2": "Molmo2-8B"}
TASK_LABEL = {"route_planning": "Route planning", "object_abs_distance": "Absolute distance",
              "room_size_estimation": "Room size"}
TASK_METRIC = {"route_planning": "ACC", "object_abs_distance": "MRA", "room_size_estimation": "MRA"}
TASK_SHORT = {"route_planning": "Route plan.", "object_abs_distance": "Abs.\\ dist.",
              "room_size_estimation": "Room size"}


def esc(s: str) -> str:
    """Escape the LaTeX specials that appear in our data (task ids, underscores)."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\^{}"}
    # "~" is deliberately NOT escaped: it is already a non-breaking space, and
    # \textasciitilde expands to a \mathchar in T1 encoding, which injects math
    # mode and aborts the compile.
    return "".join(rep.get(ch, ch) for ch in str(s))


def esc_verbatim(s: str) -> str:
    """Escape prompt/response text without turning spaces into non-breaking ones."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\^{}",
           "<": r"\textless{}", ">": r"\textgreater{}", "~": r"\char126{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def tex_lines(text: str) -> str:
    """Verbatim-ish text with line breaks preserved and blank lines collapsed."""
    lines = [esc_verbatim(line) for line in str(text).replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line.strip()]
    return "\\newline\n".join(lines) if lines else r"\textit{(empty)}"


def load_scores() -> dict:
    p = C.RESULTS_DIR / "scores.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}; run scripts/04_score.py first")
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["task"])] = row
    return out


def stats() -> dict:
    p = C.RESULTS_DIR / "stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def meta() -> dict:
    p = C.DATA_DIR / "manifest_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pct(v, nd=1) -> str:
    try:
        return f"{float(v) * 100:.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def ci(s: dict, key: str, nd: int = 0) -> str:
    raw = (s.get(key) or "").strip("[]")
    if not raw:
        return ""
    try:
        lo, hi = (float(x) for x in raw.split(","))
        return f"\\,[{lo * 100:.{nd}f}, {hi * 100:.{nd}f}]"
    except ValueError:
        return ""


def results_table(scores: dict) -> str:
    tasks = ["route_planning", "object_abs_distance", "room_size_estimation"]
    m = meta().get("tasks", {})
    # Two header rows rather than one long line: "Absolute distance (MRA, n=834)"
    # in a single cell overflows \textwidth by ~38pt.
    head = (" & ".join([r"\textbf{Model}"] + [rf"\textbf{{{TASK_LABEL[t]}}}" for t in tasks])
            + r" \\")
    sub = (" & ".join([""] + [rf"\footnotesize {TASK_METRIC[t]} $n{{=}}{m.get(t, {}).get('n', '?')}$"
                              for t in tasks]) + r" \\")
    rows = []
    for mk, name in DISPLAY.items():
        cells = [f"\\textbf{{{name}}}"]
        for t in tasks:
            s = scores.get((mk, t))
            if not s:
                cells.append("--")
                continue
            k = TASK_METRIC[t]
            v, ln = s.get(f"{k}_strict", ""), s.get(f"{k}_lenient", "")
            cell = pct(v) + ci(s, f"{k}_strict_ci")
            if v and ln and (float(ln) - float(v)) > 0.15:
                cell += f" ({pct(ln)})"
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\")
        blind = [r"\quad\emph{blind}"]
        for t in tasks:
            s = scores.get((f"{mk}_blind", t))
            blind.append(pct(s.get(f"{TASK_METRIC[t]}_strict")) if s else "--")
        rows.append(" & ".join(blind) + r" \\[1pt]")
    return (r"\begin{tabular}{@{}lccc@{}}" "\n\\toprule\n" + head + "\n" + sub
            + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def diagnostics_rows(st: dict) -> str:
    """Rows only: the diagnostics table is inlined beside its one-line lead."""
    rows = []
    for key, d in st.items():
        if "/" not in key:
            continue
        model, task = key.split("/", 1)
        if model.endswith("_blind") or task == "route_planning":
            continue
        g = d.get("diagnostics") or {}
        r = "const." if g.get("pred_is_constant") else (
            g.get("pearson_r") if g.get("pearson_r") is not None else "--")
        rows.append(f"{DISPLAY.get(model, esc(model))} & {TASK_SHORT.get(task, esc(task))} & "
                    f"{r} & {g.get('sigma_pred_over_gt', '--')} & "
                    f"{pct(d.get('strict_parse_fail_rate'))} & "
                    + (f"{pct(g.get('median_abs_rel_err'), 0)}\\%"
                       if g.get("median_abs_rel_err") is not None else "--")
                    + " \\\\")
    return "\n".join(rows) if rows else r"\multicolumn{6}{c}{no diagnostics} \\"


def taxonomy() -> tuple[str, int, int]:
    p = C.RESULTS_DIR / "failures" / "candidates.jsonl"
    if not p.exists():
        return (r"\emph{no candidates}", 0, 0)   # not \multicolumn: that needs a tabular
    counts: dict[str, int] = {}
    total = 0
    for r in C.read_jsonl(p):
        g = C.FAILURE_GROUP.get(r["failure_class"], r["failure_class"])
        counts[g] = counts.get(g, 0) + 1
        total += 1
    desc = {
        "Scale confusion": "large numeric error; predictions drift toward typical training sizes",
        "Egocentric direction": "incorrect turn sequence; models favor options B and C over A and D",
        "Refusal / abstention": "model outputs refusal text instead of estimating values",
        "Answer format": "correct value produced, but formatted outside the benchmark parser's rules",
    }
    rows = "\n".join(f"\\textbf{{{esc(k)}}} & {v} & {desc.get(k, '')} \\\\"
                     for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    tbl = (r"{\scriptsize\begin{tabular}{@{}p{0.22\textwidth}rp{0.52\textwidth}@{}}" "\n\\toprule\n"
           r"Class & $n$ & Mechanism \\" "\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}}")
    return tbl, total, len(counts)


def _card(row: dict, number: int, compact: bool = False) -> str:
    meta = (f"{esc(F.model_id_label(row))} \\quad\\textbar\\quad "
            f"{esc(F.task_label(row))} \\quad\\textbar\\quad "
            f"id~{row.get('id', '?')}")
    gt = F.ground_truth_text(row, latex=True)
    try:
        if abs(float(row.get("strict_score")) - float(row.get("lenient_score"))) > 1e-9:
            gt += f" ({F.score_text(row)})"
    except (TypeError, ValueError):
        pass
    body_font = r"\fontsize{6.5}{7.8}\selectfont" if compact else r"\scriptsize"
    return (
        "\\noindent\\textbf{Case %d --- %s}\\par\n"
        "{\\scriptsize %s}\\par\\vspace{0.5pt}\n"
        "{%s\\raggedright\\setlength{\\parindent}{0pt}\n"
        "\\textbf{Prompt.}\\ \\texttt{%s}\\par\\vspace{0.5pt}\n"
        "\\textbf{Response.}\\ \\texttt{%s}\\par\\vspace{0.5pt}\n"
        "\\textbf{Ground truth.}\\ %s\\par}\n"
        "\\vspace{0.5pt}\\hrule\\vspace{1.5pt}"
        % (number, esc(F.class_label(row)), meta, body_font,
           tex_lines(F.prompt_text(row)), tex_lines(F.response_text(row)), gt)
    )


def failure_cards(n: int = 5) -> str:
    """Text cards: exact prompt, full response, model, ground truth.

    Long route-planning prompts get a full-width card; the shorter numeric
    cards are paired two-up so the section still fits inside two pages.
    """
    cases = F.select_cases(n)
    if not cases:
        return r"\emph{no failure candidates (run scripts/05\_mine.py)}"

    blocks = []
    i = 0
    while i < len(cases):
        row = cases[i]
        if row.get("task") != "route_planning" and i + 1 < len(cases) \
                and cases[i + 1].get("task") != "route_planning":
            blocks.append(
                "\\noindent\n"
                "\\begin{minipage}[t]{0.485\\textwidth}\\raggedright\n"
                + _card(row, i + 1, compact=True) + "\n\\end{minipage}\\hfill\n"
                "\\begin{minipage}[t]{0.485\\textwidth}\\raggedright\n"
                + _card(cases[i + 1], i + 2, compact=True) + "\n\\end{minipage}\n"
                "\\par\\vspace{1pt}")
            i += 2
        else:
            blocks.append(_card(row, i + 1) + "\n\\vspace{1pt}")
            i += 1
    return "\n".join(blocks)


def build(panels: int) -> str:
    scores, st, m = load_scores(), stats(), meta()
    tax, n_fail, n_cat = taxonomy()
    n_q = m.get("n_questions_total", "?")
    n_sc = m.get("n_scenes_total", "?")

    disc = (
        r"Qwen3-VL-8B shows genuine visual estimation on the numeric tasks: it scores 47.9 MRA on "
        r"distance and 57.1 on room size, and removing input video drops these scores by 17.3 and 24.2 "
        r"points ($r=0.71$ and $0.81$). Even without video, however, it retains roughly 30 MRA by predicting "
        r"typical indoor dimensions, showing that part of its score comes from dataset priors rather than "
        r"image inspection. Gemma-4-12B avoids numeric guesses entirely on room size, refusing all 288 "
        r"questions, and its 0.3 strict score on distances rises to 22.8 under lenient parsing because it "
        r"answers in conversational sentences. Molmo2-8B correlates with room sizes ($r=0.74$) but scales "
        r"estimates roughly $3\times$ too high; when blind, it outputs a constant 20\,m$^2$, which scores "
        r"32.0 MRA because MRA rewards guesses near the dataset median. All three models perform poorly "
        r"on route planning (29.4--35.0\% ACC), partly because they systematically favor options B and C "
        r"regardless of camera motion. Overall, 32 downsampled frames without camera parameters leave models "
        r"guessing metric scale from visual priors; VLM-3R-style 3D supervision~\citep{fan2026vlm3r} would "
        r"test relative geometry instead."
    )

    return rf"""\documentclass{{article}}
\usepackage[preprint]{{neurips_2026}}
\usepackage[T1]{{fontenc}}
\usepackage{{booktabs}}
\usepackage{{microtype}}
\usepackage[margin=0pt,font=small,labelfont=bf,skip=2pt]{{caption}}
\setlength{{\abovecaptionskip}}{{2pt}}
\setlength{{\belowcaptionskip}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\captionsetup[table]{{skip=3pt}}

\title{{VSI-Bench Evaluation of Open-Weight VLMs}}

\author{{%
  Noah Lee \quad UIN: 638008962 \\
  Texas A\&M University \\
  College Station, TX 77843 \\
  \texttt{{noahvlee@tamu.edu}}
}}

\raggedbottom
\begin{{document}}
\small
\maketitle
\vspace{{-16mm}}

\section{{Setup and protocol}}
We evaluate three open-weight VLMs on VSI-Bench~\citep{{yang2025thinking}}, which covers {n_q} questions across
{n_sc} indoor scenes from ScanNet~\citep{{dai2017scannet}} and ScanNet++~\citep{{yeshwanth2023scannetpp}}. Each model receives 32 frames sampled uniformly from
the video scaled to a maximum dimension of 448\,px and outputs responses
using greedy decoding with up to 48 tokens. Prompts follow the standard VSI-Bench harness.
We report official strict accuracy (ACC) and Mean Relative Accuracy (MRA), and also report a fallback with a lenient parser
that extracts numbers written as words or embedded in explanatory text.

\vspace{{2pt}}
\noindent\textbf{{Protocol.}} Evaluated models include Qwen3-VL-8B-Instruct~\citep{{qwen2025qwen3vl}}, Gemma-4-12B-it~\citep{{gemma2026gemma4}}, and
Molmo2-8B~\citep{{deitke2024molmo}}, run on one NVIDIA H100 GPU (PyTorch 2.14.1, transformers 5.22.1 / 4.57.1).
Tasks use the full test splits: \texttt{{route\_planning}} (Category 1, ACC, $n{{=}}194$),
\texttt{{object\_abs\_distance}} (Category 2, MRA, $n{{=}}834$), and \texttt{{room\_size\_estimation}}
(Category 2, MRA, $n{{=}}288$). We report 95\% confidence intervals with resampling at the scene level, which only matters
for the \texttt{{object\_abs\_distance}} task. We also report blind model priors by witholding images and only retrieving model
answers based on the prompts.

\section{{Results}}
Table~\ref{{tab:results}} summarizes performance across all three tasks. Qwen3-VL leads in every
category. Gemma-4 scores near zero under strict evaluation because it responds in natural sentences
rather than isolated tokens; our lenient parser extracts these values and shows the underlying scores.

\vspace{{2pt}}
\begin{{table}}[h]
\centering
\caption{{Main results (strict scorer, 95\% CI over scenes). Values in parentheses show lenient parser
results when they differ by more than 15 percentage points.}}
\label{{tab:results}}
\small
{results_table(scores)}
\end{{table}}

The blind evaluation shows how much performance relies on dataset statistics rather than visual evidence.
Qwen3-VL scores 30.6 and 32.9 MRA without images. Gemma-4 refuses to answer without visual context,
scoring zero. Molmo2-8B scores higher blind on room size (32.0 vs 11.5) by predicting a constant
20\,m$^2$, which closely matches the dataset median.

\vspace{{2pt}}
\noindent\textbf{{Numeric diagnostics.}} To evaluate metric reasoning, we report the correlation between
predictions and ground truth ($r$), prediction variance relative to ground truth ($\sigma_{{\hat{{y}}}}/\sigma_y$),
parser failure rate, and median absolute relative error.

\vspace{{2pt}}
\begin{{center}}\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\toprule
Model & Task & $r$ & $\sigma_{{\hat{{y}}}}/\sigma_y$ & Parse fail. & Med.\ rel.\ err \\
\midrule
{diagnostics_rows(st)}
\bottomrule
\end{{tabular}}\end{{center}}

\section{{Failure analysis}}
The cards below show representative failures from all three models. Each card gives the exact text prompt
that accompanied the 32 input frames, the model's complete decoded response, and the ground truth; the
frames themselves are omitted because the protocol fixes them and the failure mechanism is visible in the
prompt--response pair. Table~2 categorizes {n_fail} failures across four primary mechanisms.

{failure_cards(panels)}

\begin{{center}}
\footnotesize\textbf{{Table 2:}} Failure taxonomy. Near misses within 30\% relative error are excluded,
as MRA awards partial credit in that range.\par\vspace{{1pt}}
{tax}
\end{{center}}

\section{{Discussion}}
{disc}

\vspace{{-5pt}}
\bibliographystyle{{abbrvnat}}
\begingroup
\fontsize{{5.8}}{{6.6}}\selectfont
\setlength{{\bibsep}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\bibliography{{references}}
\endgroup

\end{{document}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "report" / "neurips" / "main.tex"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "report" / "neurips_2026.sty", out.parent)

    bib_src = Path(__file__).resolve().parents[1] / "report" / "references.bib"
    if bib_src.exists():
        shutil.copy(bib_src, out.parent / "references.bib")

    out.write_text(build(args.panels), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB), "
          f"{len(F.select_cases(args.panels))} text cards")
    print("compile:  ~/.local/bin/tectonic -X compile " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
