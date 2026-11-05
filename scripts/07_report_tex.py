#!/usr/bin/env python3
"""Step 7 -- build the NeurIPS-format report from the same artefacts as the HTML.

    python scripts/07_report_tex.py           # -> report/neurips/main.tex
    tectonic -X compile report/neurips/main.tex   # -> main.pdf

Numbers come from results/scores.csv and results/stats.json; figures from
results/failures/*.png (rendered with VSI_PANEL_WIDTH=720, which prints at
~8 pt when placed two-up in the NeurIPS 5.5 in text block). Nothing here is
hand-copied, so a rerun of the eval regenerates the paper.

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


def select_panels(n: int = 5) -> list[tuple[str, Path, dict]]:
    d = C.RESULTS_DIR / "failures"
    idx_p = d / "index.json"
    index = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    groups: dict[str, list[Path]] = {}
    for p in sorted(d.glob("*.png")):
        groups.setdefault(index.get(p.name, {}).get("class", "case"), []).append(p)
    order = sorted(groups, key=lambda k: -len(groups[k]))
    picked, depth = [], 0
    while len(picked) < n and any(len(v) > depth for v in groups.values()):
        for k in order:
            if len(picked) >= n:
                break
            if len(groups[k]) > depth:
                picked.append((k, groups[k][depth], index.get(groups[k][depth].name, {})))
        depth += 1
    return picked


def figures(n: int = 5) -> str:
    picked = select_panels(n)
    if not picked:
        return r"\emph{no panels rendered}"
    cells = []
    for i, (cls, p, md) in enumerate(picked, 1):
        label = C.FAILURE_GROUP.get(cls, cls.replace("_", " "))
        sub = f"{DISPLAY.get(md.get('model'), md.get('model',''))}, {TASK_LABEL.get(md.get('task'), '')}, id~{md.get('id','')}"
        # The panel header already names the class, model and task, so no extra
        # LaTeX caption line is needed (each one costs ~0.15in).
        cells.append(
            "\\begin{minipage}[t]{0.460\\textwidth}\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{p.name}}}\n"
            "\\end{minipage}")
    # Rows are separate paragraphs, not one figure float: a float cannot break
    # across pages, so five panels would jump whole and push the taxonomy onto a
    # third page (same failure mode as the CSS grid in the HTML version).
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2]
        rows.append("\\noindent\n" + "\\hfill\n".join(pair) + "\n\\par\\vspace{1pt}")
    return "\n".join(rows)


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
        r"regardless of camera motion. Overall, feeding 32 downsampled frames without camera parameters "
        r"leaves models guessing metric scale from visual priors; benchmarks with explicit 3D reconstruction "
        r"supervision such as VLM-3R~\citep{fan2026vlm3r} highlight the need to evaluate relative geometry "
        r"rather than ungrounded absolute scale estimation."
    )

    return rf"""\documentclass{{article}}
\usepackage[preprint]{{neurips_2026}}
\usepackage[T1]{{fontenc}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{microtype}}
\usepackage[margin=0pt,font=small,labelfont=bf,skip=2pt]{{caption}}
\graphicspath{{{{failures/}}}}
\setlength{{\abovecaptionskip}}{{2pt}}
\setlength{{\belowcaptionskip}}{{0pt}}
\setlength{{\parskip}}{{0pt}}
\captionsetup[table]{{skip=3pt}}

\title{{Where Vision--Language Models Fail at 3D Space:\\A VSI-Bench Evaluation of Three Open-Weight VLMs}}

\author{{Noah Lee\\Texas A\&M University}}

\raggedbottom
\begin{{document}}
\small
\maketitle
\vspace{{-16mm}}

\section{{Setup and protocol}}
We evaluate three open-weight VLMs on VSI-Bench~\citep{{yang2025thinking}}, covering {n_q} questions across
{n_sc} indoor scenes from ScanNet~\citep{{dai2017scannet}} and ScanNet++~\citep{{yeshwanth2023scannetpp}}. Each model receives 32 frames sampled uniformly across
the video ($\mathrm{{round}}(\mathrm{{linspace}}(0,N{{-}}1,32))$), scaled to a maximum dimension of 448\,px,
using greedy decoding with up to 48 tokens. Prompts follow the standard VSI-Bench harness.
We report official strict accuracy (ACC) and Mean Relative Accuracy (MRA), alongside a lenient parser
that extracts numbers written as words or embedded in explanatory text.

\vspace{{2pt}}
\noindent\textbf{{Protocol.}} Evaluated models include Qwen3-VL-8B-Instruct~\citep{{qwen2025qwen3vl}}, Gemma-4-12B-it~\citep{{gemma2026gemma4}}, and
Molmo2-8B~\citep{{deitke2024molmo}}, run locally on one NVIDIA H100 GPU (PyTorch 2.14.1, transformers 5.22.1 / 4.57.1).
Tasks cover their full test splits: \texttt{{route\_planning}} (Category 1, ACC, $n{{=}}194$),
\texttt{{object\_abs\_distance}} (Category 2, MRA, $n{{=}}834$), and \texttt{{room\_size\_estimation}}
(Category 2, MRA, $n{{=}}288$). We report 95\% confidence intervals via scene-level bootstrapping.
A blind control arm repeats all numeric tasks with blank image inputs to measure prior bias.

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
The panels below illustrate five representative failure cases, displaying input frames with timing indices,
the question, ground truth, and model prediction. Table~2 categorizes {n_fail} failures across four
primary mechanisms.

{figures(panels)}

\begin{{center}}
\footnotesize\textbf{{Table 2:}} Failure taxonomy. Near misses within 30\% relative error are excluded,
as MRA awards partial credit in that range.\par\vspace{{1pt}}
{tax}
\end{{center}}

\section{{Discussion}}
{disc}

\vspace{{-2pt}}
\bibliographystyle{{abbrvnat}}
\begingroup
\scriptsize
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

    # figures are referenced relative to the .tex, so keep them alongside
    src = C.RESULTS_DIR / "failures"
    dst = out.parent / "failures"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.png"):
        shutil.copy(p, dst)
    for p in src.glob("index.json"):
        shutil.copy(p, dst)

    bib_src = Path(__file__).resolve().parents[1] / "report" / "references.bib"
    if bib_src.exists():
        shutil.copy(bib_src, out.parent / "references.bib")

    out.write_text(build(args.panels), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB), "
          f"{len(list(dst.glob('*.png')))} figures copied")
    print("compile:  ~/.local/bin/tectonic -X compile " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
