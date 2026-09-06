#!/usr/bin/env python3
"""Read the line-graph x bond-angle-diffusion 2x2 out of a collected run tree.

    python analyze.py [--variant sym|raw|rawsym|nosym]

Writes ``40_stats/matrix.json``, ``40_stats/matrix.tex`` and
``60_report/REPORT.md``.

**Stdlib only, on purpose.**  Every number here is a pure function of files the
run already wrote -- ``metrics_*.json`` (AtomBench's own metric code),
``history.json``, ``angle_eval.json`` and the per-stage markers under
``stages/`` -- so the analysis runs in either conda environment, or on a
laptop with the results tree and nothing else.  The sibling harness's
``analyze.py`` needs pymatgen because it recomputes per-target structure
matches for paired tests; this experiment does not ask for those, so it does
not pay for them.

Four families are reported.

**Denoising loss.**  Reported as the *structural* validation loss,

    L_struct = w_lat * L_lattice + w_frac * L_frac ,

and NOT as the trained total.  The trained objective is
``L_lat + 10*L_frac + L_angle``, and ``L_angle`` exists in only two of the four
cells, so a total-loss comparison across the angular factor compares two
different objectives and the arm that has an extra term to minimise looks
better or worse for a reason that is not about fidelity.  ``L_struct`` is the
part all four cells optimise, it is what ``--select-on structural`` selects
``best_model.pt`` on, and it is therefore the quantity a percent change can
honestly be quoted on.  ``L_angle`` is reported too, but only ever *within* the
two arms that have it.

**AtomBench metrics.**  Match rate, coordinate RMSD, ccRMSD, lattice MAE
(lengths and angles) and KLD, extracted from the ``metrics.json`` AtomBench's
own ``compute_metrics.py`` wrote.  Not recomputed here -- a second
implementation is a second thing that can drift.

**Bond-angle Wasserstein.**  The 1-D earth-mover distance in degrees between
the pooled bond-angle histogram of the generated structures and of the held-out
real ones, from ``angle_eval.json``.  This is the metric that speaks directly
to the hypothesis: a model with an angular channel should reproduce the natural
angular distribution.  Prefer the ``raw``/``rawsym`` variant for it -- see the
note in the generated report.

**GPU-hours.**  The matrix is parameter-matched, not compute-matched: the
line-graph cells run their edge update over the triplet set, which on a real
batch outnumbers pairs 5.4 to 1, so they cost measurably more wall-clock per
training step than the no-line-graph cells at this matched parameter count
(measured 1.50x on a GB10).  Rather than a second, compute-matched matrix to
correct for that, this is reported directly: every stage's ``elapsed_s`` (from
``stages/*.json``, written by the task runner) summed per cell and per
benchmark.  Every stage of a unit runs inside one SLURM element that holds
``--gres`` for its whole duration, so this sum is a measurement of GPU-hours
billed under this cluster's allocation model, not an estimate of utilisation.

No p-values.  Four cells at one to a few seeds cannot support them, and the
sibling suite's eight arms at three seeds returned every pre-registered
contrast at Holm-adjusted p = 1.000 on this benchmark.  What is reported
instead is the mean, the seed spread, the percent change, and an explicit flag
when a change is smaller than the spread it sits in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --- the matrix ------------------------------------------------------------
# Single source of truth is alignn.inverse.ablations.  It is a pure dict module
# with no torch import, so it is usually importable even from the scoring
# environment; fall back to a literal copy and say which was used.
#
# LGM_MATRIX=parameter (default) reads the derived-target 2x2 (MATRIX, the
# lg-angle-matrix task); LGM_MATRIX=state reads the angular-state 2x2
# (MATRIX_STATE, the lg-angle-state-matrix task).  Chosen by environment
# because the cell table is fixed at import, before argparse runs.  The two
# share their non-angular cells, so a collected tree holds both; the state
# run's outputs carry a ``_state`` suffix so neither overwrites the other.
MATRIX_NAME = os.environ.get("LGM_MATRIX", "parameter")
OUT_SUFFIX = "" if MATRIX_NAME == "parameter" else f"_{MATRIX_NAME}"
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alignn"))
    from alignn.inverse import ablations as _abl

    MATRIX = _abl.matrix_cells(MATRIX_NAME)
    if MATRIX_NAME == "parameter":
        MATRIX_COMPARISONS = _abl.MATRIX_COMPARISONS
        MATRIX_DESCRIPTIONS = _abl.MATRIX_DESCRIPTIONS
    else:
        MATRIX_COMPARISONS = getattr(_abl, f"MATRIX_{MATRIX_NAME.upper()}_COMPARISONS")
        MATRIX_DESCRIPTIONS = getattr(_abl, f"MATRIX_{MATRIX_NAME.upper()}_DESCRIPTIONS")

    def all_cells():
        return dict(MATRIX)

    MATRIX_SOURCE = f"alignn.inverse.ablations ({MATRIX_NAME})"
except Exception:  # noqa: BLE001
    MATRIX = {
        "neither": {"config": "nolg", "alignn_layers": 0, "gcn_layers": 9},
        "line graph": {"config": "A0", "alignn_layers": 3, "gcn_layers": 3},
        "angle diffusion": {
            "config": "nolg_ad", "alignn_layers": 0, "gcn_layers": 9,
        },
        "both": {"config": "A3", "alignn_layers": 3, "gcn_layers": 3},
    }
    MATRIX_DESCRIPTIONS = {
        "neither": "no angular channel at all; nine pair-graph convolutions",
        "line graph": "angles as ALIGNN input features, no angular objective",
        "angle diffusion": "angular denoising objective with no line graph",
        "both": "angles as features and as a denoised variable (= A3)",
    }
    MATRIX_COMPARISONS = {
        "line graph, without angle diffusion": ("neither", "line graph"),
        "line graph, with angle diffusion": ("angle diffusion", "both"),
        "angle diffusion, without a line graph": ("neither", "angle diffusion"),
        "angle diffusion, with a line graph": ("line graph", "both"),
        "both together, against neither": ("neither", "both"),
    }

    def all_cells():
        return dict(MATRIX)

    MATRIX_SOURCE = "literal copy (alignn.inverse.ablations not importable)"

#: config directory name -> matrix cell label
CONFIG_TO_LABEL = {c["config"]: label for label, c in all_cells().items()}

#: (key, label, decimals, lower_is_better).  Denoising loss first, because it
#: is the one quantity measured without any generation, sampling or force field
#: in the loop, and so the one an ablation can read most directly.
METRICS = [
    ("loss_struct", "denoising loss (structural)", 4, True),
    ("match", "match rate", 4, False),
    ("rmsd", "coordinate RMSD (A)", 4, True),
    ("ccrmsd", "ccRMSD", 4, True),
    ("abc", "lattice MAE, abc (A)", 4, True),
    ("ang", "lattice MAE, angles (deg)", 3, True),
    ("kld", "KLD", 5, True),
    ("wasserstein", "bond-angle Wasserstein (deg)", 4, True),
]

LATEX_HEADERS = {
    "loss_struct": r"Denoising loss $\downarrow$",
    "match": r"Match rate $\uparrow$",
    "rmsd": r"Coordinate RMSD (\AA) $\downarrow$",
    "ccrmsd": r"ccRMSD $\downarrow$",
    "abc": r"Lattice MAE, $abc$ (\AA) $\downarrow$",
    "ang": r"Lattice MAE, angles ($^{\circ}$) $\downarrow$",
    "kld": r"KLD $\downarrow$",
    "wasserstein": r"Bond-angle $W_1$ ($^{\circ}$) $\downarrow$",
}

#: Reported per arm but never across the angular factor: only two cells have
#: this term at all, so there is nothing to compare the other two against.
WITHIN_FAMILY = [("loss_angle", "angular denoising loss", 4)]

#: Stages whose elapsed_s counts toward GPU-hours -- everything that runs
#: inside a SLURM element holding --gres, which on this harness's sbatch
#: templates is every stage of a unit (train through score-sym; angle_eval
#: too, if 40_mechanism.sh has been run).  Data-prep stages are excluded by
#: construction: they live under a "data" config, not a matrix cell, so
#: `load()` below never looks at them at all.
GPU_STAGES = (
    "train", "generate", "symmetrize", "score-nosym", "score-sym",
    "angle_eval", "unrelaxed",
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def jload(path: Path):
    try:
        return json.loads(
            path.read_text().replace("NaN", "null").replace("Infinity", "null")
        )
    except Exception:  # noqa: BLE001
        return None


def _nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None and not _nan(v)]
    return sum(vals) / len(vals) if vals else None


def atombench_metrics(path: Path) -> Dict:
    """AtomBench's numbers, in the shape this report uses.

    Mirrors ``scripts/atombench/collect_results.py:extract`` exactly, including
    the ccRMSD/ccRMSE spelling fallback.
    """
    raw = jload(path)
    if raw is None:
        return {}
    mae = raw.get("MAE", {}).get("average_mae", {})
    kld = raw.get("KLD", {})
    rmse = raw.get("RMSE", {}).get("AtomGen", {})
    cc = raw.get("ccRMSD", raw.get("ccRMSE", {}))
    return {
        "match": rmse.get("match_rate"),
        "rmsd": rmse.get("mean_cartesian_rms_angstrom"),
        "ccrmsd": cc.get("value"),
        "abc": _mean([mae.get(k) for k in ("a", "b", "c")]),
        "ang": _mean([mae.get(k) for k in ("alpha", "beta", "gamma")]),
        "kld": _mean(
            [kld.get(k) for k in ("a", "b", "c", "alpha", "beta", "gamma")]
        ),
        "n_matched": rmse.get("n_matched"),
        "n_total": rmse.get("n_total"),
    }


def losses_from_history(hist: Path, cfg: Dict) -> Dict:
    """Best structural validation loss, and the angular one beside it.

    ``loss_structural`` is written directly by train_csp.py.  For a run
    predating that it is reconstructed from the components and the weights the
    run actually used, which ``config.json`` records -- never from assumed
    defaults, because a comparison that silently used the wrong weights would
    still look plausible.
    """
    rows = jload(hist)
    if not isinstance(rows, list) or not rows:
        return {}
    w_lat = cfg.get("lattice_weight")
    w_frac = cfg.get("frac_weight")

    def struct(v: Dict) -> Optional[float]:
        if v.get("loss_structural") is not None:
            return v["loss_structural"]
        if w_lat is None or w_frac is None:
            return None
        if v.get("loss_lattice") is None or v.get("loss_frac") is None:
            return None
        return w_lat * v["loss_lattice"] + w_frac * v["loss_frac"]

    vals = [(struct(r["val"]), r) for r in rows if "val" in r]
    vals = [(s, r) for s, r in vals if s is not None]
    if not vals:
        return {}
    best_s, best_row = min(vals, key=lambda p: p[0])
    return {
        "loss_struct": best_s,
        "loss_angle": best_row["val"].get("loss_angle"),
        "loss_total": best_row["val"].get("loss"),
        "best_epoch": best_row.get("epoch"),
        "n_epochs": rows[-1].get("epoch"),
        "reconstructed": best_row["val"].get("loss_structural") is None,
    }


def wasserstein_from(path: Path) -> Optional[float]:
    d = jload(path)
    if not isinstance(d, dict):
        return None
    return (d.get("angle_distribution") or {}).get("wasserstein_deg")


def gpu_hours_from(seed_dir: Path) -> Optional[float]:
    """Sum of every recorded stage's ``elapsed_s`` for this run, in hours.

    Every stage of a unit runs inside one SLURM element that holds --gres for
    its whole wall-clock duration (see this harness's sbatch templates), so
    this sum is a measurement of GPU-hours billed under this cluster's
    allocation model -- not an estimate of GPU utilisation, which would need
    the trace in ``gpu_trace.csv`` instead.
    """
    stages_dir = seed_dir / "stages"
    if not stages_dir.is_dir():
        return None
    total, found = 0.0, False
    for marker in stages_dir.glob("*.json"):
        if marker.stem not in GPU_STAGES:
            continue
        d = jload(marker)
        if isinstance(d, dict) and d.get("elapsed_s") is not None:
            total += float(d["elapsed_s"])
            found = True
    return total / 3600.0 if found else None


def load(results: Path, variant: str) -> Dict[str, Dict[int, Dict]]:
    """{cell label: {seed: record}} from the collected tree."""
    out: Dict[str, Dict[int, Dict]] = {}
    unknown = []
    for seed_dir in sorted((results / "10_runs").glob("*/seed*")):
        config = seed_dir.parent.name
        label = CONFIG_TO_LABEL.get(config)
        if label is None:
            unknown.append(config)
            continue
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
            continue
        cfg = jload(seed_dir / "config.json") or {}
        rec: Dict = {"config": config, "params": cfg.get("n_parameters")}
        rec["select_on"] = cfg.get("select_on")
        rec["alignn_layers"] = cfg.get("alignn_layers")
        rec["gcn_layers"] = cfg.get("gcn_layers")
        rec["angle_diffusion"] = cfg.get("angle_diffusion")
        rec["topology"] = cfg.get("topology")
        rec.update(losses_from_history(seed_dir / "history.json", cfg))
        rec.update(atombench_metrics(seed_dir / f"metrics_{variant}.json"))
        ae = (
            "angle_eval_rawsym.json"
            if variant in ("raw", "rawsym")
            else "angle_eval.json"
        )
        rec["wasserstein"] = wasserstein_from(seed_dir / ae)
        rec["gpu_hours"] = gpu_hours_from(seed_dir)
        out.setdefault(label, {})[seed] = rec
    if unknown:
        print(
            f"  note: ignored {len(set(unknown))} run dir(s) not in the "
            f"matrix: {sorted(set(unknown))}",
            file=sys.stderr,
        )
    return out


# ---------------------------------------------------------------------------
# Statistics -- deliberately just these
# ---------------------------------------------------------------------------
def stat(recs: Sequence[Dict], key: str):
    """(mean, sd, n) over the seeds that actually have this metric."""
    vals = [
        r.get(key) for r in recs
        if r.get(key) is not None and not _nan(r.get(key))
    ]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.fmean(vals), statistics.stdev(vals), len(vals)


def total(recs: Sequence[Dict], key: str) -> Optional[float]:
    """Sum over the seeds that have this metric, or None if none do."""
    vals = [
        r.get(key) for r in recs
        if r.get(key) is not None and not _nan(r.get(key))
    ]
    return sum(vals) if vals else None


def effect(cells: Dict, ref: str, test: str, key: str) -> Optional[Dict]:
    """One cell against another, on one metric."""
    if ref not in cells or test not in cells:
        return None
    m_ref, s_ref, n_ref = stat(list(cells[ref].values()), key)
    m_test, s_test, n_test = stat(list(cells[test].values()), key)
    if m_ref is None or m_test is None:
        return None
    delta = m_test - m_ref
    spread = max(s_ref or 0.0, s_test or 0.0)
    return {
        "ref": ref, "test": test,
        "ref_mean": m_ref, "ref_sd": s_ref, "ref_n": n_ref,
        "test_mean": m_test, "test_sd": s_test, "test_n": n_test,
        "delta": delta,
        "percent": (delta / m_ref * 100.0) if m_ref else None,
        # A change smaller than the run-to-run spread it sits inside is not a
        # measurement of anything.  Said once, here, rather than left for a
        # reader to work out from two columns.
        "within_spread": bool(spread) and abs(delta) < spread,
        "spread": spread or None,
    }


def interaction(cells: Dict, key: str) -> Optional[Dict]:
    """Does the effect of one factor depend on the level of the other?

    The whole reason to cross the factors rather than test them separately.
    Cell order is (neither, line graph, angle diffusion, both), so both main
    effects and their difference read straight off it.
    """
    need = list(MATRIX)
    if not all(c in cells for c in need):
        return None
    m = {}
    for c in need:
        mean, _, _ = stat(list(cells[c].values()), key)
        if mean is None:
            return None
        m[c] = mean
    none_, lg, ad, both = (m[c] for c in need)
    ad_without, ad_with = ad - none_, both - lg
    return {
        "lg_effect_without_ad": lg - none_,
        "lg_effect_with_ad": both - ad,
        "ad_effect_without_lg": ad_without,
        "ad_effect_with_lg": ad_with,
        # Identical whichever factor is differenced first, by the algebra of a
        # 2x2, so it is reported once.
        "interaction": ad_with - ad_without,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt(mean, sd, dp: int) -> str:
    if mean is None:
        return "—"
    return f"{mean:.{dp}f}" if sd is None else f"{mean:.{dp}f} ± {sd:.{dp}f}"


def order(cells: Dict) -> List[str]:
    return [c for c in MATRIX if c in cells]


def write_tex(path: Path, cells: Dict, variant: str) -> None:
    labels = order(cells)
    cols = "l" + "c" * len(labels)
    L = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Line graph and bond-angle diffusion, crossed.} "
        r"Mean $\pm$ one standard deviation over seeds. The two left columns "
        r"are the line-graph ablation of Table~\ref{tab:inverse_ablation}; "
        r"the two right columns add the angular denoising objective. "
        r"Denoising loss is the structural term "
        r"$w_{\mathrm{lat}}L_{\mathrm{lat}}+w_{\mathrm{frac}}"
        r"L_{\mathrm{frac}}$, which all four configurations optimise; the "
        r"angular term is reported separately in the text because only two "
        r"of them have it. Parameters are matched to within $1.03\%$ (the "
        r"residual is the angle encoder); the line-graph columns cost "
        r"$1.50\times$ the measured wall-clock per training step at that "
        r"matched parameter count -- see the GPU-hours row. Arrows give the "
        r"favourable direction.}",
        r"\label{tab:lg_angle_matrix}",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        "Metric & " + " & ".join(lab.replace("_", " ") for lab in labels)
        + r" \\",
        r"\midrule",
    ]
    for key, _label, dp, lower in METRICS:
        vals, best_val, best_i = [], None, None
        for i, lab in enumerate(labels):
            mean, sd, _ = stat(list(cells[lab].values()), key)
            vals.append((mean, sd))
            if mean is not None and (
                best_val is None or (mean < best_val) == lower
            ):
                best_val, best_i = mean, i
        # A tie is not a winner: bolding one of two identical numbers invents
        # a result.  The published table has exactly this case -- match rate
        # agreed to four decimals between the two arms.
        tied = (
            best_val is not None
            and sum(1 for m, _ in vals if m is not None and m == best_val) > 1
        )
        cells_txt = []
        for i, (mean, sd) in enumerate(vals):
            if mean is None:
                cells_txt.append("—")
                continue
            body = (
                f"{mean:.{dp}f}"
                if sd is None
                else f"{mean:.{dp}f}\\pm{sd:.{dp}f}"
            )
            # \mathbf inside the maths, not \textbf around it: \textbf{$x$}
            # leaves the maths at its normal weight.  The manuscript's own
            # table writes $\mathbf{2.011\pm0.018}$.
            if i == best_i and not tied:
                cells_txt.append(r"$\mathbf{" + body + "}$")
            else:
                cells_txt.append(f"${body}$")
        L.append(f"{LATEX_HEADERS[key]} & " + " & ".join(cells_txt) + r" \\")
    # GPU-hours: a cost row, not a fidelity metric, so it is not bolded for a
    # "best" direction and it is a SUM over seeds, not a mean -- the natural
    # reading of "what did this cell cost", not "what does one seed cost".
    gpu_cells = []
    for lab in labels:
        g = total(list(cells[lab].values()), "gpu_hours")
        gpu_cells.append("—" if g is None else f"${g:.2f}$")
    L.append("GPU-hours & " + " & ".join(gpu_cells) + r" \\")
    L += [
        r"\bottomrule", r"\end{tabular}",
        rf"\\[2pt]\footnotesize Scored on the \texttt{{{variant}}} "
        r"predictions.",
        r"\end{table}", "",
    ]
    path.write_text("\n".join(L))


def write_report(results: Path, payload: Dict) -> None:
    cells = payload["cells"]
    labels = order(cells)
    variant = payload["variant"]
    L = [
        "# Line graph x bond-angle diffusion — 2x2",
        "",
        f"Dataset `{payload['dataset']}`, run `{payload['run_id']}`, scored on "
        f"the `{variant}` predictions. Cells from `{payload['matrix_source']}`.",
        "",
        "## The matrix",
        "",
        "|  | no angle diffusion | angle diffusion |",
        "|---|---|---|",
        "| **no line graph** | `neither` | `angle diffusion` |",
        "| **line graph** | `line graph` | `both` (= A3, proposed) |",
        "",
        "Parameter-matched, not compute-matched: the line-graph cells run "
        "their edge update over the triplet set, which on a real batch "
        "outnumbers pairs 5.4 to 1, so they cost more wall-clock per step "
        "than the no-line-graph cells at this matched parameter count "
        "(measured 1.50x on a GB10). Reported directly below as GPU-hours, "
        "rather than run a second, compute-matched matrix to correct for it.",
        "",
    ]
    for lab in labels:
        recs = list(cells[lab].values())
        d = MATRIX_DESCRIPTIONS.get(lab, "")
        p, _, _ = stat(recs, "params")
        depth = f"{recs[0].get('alignn_layers')}/{recs[0].get('gcn_layers')}"
        pt = "—" if p is None else f"{p/1e6:.4f} M"
        L.append(
            f"- **{lab}** — {d}  \n  {depth} ALIGNN/pair convolutions, "
            f"{pt} parameters, {len(recs)} seed(s)"
        )
    L += ["", "## Results", "",
          "| metric | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    for key, label, dp, _lower in METRICS:
        row = [label]
        for lab in labels:
            mean, sd, _ = stat(list(cells[lab].values()), key)
            row.append(fmt(mean, sd, dp))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # -- GPU-hours: per cell, plus the benchmark total ----------------------
    L += ["## GPU-hours", "",
          "Sum of every recorded stage's wall-clock (train through "
          "score-sym, and angle_eval/unrelaxed if run) for each cell's "
          "seed(s) -- the GPU-hours this benchmark actually consumed, "
          "since every stage of a unit runs inside one SLURM element "
          "holding `--gres` for its whole duration. Not an estimate.", "",
          "| cell | seeds | GPU-hours |", "|---|---|---|"]
    grand_total = 0.0
    any_total = False
    for lab in labels:
        recs = list(cells[lab].values())
        g = total(recs, "gpu_hours")
        n_g = sum(1 for r in recs if r.get("gpu_hours") is not None)
        if g is not None:
            grand_total += g
            any_total = True
        L.append(f"| {lab} | {n_g}/{len(recs)} | "
                  f"{'—' if g is None else f'{g:.2f}'} |")
    L.append(f"| **total** | | **{grand_total:.2f}** |" if any_total
              else "| **total** | | — (no stage markers found) |")
    L.append("")

    inter = payload.get("interaction") or {}
    if inter:
        L += ["## Interaction", "",
              "The reason to cross the factors rather than test them "
              "separately. If the angular objective helps by the same "
              "amount with and without the line graph, the two are "
              "independent and each main effect stands alone. If it helps "
              "only with the line graph, the objective needs the "
              "architecture; if only without, it is substituting for it.",
              "",
              "| metric | Δ(LG) no AD | Δ(LG) with AD | Δ(AD) no LG | "
              "Δ(AD) with LG | interaction |",
              "|---|---|---|---|---|---|"]
        for key, label, dp, _ in METRICS:
            i = inter.get(key)
            if i is None:
                continue
            L.append(
                f"| {label} | {i['lg_effect_without_ad']:+.{dp}f} | "
                f"{i['lg_effect_with_ad']:+.{dp}f} | "
                f"{i['ad_effect_without_lg']:+.{dp}f} | "
                f"{i['ad_effect_with_lg']:+.{dp}f} | "
                f"{i['interaction']:+.{dp}f} |")
        L.append("")

    L += ["## Angular denoising loss", "",
          "Only two cells have this term, so it is reported *within* them "
          "and never across the angular factor — there is nothing in the "
          "other two cells to compare it against.", "",
          "| cell | angular denoising loss |", "|---|---|"]
    for lab in labels:
        recs = list(cells[lab].values())
        # train_csp.py accumulates a zero into loss_angle when the term is
        # off, so a cell without the objective would otherwise report a
        # flawless 0.0000 -- which reads as the best score in the column
        # rather than as "this quantity does not exist here".
        if not any(r.get("angle_diffusion") for r in recs):
            L.append(f"| {lab} | n/a — no angular term |")
            continue
        mean, sd, _ = stat(recs, "loss_angle")
        L.append(f"| {lab} | {fmt(mean, sd, 4)} |")

    L += ["", "## Effects", "",
          "Each row is one cell against another, on every metric. `Δ%` is "
          "the change in the second cell relative to the first; `~` marks a "
          "change smaller than the larger of the two arms' seed spreads, "
          "which is not a measurement of anything.", ""]
    for question, (ref, test) in payload["effect_order"]:
        if ref not in cells or test not in cells:
            continue
        L += [f"### {question}", "", f"`{ref}` → `{test}`", "",
              "| metric | ref | test | Δ | Δ% | |",
              "|---|---|---|---|---|---|"]
        for key, label, dp, lower in METRICS:
            e = payload["effects"].get(question, {}).get(key)
            if e is None:
                continue
            direction = (
                "—" if e["percent"] is None or abs(e["percent"]) < 0.05
                else ("better" if (e["delta"] < 0) == lower else "worse")
            )
            flag = "~" if e["within_spread"] else ""
            pct = "—" if e["percent"] is None else f"{e['percent']:+.1f}%"
            note = f"{direction} {flag}".strip()
            L.append(
                f"| {label} | {fmt(e['ref_mean'], e['ref_sd'], dp)} | "
                f"{fmt(e['test_mean'], e['test_sd'], dp)} | "
                f"{e['delta']:+.{dp}f} | {pct} | {note} |"
            )
        L.append("")

    L += ["## How to read this", "",
          "- **Denoising loss is the structural term only.** The trained "
          "objective is `w_lat*L_lat + w_frac*L_frac + w_ang*L_ang`, and "
          "`L_ang` exists in only two cells. Quoting the trained total "
          "across the angular factor would compare two different "
          "objectives. All four cells also select `best_model.pt` on this "
          "same structural loss (`--select-on structural`), so the "
          "checkpoint being scored and the loss being quoted agree.",
          "- **Loss and fidelity are not the same claim, and have already "
          "come apart on this benchmark.** The published line-graph "
          "ablation moved denoising loss by 14.5% (2.011 vs 2.351, no "
          "overlap across twelve runs) and moved match rate by *exactly "
          "nothing* (0.4709 both arms). A large loss change here is "
          "evidence about score fitting; it is not by itself evidence "
          "about generation.",
          "- **Prefer the `raw`/`rawsym` variant for the bond-angle "
          "Wasserstein.** The headline pipeline ranks 32 candidates by "
          "ALIGNN-FF energy and relaxes the survivors, which snaps every "
          "sample into the force field's nearest local minimum — precisely "
          "where a local-geometry advantage would be erased, and local "
          "geometry is what the angular channel claims to fix. The relaxed "
          "number measures the diffusion model and ALIGNN-FF together.",
          "- **A 180° spike appears in the generated *and* the real "
          "bond-angle histograms.** It is back-tracking triplets from the "
          "graph builder, identical across cells. It is not physics and "
          "does not bias a comparison.",
          "- **Parameter budget.** `angle diffusion` and `both` match to "
          "the parameter (the angular path adds no weights beyond the "
          "shared angle encoder and head). `neither` and `line graph` "
          "differ by 1.1%, the angle encoder, which is the manuscript's "
          "existing arm-A/arm-B design: the budget is matched at nine "
          "convolution blocks. The GPU-hours row above is what that "
          "budget choice costs in wall-clock, measured, not modelled.",
          "- **No p-values, deliberately.** The cells carry "
          f"{max((len(v) for v in cells.values()), default=0)} seed(s), "
          "which cannot support them. The sibling eight-arm suite returned "
          "every pre-registered contrast at Holm-adjusted p = 1.000 on the "
          "103-target split; the honest reading was underpowered, not "
          "null.",
          ""]
    (results / "60_report").mkdir(parents=True, exist_ok=True)
    (results / "60_report" / f"REPORT{OUT_SUFFIX}.md").write_text("\n".join(L))


def print_table(cells: Dict) -> None:
    labels = order(cells)
    w = max([len(label) for _, label, _, _ in METRICS] + [12]) + 2
    print("\n" + "metric".ljust(w) + "".join(f"{c:>22}" for c in labels))
    print("-" * (w + 22 * len(labels)))
    for key, label, dp, _ in METRICS:
        line = label.ljust(w)
        for lab in labels:
            mean, sd, _ = stat(list(cells[lab].values()), key)
            line += f"{fmt(mean, sd, dp):>22}"
        print(line)
    print("-" * (w + 22 * len(labels)))
    line = "GPU-hours (total)".ljust(w)
    grand_total, any_total = 0.0, False
    for lab in labels:
        g = total(list(cells[lab].values()), "gpu_hours")
        if g is not None:
            grand_total += g
            any_total = True
        line += f"{('—' if g is None else f'{g:.2f}'):>22}"
    print(line)
    if any_total:
        print(f"{'benchmark total'.ljust(w)}{grand_total:>22.2f}  GPU-hours")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--variant", default="sym",
        choices=("sym", "nosym", "raw", "rawsym"),
        help="which scored CSV's metrics to read. 'sym' is what the "
        "manuscript's lattice columns are quoted on; 'rawsym' is the "
        "unrelaxed generation, which is the honest denominator for the "
        "bond-angle comparison",
    )
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    args = ap.parse_args()

    if not args.results:
        print("set RESULTS (source ./env.sh)", file=sys.stderr)
        return 2
    results = Path(args.results)
    if not (results / "10_runs").is_dir():
        print(f"no {results/'10_runs'}; run collect.py first", file=sys.stderr)
        return 1

    cells = load(results, args.variant)
    if not cells:
        print("no matrix cells found in the run tree", file=sys.stderr)
        return 1
    n_runs = sum(len(v) for v in cells.values())
    print(f"loaded {n_runs} run(s) across {len(cells)}/{len(all_cells())} "
          f"cell(s)")
    print(f"  matrix from {MATRIX_SOURCE}")

    sel = {
        r.get("select_on")
        for seeds in cells.values() for r in seeds.values()
    }
    if sel - {"structural"}:
        print(
            f"  WARNING: not every run selected its checkpoint on the "
            f"structural loss (saw {sorted(x for x in sel if x)}). Cells "
            f"selected on the trained total are not comparable across the "
            f"angular factor.",
            file=sys.stderr,
        )
    recon = [
        r["config"] for seeds in cells.values() for r in seeds.values()
        if r.get("reconstructed")
    ]
    if recon:
        print(
            f"  note: structural loss reconstructed from components for "
            f"{len(recon)} run(s) (history predates loss_structural)"
        )

    effects = {}
    effect_order: List = []
    for question, (ref, test) in MATRIX_COMPARISONS.items():
        per_metric = {}
        for key, _, _, _ in METRICS:
            e = effect(cells, ref, test, key)
            if e is not None:
                per_metric[key] = e
        if per_metric:
            effects[question] = per_metric
            effect_order.append((question, (ref, test)))

    inter = {}
    for key, _, _, _ in METRICS:
        i = interaction(cells, key)
        if i is not None:
            inter[key] = i

    payload = {
        "dataset": os.environ.get("DATASET", "?"),
        "run_id": os.environ.get("RUN_ID", results.name),
        "variant": args.variant,
        "matrix_source": MATRIX_SOURCE,
        "n_runs": n_runs,
        "cells": cells,
        "effects": effects,
        "effect_order": effect_order,
        "interaction": inter,
    }
    outdir = results / "40_stats"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"matrix{OUT_SUFFIX}.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    write_tex(outdir / f"matrix{OUT_SUFFIX}.tex", cells, args.variant)
    write_report(results, payload)

    print_table(cells)
    print(f"\nwrote {outdir}/matrix{OUT_SUFFIX}.json, {outdir}/matrix{OUT_SUFFIX}.tex")
    print(f"      {results/'60_report'/'REPORT.md'}")
    missing = [c for c in all_cells() if c not in cells]
    if missing:
        print(f"\nstill missing cell(s): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
