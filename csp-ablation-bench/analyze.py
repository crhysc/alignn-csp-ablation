#!/usr/bin/env python3
"""Statistical analysis of the angular-diffusion ablation.

    python analyze.py [--variant sym|nosym] [--recompute]

Run this with the *scoring* environment's python: it needs pymatgen, and it
imports AtomBench so that every metric definition stays AtomBench's rather
than becoming a second implementation that can drift.

Why this file exists at all
---------------------------
AtomBench computes per-benchmark metrics, plots them and writes a LaTeX table.
It has no notion of repeated seeds — grep the package for ``ttest``,
``bootstrap``, ``std(``, ``wilcoxon``: nothing.  Each CSV is one independent
benchmark and the output is a point estimate.

That is a specific problem here.  The inverse-design README records match rate
across fifteen independently trained models spanning **0.437–0.524** on a
103-target split.  A bar chart of 21 point estimates would be actively
misleading, and a difference of a few percent between two single runs is not a
result.

Three tiers
-----------
**Tier 1 — arm level, over seeds.**  Mean ± s.d. per arm, Welch two-sided p for
each declared contrast.  This duplicates ``run_task.py --aggregate`` on
purpose: the two are cross-checked against each other.  n = 3 per arm, so
every p here is descriptive.

**Tier 2 — per-target paired tests, where the power actually is.**  Every arm
is scored on the *same* targets; comparing three seed-means throws that
pairing away.  Joining on ``id``:

  * match rate is paired **binary** -> McNemar exact on the discordant pairs,
    plus a pooled Cochran-Mantel-Haenszel across seeds.
  * RMSD and lattice error are paired **continuous** -> Wilcoxon signed-rank
    plus a paired BCa bootstrap CI on the mean difference.
  * Hedges' g with the small-sample correction, so effect size sits beside the
    p-value rather than instead of it.

This is a more powerful test of the *same* pre-registered metrics — not a new
metric and not a fishing expedition.  The README fixed the metric suite before
any run precisely so a favourable one could not be chosen afterwards, and that
constraint is respected here.

**Tier 3 — multiplicity.**  Holm-Bonferroni across the declared contrasts
within each metric family.  The three primary questions (A0-A1, A0-A2, A4-A3)
are marked pre-registered; everything else is labelled exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np

# --- the declared contrasts ------------------------------------------------
# Single source of truth is alignn.inverse.ablations, but that lives in the
# training environment and this script runs in the scoring one.  Import when
# possible, fall back to a literal copy and say so.
try:
    from alignn.inverse.ablations import COMPARISONS, DESCRIPTIONS
    COMPARISONS_SOURCE = "alignn.inverse.ablations"
except Exception:
    COMPARISONS = {
        "does explicit angular denoising help": ("A0", "A1"),
        "does smooth topology alone help": ("A0", "A2"),
        "do the two together help": ("A0", "A3"),
        "is the coupling doing the work (not just auxiliary loss)": ("A4", "A3"),
        "A5: hard kNN vs smooth radius, with angles on": ("A1", "A3"),
        "A5: hard kNN vs smooth radius, with angles off": ("A0", "A2"),
        "A6: does the angular basis matter": ("A3", "A6"),
    }
    DESCRIPTIONS = {
        "A0": "baseline: angles as features only",
        "A1": "explicit angular denoising, baseline kNN topology",
        "A2": "smooth radius topology, no angular objective",
        "A3": "proposed: angular denoising + smooth topology",
        "A4": "control: angular objective, angle->bond coupling removed",
        "A6": "A3 with the Fourier angular basis",
    }
    COMPARISONS_SOURCE = "literal copy (alignn not importable here)"

# --- metric extraction -----------------------------------------------------
# Tier 1 reads metrics.json and needs no pymatgen; only tier 2's per-target
# matching does.  So the extractor is resolved once here, preferring
# AtomBench's own so the definitions cannot drift, and falling back to the
# equivalent in ALIGNN's collect_results (stdlib only, and it already handles
# both the ccRMSD and ccRMSE spellings).  Either way tier 1 runs in either
# environment, and only tier 2 is skipped when pymatgen is absent.
try:
    from atombench.tables import extract_metrics as _extract

    def extract_metrics(raw):
        e = _extract(raw)
        return {"match_rate": e["match_rate"], "rmsd": e["RMSD"],
                "ccrmsd": e["ccRMSD"], "mae_abc": e["MAE"]["mean_abc"],
                "mae_ang": e["MAE"]["mean_angles"], "kld": e["KLD"]["mean"],
                "n_total": e["n_total"], "n_matched": e["n_matched"]}

    EXTRACT_SOURCE = "atombench.tables.extract_metrics"
except Exception:
    def extract_metrics(raw):
        def mean(vals):
            vals = [v for v in vals
                    if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return sum(vals) / len(vals) if vals else None
        mae = raw.get("MAE", {}).get("average_mae", {})
        kld = raw.get("KLD", {})
        rmse = raw.get("RMSE", {}).get("AtomGen", {})
        cc = raw.get("ccRMSD", raw.get("ccRMSE", {}))
        return {
            "match_rate": rmse.get("match_rate"),
            "rmsd": rmse.get("mean_cartesian_rms_angstrom"),
            "ccrmsd": cc.get("value"),
            "mae_abc": mean([mae.get(k) for k in ("a", "b", "c")]),
            "mae_ang": mean([mae.get(k) for k in ("alpha", "beta", "gamma")]),
            "kld": mean([kld.get(k) for k in
                         ("a", "b", "c", "alpha", "beta", "gamma")]),
            "n_total": rmse.get("n_total"), "n_matched": rmse.get("n_matched"),
        }

    EXTRACT_SOURCE = "local fallback (atombench not importable in this env)"


#: The questions the branch was designed to answer, fixed in advance.
PRIMARY = {("A0", "A1"), ("A0", "A2"), ("A4", "A3")}

#: Metrics compared arm-to-arm.  (key, label, lower_is_better)
METRICS = [
    ("match_rate", "match rate", False),
    ("rmsd", "coordinate RMSD", True),
    ("ccrmsd", "ccRMSD", True),
    ("mae_abc", "lattice MAE abc", True),
    ("mae_ang", "lattice MAE angles", True),
    ("kld", "KLD", True),
]

#: Mechanism metrics from angle_eval.json — the branch's actual hypothesis.
MECHANISM = [
    ("angle_wasserstein_deg", "bond-angle Wasserstein (deg)", True),
    ("angle_js", "bond-angle Jensen-Shannon", True),
    ("relax_rmsd", "relaxation displacement (A)", True),
]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def welch_p(a, b):
    try:
        from scipy import stats
    except ImportError:
        return None
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def hedges_g(a, b):
    """Standardised mean difference with the small-sample correction."""
    a = np.asarray([x for x in a if x is not None], float)
    b = np.asarray([x for x in b if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                   / (na + nb - 2))
    if sp == 0:
        return None
    d = (b.mean() - a.mean()) / sp
    j = 1 - 3 / (4 * (na + nb) - 9)          # Hedges' correction
    return float(d * j)


def mcnemar_exact(both, only_a, only_b, neither):
    """Exact McNemar on a paired binary outcome.

    Only the discordant cells carry information: targets that both arms match,
    or neither does, say nothing about which arm is better.  Under H0 each
    discordant target is a fair coin, so the p-value is an exact two-sided
    binomial test on ``only_b`` out of ``only_a + only_b``.
    """
    n = only_a + only_b
    if n == 0:
        return {"n_discordant": 0, "p": None, "only_a": 0, "only_b": 0,
                "both": both, "neither": neither}
    try:
        from scipy import stats
        p = float(stats.binomtest(only_b, n, 0.5).pvalue)
    except Exception:
        p = None
    return {"n_discordant": n, "p": p, "only_a": only_a, "only_b": only_b,
            "both": both, "neither": neither}


def cmh(tables):
    """Cochran-Mantel-Haenszel across seed-matched 2x2 tables.

    Pools the paired evidence over seeds without pretending the seeds are extra
    targets.  Implemented for the matched-pairs case, where it reduces to a
    stratified McNemar.
    """
    num = den = 0.0
    for t in tables:
        b, c = t["only_a"], t["only_b"]
        if b + c == 0:
            continue
        num += c - (b + c) / 2
        den += (b + c) / 4
    if den <= 0:
        return {"chi2": None, "p": None, "n_strata": 0}
    chi2 = (abs(num) - 0.5) ** 2 / den
    try:
        from scipy import stats
        p = float(stats.chi2.sf(chi2, 1))
    except Exception:
        p = None
    return {"chi2": float(chi2), "p": p, "n_strata": len(tables)}


def wilcoxon(diff):
    diff = [d for d in diff if d is not None and not math.isnan(d)]
    if len(diff) < 6 or all(d == 0 for d in diff):
        return None
    try:
        from scipy import stats
        return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except Exception:
        return None


def bca_bootstrap(diff, n_boot=10000, alpha=0.05, seed=0):
    """Bias-corrected and accelerated CI for the mean paired difference.

    Resampling is over *targets*, which is the unit the pairing is defined on.
    """
    x = np.asarray([d for d in diff if d is not None and not math.isnan(d)], float)
    n = len(x)
    if n < 8:
        return None
    rng = np.random.default_rng(seed)
    theta = x.mean()
    boots = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    prop = float((boots < theta).mean())
    if prop <= 0 or prop >= 1:
        return {"mean": float(theta), "lo": float(np.percentile(boots, 100*alpha/2)),
                "hi": float(np.percentile(boots, 100*(1-alpha/2))),
                "method": "percentile (z0 undefined)", "n": n}
    from scipy import stats as sps
    z0 = sps.norm.ppf(prop)
    # jackknife acceleration
    jack = (x.sum() - x) / (n - 1)
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    def endpoint(q):
        z = sps.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        return float(np.percentile(boots, 100 * sps.norm.cdf(adj)))

    return {"mean": float(theta), "lo": endpoint(alpha / 2),
            "hi": endpoint(1 - alpha / 2), "method": "BCa", "n": n,
            "z0": float(z0), "a": float(a)}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    if not items:
        return {k: None for k in pvals}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # enforce monotonicity
        out[k] = running
    for k in pvals:
        out.setdefault(k, None)
    return out


# ---------------------------------------------------------------------------
# Per-target values, using AtomBench's own matcher
# ---------------------------------------------------------------------------
def matcher_and_reduce():
    """AtomBench's StructureMatcher settings and Niggli reduction.

    Imported rather than reimplemented: a second copy of the matcher settings
    is a second thing that can drift from the metric being reported.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    try:
        from atombench.cli import _reduced_struct
        source = "atombench.cli._reduced_struct"
    except Exception:
        from atombench._structure_io import parse_structure

        def _reduced_struct(cell_str):
            s = parse_structure(cell_str).get_primitive_structure()
            return s.get_reduced_structure(reduction_algo="niggli")
        source = "local reimplementation (atombench.cli import failed)"
    # STOL/angle_tol/ltol are AtomBench's, from _compute_atomgen_rmse.
    return StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3), _reduced_struct, source


def per_target(csv_path: Path, cache: Path, recompute=False):
    """[(id, matched, rmsd_normalised)] for one benchmark CSV."""
    if cache.exists() and not recompute:
        with cache.open() as fh:
            return [(r["id"], int(r["matched"]),
                     float(r["rmsd"]) if r["rmsd"] not in ("", "None") else None)
                    for r in csv.DictReader(fh)]

    matcher, reduce_s, _ = matcher_and_reduce()
    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().lower(): v for k, v in row.items()}
            rid = row.get("id", "")
            try:
                st = reduce_s(str(row["target"]))
                sp = reduce_s(str(row["prediction"]))
            except Exception:
                rows.append((rid, 0, None))
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rms = matcher.get_rms_dist(sp, st)
            except Exception:
                rms = None
            if rms is None:
                rows.append((rid, 0, None))
            else:
                rows.append((rid, 1, float(rms[0])))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "matched", "rmsd"])
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
def load_arm_metrics(results: Path, variant: str):
    """Tier-1 inputs: one record per (arm, seed), read not recomputed."""
    out = {}
    for seed_dir in sorted((results / "10_runs").glob("*/seed*")):
        arm = seed_dir.parent.name
        seed = int(re.search(r"seed(\d+)", seed_dir.name).group(1))
        mj = seed_dir / f"metrics_{variant}.json"
        if not mj.exists():
            continue
        try:
            raw = json.loads(mj.read_text().replace("NaN", "null")
                             .replace("Infinity", "null"))
            rec = extract_metrics(raw)
        except Exception as exc:
            print(f"  warn: unreadable {mj}: {exc}", file=sys.stderr)
            continue

        hist = seed_dir / "history.json"
        if hist.exists():
            try:
                rows = json.loads(hist.read_text())
                losses = [r["val"]["loss"] for r in rows if "val" in r]
                rec["val_loss"] = min(losses) if losses else None
            except Exception:
                pass
        ae = seed_dir / "angle_eval.json"
        if ae.exists():
            try:
                rec["mechanism"] = json.loads(ae.read_text())
            except Exception:
                pass
        out.setdefault(arm, {})[seed] = rec
    return out


def flatten_mechanism(rec: dict, key: str):
    """angle_eval.json nests differently by version; search for the key."""
    mech = rec.get("mechanism")
    if not isinstance(mech, dict):
        return None
    stack = [mech]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k == key or k.replace("-", "_") == key:
                return v if isinstance(v, (int, float)) else None
            if isinstance(v, dict):
                stack.append(v)
    return None


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="sym", choices=("sym", "nosym"))
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the per-target cache (slow: StructureMatcher "
                         "over every target of every run)")
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    if not args.results:
        print("set RESULTS (source ./env.sh)", file=sys.stderr)
        return 2
    results = Path(args.results)
    outdir = results / "40_stats"
    outdir.mkdir(parents=True, exist_ok=True)

    arms = load_arm_metrics(results, args.variant)
    if not arms:
        print("no metrics found; run collect.py (and score first)", file=sys.stderr)
        return 1
    print(f"loaded {sum(len(v) for v in arms.values())} run(s) across "
          f"{len(arms)} arm(s)")
    print(f"  metrics via {EXTRACT_SOURCE}")
    print(f"  contrasts from {COMPARISONS_SOURCE}")

    # -- tier 1 --------------------------------------------------------------
    tier1 = {}
    for arm, seeds in sorted(arms.items()):
        entry = {"n_seeds": len(seeds), "seeds": sorted(seeds)}
        for key, _, _ in METRICS + [("val_loss", "", True)]:
            vals = [r.get(key) for r in seeds.values() if r.get(key) is not None]
            entry[key] = {
                "mean": round(statistics.fmean(vals), 6) if vals else None,
                "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else (0.0 if vals else None),
                "n": len(vals), "values": vals,
            }
        for key, _, _ in MECHANISM:
            vals = [flatten_mechanism(r, key) for r in seeds.values()]
            vals = [v for v in vals if v is not None]
            if vals:
                entry[key] = {"mean": round(statistics.fmean(vals), 6),
                              "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                              "n": len(vals), "values": vals}
        tier1[arm] = entry

    contrasts = {}
    for question, (ref, test) in COMPARISONS.items():
        if ref not in arms or test not in arms:
            continue
        c = {"ref": ref, "test": test,
             "preregistered": (ref, test) in PRIMARY, "metrics": {}}
        for key, label, lower_better in METRICS:
            a = [r.get(key) for r in arms[ref].values()]
            b = [r.get(key) for r in arms[test].values()]
            av = [x for x in a if x is not None]
            bv = [x for x in b if x is not None]
            c["metrics"][key] = {
                "label": label, "lower_is_better": lower_better,
                "ref_mean": round(statistics.fmean(av), 6) if av else None,
                "test_mean": round(statistics.fmean(bv), 6) if bv else None,
                "delta": round(statistics.fmean(bv) - statistics.fmean(av), 6)
                         if av and bv else None,
                "welch_p": welch_p(a, b),
                "hedges_g": hedges_g(a, b),
            }
        contrasts[question] = c

    # -- tier 2 --------------------------------------------------------------
    per_target_cache: dict[tuple[str, int], dict] = {}
    try:
        matcher_and_reduce()
        tier2 = True
        print("\ntier 2: computing per-target values (StructureMatcher; cached)")
    except Exception as exc:
        tier2 = False
        print(f"\ntier 2 SKIPPED: {exc}", file=sys.stderr)
        print("  The paired per-target tests need pymatgen. Re-run with the",
              file=sys.stderr)
        print("  scoring environment:  $SCORE_ENV_PATH/bin/python analyze.py",
              file=sys.stderr)
        print("  Tier 1 (seed-level) results below are complete either way.",
              file=sys.stderr)
    if tier2:
        for arm, seeds in sorted(arms.items()):
            for seed in sorted(seeds):
                csv_path = (results / "10_runs" / arm / f"seed{seed}"
                            / f"bench_{args.variant}.csv")
                if not csv_path.exists():
                    continue
                cache = outdir / "per_target" / f"{arm}_seed{seed}_{args.variant}.csv"
                try:
                    rows = per_target(csv_path, cache, args.recompute)
                except Exception as exc:
                    print(f"  warn: {arm} seed{seed}: {exc}", file=sys.stderr)
                    continue
                per_target_cache[(arm, seed)] = {rid: (m, r) for rid, m, r in rows}
                print(f"  {arm} seed{seed}: "
                      f"{sum(m for _, m, _ in rows)}/{len(rows)} matched")

    for question, c in contrasts.items():
        ref, test = c["ref"], c["test"]
        tables, rmsd_diffs = [], []
        shared_seeds = sorted(set(arms[ref]) & set(arms[test]))
        for seed in shared_seeds:
            A = per_target_cache.get((ref, seed))
            B = per_target_cache.get((test, seed))
            if not A or not B:
                continue
            ids = sorted(set(A) & set(B))
            both = onlyA = onlyB = neither = 0
            for i in ids:
                ma, mb = A[i][0], B[i][0]
                both += ma and mb
                onlyA += ma and not mb
                onlyB += mb and not ma
                neither += not ma and not mb
                if A[i][1] is not None and B[i][1] is not None:
                    rmsd_diffs.append(B[i][1] - A[i][1])
            tables.append(mcnemar_exact(both, onlyA, onlyB, neither))
        if tables:
            c["paired_match"] = {
                "per_seed": tables,
                "pooled_cmh": cmh(tables),
                "n_targets_shared": len(shared_seeds),
            }
        if rmsd_diffs:
            c["paired_rmsd"] = {
                "n_pairs": len(rmsd_diffs),
                "mean_diff": round(float(np.mean(rmsd_diffs)), 6),
                "wilcoxon_p": wilcoxon(rmsd_diffs),
                "bca": bca_bootstrap(rmsd_diffs, n_boot=args.n_boot),
            }

    # -- tier 3 --------------------------------------------------------------
    # Holm is applied over DISTINCT ARM PAIRS, not over question names.
    # COMPARISONS asks ("A0","A2") twice -- once as "does smooth topology alone
    # help" and once as "A5: hard kNN vs smooth radius, with angles off" -- but
    # that is one statistical test with two names.  Counting it twice inflates
    # the correction, so the adjustment runs over unique pairs and the result
    # is mapped back onto every question that uses each pair.
    pair_of = {q: (c["ref"], c["test"]) for q, c in contrasts.items()}
    unique_pairs = sorted(set(pair_of.values()))
    aliased = {pair: [q for q, pr in pair_of.items() if pr == pair]
               for pair in unique_pairs}

    def adjust(getter):
        by_pair = {}
        for pair, questions in aliased.items():
            vals = [v for v in (getter(contrasts[q]) for q in questions)
                    if v is not None]
            by_pair[pair] = vals[0] if vals else None
        return holm(by_pair)

    families = {key: adjust(lambda c, k=key: c["metrics"][k]["welch_p"])
                for key, _, _ in METRICS}
    families["paired_match"] = adjust(
        lambda c: (c.get("paired_match", {}).get("pooled_cmh") or {}).get("p"))
    families["paired_rmsd"] = adjust(
        lambda c: (c.get("paired_rmsd") or {}).get("wilcoxon_p"))

    for family, by_pair in families.items():
        for pair, adj in by_pair.items():
            for q in aliased[pair]:
                contrasts[q].setdefault("holm", {})[family] = adj
    for q, c in contrasts.items():
        others = [o for o in aliased[pair_of[q]] if o != q]
        if others:
            c["same_test_as"] = others
    n_tests = len(unique_pairs)
    print(f"  tier 3: Holm over {n_tests} distinct arm pair(s) "
          f"from {len(contrasts)} question(s)")

    payload = {"variant": args.variant, "contrasts_source": COMPARISONS_SOURCE,
               "extract_source": EXTRACT_SOURCE, "tier2_ran": tier2,
               "n_distinct_tests": n_tests,
               "arms": tier1, "contrasts": contrasts,
               "descriptions": DESCRIPTIONS}
    (outdir / "stats.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    write_report(results, payload)
    write_contrast_tex(outdir, payload)
    print(f"\nwrote {outdir} and {results/'60_report'/'REPORT.md'}")
    return 0


# ---------------------------------------------------------------------------
def _f(v, dp=4, dash="—"):
    return dash if v is None else f"{v:.{dp}f}"


def write_contrast_tex(outdir: Path, payload: dict) -> None:
    L = [r"% Requires: \usepackage{booktabs}", r"\begin{table}[htbp]",
         r"\centering", r"\small",
         r"\caption{Declared contrasts. $\Delta$ is the second arm minus the "
         r"first. Paired $p$ is the per-target test (exact McNemar pooled by "
         r"Cochran--Mantel--Haenszel for match rate); Holm is adjusted across "
         r"the contrasts within each metric family.}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Question & Arms & $\Delta$ match & Welch $p$ & Paired $p$ & Holm \\",
         r"\midrule"]
    for q, c in payload["contrasts"].items():
        m = c["metrics"]["match_rate"]
        pm = (c.get("paired_match", {}).get("pooled_cmh") or {}).get("p")
        holm_p = (c.get("holm") or {}).get("paired_match")
        star = r"$^{\dagger}$" if c["preregistered"] else ""
        L.append(f"{q.replace('&', 'and')}{star} & {c['ref']}$\\to${c['test']} & "
                 f"{_f(m['delta'], 4, '---')} & {_f(m['welch_p'], 3, '---')} & "
                 f"{_f(pm, 3, '---')} & {_f(holm_p, 3, '---')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[2pt]\footnotesize $\dagger$ pre-registered primary question.",
          r"\end{table}", ""]
    (outdir / "contrasts.tex").write_text("\n".join(L))


def write_report(results: Path, payload: dict) -> None:
    arms, contrasts = payload["arms"], payload["contrasts"]
    L = ["# Angular-diffusion ablation — results", "",
         f"Variant: `{payload['variant']}`. Contrasts from "
         f"`{payload['contrasts_source']}`.", "",
         "## Arms", "",
         "| arm | n | match rate | RMSD | ccRMSD | MAE abc | MAE ang | KLD |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, e in arms.items():
        cells = []
        for key, _, _ in METRICS:
            m, sd = e[key]["mean"], e[key]["sd"]
            cells.append("—" if m is None else f"{m:.4f} ± {sd:.4f}")
        L.append(f"| {arm} | {e['n_seeds']} | " + " | ".join(cells) + " |")
    L += ["", "Arm descriptions:", ""]
    for arm in arms:
        if arm in payload["descriptions"]:
            L.append(f"- **{arm}** — {payload['descriptions'][arm]}")
    L += ["",
          "> **Denoising validation loss is not in this table on purpose.**",
          "> `L_ang` is a new term being optimised in the angular arms, so it",
          "> falls there *by construction*. It is reported separately below and",
          "> is not comparable between arms with and without the objective.",
          "", "## Denoising validation loss (within-family only)", "",
          "| arm | best val loss |", "|---|---|"]
    for arm, e in arms.items():
        v = e.get("val_loss") or {}
        cell = "—" if v.get("mean") is None else f"{v['mean']:.4f} ± {v['sd']:.4f}"
        L.append(f"| {arm} | {cell} |")
    L += ["", "## Contrasts", ""]
    for q, c in contrasts.items():
        tag = "**pre-registered**" if c["preregistered"] else "exploratory"
        L += [f"### {q}", "", f"`{c['ref']}` → `{c['test']}` ({tag})", ""]
        if c.get("same_test_as"):
            L += ["> Same arms, and therefore the same statistical test, as: "
                  + ", ".join(f"*{o}*" for o in c["same_test_as"])
                  + ". Counted once in the Holm adjustment.", ""]
        L += ["| metric | ref | test | Δ | Welch p | Hedges g | Holm |",
              "|---|---|---|---|---|---|---|"]
        for key, label, lower in METRICS:
            m = c["metrics"][key]
            holm_p = (c.get("holm") or {}).get(key)
            L.append(f"| {label} | {_f(m['ref_mean'])} | {_f(m['test_mean'])} | "
                     f"{_f(m['delta'])} | {_f(m['welch_p'], 3)} | "
                     f"{_f(m['hedges_g'], 2)} | {_f(holm_p, 3)} |")
        L.append("")
        pm = c.get("paired_match")
        if pm:
            cm = pm["pooled_cmh"]
            L += ["**Paired match-rate test.** Both arms are scored on the same "
                  "targets, so the informative quantity is the discordant pairs, "
                  "not the difference of two rates.", "",
                  "| seed | both | only ref | only test | neither | McNemar p |",
                  "|---|---|---|---|---|---|"]
            for i, t in enumerate(pm["per_seed"]):
                L.append(f"| {i} | {t['both']} | {t['only_a']} | {t['only_b']} | "
                         f"{t['neither']} | {_f(t['p'], 3)} |")
            L += ["",
                  f"Pooled (Cochran–Mantel–Haenszel over {cm['n_strata']} seeds): "
                  f"χ² = {_f(cm['chi2'], 2)}, p = {_f(cm['p'], 3)}", ""]
        pr = c.get("paired_rmsd")
        if pr:
            b = pr["bca"] or {}
            L += ["**Paired RMSD.** Wilcoxon signed-rank over per-target "
                  "differences, with a BCa bootstrap CI resampled over targets.",
                  "",
                  f"- n pairs: {pr['n_pairs']}",
                  f"- mean difference (test − ref): {_f(pr['mean_diff'])} Å",
                  f"- Wilcoxon p: {_f(pr['wilcoxon_p'], 4)}",
                  f"- 95% CI ({b.get('method', '—')}): "
                  f"[{_f(b.get('lo'))}, {_f(b.get('hi'))}]", ""]
    L += ["## How to read this", "",
          "- Tier-1 Welch p-values have n = 3 per arm and are **descriptive**. "
          "The README records match rate spanning 0.437–0.524 across fifteen "
          "independently trained models on a 103-target split; three seeds "
          "cannot resolve a few percent.",
          "- The paired tests are the powered ones. They test the *same* "
          "pre-registered metrics, using the fact that every arm sees the same "
          "targets — not a new metric, and not a post-hoc selection.",
          "- Holm columns adjust across the **distinct arm pairs** within each "
          "metric family. `COMPARISONS` names one pair twice, and two names for "
          "one test is still one test. Only the three marked pre-registered "
          "were fixed in advance; the rest are exploratory.",
          "- **Information level.** All arms receive the same conditioning "
          "(per-atom species, `natoms`, Tc) and no ground-truth geometry, so "
          "these contrasts are not confounded by what the model was shown. "
          "That is *not* true of any comparison against the published "
          "AtomBench baselines, which sit at different information tiers "
          "(CDVAE: full structure; AtomGPT: formula + Tc; FlowMM: composition "
          "only). Do not read across tiers without saying so.",
          "- **A4 is not a second baseline.** Its structural trunk is A3's with "
          "one aggregation zeroed, a different function from A0's. It controls "
          "for A3 only.",
          "- A generated-vs-real bond-angle histogram will show a large spike at "
          "180° in *both* distributions: back-tracking triplets, inherited from "
          "the graph builder and identical across arms. It does not bias a "
          "comparison and is not physics.", ""]
    (results / "60_report").mkdir(parents=True, exist_ok=True)
    (results / "60_report" / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
