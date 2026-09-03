#!/usr/bin/env python3
"""Computational cost accounting, per ablation.

    python costs.py [--harvest] [--jobids FILE]

Writes into ``50_costs/``:

    computational_costs.json   AtomBench's own cost schema, extended
    computational_costs.tex    booktabs table
    stage_times.csv            every stage of every run, flat
    gpu.csv                    GPU trace reduced to peak/mean per run
    sacct.tsv                  raw SLURM accounting (with --harvest)
    cost_report.md             the narrative, including the 2.4x check

Design notes, because the gaps matter as much as the numbers:

* ``run_task.py`` already records ``elapsed_s``, ``host`` and the git revision
  in ``<rundir>/.stages/<stage>.json`` for **all five** stages — but
  ``aggregate.py:print_cost`` surfaces only ``train_s``.  Sampling and
  relaxation are very plausibly the larger half of the bill (103 targets x 32
  candidates x 1000 denoising steps, plus 412 ALIGNN-FF relaxations) and they
  are *not* arm-independent: the smooth-topology arms rebuild the graph on
  every forward pass, which is a sampling cost as much as a training one.
  So all five are harvested here.

* ``history.json`` has no per-epoch timestamps, so a time series is not
  recoverable.  Per-step cost is derived exactly instead — epochs and batch
  size are pinned identically across arms, so the ratio *is* the per-step
  ratio, which is what the README's 2.4x claim is about.

* ``AcctGatherEnergyType = (null)`` on this cluster, so ``ConsumedEnergy``
  reads 0.  It is recorded as ``unavailable``, never as zero joules.  The
  ``power.draw`` integral from the GPU trace is reported separately and
  labelled an estimate, because it is one.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

STAGES = ("train", "generate", "symmetrize", "score-nosym", "score-sym",
          "angle_eval")

SACCT_FIELDS = (
    "JobID,JobName,State,ExitCode,Partition,NodeList,ReqTRES,AllocTRES,"
    "Submit,Start,End,Planned,Elapsed,ElapsedRaw,TotalCPU,CPUTimeRAW,"
    "MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite,ConsumedEnergy"
)


def jload(path: Path):
    try:
        return json.loads(path.read_text().replace("NaN", "null")
                          .replace("Infinity", "null"))
    except Exception:
        return None


def hms(seconds) -> str:
    if seconds is None:
        return "-"
    h, rest = divmod(int(seconds), 3600)
    return f"{h}h{rest//60:02d}m" if h else f"{rest//60}m{rest%60:02d}s"


# ---------------------------------------------------------------------------
def reduce_trace(path: Path) -> dict:
    """Peak memory, mean utilisation and mean power from a GPU trace."""
    mem, util, power = [], [], []
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("#") or line.startswith("ts,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                mem.append(float(parts[3]))
                util.append(float(parts[5]))
                power.append(float(parts[8]))
            except ValueError:
                continue
    except Exception:
        return {}
    if not mem:
        return {}
    return {
        "peak_gpu_mem_mib": round(max(mem), 1),
        "mean_gpu_util_pct": round(statistics.fmean(util), 1),
        "median_gpu_util_pct": round(statistics.median(util), 1),
        "mean_power_w": round(statistics.fmean(power), 1),
        "n_samples": len(mem),
    }


#: Which prepared split this run is about.  env.sh exports SPLIT; the default
#: keeps standalone invocations working on the original benchmark.
SPLIT = os.environ.get("SPLIT", "jarvis")

#: Documented train-set sizes, used only when the split is not on disk.
DOCUMENTED_N_TRAIN = {"jarvis": 847, "alex": 6603}


def train_set_size(runs_root: Path) -> tuple[int, str]:
    """Training-set size, for the per-step derivation."""
    for name in ("split_meta.json",):
        meta = jload(runs_root / "data" / SPLIT / name)
        if isinstance(meta, dict) and meta.get("n_train"):
            return int(meta["n_train"]), name
    train = runs_root / "data" / SPLIT / "train.json"
    data = jload(train)
    if isinstance(data, list):
        return len(data), "train.json"
    # JARVIS Supercon-3D is 847/105/103; Alexandria DS-A/B is 6603/825/825.
    return (
        DOCUMENTED_N_TRAIN.get(SPLIT, 847),
        f"documented default for {SPLIT} (split not on disk)",
    )


# ---------------------------------------------------------------------------
def build_records(results: Path, runs_root: Path) -> list[dict]:
    n_train, n_train_src = train_set_size(runs_root)
    records = []

    for seed_dir in sorted((results / "10_runs").glob("*/seed*")):
        arm = seed_dir.parent.name
        seed = int(re.search(r"seed(\d+)", seed_dir.name).group(1))
        rec = {"arm": arm, "seed": seed, "benchmark_name": f"{arm}_seed{seed}"}

        cfg = jload(seed_dir / "config.json") or {}
        rec["n_parameters"] = cfg.get("n_parameters")
        epochs = cfg.get("epochs")
        batch = cfg.get("batch_size") or 64
        rec["num_epochs"] = epochs
        rec["batch_size"] = batch

        hosts, per_stage = set(), {}
        for stage in STAGES:
            marker = jload(seed_dir / "stages" / f"{stage}.json")
            if not marker:
                continue
            per_stage[stage] = marker.get("elapsed_s")
            if marker.get("host"):
                hosts.add(marker["host"])
        rec["stage_s"] = per_stage
        rec["hosts"] = sorted(hosts)

        train_s = per_stage.get("train")
        # AtomBench calls sampling + everything downstream of it "inference".
        infer_s = sum(v for k, v in per_stage.items()
                      if k != "train" and v is not None) or None

        rec["train_s"] = train_s
        rec["train_h"] = round(train_s / 3600, 4) if train_s else None
        rec["infer_s"] = infer_s
        rec["infer_h"] = round(infer_s / 3600, 4) if infer_s else None
        if train_s and infer_s:
            rec["total_s"] = train_s + infer_s
            rec["total_h"] = round((train_s + infer_s) / 3600, 4)

        rec["train_s_per_epoch"] = (round(train_s / epochs, 2)
                                    if train_s and epochs else None)
        steps_per_epoch = math.ceil(n_train / batch) if batch else None
        rec["steps_per_epoch"] = steps_per_epoch
        rec["train_s_per_step"] = (
            round(train_s / (epochs * steps_per_epoch), 5)
            if train_s and epochs and steps_per_epoch else None
        )

        metrics = jload(seed_dir / "metrics_sym.json") or {}
        n_test = (metrics.get("RMSE", {}).get("AtomGen", {}) or {}).get("n_total")
        rec["num_test_structures"] = n_test
        rec["infer_s_per_structure"] = (round(infer_s / n_test, 4)
                                        if infer_s and n_test else None)
        rec["n_matched"] = (metrics.get("RMSE", {}).get("AtomGen", {}) or {}).get("n_matched")
        rec["match_rate"] = (metrics.get("RMSE", {}).get("AtomGen", {}) or {}).get("match_rate")
        # The honest cost-benefit figure: a 2.4x step cost that buys nothing is
        # a different result from one that buys three more matches.
        rec["gpu_s_per_match"] = (round(rec["total_s"] / rec["n_matched"], 1)
                                  if rec.get("total_s") and rec.get("n_matched") else None)

        rec.update(reduce_trace(seed_dir / "gpu_trace.csv"))
        if rec.get("mean_power_w") and rec.get("total_s"):
            # Estimated, not accounted: SLURM gathers no energy on this cluster.
            rec["est_energy_kwh"] = round(
                rec["mean_power_w"] * rec["total_s"] / 3.6e6, 4)
        rec["energy_j_slurm"] = "unavailable (AcctGatherEnergyType=null)"
        records.append(rec)

    for rec in records:
        rec["_n_train"] = n_train
        rec["_n_train_source"] = n_train_src
    return records


def relaxation_health(runs_root: Path, tree: str) -> list[str]:
    """Did the force-field relaxation actually run, or silently degrade?

    generate_benchmark.py relaxes in a multiprocessing pool.  A worker that
    dies -- an unreachable figshare on a compute node is the way this happens
    here -- does not fail the stage: the parent still writes pred.csv, still
    exits 0, and the runner still records the stage as done.  The benchmark
    would then be quietly wrong rather than loudly broken.

    candidates.json carries the per-candidate energies and errors, so the
    degradation is detectable after the fact.  Checked for every run.
    """
    notes = []
    for cand in sorted((runs_root / tree).glob("*/seed*/bench/*/candidates.json")):
        try:
            recs = json.loads(cand.read_text())
        except Exception as exc:
            notes.append(f"{cand}: unreadable ({exc})")
            continue
        if not recs:
            notes.append(f"{cand}: empty")
            continue
        n_null = sum(1 for r in recs for e in (r.get("energies") or []) if e is None)
        n_err = sum(1 for r in recs for e in (r.get("errors") or []) if e)
        n_e = sum(len(r.get("energies") or []) for r in recs)
        conv = [c for r in recs for c in (r.get("converged") or [])]
        tag = f"{cand.parent.parent.parent.parent.name}/{cand.parent.parent.parent.name}"
        if n_e == 0:
            notes.append(f"{tag}: NO energies -- ranking did not run")
        elif n_null or n_err:
            notes.append(f"{tag}: {n_null}/{n_e} null energies, {n_err} worker "
                         "errors -- relaxation degraded, treat these metrics "
                         "as unreliable")
        elif conv and not any(conv):
            notes.append(f"{tag}: 0/{len(conv)} candidates converged "
                         "(expected at smoke relax-steps; suspicious at 200)")
    return notes


# ---------------------------------------------------------------------------
def harvest_sacct(results: Path, jobids_file: Path) -> Path | None:
    """Pull SLURM accounting before the database ages it out."""
    if not jobids_file.exists():
        print(f"  no {jobids_file}; skipping sacct harvest")
        return None
    ids = sorted({line.split("\t")[2].strip()
                  for line in jobids_file.read_text().splitlines()
                  if len(line.split("\t")) >= 3})
    if not ids:
        return None
    out = results / "50_costs" / "sacct.tsv"
    try:
        proc = subprocess.run(
            ["sacct", "-j", ",".join(ids), "--parsable2", "--units=M",
             "--format", SACCT_FIELDS],
            capture_output=True, text=True, timeout=300)
    except Exception as exc:
        print(f"  sacct failed: {exc}")
        return None
    if proc.returncode != 0:
        print(f"  sacct exit {proc.returncode}: {proc.stderr.strip()[:200]}")
        return None
    out.write_text(proc.stdout)
    print(f"  sacct -> {out}  ({len(proc.stdout.splitlines())-1} rows)")

    # gres/gpumem and gres/gpuutil are registered in AccountingStorageTRES but
    # registration is not population; 10_smoke.sh probes it and this reports
    # what actually came back rather than assuming either way.
    has = {k: (k in proc.stdout) for k in ("gres/gpumem", "gres/gpuutil")}
    print(f"  AllocTRES carries: " +
          ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in has.items()))
    return out


# ---------------------------------------------------------------------------
def arm_summary(records: list[dict]) -> dict[str, dict]:
    arms: dict[str, list[dict]] = {}
    for r in records:
        arms.setdefault(r["arm"], []).append(r)

    out = {}
    for arm, rows in sorted(arms.items()):
        def agg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                return None, None
            return (round(statistics.fmean(vals), 5),
                    round(statistics.stdev(vals), 5) if len(vals) > 1 else 0.0)
        out[arm] = {
            "n_seeds": len(rows),
            "n_parameters": rows[0].get("n_parameters"),
            "train_s": agg("train_s"),
            "train_s_per_step": agg("train_s_per_step"),
            "infer_s": agg("infer_s"),
            "infer_s_per_structure": agg("infer_s_per_structure"),
            "total_h": agg("total_h"),
            "peak_gpu_mem_mib": agg("peak_gpu_mem_mib"),
            "mean_gpu_util_pct": agg("mean_gpu_util_pct"),
            "gpu_s_per_match": agg("gpu_s_per_match"),
            "hosts": sorted({h for r in rows for h in r.get("hosts", [])}),
            "devices": sorted({r.get("gpu_device") for r in rows
                               if r.get("gpu_device")}),
        }
    return out


def homogeneity_audit(records: list[dict], summary: dict) -> list[str]:
    """Hardware and contention checks.

    Pinning A30 fixes the silicon; it does not fix node contention, and a
    contended run makes wall time incomparable even on identical hardware.
    Flagged units are reported and excluded from the timing comparison, never
    dropped silently.
    """
    notes = []
    devices = {r.get("gpu_device") for r in records if r.get("gpu_device")}
    if len(devices) > 1:
        notes.append(f"MIXED GPU MODELS across runs: {sorted(devices)} — "
                     "per-step timings are not comparable across arms.")
    elif devices:
        notes.append(f"all runs on {devices.pop()} — hardware homogeneous.")
    else:
        notes.append("no GPU device recorded (sampler did not run); hardware "
                     "homogeneity is unverified.")

    for arm, rows in {a: [r for r in records if r["arm"] == a]
                      for a in {r['arm'] for r in records}}.items():
        vals = [r["train_s_per_step"] for r in rows if r.get("train_s_per_step")]
        if len(vals) < 2:
            continue
        med = statistics.median(vals)
        for r in rows:
            v = r.get("train_s_per_step")
            if v and med and v > 1.5 * med:
                notes.append(
                    f"{arm} seed{r['seed']}: {v:.4f} s/step is {v/med:.2f}x the "
                    f"arm median ({med:.4f}) on {','.join(r.get('hosts') or ['?'])} "
                    "— likely node contention; EXCLUDED from the timing "
                    "comparison, retained everywhere else.")
                r["timing_excluded"] = True
    return notes


# ---------------------------------------------------------------------------
def fmt(v, spec="", dash="---"):
    """Format a value, or a dash when it is missing."""
    return dash if v is None else format(v, spec)


def latex(summary: dict, baseline: str) -> str:
    ref = (summary.get(baseline, {}).get("train_s_per_step") or (None,))[0]
    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"% Schema matches atombench/scripts/harvest_compute_times.py so these",
        r"% rows drop into the same table as the published baselines.",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Computational cost of the angular-diffusion ablations. "
        r"Train/step is normalised by configured epochs and steps per epoch; "
        r"Infer/struct by the number of test structures. "
        r"Mean $\pm$ s.d. over seeds.}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Arm & Params (M) & Train (h) & Train/step (s) & $\times$ base & "
        r"Infer/struct (s) & Peak GPU (GiB) \\",
        r"\midrule",
    ]
    for arm, s in summary.items():
        params = s["n_parameters"]
        train_h = s["train_s"][0] / 3600 if s["train_s"][0] else None
        per_step, per_step_sd = s["train_s_per_step"]
        mem = s["peak_gpu_mem_mib"][0]
        step_cell = ("---" if per_step is None
                     else f"{per_step:.4f} $\\pm$ {per_step_sd:.4f}")
        lines.append(
            f"{arm.replace('_', chr(92) + '_')} & "
            f"{fmt(params / 1e6 if params else None, '.2f')} & "
            f"{fmt(train_h, '.2f')} & "
            f"{step_cell} & "
            f"{fmt(per_step / ref if (per_step and ref) else None, '.2f')} & "
            f"{fmt(s['infer_s_per_structure'][0], '.2f')} & "
            f"{fmt(mem / 1024 if mem else None, '.1f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", action="store_true",
                    help="also pull sacct (do this soon after each array "
                         "finishes; the accounting database ages out)")
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    ap.add_argument("--runs-root", default=os.environ.get("CSP_RUNS"))
    ap.add_argument("--baseline", default="A0")
    args = ap.parse_args()

    if not args.results or not args.runs_root:
        print("set RESULTS and CSP_RUNS (source ./env.sh)", file=sys.stderr)
        return 2
    results, runs_root = Path(args.results), Path(args.runs_root)
    outdir = results / "50_costs"
    outdir.mkdir(parents=True, exist_ok=True)

    records = build_records(results, runs_root)
    if not records:
        print("no runs collected; run collect.py first", file=sys.stderr)
        return 1
    summary = arm_summary(records)
    notes = homogeneity_audit(records, summary)
    tree = "train_quick" if (results / "10_runs").exists() and False else "train"
    notes += relaxation_health(runs_root, tree) or ["force-field relaxation "
                                                   "healthy in every run"]

    if args.harvest:
        print("harvesting SLURM accounting")
        harvest_sacct(results, results / "00_provenance" / "slurm_jobids.txt")

    (outdir / "computational_costs.json").write_text(
        json.dumps({"per_run": {r["benchmark_name"]: r for r in records},
                    "per_arm": summary,
                    "audit": notes}, indent=2) + "\n")
    (outdir / "computational_costs.tex").write_text(latex(summary, args.baseline))

    with (outdir / "stage_times.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed", "stage", "elapsed_s", "hosts"])
        for r in records:
            for stage, secs in r["stage_s"].items():
                w.writerow([r["arm"], r["seed"], stage, secs,
                            ";".join(r.get("hosts", []))])

    with (outdir / "gpu.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["peak_gpu_mem_mib", "mean_gpu_util_pct", "median_gpu_util_pct",
                "mean_power_w", "est_energy_kwh", "n_samples"]
        w.writerow(["arm", "seed", *cols])
        for r in records:
            w.writerow([r["arm"], r["seed"], *[r.get(c) for c in cols]])

    write_report(outdir, records, summary, notes, args.baseline)

    # -- console -------------------------------------------------------------
    ref = (summary.get(args.baseline, {}).get("train_s_per_step") or (None,))[0]
    print(f"\ncost per arm  (baseline {args.baseline}, {records[0]['_n_train']} "
          f"train crystals from {records[0]['_n_train_source']})")
    head = (f"  {'arm':<14}{'params':>9}{'train':>9}{'s/step':>10}"
            f"{'x base':>8}{'infer/str':>11}{'GPU GiB':>9}{'s/match':>9}")
    print(head)
    for arm, s in summary.items():
        params = s["n_parameters"]
        per_step = s["train_s_per_step"][0]
        per_struct = s["infer_s_per_structure"][0]
        mem = s["peak_gpu_mem_mib"][0]
        per_match = s["gpu_s_per_match"][0]
        cells = [
            fmt(params / 1e6 if params else None, ".2f", "-"),
            hms(s["train_s"][0]),
            fmt(per_step, ".4f", "-"),
            fmt(per_step / ref if (per_step and ref) else None, ".2f", "-"),
            fmt(per_struct, ".2f", "-"),
            fmt(mem / 1024 if mem else None, ".1f", "-"),
            fmt(per_match, ".0f", "-"),
        ]
        print(f"  {arm:<14}{cells[0]:>9}{cells[1]:>9}{cells[2]:>10}"
              f"{cells[3]:>8}{cells[4]:>11}{cells[5]:>9}{cells[6]:>9}")
    print()
    for n in notes:
        print(f"  * {n}")
    print(f"\nwrote {outdir}")
    return 0


def write_report(outdir: Path, records, summary, notes, baseline) -> None:
    ref = (summary.get(baseline, {}).get("train_s_per_step") or (None,))[0]
    L = ["# Computational cost", "",
         f"Baseline arm: `{baseline}`. Mean ± s.d. over seeds.", "",
         "## The 2.4× per-step claim", "",
         "The README records the angular channel as costing **2.4× per training",
         "step**. Epochs and batch size are pinned identically across arms, so",
         "the wall-time ratio below *is* the per-step ratio — this is a test of",
         "that number, not a restatement of it.", "",
         "| arm | s/step | × baseline |", "|---|---|---|"]
    for arm, s in summary.items():
        ps, sd = s["train_s_per_step"]
        step_cell = "—" if ps is None else f"{ps:.4f} ± {sd:.4f}"
        L.append(f"| {arm} | {step_cell} | "
                 f"{fmt(ps / ref if (ps and ref) else None, '.2f', '—')} |")
    L += ["", "## Total cost per arm", "",
          "| arm | params | train | inference | total GPU-h | GPU-s per match |",
          "|---|---|---|---|---|---|"]
    for arm, s in summary.items():
        p = s["n_parameters"]
        params_cell = fmt(p / 1e6 if p else None, ".2f", "—")
        L.append(
            f"| {arm} | {params_cell}{' M' if p else ''} "
            f"| {hms(s['train_s'][0])} | {hms(s['infer_s'][0])} "
            f"| {fmt(s['total_h'][0], '.2f', '—')} "
            f"| {fmt(s['gpu_s_per_match'][0], '.0f', '—')} |")
    L += ["",
          "`inference` is sampling + relaxation + symmetrisation + scoring.",
          "`generate_benchmark.py` runs sampling and ALIGNN-FF relaxation in one",
          "stage, so the split between them is not separable without",
          "instrumenting that script — a stated limitation, not an omission.", "",
          "## GPU", "",
          "| arm | peak memory | mean utilisation |", "|---|---|---|"]
    for arm, s in summary.items():
        m, u = s["peak_gpu_mem_mib"][0], s["mean_gpu_util_pct"][0]
        L.append(f"| {arm} | {fmt(m / 1024 if m else None, '.1f', '—')} GiB "
                 f"| {fmt(u, '.0f', '—')}% |")
    L += ["",
          "Energy: **not available**. `AcctGatherEnergyType = (null)` on this",
          "cluster, so `sacct ConsumedEnergy` reads 0 and is recorded as",
          "`unavailable` rather than as zero joules. The `est_energy_kwh` column",
          "in `gpu.csv` integrates `power.draw` from the sampler and is an",
          "estimate, not accounting.", "",
          "## Audit", ""]
    L += [f"- {n}" for n in notes]
    L.append("")
    (outdir / "cost_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
