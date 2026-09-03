#!/usr/bin/env python3
"""Collect the run tree into a self-contained, archivable results directory.

    python collect.py [--quick] [--force]

Two tiers, split on one property: *the light tier must be sufficient to
re-derive every number without a GPU and without the checkpoints*.  Every
AtomBench metric is a pure function of ``pred.csv``, so keeping the CSVs keeps
the science; the checkpoints stay on scratch where the space is.

    heavy   /scratch/.../alignn_csp/     checkpoints, candidates.json   tens of GB
    light   results/<RUN_ID>/            everything else                 <1 GB

Idempotent.  Refuses to overwrite a RUN_ID that already has a manifest unless
given --force, because a results tree that silently changed under you is worse
than one that refused to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parent

# Files copied out of each run directory.  Small, and between them they carry
# the predictions, the metrics, the training curve, the exact arguments, the
# per-stage cost and the mechanism metrics.
RUN_FILES = {
    "bench/nosym/pred.csv": "bench_nosym.csv",
    "bench/nosym/metrics.json": "metrics_nosym.json",
    "bench/sym/pred.csv": "bench_sym.csv",
    "bench/sym/metrics.json": "metrics_sym.json",
    "bench/sym/angle_eval.json": "angle_eval.json",
    # The unrelaxed variants: the model's own output, with no force field in
    # the loop.  Scored separately because the relaxed numbers measure the
    # diffusion model *and* ALIGNN-FF together, which is the wrong denominator
    # for an ablation.
    "bench/raw/pred.csv": "bench_raw.csv",
    "bench/raw/metrics.json": "metrics_raw.json",
    "bench/rawsym/pred.csv": "bench_rawsym.csv",
    "bench/rawsym/metrics.json": "metrics_rawsym.json",
    "bench/rawsym/angle_eval.json": "angle_eval_rawsym.json",
    "history.json": "history.json",
    "config.json": "config.json",
}


def sh(*cmd, cwd=None) -> str:
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance must never crash
        return f"<unavailable: {exc}>"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
SPLIT = os.environ.get("SPLIT", "jarvis")


def discover_runs(runs_root: Path, tree: str) -> list[tuple[str, int, Path]]:
    """(arm, seed, rundir) for every training run on disk.

    Globbing beats importing tasks.py here: it picks up whatever actually ran,
    including the confound arm and <split>_nolg, neither of which the runner
    knows how to enumerate as part of the angle-ablation task.
    """
    out = []
    # Config names are prefixed with the split ("jarvis_A0", "alex_A0") so the
    # two benchmarks never collide in one run tree; strip whichever applies so
    # arm labels stay dataset-agnostic in the tables.
    prefixes = tuple(f"{s}_" for s in (SPLIT, "jarvis", "alex"))
    for rundir in sorted((runs_root / tree).glob("*/seed*")):
        if not rundir.is_dir():
            continue
        arm = rundir.parent.name
        for pref in prefixes:
            if arm.startswith(pref):
                arm = arm[len(pref):]
                break
        m = re.search(r"seed(\d+)$", rundir.name)
        out.append((arm, int(m.group(1)) if m else -1, rundir))
    return out


def run_window(rundir: Path) -> tuple[float, float] | None:
    """Wall-clock span of a run, from its stage markers.

    Used to attach GPU traces, which are keyed by (job id, array index)
    because the sampler hook fires before run_task.py has resolved which unit
    it is.  Host plus time window identifies it unambiguously.
    """
    lo = hi = None
    for marker in sorted((rundir / ".stages").glob("*.json")):
        try:
            data = json.loads(marker.read_text())
            end = datetime.fromisoformat(data["finished"]).timestamp()
            start = end - float(data.get("elapsed_s", 0))
        except Exception:
            continue
        lo = start if lo is None else min(lo, start)
        hi = end if hi is None else max(hi, end)
    return None if lo is None else (lo, hi)


def marker_hosts(rundir: Path) -> set[str]:
    hosts = set()
    for marker in (rundir / ".stages").glob("*.json"):
        try:
            host = json.loads(marker.read_text()).get("host")
        except Exception:
            continue
        if host:
            hosts.add(host)
    return hosts


def trace_meta(path: Path) -> dict:
    """Header line written by lib/gpu_sampler.sh."""
    try:
        first = path.read_text(errors="replace").split("\n", 1)[0]
    except Exception:
        return {}
    if not first.startswith("#"):
        return {}
    meta = dict(
        part.split("=", 1)
        for part in first[1:].strip().split(" ")
        if "=" in part
    )
    if "started" in meta:
        try:
            meta["started_ts"] = datetime.fromisoformat(meta["started"]).timestamp()
        except ValueError:
            pass
    return meta


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="collect the train_quick/ tree instead of train/")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a results tree that already has a manifest")
    ap.add_argument("--verify", action="store_true",
                    help="re-check every collected file against the manifest's "
                         "sha256 and exit; catches anything that mutated the "
                         "archival tier after collection")
    ap.add_argument("--runs-root", default=os.environ.get("CSP_RUNS"))
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    args = ap.parse_args()

    if not args.runs_root or not args.results:
        print("set CSP_RUNS and RESULTS (source ./env.sh)", file=sys.stderr)
        return 2

    runs_root = Path(args.runs_root)
    results = Path(args.results)
    tree = "train_quick" if args.quick else "train"

    manifest_path = results / "00_provenance" / "manifest.json"

    if args.verify:
        return verify(results, manifest_path)

    if manifest_path.exists() and not args.force:
        print(f"{manifest_path} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    for sub in ("00_provenance", "10_runs", "20_benchmarks", "30_atombench",
                "40_stats", "50_costs", "60_report"):
        (results / sub).mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_root, tree)
    if not runs:
        print(f"no runs under {runs_root/tree}", file=sys.stderr)
        return 1
    print(f"found {len(runs)} run(s) under {runs_root/tree}")

    # -- GPU traces, matched by host and time window -------------------------
    traces = []
    trace_dir = results / "_gputrace"
    for tpath in sorted(trace_dir.glob("*.csv")) if trace_dir.is_dir() else []:
        traces.append((tpath, trace_meta(tpath)))
    used_traces: set[Path] = set()

    collected, missing = [], []
    for arm, seed, rundir in runs:
        dest = results / "10_runs" / arm / f"seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        record = {"arm": arm, "seed": seed, "rundir": str(rundir), "files": {}}

        for rel, name in RUN_FILES.items():
            src = rundir / rel
            if not src.exists():
                missing.append(f"{arm}/seed{seed}: {rel}")
                continue
            shutil.copy2(src, dest / name)
            record["files"][name] = {"sha256": sha256(dest / name),
                                     "bytes": (dest / name).stat().st_size}

        stages_dir = dest / "stages"
        stages_dir.mkdir(exist_ok=True)
        for marker in sorted((rundir / ".stages").glob("*.json")):
            shutil.copy2(marker, stages_dir / marker.name)

        window, hosts = run_window(rundir), marker_hosts(rundir)
        if window:
            lo, hi = window
            for tpath, meta in traces:
                ts = meta.get("started_ts")
                if ts is None or meta.get("host") not in hosts:
                    continue
                if lo - 300 <= ts <= hi + 300:
                    shutil.copy2(tpath, dest / "gpu_trace.csv")
                    record["gpu_trace"] = tpath.name
                    record["gpu_device"] = meta.get("device")
                    used_traces.add(tpath)
                    break
        record["hosts"] = sorted(hosts)
        collected.append(record)

    orphan = [t.name for t, _ in traces if t not in used_traces]

    # -- provenance ----------------------------------------------------------
    prov = results / "00_provenance"
    alignn_repo = Path(os.environ.get("ALIGNN_REPO", ""))
    atombench_repo = Path(os.environ.get("ATOMBENCH_REPO", ""))

    cluster_env = alignn_repo / "task_runners" / "cluster.env"
    if cluster_env.exists():
        shutil.copy2(cluster_env, prov / "cluster.env")
    if (HARNESS / "env.sh").exists():
        shutil.copy2(HARNESS / "env.sh", prov / "env.sh")

    for env_name, env_path in (("train", os.environ.get("TRAIN_ENV")),
                               ("score", os.environ.get("SCORE_ENV_PATH"))):
        pip = Path(env_path or "/nonexistent") / "bin" / "pip"
        if pip.exists():
            (prov / f"pip_freeze_{env_name}.txt").write_text(sh(str(pip), "freeze"))

    manifest = {
        "run_id": os.environ.get("RUN_ID"),
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "tree": tree,
        "runs_root": str(runs_root),
        "alignn": {
            "path": str(alignn_repo),
            "rev": sh("git", "rev-parse", "HEAD", cwd=alignn_repo or None),
            "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=alignn_repo or None),
            "dirty": bool(sh("git", "status", "--porcelain", cwd=alignn_repo or None)),
        },
        "atombench": {
            "path": str(atombench_repo),
            "rev": sh("git", "rev-parse", "HEAD", cwd=atombench_repo or None),
        },
        "n_runs": len(collected),
        "runs": collected,
        "missing_files": missing,
        "orphan_gpu_traces": orphan,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    write_index(results, manifest)

    print(f"collected {len(collected)} run(s) -> {results}")
    if missing:
        print(f"  {len(missing)} missing file(s) (unfinished stages?):")
        for m in missing[:12]:
            print(f"    {m}")
        if len(missing) > 12:
            print(f"    ... and {len(missing)-12} more")
    if orphan:
        print(f"  {len(orphan)} GPU trace(s) matched no run: {orphan[:5]}")
    return 0


def verify(results: Path, manifest_path: Path) -> int:
    """Re-check the archival tier against the manifest.

    10_runs/ is meant to be immutable once collected.  Anything that writes
    through into it -- most easily the AtomBench CLI, which recomputes
    metrics.json in place and will follow a symlink to do it -- shows up here
    as a checksum mismatch rather than as a silently different number three
    steps later.
    """
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    checked = changed = gone = 0
    for rec in manifest["runs"]:
        base = results / "10_runs" / rec["arm"] / f"seed{rec['seed']}"
        for name, meta in rec["files"].items():
            path = base / name
            checked += 1
            if not path.exists():
                gone += 1
                print(f"  MISSING  {rec['arm']}/seed{rec['seed']}/{name}")
            elif sha256(path) != meta["sha256"]:
                changed += 1
                print(f"  CHANGED  {rec['arm']}/seed{rec['seed']}/{name}")
    print(f"\nverified {checked} file(s): {changed} changed, {gone} missing")
    if changed or gone:
        print("The archival tier has been modified since collection. Re-run "
              "`python collect.py --force` to re-baseline, but first work out "
              "what wrote to it.", file=sys.stderr)
        return 1
    print("archival tier intact")
    return 0


def write_index(results: Path, manifest: dict) -> None:
    """INDEX.md is generated, never hand-written, so it cannot drift."""
    arms = sorted({r["arm"] for r in manifest["runs"]})
    lines = [
        f"# Results index — `{manifest['run_id']}`",
        "",
        f"Collected {manifest['collected_utc']} from `{manifest['runs_root']}`",
        f"(`{manifest['tree']}/` tree). ALIGNN `{manifest['alignn']['rev'][:8]}` "
        f"on `{manifest['alignn']['branch']}`"
        + ("  **working tree dirty**" if manifest["alignn"]["dirty"] else "")
        + f", AtomBench `{manifest['atombench']['rev'][:8]}`.",
        "",
        f"{manifest['n_runs']} runs over {len(arms)} arms: {', '.join(arms)}.",
        "",
        "This directory is self-contained: every metric here is a pure function",
        "of the prediction CSVs, so the whole analysis re-runs with no GPU and",
        "no checkpoints. The checkpoints stay in the run tree above.",
        "",
        "## What is where",
        "",
        "| path | contents | regenerate with |",
        "|---|---|---|",
        "| `00_provenance/manifest.json` | git revisions, env freezes, per-file sha256 | `python collect.py` |",
        "| `00_provenance/cluster.env` | the generated site config, as submitted | `bash env.sh --write` |",
        "| `00_provenance/doctor.txt` | dependency check at submission time | `bash 00_setup.sh` |",
        "| `00_provenance/phase1_probe.txt` | CUDA / MIG / sampler / sacct-TRES probe | `bash 10_smoke.sh` |",
        "| `00_provenance/slurm_jobids.txt` | array job ids, for the sacct harvest | `bash 30_full.sh` |",
        "| `10_runs/<arm>/seed<N>/` | predictions, metrics, history, config, stage costs, GPU trace | `python collect.py` |",
        "| `20_benchmarks/<arm>_seed<N>/` | AtomBench staging tree (symlinks) | `python stage_benchmarks.py` |",
        "| `30_atombench/` | AtomBench's own figures and metrics table | `atombench 20_benchmarks 30_atombench` |",
        "| `40_stats/` | seed statistics, paired tests, contrasts | `python analyze.py` |",
        "| `50_costs/` | cost table, sacct record, GPU traces reduced | `python costs.py --harvest` |",
        "| `60_report/REPORT.md` | the narrative document | `python analyze.py` |",
        "",
        "## Per-run files",
        "",
        "| file | what it is |",
        "|---|---|",
        "| `bench_sym.csv` / `bench_nosym.csv` | predictions, symmetrised and not. Both are kept because they answer different questions: the manuscript's lattice columns are measured after symmetrisation, the pipeline ablation before it. |",
        "| `metrics_sym.json` / `metrics_nosym.json` | AtomBench metrics for each |",
        "| `history.json` | per-epoch train/val denoising loss (no timestamps — per-step cost is derived in `50_costs/`) |",
        "| `config.json` | the full argument namespace plus `n_parameters` |",
        "| `stages/*.json` | per-stage `elapsed_s`, `host`, git revision, exact argv |",
        "| `angle_eval.json` | bond-angle distribution distance and relaxation displacement |",
        "| `gpu_trace.csv` | 30 s samples of GPU memory, utilisation and power |",
        "",
    ]
    if manifest["missing_files"]:
        lines += ["## Incomplete", "",
                  f"{len(manifest['missing_files'])} expected file(s) were absent "
                  "at collection time — unfinished or failed stages:", "",
                  "```"] + manifest["missing_files"][:40] + ["```", ""]
    (results / "INDEX.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
