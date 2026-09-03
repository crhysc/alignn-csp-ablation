#!/usr/bin/env python3
"""Build the AtomBench staging tree.

    python stage_benchmarks.py [--variant sym|nosym]

Every prediction file in the run tree is called ``pred.csv``, and AtomBench
names each benchmark after the directory containing it
(``discover_benchmark_csvs``: a structured directory is one benchmark CSV per
immediate subdirectory, labelled by the subdirectory name).  So without
staging, all 21 benchmarks would be called ``sym`` and every figure would be
unreadable.

This writes ``20_benchmarks/<arm>_seed<N>/`` containing:

    pred.csv      symlink into 10_runs/   (only ever read)
    metrics.json  *copy* of 10_runs/      (the CLI overwrites this in place)

The asymmetry is deliberate and was found the hard way.  The ``atombench`` CLI
recomputes metrics and writes ``metrics.json`` beside each CSV; Python's
``open(path, "w")`` follows a symlink and writes to its target, so linking that
file made the CLI silently overwrite the collected record and invalidate its
manifest checksum.  Copying it keeps ``10_runs/`` immutable — the archival tier
stays exactly as collected, and the recomputed values in ``20_benchmarks/`` can
be diffed against it as a genuine cross-check rather than replacing it.

Staging ``metrics.json`` at all is what makes the analysis cheap:
``atombench.tables.collect_metrics`` reads it from beside each CSV rather than
recomputing, so ``analyze.py`` costs nothing once the array has scored, whether
or not you also run the CLI.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="sym",
                    choices=("sym", "nosym", "raw", "rawsym"),
                    help="which scored CSV to stage (default: sym, which is "
                         "what the manuscript's lattice columns are quoted on)")
    ap.add_argument("--results", default=os.environ.get("RESULTS"))
    args = ap.parse_args()

    if not args.results:
        print("set RESULTS (source ./env.sh)", file=sys.stderr)
        return 2
    results = Path(args.results)
    runs = results / "10_runs"
    if not runs.is_dir():
        print(f"no {runs}; run collect.py first", file=sys.stderr)
        return 1

    staging = results / ("20_benchmarks" if args.variant == "sym"
                         else f"20_benchmarks_{args.variant}")
    staging.mkdir(parents=True, exist_ok=True)
    for stale in staging.iterdir():
        if stale.is_dir():
            for f in stale.iterdir():
                f.unlink()          # unlink() removes the link, not its target
            stale.rmdir()

    made, skipped = 0, []
    for seed_dir in sorted(runs.glob("*/seed*")):
        arm = seed_dir.parent.name
        csv = seed_dir / f"bench_{args.variant}.csv"
        mj = seed_dir / f"metrics_{args.variant}.json"
        if not csv.exists():
            skipped.append(f"{arm}/{seed_dir.name}: no bench_{args.variant}.csv")
            continue
        dest = staging / f"{arm}_{seed_dir.name}"
        dest.mkdir(parents=True, exist_ok=True)
        # Relative link so the whole results tree stays movable.  Read-only:
        # nothing downstream writes to a benchmark CSV.
        (dest / "pred.csv").symlink_to(os.path.relpath(csv, dest))
        if mj.exists():
            # COPY, not a link -- the CLI writes metrics.json in place and a
            # symlink would carry that write back into the archival tier.
            shutil.copy2(mj, dest / "metrics.json")
        else:
            skipped.append(f"{arm}/{seed_dir.name}: no metrics_{args.variant}.json "
                           "(scored yet?)")
        made += 1

    print(f"staged {made} benchmark(s) -> {staging}  (variant: {args.variant})")
    for s in skipped:
        print(f"  skipped: {s}")
    if made:
        print("\nnext:")
        print(f"  $SCORE_ENV_PATH/bin/atombench {staging} {results/'30_atombench'}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
