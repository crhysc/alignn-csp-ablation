#!/usr/bin/env bash
# Phases 2 and 3 -- the cheap filter, then the decision point.
#
#   bash 20_pilot.sh loss     # phase 2: 12 elements, training only, inter_a30
#   bash 20_pilot.sh quick    # phase 3: full --quick pipeline, gpu_2day
#   bash 20_pilot.sh measure  # read the pilot's cost back, print phase-4 --time
#
# Phase 3 is the decision point: --quick keeps everything that makes two arms
# comparable (the whole test split, the same pipeline, the same scoring code)
# and cuts only what scales cost.  Its numbers are not comparable to the
# published ones -- `verify --quick` says so rather than letting you read
# across -- but arm-vs-arm it is a real measurement at about a tenth of the
# price, and it debugs collect/costs/analyze on real data before the
# expensive run exists.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)_quick}"; export RUN_ID
source ./env.sh

MODE="${1:-quick}"
JOBLOG="$RESULTS/00_provenance/slurm_jobids.txt"
mkdir -p "$RESULTS/00_provenance"

record_jobs() {  # <task> <submit.sh output>
    awk -v t="$1" '/submitted array job/{print t"\tarray\t"$4}
                   /submitted aggregation job/{print t"\tagg\t"$4}' \
        >> "$JOBLOG"
}

case "$MODE" in

loss)
    echo "=== phase 2: --quick --loss-only on $PART_INTER"
    # 6 arms x 2 seeds, 300 epochs, training only.  No scoring environment, no
    # sampling, no AtomBench.  The README calls the denoising validation loss
    # the most reproducible arm-vs-arm signal there is -- it repeated to three
    # decimals across two machines.  It is a filter for a broken arm, not a
    # verdict on a real one: the line-graph ablation moved that loss by 14.5%
    # and moved match rate by exactly nothing.
    csp_submit "$PART_INTER" "$PILOT_TIME" angle-ablation \
        --quick --loss-only --relax-workers "$RELAX_WORKERS" \
        | tee /dev/stderr | record_jobs angle-ablation-loss
    echo
    echo "when it finishes:"
    echo "  source env.sh && run_task angle-ablation --quick --aggregate"
    ;;

quick)
    echo "=== phase 3: full --quick on $PART_PILOT"
    run_task_data data-jarvis
    csp_submit "$PART_PILOT" "$PILOT_TIME" angle-ablation \
        --quick --relax-workers "$RELAX_WORKERS" --symprec "$SYMPREC" \
        | tee /dev/stderr | record_jobs angle-ablation-quick
    echo
    echo "when it finishes, exercise the whole analysis chain on it:"
    echo "  bash 20_pilot.sh measure"
    echo "  python collect.py --quick && python stage_benchmarks.py"
    echo "  $SCORE_ENV_PATH/bin/atombench $RESULTS/20_benchmarks $RESULTS/30_atombench"
    echo "  python costs.py --harvest && python analyze.py"
    ;;

measure)
    # The shipped walltimes are placeholders the repo explicitly disclaims.
    # This turns the pilot into the measurement, rather than trusting them.
    echo "=== measured pilot cost -> phase-4 walltime"
    ids=$(awk '$1 ~ /quick/ && $2=="array" {print $3}' "$JOBLOG" | paste -sd, -)
    [ -n "$ids" ] || { echo "no quick array in $JOBLOG; run 'quick' first" >&2; exit 1; }
    sacct -j "$ids" --parsable2 --units=M \
        --format=JobID,JobName,State,ElapsedRaw,Planned,TotalCPU,MaxRSS,NodeList \
        | tee "$RESULTS/50_costs/pilot_sacct.tsv"
    "$TRAIN_ENV/bin/python" - "$RESULTS/50_costs/pilot_sacct.tsv" <<'PY'
import sys, csv, math
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="|")
        if r["JobID"].count("_") == 1 and "." not in r["JobID"]]
el = sorted(int(r["ElapsedRaw"]) for r in rows if r["ElapsedRaw"].isdigit())
if not el:
    sys.exit("no completed array elements yet")
worst = el[-1]
# --quick is 300 epochs and 8 candidates; the full run is 3000 and 32.
# Training scales with epochs, generation with candidates; taking the whole
# element as if it were all training is deliberately conservative.
est = worst * 10
pad = int(est * 1.5)
h, m = divmod(math.ceil(pad / 60), 60)
print(f"\n  slowest quick element : {worst/3600:.2f} h")
print(f"  x10 (epochs)          : {est/3600:.2f} h")
print(f"  +50% headroom         : {pad/3600:.2f} h")
print(f"\n  put this in env.sh:  PHASE4_TIME=\"{h:02d}:{m:02d}:00\"")
print("  (gpu_7day allows up to 7-00:00:00; over-asking costs queue priority,")
print("   under-asking costs a restart from zero -- the runner resumes at")
print("   stage granularity, so a training killed at 90% restarts at 0%.)")
PY
    ;;

*) echo "usage: $0 {loss|quick|measure}" >&2; exit 2 ;;
esac
