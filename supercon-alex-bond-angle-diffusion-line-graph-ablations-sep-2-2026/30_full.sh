#!/usr/bin/env bash
# The run.  Six cells, one dataset, chained on the data task.
#
#   DATASET=jarvis bash 30_full.sh all     # 6 cells x 3 seeds = 18 elements
#   DATASET=alex   bash 30_full.sh all     # 6 cells x 1 seed  =  6 elements
#
# Six rather than four because depth cannot level parameters and compute at
# once: the suite carries a parameter-matched 2x2 and a compute-matched one,
# and they share their two line-graph cells.
#
#   bash 30_full.sh data       # just the split
#   bash 30_full.sh train      # the matrix, chained on the split
#   bash 30_full.sh symprec    # tolerance sweep, on VALIDATION
#   bash 30_full.sh choose     # read the sweep back and pick the tolerance
#
# DRY=1 prints every sbatch that would be issued and submits nothing.  It is
# the last checkpoint before this costs anything.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

STEP="${1:-all}"
DRY="${DRY:-0}"
[ "$DRY" = "1" ] && DRYFLAG="--dry-run" || DRYFLAG=""
JOBLOG="$RESULTS/00_provenance/slurm_jobids.txt"
mkdir -p "$RESULTS"/{00_provenance,50_costs}

record_jobs() {
    tee /dev/stderr | awk -v t="$1" '/submitted array job/{print t"\tarray\t"$4}
                                     /submitted aggregation job/{print t"\tagg\t"$4}' \
        >> "$JOBLOG"
}
last_array_job() { awk -v t="$1" '$1==t && $2=="array"{j=$3} END{print j}' "$JOBLOG"; }

# A dependency is only valid while SLURM still tracks the job.  Once an array
# has completed and been purged from the active list, --dependency=afterok on
# it is rejected outright ("Job dependency problem") rather than treated as
# already satisfied -- so only attach one to a job that is still queued.
dep_flag() {
    local jid="$1"
    [ -n "$jid" ] || return 0
    if [ -n "$(squeue -j "$jid" -h -o %i 2>/dev/null)" ]; then
        echo "--dependency=afterok:$jid"
    else
        echo "  (job $jid already finished; no dependency needed)" >&2
    fi
}

# ---------------------------------------------------------------------------
do_data() {
    if [ -f "$CSP_RUNS/data/$SPLIT/train.json" ]; then
        echo "=== $DATA_TASK: already prepared, skipping"
        echo "    $(tr -d '\n ' < "$CSP_RUNS/data/$SPLIT/split_meta.json" 2>/dev/null)"
        return 0
    fi
    # CPU-only, but it still needs the aarch64 SCORING environment (pymatgen),
    # which cannot run on the x86_64 login node -- so it is submitted, not run
    # here.  Its sbatch declares no --gres and so holds no GPU.
    echo "=== $DATA_TASK (CPU-only; submitted to $PART_FULL, no GPU held)"
    csp_submit "$PART_FULL" "04:00:00" "$DATA_TASK" $DRYFLAG \
        | record_jobs "$DATA_TASK"
}

do_train() {
    local datadep=""
    if [ "$DRY" != "1" ] && [ ! -f "$CSP_RUNS/data/$SPLIT/train.json" ]; then
        # Making data prep an asynchronous job silently drops the ordering a
        # synchronous call used to get for free.  Every cell reads the split,
        # so the dependency has to be explicit or the array starts against a
        # missing file.
        local j; j="$(last_array_job "$DATA_TASK")"
        [ -n "$j" ] && datadep="--dependency=afterok:$j"
        echo "    (chained on the data job: ${datadep:-none})"
    fi
    local nseeds; nseeds=$(tr ',' '\n' <<<"$SEEDS" | wc -l)
    local ncells; ncells=$(python3 -c "
import sys; sys.path.insert(0, '$ALIGNN_REPO')
from alignn.inverse.ablations import all_cells; print(len(all_cells()))")
    echo "=== $MATRIX_TASK: $ncells cells x $nseeds seed(s) = $((ncells*nseeds)) elements"
    echo "    parameter-matched: neither / line graph / angle diffusion / both"
    echo "    compute-matched:   neither (compute) / angle diffusion (compute)"
    echo "    (both line-graph cells are shared, so this is 6 cells, not 8)"
    echo "    checkpoint selection: --select-on $SELECT_ON (declared by the task)"
    CSP_SBATCH_EXTRA="$datadep" \
    csp_submit "$PART_FULL" "$PHASE_TIME" "$MATRIX_TASK" \
        --seeds "$SEEDS" --relax-workers "$RELAX_WORKERS" --symprec "$SYMPREC" \
        $DRYFLAG | record_jobs "$MATRIX_TASK"
}

# ---------------------------------------------------------------------------
# The symmetrisation tolerance.  Chosen on VALIDATION, once, and applied
# uniformly to every cell -- per-cell tuning would be selecting a
# hyperparameter on the metric under test.  It does not transfer between
# splits, so it is swept per dataset.
#
# Reduced candidates on purpose: the tolerance is a property of the predicted
# cell distribution, not of how many candidates were drawn, and a full 32
# candidates over the whole 825-target Alexandria validation split would cost
# as much as an entire arm for a preprocessing constant.
do_symprec() {
    local dep="${1:-}"
    local task="symprec-sweep"
    [ "$DATASET" = "alex" ] && task="symprec-sweep-alex"
    echo "=== $task (validation split only, ${SYMPREC_CANDIDATES:-8} candidates, first ${SYMPREC_LIMIT:-150} targets)"
    CSP_SBATCH_EXTRA="$(dep_flag "$dep")" \
    csp_submit "$PART_FULL" "12:00:00" "$task" \
        --checkpoint "$CSP_RUNS/train/${SPLIT}_A3/seed$(cut -d, -f1 <<<"$SEEDS")/best_model.pt" \
        --num-candidates "${SYMPREC_CANDIDATES:-8}" \
        --limit "${SYMPREC_LIMIT:-150}" \
        --relax-workers "$RELAX_WORKERS" $DRYFLAG | record_jobs "$task"
}

do_choose() {
    echo "=== choosing the symmetrisation tolerance on validation"
    train_py - "$CSP_RUNS/symprec" <<'CHOOSE' | tee "$RESULTS/50_costs/symprec_choice.txt"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for mj in sorted(root.rglob("metrics.json")):
    raw = json.loads(mj.read_text().replace("NaN", "null").replace("Infinity", "null"))
    kld = [v for v in raw.get("KLD", {}).values() if v is not None]
    mae = raw.get("MAE", {}).get("average_mae", {})
    ang = [mae[k] for k in ("alpha", "beta", "gamma") if mae.get(k) is not None]
    rm = raw.get("RMSE", {}).get("AtomGen", {})
    tag = mj.parent.name if mj.parent.name != "val" else "none"
    rows.append((tag, sum(kld)/len(kld) if kld else None,
                 sum(ang)/len(ang) if ang else None, rm.get("match_rate")))
print(f"{'symprec':<18}{'KLD':>10}{'angle MAE':>12}{'match':>10}")
for tag, kld, ang, mr in rows:
    f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"
    print(f"{tag:<18}{f(kld):>10}{f(ang,2):>12}{f(mr):>10}")
# Symmetrisation does not move match rate and does move KLD and the angle MAE
# a great deal (0.030 -> 0.018, 15.9 -> 8.4 on JARVIS), so the choice is made
# on what it actually affects.
scored = [r for r in rows if r[1] is not None and r[0] != "none"]
if scored:
    best = min(scored, key=lambda r: (r[1], r[2] if r[2] is not None else 9e9))
    val = best[0].replace("symprec", "").replace("p", ".")
    print(f"\nchosen: {val}   (argmin KLD, tie-break lattice-angle MAE)")
    print(f'put SYMPREC="{val}" in env.sh, then re-run 30_full.sh train')
    print("(the stage markers will re-run only symmetrize and score, not training)")
CHOOSE
}

# ---------------------------------------------------------------------------
case "$STEP" in
    data)     do_data ;;
    train)    do_data; do_train ;;
    symprec)  do_symprec "$(last_array_job "$MATRIX_TASK")" ;;
    choose)   do_choose ;;
    all)
        do_data
        do_train
        echo
        echo "The symmetrisation tolerance is NOT chained: it needs a trained"
        echo "checkpoint first, and choosing it changes how the test split is"
        echo "scored.  When the array finishes:"
        echo "    bash 30_full.sh symprec"
        echo "    bash 30_full.sh choose        # then set SYMPREC in env.sh"
        echo "    bash 30_full.sh train         # re-scores; training is skipped"
        echo
        echo "Then the mechanism metrics and the unrelaxed variant:"
        echo "    bash 40_mechanism.sh"
        echo "    bash 50_unrelaxed.sh"
        ;;
    *) echo "usage: $0 {all|data|train|symprec|choose}" >&2; exit 2 ;;
esac

echo
echo "job ids -> $JOBLOG"
echo "collect with: python collect.py && python analyze.py"
