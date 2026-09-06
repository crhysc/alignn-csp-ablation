#!/usr/bin/env bash
# The run.  Four cells, one dataset.
#
#   DATASET=jarvis bash 30_full.sh all     # 4 cells x 1 seed = 4 SLURM jobs
#   DATASET=alex   bash 30_full.sh all     # 4 cells x 1 seed = 4 SLURM jobs
#
# Each ablation is its own SLURM job, not one array element among several --
# its own job id, queue position and sacct record.  See submit_matrix_jobs()
# below.  Running both datasets is 8 jobs in total.
#
# Both datasets may be queued at the same time.  Every unit passes its own
# --runs-root, so it does not matter which dataset task_runners/cluster.env
# happens to name when the job starts (see submit_matrix_jobs()).  To have
# ONE aggregate wait on both datasets rather than one per dataset:
#
#   DATASET=alex AGG_JOIN_DATASET=jarvis bash 30_full.sh all
#
# which cancels the still-pending jarvis-only aggregate, and queues a single
# aggregate after every jarvis and alex ablation job that is still tracked.
#
# Data preparation is NOT one of those jobs.  It is pure Python/numpy
# (pymatgen, jarvis-tools), no GPU anywhere in it, so it runs synchronously
# right here on the login node (see do_data(), env.sh's DATA_ENV) and never
# touches SLURM -- the scheduler is reserved for what actually needs a GPU:
# training and generation.
#
# Parameter-matched, not compute-matched: the line-graph cells cost more
# wall-clock per step than the no-line-graph ones at this matched parameter
# count (measured 1.50x on a GB10). Rather than a second, compute-matched
# 2x2 to correct for that, analyze.py sums each cell's own measured
# wall-clock into GPU-hours, so the asymmetry is a reported number rather
# than a doubled run count.
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
# Every individually-submitted ("job"-tagged) SLURM id recorded for a task,
# one per line.  Used for the matrix task, which submits one job per ablation
# rather than one array -- see submit_matrix_jobs() below.
matrix_job_ids() { awk -v t="$1" '$1==t && $2=="job"{print $3}' "${2:-$JOBLOG}"; }
# The most recent aggregate recorded for a task in a job log, if any.
matrix_agg_id()  { awk -v t="$1" '$1==t && $2=="agg"{id=$3} END{if(id)print id}' "${2:-$JOBLOG}"; }
# The job ids among the arguments that SLURM still tracks (pending or running).
still_queued() {
    local id
    for id in "$@"; do
        [ -n "$(squeue -j "$id" -h -o %i 2>/dev/null)" ] && echo "$id"
    done
    return 0
}

# A dependency is only valid while SLURM still tracks the job.  Once a job has
# completed and been purged from the active list, --dependency=afterok on it
# is rejected outright ("Job dependency problem") rather than treated as
# already satisfied -- so only name jobs that are still queued.
# --dependency=afterok:id1:id2:... over every one of a task's individually
# submitted jobs, so a downstream step (the symprec sweep) waits on every
# ablation that is still running and on none that has already finished.
matrix_dep_flag() {
    local ids=() still=() id
    while read -r id; do [ -n "$id" ] && ids+=("$id"); done < <(matrix_job_ids "$1")
    [ "${#ids[@]}" -gt 0 ] || return 0
    for id in "${ids[@]}"; do
        if [ -n "$(squeue -j "$id" -h -o %i 2>/dev/null)" ]; then
            still+=("$id")
        else
            echo "  (job $id already finished; dropped from the dependency)" >&2
        fi
    done
    [ "${#still[@]}" -gt 0 ] || return 0
    local joined; joined=$(IFS=:; echo "${still[*]}")
    echo "--dependency=afterok:$joined"
}

# ---------------------------------------------------------------------------
do_data() {
    if [ -f "$CSP_RUNS/data/$SPLIT/train.json" ]; then
        echo "=== $DATA_TASK: already prepared, skipping"
        echo "    $(tr -d '\n ' < "$CSP_RUNS/data/$SPLIT/split_meta.json" 2>/dev/null)"
        return 0
    fi
    # Pure Python/numpy (pymatgen, jarvis-tools) -- no GPU, no CUDA, nothing
    # SLURM-worthy about it.  Runs right here, on the login node, in DATA_ENV
    # (see env.sh) -- never submitted, never queued.  Blocks until it is
    # done, which is deliberate: by the time submit_matrix_jobs() runs below,
    # the split unconditionally exists, so there is no data-job dependency
    # for the ablation jobs to wait on any more.
    echo "=== $DATA_TASK (login node, $DATA_ENV, no SLURM)"
    if [ "$DRY" = "1" ]; then
        echo "  [dry-run] run_task_local $DATA_TASK"
        return 0
    fi
    run_task_local "$DATA_TASK"
}

# One SLURM job per ablation, not one job array of N elements.  Each cell
# gets its own job id, its own queue position and its own sacct record;
# nothing about what runs changes -- same run_task.py invocation, same
# environment, same walltime, just one `sbatch` call per unit instead of one
# `--array=0-N` call for all of them.  The shipped $MATRIX_TASK.sbatch
# template is array-shaped (reads SLURM_ARRAY_TASK_ID), so this writes its
# own single-job template per unit rather than reusing it -- the same
# heredoc-sbatch pattern 20_pilot.sh and 40_mechanism.sh already use for
# their own one-off jobs, not a new idiom.
submit_matrix_jobs() {
    local dep="${1:-}"   # bare "afterok:<jobid>", or empty
    # Sized by the task itself, not by a hard-coded cell list, so the same
    # path drives any matrix the task table defines.  tasks.py is stdlib plus
    # alignn.inverse.ablations, so the system python3 can import it here.
    local n_units
    n_units=$(cd "$ALIGNN_REPO" && python3 task_runners/run_task.py "$MATRIX_TASK" \
                  --count --seeds "$SEEDS" --runs-root "$CSP_RUNS")
    # UNITS="2,3" submits only those unit indices.  For a matrix that shares
    # cells with one already run (the angular-state 2x2 reuses the two
    # non-angular cells' config names), the shared units would only start,
    # find every stage marker present, and exit -- a GPU slot for nothing.
    local -a units=()
    if [ -n "${UNITS:-}" ]; then
        IFS=',' read -r -a units <<<"$UNITS"
    else
        for ((i = 0; i < n_units; i++)); do units+=("$i"); done
    fi
    echo "=== $MATRIX_TASK: ${#units[@]} of $n_units unit(s) as separate SLURM job(s), one per ablation (not an array)"
    (cd "$ALIGNN_REPO" && python3 task_runners/run_task.py "$MATRIX_TASK" --list \
         --seeds "$SEEDS" --runs-root "$CSP_RUNS" | grep -E '^\s+[0-9]+ ' | sed 's/^/   /')
    echo "    checkpoint selection: --select-on $SELECT_ON (declared by the task)"

    # cluster.env is the one file shared with every other dataset and with the
    # sibling harness, and common.sh sources it INSIDE each job, at start time
    # -- so a job that is still pending reads whatever the file says when it
    # finally runs, not what it said at submission.  The only line in it that
    # differs between datasets is CSP_RUNS, so nothing dataset-specific is
    # allowed to travel through it: each unit below passes its own
    # --runs-root instead, and the file is written only when it is missing or
    # was stamped by a different harness.  Rewriting it here for this dataset
    # would repoint every other dataset's still-pending job at this run root.
    local ce="$ALIGNN_REPO/task_runners/cluster.env"
    if ! grep -qs "^# HARNESS=$HARNESS_ID\$" "$ce"; then
        CSP_PARTITION="$PART_FULL" \
        CSP_SBATCH_EXTRA="--time=$PHASE_TIME --cpus-per-task=$CPUS_PER_TASK --mem=$MEM_PER_TASK ${CSP_SBATCH_EXTRA:-}" \
            bash "$HARNESS/env.sh" --write >/dev/null
    else
        echo "    cluster.env left as is ($(grep -m1 '^# DATASET=' "$ce")); run root pinned per unit"
    fi
    mkdir -p "$RESULTS/00_provenance" "$GPU_TRACE_DIR"

    local jids=() i sb jid
    for i in "${units[@]}"; do
        sb="$RESULTS/00_provenance/${MATRIX_TASK}_unit${i}.sbatch"
        cat > "$sb" <<UNITSB
#!/usr/bin/env bash
#SBATCH --job-name=csp-${MATRIX_TASK}-u${i}
#SBATCH --output=$ALIGNN_REPO/task_runners/logs/%x-%j.out
#SBATCH --error=$ALIGNN_REPO/task_runners/logs/%x-%j.err
#SBATCH --time=$PHASE_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_FULL
cd "$ALIGNN_REPO"
source task_runners/common.sh
# The run root is this dataset's, whatever cluster.env says by the time this
# job starts (common.sh has just set ALIGNN_RUNS from it).
export ALIGNN_RUNS="$CSP_RUNS"

python task_runners/run_task.py $MATRIX_TASK \\
    --runs-root "$CSP_RUNS" \\
    --seeds "$SEEDS" --relax-workers "$RELAX_WORKERS" --symprec "$SYMPREC" \\
    --unit $i
UNITSB
        if [ "$DRY" = "1" ]; then
            echo "  [dry-run] unit $i -> sbatch ${dep:+--dependency=$dep }$sb"
            continue
        fi
        if [ -n "$dep" ]; then
            jid=$(sbatch --parsable --export=ALL --dependency="$dep" "$sb")
        else
            jid=$(sbatch --parsable --export=ALL "$sb")
        fi
        echo "  unit $i -> job $jid"
        printf '%s\tjob\t%s\n' "$MATRIX_TASK" "$jid" >> "$JOBLOG"
        jids+=("$jid")
    done

    submit_aggregate "${jids[@]}"
}

# One aggregate after every ablation job -- the same summary submit.sh would
# queue after a single array job, depending on all N individually instead of
# on one array job id.
#
# The repo's aggregate.sbatch is not used: it names no partition, so it lands
# on the default (main, x86_64) where the aarch64 environment cannot execute,
# and it takes its run root from cluster.env, which is exactly the coupling
# the units above avoid.  This writes its own, on the GPU partition with no
# --gres (the summary needs the environment, not a GPU), with the run root of
# each dataset it summarises spelled out.
#
# AGG_JOIN_DATASET=<other dataset> makes this ONE aggregate for both: it waits
# on that dataset's matrix jobs too (those still tracked by SLURM), summarises
# both run roots, and cancels that dataset's own still-pending aggregate, which
# it supersedes.  The other dataset's job log gets the new id as well.
submit_aggregate() {
    local jids=("$@")
    [ "$DRY" = "1" ] && [ "${#jids[@]}" -eq 0 ] && jids=(DRYRUN)
    [ "${#jids[@]}" -gt 0 ] || return 0

    local -a tasks=("$MATRIX_TASK") roots=("$CSP_RUNS") deps=("${jids[@]}")
    local other="${AGG_JOIN_DATASET:-}" other_log="" other_task="" other_agg=""
    if [ -n "$other" ]; then
        [ "$other" != "$DATASET" ] || { echo "AGG_JOIN_DATASET is this dataset" >&2; return 1; }
        local other_root
        # The other dataset's names, from env.sh, without disturbing ours.
        read -r other_task other_root < <(
            env -i HOME="$HOME" PATH="$PATH" DATASET="$other" RUN_ID="$RUN_ID" \
                bash -c 'source "$1" && printf "%s %s\n" "$MATRIX_TASK" "$CSP_RUNS"' _ "$HARNESS/env.sh")
        other_log="$HARNESS/results/$other/$RUN_ID/00_provenance/slurm_jobids.txt"
        [ -f "$other_log" ] || { echo "no job log for DATASET=$other at $other_log" >&2; return 1; }
        local -a other_jobs
        mapfile -t other_jobs < <(still_queued $(matrix_job_ids "$other_task" "$other_log"))
        [ "${#other_jobs[@]}" -gt 0 ] || {
            echo "AGG_JOIN_DATASET=$other: none of its $other_task jobs is still queued;" >&2
            echo "  a dependency on a finished job is rejected by SLURM, so nothing to join." >&2
            return 1; }
        echo "=== joining $other: $other_task jobs still queued: ${other_jobs[*]}"
        tasks+=("$other_task"); roots+=("$other_root"); deps+=("${other_jobs[@]}")
        other_agg=$(matrix_agg_id "$other_task" "$other_log")
        [ -n "$other_agg" ] && [ -z "$(still_queued "$other_agg")" ] && other_agg=""
    fi

    local sb="$RESULTS/00_provenance/aggregate_${MATRIX_TASK}${other:+_with_$other}.sbatch"
    {
        cat <<AGGSB
#!/usr/bin/env bash
#SBATCH --job-name=csp-aggregate-${MATRIX_TASK}${other:++$other}
#SBATCH --output=$RESULTS/00_provenance/aggregate-%j.out
#SBATCH --error=$RESULTS/00_provenance/aggregate-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
$(sbatch_account_line)
#SBATCH --partition=$PART_FULL
# No --gres: the summary needs the aarch64 environment, not a GPU.
cd "$ALIGNN_REPO"
source task_runners/common.sh
AGGSB
        local k
        for k in "${!tasks[@]}"; do
            cat <<AGGSB

echo; echo "=== ${tasks[$k]}  (${roots[$k]})"
python task_runners/run_task.py "${tasks[$k]}" --aggregate --latex --runs-root "${roots[$k]}"
AGGSB
        done
        echo; echo "echo; echo \"=== ${MATRIX_TASK}${other:+ + $other} complete: \$(date -Is)\""
    } > "$sb"

    local aggdep; aggdep=$(IFS=:; echo "afterok:${deps[*]}")
    if [ "$DRY" = "1" ]; then
        echo "  [dry-run] aggregate -> sbatch --dependency=$aggdep $sb"
        [ -n "$other_agg" ] && echo "  [dry-run] scancel $other_agg  (the $other-only aggregate it supersedes)"
        return 0
    fi
    local agg
    agg=$(sbatch --parsable --export=ALL --dependency="$aggdep" "$sb")
    echo "  aggregation -> job $agg (after all ${#deps[@]} matrix jobs: ${deps[*]})"
    printf '%s\tagg\t%s\n' "$MATRIX_TASK" "$agg" >> "$JOBLOG"
    if [ -n "$other" ]; then
        if [ -n "$other_agg" ]; then
            scancel "$other_agg" && echo "  cancelled $other_agg (the $other-only aggregate; superseded by $agg)"
            printf '# %s\tagg\t%s\tcancelled %s, superseded by combined aggregate %s\n' \
                "$other_task" "$other_agg" "$(date -Is)" "$agg" >> "$other_log"
        fi
        printf '%s\tagg\t%s\n' "$other_task" "$agg" >> "$other_log"
        echo "  recorded $agg in $other_log as well"
    fi
}

do_train() {
    # No data-job dependency to build any more: do_data() (called by every
    # caller of do_train() below) runs synchronously and blocks until the
    # split exists, so by the time this runs it unconditionally already does
    # -- there is nothing left for the ablation jobs to wait on.
    submit_matrix_jobs ""
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
    local task="symprec-sweep"
    [ "$DATASET" = "alex" ] && task="symprec-sweep-alex"
    echo "=== $task (validation split only, ${SYMPREC_CANDIDATES:-8} candidates, first ${SYMPREC_LIMIT:-150} targets)"
    # Depends on every individually-submitted ablation job still queued, not
    # on one array job -- see submit_matrix_jobs().
    CSP_SBATCH_EXTRA="$(matrix_dep_flag "$MATRIX_TASK")" \
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
    symprec)  do_symprec ;;
    choose)   do_choose ;;
    all)
        do_data
        do_train
        echo
        echo "The symmetrisation tolerance is NOT chained: it needs a trained"
        echo "checkpoint first, and choosing it changes how the test split is"
        echo "scored.  When the four ablation jobs finish:"
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
