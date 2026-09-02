#!/usr/bin/env bash
# Phase 4 -- the real run, on gpu_7day with A30 pinned.
#
#   bash 30_full.sh all        # everything, chained with --dependency=afterok
#   bash 30_full.sh train      # data-jarvis, angle-ablation, linegraph, confound
#   bash 30_full.sh symprec    # the tolerance sweep (needs the A0 checkpoint)
#   bash 30_full.sh choose     # read the sweep back, pick the tolerance
#   bash 30_full.sh pipeline   # pipeline-ablation, per arm, per root
#
# gpu_7day rather than gpu_2day because the runner resumes at *stage*
# granularity: a training killed at 90% restarts from zero, not from a
# checkpoint, so a walltime kill costs the whole element plus a fresh queue
# wait.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

STEP="${1:-all}"
# DRY=1 prints every sbatch that would be issued and submits nothing.  This is
# the last checkpoint before the run costs anything.
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
    if squeue -j "$jid" -h -o %i >/dev/null 2>&1 && [ -n "$(squeue -j "$jid" -h -o %i 2>/dev/null)" ]; then
        echo "--dependency=afterok:$jid"
    else
        echo "  (job $jid already finished; no dependency needed)" >&2
    fi
}

# ---------------------------------------------------------------------------
do_train() {
    echo "=== data-jarvis (CPU, minutes; runs in the scoring env, see env.sh)"
    run_task_data data-jarvis

    echo "=== angle-ablation: 6 arms x $(tr -cd , <<<"$SEEDS" | wc -c | awk '{print $1+1}') seeds"
    csp_submit "$PART_FULL" "$PHASE4_TIME" angle-ablation \
        --seeds "$SEEDS" --relax-workers "$RELAX_WORKERS" --symprec "$SYMPREC" \
        $DRYFLAG | record_jobs angle-ablation

    # Arm A of ablation-linegraph is byte-identical to A0 and is skipped by the
    # stage markers; only the three jarvis_nolg runs are new work.  It is the
    # calibration for reading A0->A3: deleting the line graph moves denoising
    # loss a lot (2.351 vs 2.011) and match rate not at all.
    echo "=== ablation-linegraph (+3 new: jarvis_nolg)"
    csp_submit "$PART_FULL" "$PHASE4_TIME" ablation-linegraph \
        --seeds "$SEEDS" --relax-workers "$RELAX_WORKERS" --symprec "$SYMPREC" \
        $DRYFLAG | record_jobs ablation-linegraph

    [ "$RUN_CONFOUND_ARM" = "1" ] && do_confound
}

# ---------------------------------------------------------------------------
# The confound arm.  run_task.py has no --gate-pair-messages flag and tasks.py
# never passes one, so this arm cannot go through the runner; it is submitted
# directly against the same scripts, with train_stage()'s argv reproduced
# exactly and one switch added.  Everything else -- hidden size, depth, knn,
# schedule, batch size, lr, augment, guidance, candidates, symprec -- is copied
# from tasks.py, because an arm that differs in anything else is not a control.
do_confound() {
    echo "=== confound arm: A3 with the pair channel ungated"
    local nseeds; nseeds=$(tr ',' '\n' <<<"$SEEDS" | wc -l)
    local sb="$RESULTS/00_provenance/confound.sbatch"
    cat > "$sb" <<CONF
#!/usr/bin/env bash
#SBATCH --job-name=csp-confound
#SBATCH --output=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.out
#SBATCH --error=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-$((nseeds-1))%$CSP_MAX_CONCURRENT
#SBATCH --time=$PHASE4_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
#SBATCH --account=$CSP_ACCOUNT
#SBATCH --partition=$PART_FULL
cd "$ALIGNN_REPO"
source task_runners/common.sh

SEEDS=(\$(tr ',' ' ' <<<"$SEEDS"))
S=\${SEEDS[\${SLURM_ARRAY_TASK_ID:-0}]}
DATA="$CSP_RUNS/data/jarvis"
RUN="$CSP_RUNS/train/$CONFOUND_CONFIG/seed\$S"
mkdir -p "\$RUN/bench/nosym" "\$RUN/bench/sym" "\$RUN/.stages"

t0=\$SECONDS
python -u -m alignn.inverse.train_csp \\
    --data-dir "\$DATA" --output "\$RUN" --epochs 3000 --seed "\$S" \\
    --ablation A3 --gate-pair-messages 0 \\
    --alignn-layers 3 --gcn-layers 3 --hidden-features 256 --knn 12 \\
    --num-steps 1000 --batch-size 64 --lr 1e-3 --augment 0 \\
    --device cuda --log-every 25
printf '{"argv":["confound-train"],"elapsed_s":%d,"host":"%s"}\n' \\
    \$((SECONDS-t0)) "\$(hostname)" > "\$RUN/.stages/train.json"

t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \\
    --checkpoint "\$RUN/best_model.pt" --data-dir "\$DATA" --split test \\
    --output-csv "\$RUN/bench/nosym/pred.csv" \\
    --num-candidates 32 --guidance 2.0 --relax cell --rank energy \\
    --relax-steps 200 --prescreen-keep 4 --relax-workers $RELAX_WORKERS \\
    --seed "\$S" --device cuda \\
    --save-candidates "\$RUN/bench/nosym/candidates.json"
printf '{"argv":["confound-generate"],"elapsed_s":%d,"host":"%s"}\n' \\
    \$((SECONDS-t0)) "\$(hostname)" > "\$RUN/.stages/generate.json"

python -u scripts/atombench/symmetrize_predictions.py \\
    --csv "\$RUN/bench/nosym/pred.csv" --out "\$RUN/bench/sym/pred.csv" \\
    --symprec $SYMPREC
bash scripts/atombench/score.sh "\$RUN/bench/nosym/pred.csv"
bash scripts/atombench/score.sh "\$RUN/bench/sym/pred.csv"
CONF
    if [ "$DRY" = "1" ]; then
        echo "would submit: sbatch --export=ALL $sb"
        echo "  (script written; array 0-$((nseeds-1))%$CSP_MAX_CONCURRENT," \
             "$PART_FULL, $CSP_GPU_GRES, $PHASE4_TIME)"
        return 0
    fi
    local jid; jid=$(sbatch --parsable --export=ALL "$sb")
    echo "submitted array job $jid" | record_jobs confound
}

# ---------------------------------------------------------------------------
do_symprec() {
    local dep="${1:-}"
    echo "=== symprec-sweep (validation split only)"
    CSP_SBATCH_EXTRA="$(dep_flag "$dep")" \
    csp_submit "$PART_FULL" "12:00:00" symprec-sweep \
        --relax-workers "$RELAX_WORKERS" $DRYFLAG | record_jobs symprec-sweep
}

do_choose() {
    echo "=== choosing the symmetrisation tolerance on validation"
    "$TRAIN_ENV/bin/python" - "$CSP_RUNS/symprec" <<'PY' \
        | tee "$RESULTS/50_costs/symprec_choice.txt"
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
    rows.append((tag,
                 sum(kld)/len(kld) if kld else None,
                 sum(ang)/len(ang) if ang else None,
                 rm.get("match_rate")))
print(f"{'symprec':<18}{'KLD':>10}{'angle MAE':>12}{'match':>10}")
for tag, kld, ang, mr in rows:
    f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"
    print(f"{tag:<18}{f(kld):>10}{f(ang,2):>12}{f(mr):>10}")
# Symmetrisation does not move match rate ("not at all", per the README) and
# does move KLD and the angle MAE a great deal (0.030 -> 0.018, 15.9 -> 8.4),
# so the choice is made on what it actually affects.  Chosen on validation,
# once, and applied to every arm -- per-arm tuning would be selecting a
# hyperparameter on the metric under test.
scored = [r for r in rows if r[1] is not None and r[0] != "none"]
if scored:
    best = min(scored, key=lambda r: (r[1], r[2] if r[2] is not None else 9e9))
    val = best[0].replace("symprec", "").replace("p", ".")
    print(f"\nchosen: {val}   (argmin KLD, tie-break lattice-angle MAE)")
    print(f"put SYMPREC=\"{val}\" in env.sh before running the pipeline step")
PY
}

# ---------------------------------------------------------------------------
# pipeline-ablation builds its run directories as ctx.out("pipeline", variant)
# -- the checkpoint is NOT in the path -- so two arms under one runs-root land
# in the same four directories and the second silently overwrites the first.
# Each arm therefore gets its own root, with data symlinked in because ctx.data
# is <root>/data while --checkpoint is absolute.
do_pipeline() {
    local dep="${1:-}"
    for arm in $PIPELINE_ARMS; do
        local root="$CSP_RUNS/pipeline_arms/$arm"
        mkdir -p "$root"
        ln -sfn "$CSP_RUNS/data" "$root/data"
        echo "=== pipeline-ablation for $arm  -> $root"
        CSP_SBATCH_EXTRA="$(dep_flag "$dep")" \
        csp_submit "$PART_FULL" "12:00:00" pipeline-ablation \
            --runs-root "$root" \
            --checkpoint "$CSP_RUNS/train/jarvis_$arm/seed$(cut -d, -f1 <<<"$SEEDS")/best_model.pt" \
            --symprec "$SYMPREC" --relax-workers "$RELAX_WORKERS" $DRYFLAG \
            | record_jobs "pipeline-$arm"
    done
}

# ---------------------------------------------------------------------------
case "$STEP" in
    train)    do_train ;;
    symprec)  do_symprec "$(last_array_job angle-ablation)" ;;
    choose)   do_choose ;;
    pipeline) do_pipeline "" ;;
    all)
        do_train
        do_symprec "$(last_array_job angle-ablation)"
        echo
        echo "pipeline-ablation is NOT chained: it needs the symprec chosen in"
        echo "between.  When the sweep finishes:"
        echo "    bash 30_full.sh choose      # then set SYMPREC in env.sh"
        echo "    bash 30_full.sh pipeline"
        ;;
    *) echo "usage: $0 {all|train|symprec|choose|pipeline}" >&2; exit 2 ;;
esac

echo
echo "job ids -> $JOBLOG   (costs.py --harvest reads this; run it soon after"
echo "each array finishes, the SLURM accounting database ages out)"
