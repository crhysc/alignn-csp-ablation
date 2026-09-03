#!/usr/bin/env bash
# The bond-angle Wasserstein distance -- one of the three things this
# experiment reports, and the only one nothing else in the pipeline computes.
#
#   bash 40_mechanism.sh          # submit as an array over finished runs
#   bash 40_mechanism.sh --list   # show what would run
#
# scripts/atombench/angle_eval.py pools every bond angle in the generated
# structures and every bond angle in the held-out real ones, histograms both on
# the same 180 bins over [0, 180] degrees, and reports the 1-D earth-mover
# distance between them in degrees -- exact for a 1-D histogram, being the
# integral of the absolute CDF difference.  It reports KL and Jensen-Shannon on
# the same binning beside it.  This is FoldingDiff's own diagnostic and the
# most direct check that an angular channel does what it claims.
#
# It also computes the relaxation displacement (how far a sample must move to
# reach the nearest ALIGNN-FF minimum), which needs the force field and is why
# this is a GPU job.
#
# angle_eval.py is NOT a stage in any task -- `grep -rn angle_eval
# task_runners/` finds nothing -- which is why this script exists.
#
# Run 50_unrelaxed.sh as well, and prefer ITS numbers for the angle
# comparison: see the note at the bottom of this file.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

LIST="$RESULTS/00_provenance/mechanism_targets.txt"
mkdir -p "$RESULTS/00_provenance"
: > "$LIST"
for csv in "$CSP_RUNS"/train/*/seed*/bench/sym/pred.csv; do
    [ -e "$csv" ] && echo "$csv" >> "$LIST"
done
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "no scored runs under $CSP_RUNS/train yet" >&2; exit 1; }

if [ "${1:-}" = "--list" ]; then
    echo "$N run(s):"; nl -ba "$LIST"; exit 0
fi

SB="$RESULTS/00_provenance/mechanism.sbatch"
cat > "$SB" <<MECH
#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-mechanism
#SBATCH --output=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.out
#SBATCH --error=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-$((N-1))%$CSP_MAX_CONCURRENT
#SBATCH --time=$MECH_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_FULL
cd "$ALIGNN_REPO"
source task_runners/common.sh

CSV=\$(sed -n "\$((\${SLURM_ARRAY_TASK_ID:-0}+1))p" "$LIST")
OUT="\$(dirname "\$CSV")/angle_eval.json"
echo "mechanism metrics: \$CSV"

t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/angle_eval.py "\$CSV" \\
    --relax --relax-steps 200 --output "\$OUT"
RUNDIR="\$(dirname "\$(dirname "\$(dirname "\$CSV")")")"
mkdir -p "\$RUNDIR/.stages"
printf '{"argv":["angle_eval"],"elapsed_s":%d,"host":"%s"}\n' \\
    \$((SECONDS-t0)) "\$(hostname)" > "\$RUNDIR/.stages/angle_eval.json"
MECH

if [ "${DRY:-0}" = "1" ]; then
    echo "would submit: sbatch --export=ALL $SB"
    echo "  array 0-$((N-1))%$CSP_MAX_CONCURRENT on $PART_FULL, $CSP_GPU_GRES, $MECH_TIME"
    exit 0
fi
JID=$(sbatch --parsable --export=ALL "$SB")
echo "submitted array job $JID  ($N elements)"
printf 'mechanism\tarray\t%s\n' "$JID" >> "$RESULTS/00_provenance/slurm_jobids.txt"
echo "watch: squeue -j $JID"
echo
echo "NOTE: these angles are measured on the RELAXED predictions, because that"
echo "is what the headline pipeline produces.  Relaxation snaps every sample"
echo "into ALIGNN-FF's nearest local minimum -- exactly where a local-geometry"
echo "advantage would be erased, and local geometry is what the angular channel"
echo "claims to fix.  Run 50_unrelaxed.sh and report the raw numbers as well;"
echo "analyze.py --variant rawsym reads them."
