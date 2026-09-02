#!/usr/bin/env bash
# The mechanism metrics -- the two numbers the branch's hypothesis actually
# turns on, which nothing else in this pipeline computes.
#
#   bash 40_mechanism.sh            # submit as an A30 array
#   bash 40_mechanism.sh --list     # show what would run
#
# scripts/atombench/angle_eval.py computes:
#   * the bond-angle distribution distance (KL / JS / 1-D Wasserstein in
#     degrees) between generated and held-out real structures.  This is
#     FoldingDiff's own diagnostic and the most direct check that the angular
#     channel does what it claims.
#   * the relaxation displacement -- how far a sample must move to reach the
#     nearest ALIGNN-FF minimum, plus volume change and energy drop.  If
#     explicit angular denoising produces locally coherent geometry, its
#     samples should need less geometric repair.  This is MatterGen's
#     evaluation.
#
# It is NOT a stage in any task: `grep -rn angle_eval task_runners/` finds
# nothing, and claims.py registers no angle-ablation claims, so the runner
# will never call it.  Hence this script.
#
# --relax runs ALIGNN-FF, so this is a GPU job, not a login-node one.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

# One line per finished run: the symmetrised CSV is what the manuscript's
# lattice columns are quoted on, so that is what the mechanism metrics use.
LIST="$RESULTS/00_provenance/mechanism_targets.txt"
mkdir -p "$RESULTS/00_provenance"
: > "$LIST"
for csv in "$CSP_RUNS"/train/*/seed*/bench/sym/pred.csv; do
    [ -e "$csv" ] || continue
    echo "$csv" >> "$LIST"
done
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "no scored runs under $CSP_RUNS/train yet" >&2; exit 1; }

if [ "${1:-}" = "--list" ]; then
    echo "$N run(s):"; nl -ba "$LIST"; exit 0
fi

SB="$RESULTS/00_provenance/mechanism.sbatch"
cat > "$SB" <<MECH
#!/usr/bin/env bash
#SBATCH --job-name=csp-mechanism
#SBATCH --output=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.out
#SBATCH --error=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-$((N-1))%$CSP_MAX_CONCURRENT
#SBATCH --time=$MECH_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
#SBATCH --account=$CSP_ACCOUNT
#SBATCH --partition=$PART_FULL
cd "$ALIGNN_REPO"
source task_runners/common.sh

CSV=\$(sed -n "\$((\${SLURM_ARRAY_TASK_ID:-0}+1))p" "$LIST")
OUT="\$(dirname "\$CSV")/angle_eval.json"
echo "mechanism metrics: \$CSV"

t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/angle_eval.py "\$CSV" \\
    --relax --relax-steps 200 --output "\$OUT"
printf '{"argv":["angle_eval"],"elapsed_s":%d,"host":"%s"}\n' \\
    \$((SECONDS-t0)) "\$(hostname)" \\
    > "\$(dirname "\$CSV")/../../.stages/angle_eval.json" 2>/dev/null || true
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
