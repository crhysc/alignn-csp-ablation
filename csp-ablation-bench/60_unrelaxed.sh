#!/usr/bin/env bash
# Score the RAW generated structures -- no force-field relaxation.
#
#   bash 60_unrelaxed.sh
#
# The main benchmark scores after `--relax cell`, which is what reproduces the
# published pipeline but measures the diffusion model *and* ALIGNN-FF together.
# For an ablation that is the wrong denominator twice over:
#
#   * relaxation snaps every sample into the force field's nearest local
#     minimum, which is exactly where a local-geometry advantage would be
#     erased -- and local geometry is what the angular channel claims to fix.
#   * the relaxation-displacement metric is meaningless on an already-relaxed
#     structure: measured 0.0007 A, because there is nowhere left to move.
#
# So this regenerates one sample per target with no ranking and no relaxation,
# scores it, and runs the mechanism metrics on it.  Generation only -- the
# checkpoints are reused, nothing retrains.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

LIST="$RESULTS/00_provenance/unrelaxed_targets.txt"
: > "$LIST"
for ckpt in "$CSP_RUNS"/train/*/seed*/best_model.pt; do
    [ -e "$ckpt" ] && echo "$(dirname "$ckpt")" >> "$LIST"
done
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "no checkpoints found" >&2; exit 1; }
echo "$N run(s) to regenerate unrelaxed"

SB="$RESULTS/00_provenance/unrelaxed.sbatch"
cat > "$SB" <<UNRELAXED
#!/usr/bin/env bash
#SBATCH --job-name=csp-unrelaxed
#SBATCH --output=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.out
#SBATCH --error=$ALIGNN_REPO/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-$((N-1))%$CSP_MAX_CONCURRENT
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
#SBATCH --account=$CSP_ACCOUNT
#SBATCH --partition=$PART_FULL
cd "$ALIGNN_REPO"
source task_runners/common.sh

RUN=\$(sed -n "\$((\${SLURM_ARRAY_TASK_ID:-0}+1))p" "$LIST")
SEED=\$(basename "\$RUN" | tr -dc 0-9)
DATA="$CSP_RUNS/data/jarvis"
mkdir -p "\$RUN/bench/raw" "\$RUN/bench/rawsym"
echo "unrelaxed generation: \$RUN (seed \$SEED)"

# One sample per target, no energy ranking, no relaxation: the model's own
# output, untouched.  Same seed, same guidance, same denoising steps as the
# relaxed run, so the only difference is the post-processing.
t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \\
    --checkpoint "\$RUN/best_model.pt" --data-dir "\$DATA" --split test \\
    --output-csv "\$RUN/bench/raw/pred.csv" \\
    --num-candidates 1 --guidance 2.0 --relax none --rank none \\
    --seed "\$SEED" --device cuda
echo "generate_raw \$((SECONDS-t0))s"

python -u scripts/atombench/symmetrize_predictions.py \\
    --csv "\$RUN/bench/raw/pred.csv" --out "\$RUN/bench/rawsym/pred.csv" \\
    --symprec $SYMPREC
bash scripts/atombench/score.sh "\$RUN/bench/raw/pred.csv"
bash scripts/atombench/score.sh "\$RUN/bench/rawsym/pred.csv"

# Now the relaxation-displacement metric means something: these structures
# have not been relaxed, so the distance to the nearest ALIGNN-FF minimum is
# a real measurement of how much geometric repair the sample needs.
OMP_NUM_THREADS=1 python -u scripts/atombench/angle_eval.py \\
    "\$RUN/bench/rawsym/pred.csv" --relax --relax-steps 200 \\
    --output "\$RUN/bench/rawsym/angle_eval.json"
UNRELAXED

JID=$(sbatch --parsable --export=ALL "$SB")
echo "submitted array job $JID ($N elements)"
printf 'unrelaxed\tarray\t%s\n' "$JID" >> "$RESULTS/00_provenance/slurm_jobids.txt"
