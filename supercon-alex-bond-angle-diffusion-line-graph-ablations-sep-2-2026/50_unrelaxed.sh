#!/usr/bin/env bash
# Score the RAW generated structures -- one sample per target, no ranking, no
# force-field relaxation.
#
#   bash 50_unrelaxed.sh
#
# This is not an extra: for an ablation it is arguably the primary measurement,
# and it is cheap (one candidate instead of 32, no relaxation at all -- far
# less than the headline run it complements).  Generation only; the checkpoints
# are reused and nothing retrains.
#
# The headline pipeline samples 32 candidates, ranks them by ALIGNN-FF energy
# and relaxes the survivors.  That reproduces the published numbers, but it
# measures the diffusion model *and* ALIGNN-FF together, which is the wrong
# denominator for this experiment twice over:
#
#   * relaxation snaps every sample into the force field's nearest local
#     minimum, which is precisely where a local-geometry advantage would be
#     erased -- and local coordination geometry is what an angular channel
#     claims to improve.  If bond-angle diffusion helps and relaxation hides
#     it, the relaxed comparison will report nothing and the raw one will not.
#   * the bond-angle Wasserstein distance on relaxed structures is largely a
#     measurement of ALIGNN-FF's preferred angles, not of the generator's.
#     Whatever the generator produced, the force field has moved it.
#
# So: same checkpoints, same seed, same guidance, same denoising steps, and the
# post-processing removed.  analyze.py --variant rawsym reads what this writes.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

LIST="$RESULTS/00_provenance/unrelaxed_targets.txt"
mkdir -p "$RESULTS/00_provenance"
: > "$LIST"
for ckpt in "$CSP_RUNS"/train/*/seed*/best_model.pt; do
    [ -e "$ckpt" ] && dirname "$ckpt" >> "$LIST"
done
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "no checkpoints under $CSP_RUNS/train" >&2; exit 1; }

if [ "${1:-}" = "--list" ]; then
    echo "$N run(s):"; nl -ba "$LIST"; exit 0
fi
echo "$N run(s) to regenerate unrelaxed"

SB="$RESULTS/00_provenance/unrelaxed.sbatch"
cat > "$SB" <<UNRELAXED
#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-unrelaxed
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

RUN=\$(sed -n "\$((\${SLURM_ARRAY_TASK_ID:-0}+1))p" "$LIST")
SEED=\$(basename "\$RUN" | tr -dc 0-9)
DATA="$CSP_RUNS/data/$SPLIT"
mkdir -p "\$RUN/bench/raw" "\$RUN/bench/rawsym" "\$RUN/.stages"
echo "unrelaxed generation: \$RUN (seed \$SEED)"

t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \\
    --checkpoint "\$RUN/best_model.pt" --data-dir "\$DATA" --split test \\
    --output-csv "\$RUN/bench/raw/pred.csv" \\
    --num-candidates 1 --guidance 2.0 --relax none --rank none \\
    --seed "\$SEED" --device cuda
echo "generate_raw \$((SECONDS-t0))s"

# Symmetrised as well as raw: the lattice-angle and KLD metrics are measured
# after Niggli reduction, which is discontinuous, so a nearly-symmetric cell
# that is slightly off reduces to a different basis and contributes a large
# angular error.  Same tolerance as the headline run.
bash scripts/atombench/symmetrize.sh \\
    --csv "\$RUN/bench/raw/pred.csv" --out "\$RUN/bench/rawsym/pred.csv" \\
    --symprec $SYMPREC
bash scripts/atombench/score.sh "\$RUN/bench/raw/pred.csv"
bash scripts/atombench/score.sh "\$RUN/bench/rawsym/pred.csv"

# Now the relaxation-displacement metric means something: these structures
# have NOT been relaxed, so the distance to the nearest ALIGNN-FF minimum is a
# real measurement of how much geometric repair the sample needs.  On the
# relaxed predictions it measures 0.0007 A, because there is nowhere left to
# move.
OMP_NUM_THREADS=1 python -u scripts/atombench/angle_eval.py \\
    "\$RUN/bench/rawsym/pred.csv" --relax --relax-steps 200 \\
    --output "\$RUN/bench/rawsym/angle_eval.json"

printf '{"argv":["unrelaxed"],"elapsed_s":%d,"host":"%s"}\n' \\
    \$((SECONDS-t0)) "\$(hostname)" > "\$RUN/.stages/unrelaxed.json"
UNRELAXED

if [ "${DRY:-0}" = "1" ]; then
    echo "would submit: sbatch --export=ALL $SB"
    exit 0
fi
JID=$(sbatch --parsable --export=ALL "$SB")
echo "submitted array job $JID ($N elements)"
printf 'unrelaxed\tarray\t%s\n' "$JID" >> "$RESULTS/00_provenance/slurm_jobids.txt"
echo "watch: squeue -j $JID"
echo
echo "then:  python collect.py --force"
echo "       python analyze.py --variant rawsym   # the generator alone"
echo "       python analyze.py --variant sym      # the published pipeline"
