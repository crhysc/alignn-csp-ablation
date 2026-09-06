#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-unrelaxed
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%A_%a.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-5%2
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
cd "/home/ccamp104/repositories/alignn-csp-ablation/alignn"
source task_runners/common.sh

RUN=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0}+1))p" "/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/unrelaxed_targets.txt")
SEED=$(basename "$RUN" | tr -dc 0-9)
DATA="/data/ccamp104/alignn_csp_lgmatrix/alex/data/alex"
mkdir -p "$RUN/bench/raw" "$RUN/bench/rawsym" "$RUN/.stages"
echo "unrelaxed generation: $RUN (seed $SEED)"

t0=$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \
    --checkpoint "$RUN/best_model.pt" --data-dir "$DATA" --split test \
    --output-csv "$RUN/bench/raw/pred.csv" \
    --num-candidates 1 --guidance 2.0 --relax none --rank none \
    --seed "$SEED" --device cuda
echo "generate_raw $((SECONDS-t0))s"

# Symmetrised as well as raw: the lattice-angle and KLD metrics are measured
# after Niggli reduction, which is discontinuous, so a nearly-symmetric cell
# that is slightly off reduces to a different basis and contributes a large
# angular error.  Same tolerance as the headline run.
bash scripts/atombench/symmetrize.sh \
    --csv "$RUN/bench/raw/pred.csv" --out "$RUN/bench/rawsym/pred.csv" \
    --symprec 0.1
bash scripts/atombench/score.sh "$RUN/bench/raw/pred.csv"
bash scripts/atombench/score.sh "$RUN/bench/rawsym/pred.csv"

# Now the relaxation-displacement metric means something: these structures
# have NOT been relaxed, so the distance to the nearest ALIGNN-FF minimum is a
# real measurement of how much geometric repair the sample needs.  On the
# relaxed predictions it measures 0.0007 A, because there is nowhere left to
# move.
OMP_NUM_THREADS=1 python -u scripts/atombench/angle_eval.py \
    "$RUN/bench/rawsym/pred.csv" --relax --relax-steps 200 \
    --output "$RUN/bench/rawsym/angle_eval.json"

printf '{"argv":["unrelaxed"],"elapsed_s":%d,"host":"%s"}\n' \
    $((SECONDS-t0)) "$(hostname)" > "$RUN/.stages/unrelaxed.json"
