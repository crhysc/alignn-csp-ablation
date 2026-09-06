#!/usr/bin/env bash
#SBATCH --job-name=csp-confound
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%A_%a.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%A_%a.err
#SBATCH --array=0-0%2
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
cd "/home/ccamp104/repositories/alignn-csp-ablation/alignn"
source task_runners/common.sh

SEEDS=($(tr ',' ' ' <<<"0"))
S=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}
DATA="/data/ccamp104/alignn_csp/alex/data/alex"
RUN="/data/ccamp104/alignn_csp/alex/train/alex_A3_nogate/seed$S"
mkdir -p "$RUN/bench/nosym" "$RUN/bench/sym" "$RUN/.stages"

t0=$SECONDS
python -u -m alignn.inverse.train_csp \
    --data-dir "$DATA" --output "$RUN" --epochs 1000 --seed "$S" \
    --ablation A3 --gate-pair-messages 0 \
    --alignn-layers 3 --gcn-layers 3 --hidden-features 256 --knn 12 \
    --num-steps 1000 --batch-size 64 --lr 1e-3 --augment 0 \
    --device cuda --log-every 25
printf '{"argv":["confound-train"],"elapsed_s":%d,"host":"%s"}\n' \
    $((SECONDS-t0)) "$(hostname)" > "$RUN/.stages/train.json"

t0=$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \
    --checkpoint "$RUN/best_model.pt" --data-dir "$DATA" --split test \
    --output-csv "$RUN/bench/nosym/pred.csv" \
    --num-candidates 32 --guidance 2.0 --relax cell --rank energy \
    --relax-steps 200 --prescreen-keep 4 --relax-workers 16 \
    --seed "$S" --device cuda \
    --save-candidates "$RUN/bench/nosym/candidates.json"
printf '{"argv":["confound-generate"],"elapsed_s":%d,"host":"%s"}\n' \
    $((SECONDS-t0)) "$(hostname)" > "$RUN/.stages/generate.json"

python -u scripts/atombench/symmetrize_predictions.py \
    --csv "$RUN/bench/nosym/pred.csv" --out "$RUN/bench/sym/pred.csv" \
    --symprec 0.1
bash scripts/atombench/score.sh "$RUN/bench/nosym/pred.csv"
bash scripts/atombench/score.sh "$RUN/bench/sym/pred.csv"
