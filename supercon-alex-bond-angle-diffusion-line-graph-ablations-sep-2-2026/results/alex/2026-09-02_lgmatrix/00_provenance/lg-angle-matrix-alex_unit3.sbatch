#!/usr/bin/env bash
#SBATCH --job-name=csp-lg-angle-matrix-alex-u3
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%j.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/alignn/task_runners/logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
cd "/home/ccamp104/repositories/alignn-csp-ablation/alignn"
source task_runners/common.sh
# The run root is this dataset's, whatever cluster.env says by the time this
# job starts (common.sh has just set ALIGNN_RUNS from it).
export ALIGNN_RUNS="/data/ccamp104/alignn_csp_lgmatrix/alex"

python task_runners/run_task.py lg-angle-matrix-alex \
    --runs-root "/data/ccamp104/alignn_csp_lgmatrix/alex" \
    --seeds "0" --relax-workers "16" --symprec "0.1" \
    --unit 3
