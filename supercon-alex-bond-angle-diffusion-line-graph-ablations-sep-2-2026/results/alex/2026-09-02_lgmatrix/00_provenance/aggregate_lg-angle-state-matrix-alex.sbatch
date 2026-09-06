#!/usr/bin/env bash
#SBATCH --job-name=csp-aggregate-lg-angle-state-matrix-alex
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/aggregate-%j.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/aggregate-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
# No --gres: the summary needs the aarch64 environment, not a GPU.
cd "/home/ccamp104/repositories/alignn-csp-ablation/alignn"
source task_runners/common.sh

echo; echo "=== lg-angle-state-matrix-alex  (/data/ccamp104/alignn_csp_lgmatrix/alex)"
python task_runners/run_task.py "lg-angle-state-matrix-alex" --aggregate --latex --runs-root "/data/ccamp104/alignn_csp_lgmatrix/alex"

echo; echo "=== lg-angle-state-matrix-alex complete: $(date -Is)"
