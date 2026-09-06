#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-pilot-train
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/pilot_train-%j.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/pilot_train-%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
set -uo pipefail
cd "/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026"; source ./env.sh
# --only-stages train: the whole point is to isolate the training cost from
# the generation cost, which `price` measures separately.
run_task lg-angle-matrix-alex --seeds "0" \
    --only-stages train --unit 3
