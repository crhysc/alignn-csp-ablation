#!/usr/bin/env bash
# The post-run analysis chain, as a SLURM job.
#
#   bash 50_analyze.sh            # submit
#   bash 50_analyze.sh --local    # run here instead (only for tiny trees)
#
# This is NOT light work and does not belong on a login node: the atombench
# CLI runs StructureMatcher plus an AMD descriptor over every structure of
# every benchmark, and analyze.py's tier-2 pass re-matches every target of
# every run to get the per-target pairing.  On 24 runs x 103 targets that is
# tens of thousands of structure comparisons.
#
# CPU only -- no --gres.  Every partition on this cluster carries GPUs, so a
# CPU job simply declines to ask for one rather than going somewhere else.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

SB="$RESULTS/00_provenance/analysis.sbatch"
mkdir -p "$RESULTS/00_provenance"
cat > "$SB" <<ANALYSIS
#!/usr/bin/env bash
#SBATCH --job-name=csp-analysis
#SBATCH --output=$RESULTS/00_provenance/analysis-%j.out
#SBATCH --error=$RESULTS/00_provenance/analysis-%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=32G
#SBATCH --account=$CSP_ACCOUNT
#SBATCH --partition=$PART_FULL
set -euo pipefail
cd "$HARNESS"
source ./env.sh

echo "=== 1/5 collect"
"\$TRAIN_ENV/bin/python" collect.py --force
echo "=== 2/5 stage"
"\$TRAIN_ENV/bin/python" stage_benchmarks.py
echo "=== 3/5 atombench (metrics, figures, tables)"
"\$SCORE_ENV_PATH/bin/atombench" "\$RESULTS/20_benchmarks" "\$RESULTS/30_atombench"
echo "=== 4/5 costs"
"\$TRAIN_ENV/bin/python" costs.py --harvest
echo "=== 5/5 statistics"
"\$SCORE_ENV_PATH/bin/python" analyze.py
echo
echo "=== integrity"
"\$TRAIN_ENV/bin/python" collect.py --verify
echo
echo "results: \$RESULTS"
ANALYSIS

if [ "${1:-}" = "--local" ]; then
    echo "running locally (use only for smoke-sized trees)"; bash "$SB"; exit
fi
JID=$(sbatch --parsable --export=ALL "$SB")
echo "submitted analysis job $JID"
printf 'analysis\tjob\t%s\n' "$JID" >> "$RESULTS/00_provenance/slurm_jobids.txt"
echo "watch: squeue -j $JID   log: $RESULTS/00_provenance/analysis-$JID.out"
