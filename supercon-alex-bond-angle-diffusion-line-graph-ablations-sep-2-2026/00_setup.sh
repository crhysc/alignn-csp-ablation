#!/usr/bin/env bash
# Phase 0 -- make this workspace ready.  Idempotent; safe to re-run.
#
#   bash 00_setup.sh                 # DATASET=jarvis
#   DATASET=alex bash 00_setup.sh
#
# This harness does NOT build the conda environments.  They already exist,
# built by the sibling angular-diffusion harness's 00_setup.sh, and rebuilding
# them would be both slow and a way for the two experiments to end up running
# against different package versions.  What this does instead: check they are
# usable, link the shared data store, generate cluster.env, and run the doctor.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

say() { printf '\n=== %s\n' "$*"; }

say "1/5  environments (built by the sibling harness; checked, not rebuilt)"
for e in "$TRAIN_ENV" "$SCORE_ENV_PATH"; do
    test -x "$e/bin/python" || {
        echo "FATAL: no python at $e" >&2
        echo "  Build it with ../supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/00_setup.sh" >&2
        exit 1; }
    echo "  found $e"
done
# The editable install has to be refreshed whenever alignn/ gains a module:
# the editable finder does not see a package that postdates it.  --no-deps
# refreshes the finder without touching a working dependency set.
"$TRAIN_ENV/bin/pip" install -e "$ALIGNN_REPO" --no-deps -q
"$SCORE_ENV_PATH/bin/pip" install -e "$ALIGNN_REPO" --no-deps -q
echo "  refreshed the editable alignn install in both"

say "2/5  AtomBench (from GitHub, never PyPI)"
test -f "$ATOMBENCH_REPO/scripts/scripts_consolidated/compute_metrics.py" \
    || { echo "FATAL: compute_metrics.py not found at $ATOMBENCH_REPO" >&2; exit 1; }
ver="$(on_env_arch "$SCORE_ENV_PATH/bin/python" -c \
    'import importlib.metadata as m; print(m.version("atombench"))' 2>/dev/null || echo NONE)"
case "$ver" in
    2022.*) echo "FATAL: the PyPI stub ($ver) is installed -- no metric code." >&2
            echo "  fix: $SCORE_ENV_PATH/bin/pip uninstall -y atombench" >&2
            echo "       $SCORE_ENV_PATH/bin/pip install -e $ATOMBENCH_REPO" >&2; exit 1 ;;
    NONE)   echo "FATAL: atombench not installed in $SCORE_ENV_PATH" >&2; exit 1 ;;
    *)      echo "  atombench $ver from $ATOMBENCH_REPO" ;;
esac

say "3/5  shared data store  ->  $CSP_RUNS/data"
# The prepared splits are shared with the sibling experiment on purpose: they
# are a pure function of the public databases and the preparation script, the
# Alexandria prep is expensive and already verified (6603/825/825, zero
# canonicalisation fallbacks), and two experiments deriving their own copies is
# just an opportunity for them to disagree about what the test set is.
#
# The RUN tree is NOT shared -- see env.sh.  Two of the four config names here
# collide with the sibling suite's, and those checkpoints are selected on a
# different criterion, so they are not interchangeable.
mkdir -p "$CSP_RUNS" "$DATA_STORE"
if [ -L "$CSP_RUNS/data" ]; then
    echo "  data -> $(readlink -f "$CSP_RUNS/data")"
elif [ -e "$CSP_RUNS/data" ]; then
    echo "  WARNING: $CSP_RUNS/data exists and is not a symlink; leaving it alone"
else
    ln -s "$DATA_STORE" "$CSP_RUNS/data"
    echo "  linked $CSP_RUNS/data -> $DATA_STORE"
fi
for s in train val test; do
    f="$CSP_RUNS/data/$SPLIT/$s.json"
    if [ -f "$f" ]; then
        n=$(python3 -c "import json;print(len(json.load(open('$f'))))")
        echo "  data/$SPLIT/$s.json  n=$n"
    else
        echo "  data/$SPLIT/$s.json  MISSING -- 30_full.sh submits $DATA_TASK"
    fi
done

say "4/5  cluster.env + results tree"
mkdir -p "$RESULTS"/{00_provenance,10_runs,20_benchmarks,30_atombench,40_stats,50_costs,60_report} \
         "$GPU_TRACE_DIR" "$ALIGNN_REPO/task_runners/logs"
bash ./env.sh --write

say "5/5  doctor"
set +e
run_task doctor 2>&1 | tee "$RESULTS/00_provenance/doctor.txt"
set -e
# doctor probes the TRAINING env, which deliberately has no pymatgen -- those
# live in the scoring env.  So check the scoring side where it belongs.
on_env_arch "$SCORE_ENV_PATH/bin/python" - <<'PY' | tee -a "$RESULTS/00_provenance/doctor.txt"
import importlib
print("\nScoring environment")
ok = True
for m in ("pymatgen", "amd", "atombench", "scipy", "matplotlib"):
    try:
        importlib.import_module(m); print(f"  [x] {m}")
    except Exception as e:
        ok = False; print(f"  [ ] {m}: {e}")
raise SystemExit(0 if ok else 1)
PY

echo
echo "setup complete.  DATASET=$DATASET  RUN_ID=$RUN_ID"
echo "results -> $RESULTS"
echo "next: bash preflight.sh"
