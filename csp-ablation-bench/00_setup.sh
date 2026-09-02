#!/usr/bin/env bash
# Phase 0 -- make this machine ready.  Idempotent; safe to re-run.
#
#   bash 00_setup.sh
#
# Installs AtomBench from GitHub (NOT from PyPI -- see PLAN.md 0.1), creates
# the scoring environment, generates cluster.env, and refuses to declare
# success until `run_task.py doctor` is clean.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

say() { printf '\n=== %s\n' "$*"; }

say "1/6  AtomBench clone -> $ATOMBENCH_REPO"
# Deliberately no --recurse-submodules: .gitmodules points at flowmm, atomgpt,
# cdvae and mattergen (three over SSH).  None are needed to score a CSV.
if [ -d "$ATOMBENCH_REPO/.git" ]; then
    git -C "$ATOMBENCH_REPO" pull --ff-only || echo "  (pull skipped)"
else
    mkdir -p "$(dirname "$ATOMBENCH_REPO")"
    git clone https://github.com/atomgptlab/atombench "$ATOMBENCH_REPO"
fi
test -f "$ATOMBENCH_REPO/scripts/scripts_consolidated/compute_metrics.py" \
    || { echo "FATAL: clone is missing compute_metrics.py" >&2; exit 1; }

say "2/6  scoring environment -> $SCORE_ENV_PATH"
if [ ! -x "$SCORE_ENV_PATH/bin/python" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -y -p "$SCORE_ENV_PATH" python=3.11
fi
# Editable from the clone gives us all three at once: the `atombench` CLI on
# PATH, compute_metrics.py at a stable path for score.sh, and the package
# importable by analyze.py so metric definitions stay AtomBench's.
"$SCORE_ENV_PATH/bin/pip" install -e "$ATOMBENCH_REPO"
"$SCORE_ENV_PATH/bin/pip" install "average-minimum-distance" scipy
# analyze.py runs here and reads the declared contrasts from
# alignn.inverse.ablations, which is a pure-python dict module -- --no-deps
# makes it importable without dragging torch into the scoring environment.
"$SCORE_ENV_PATH/bin/pip" install -e "$ALIGNN_REPO" --no-deps -q

say "3/6  guard against the PyPI stub"
# PyPI's `atombench` is version 2022.7.15: a 2.5 kB wheel with one file and no
# metric code, pinning numpy==1.19.5 / pandas==1.2.4.  If it ever displaces
# the editable install, everything downstream silently degrades.
# The two are cleanly distinguishable: the stub is 2022.7.15, carries a
# module-level __version__ and declares no console scripts; the GitHub package
# is 0.1.0, has no __version__ attribute, and declares atombench/-verify/-submit.
# So distribution metadata is the check, not the module attribute.
ver="$("$SCORE_ENV_PATH/bin/python" -c \
    'import importlib.metadata as m; print(m.version("atombench"))' 2>/dev/null || echo NONE)"
case "$ver" in
    2022.*)
        echo "FATAL: the PyPI stub ($ver) is installed, not the GitHub package." >&2
        echo "  It has no metric code and pins numpy==1.19.5." >&2
        echo "  fix: $SCORE_ENV_PATH/bin/pip uninstall -y atombench" >&2
        echo "       $SCORE_ENV_PATH/bin/pip install -e $ATOMBENCH_REPO" >&2
        exit 1 ;;
    NONE)
        echo "FATAL: atombench is not installed in $SCORE_ENV_PATH" >&2; exit 1 ;;
esac
"$SCORE_ENV_PATH/bin/python" -c 'import atombench.cli, atombench.tables, atombench.plots' \
    || { echo "FATAL: atombench package not importable" >&2; exit 1; }
test -x "$SCORE_ENV_PATH/bin/atombench" \
    || { echo "FATAL: no 'atombench' console script -- the stub declares none," >&2
         echo "  so this usually means the stub displaced the real package." >&2; exit 1; }
echo "  atombench $ver from $ATOMBENCH_REPO, importable, CLI present"

say "4/6  training environment"
test -x "$TRAIN_ENV/bin/python" || { echo "FATAL: no python at $TRAIN_ENV" >&2; exit 1; }
# INSTRUCTIONS.md: re-run the editable install even if you installed before --
# the editable finder does not see alignn/inverse if it predates it.  Verified
# on this machine: `import alignn` works while `import alignn.inverse` raises
# ModuleNotFoundError.  --no-deps refreshes the finder without touching a
# working dependency set.
"$TRAIN_ENV/bin/pip" install -e "$ALIGNN_REPO" --no-deps -q
"$TRAIN_ENV/bin/python" -c 'import alignn.inverse.train_csp, alignn.inverse.ablations' \
    || { echo "FATAL: alignn.inverse still not importable from $TRAIN_ENV" >&2
         echo "  try: $TRAIN_ENV/bin/pip install -e $ALIGNN_REPO" >&2; exit 1; }
echo "  alignn.inverse importable"

say "5/6  cluster.env + results tree"
mkdir -p "$RESULTS"/{00_provenance,10_runs,20_benchmarks,30_atombench,40_stats,50_costs,60_report} "$GPU_TRACE_DIR"
bash ./env.sh --write

say "6/6  doctor"
set +e
run_task doctor 2>&1 | tee "$RESULTS/00_provenance/doctor.txt"
set -e

# doctor scores the *training* env, which deliberately has no pymatgen -- those
# live in the scoring env, which is what CSP_SCORE_ENV is for.  So check the
# scoring dependencies where they actually belong.
"$SCORE_ENV_PATH/bin/python" - <<'PY' | tee -a "$RESULTS/00_provenance/doctor.txt"
import importlib
print("\nScoring environment")
ok = True
for m in ("pymatgen", "amd", "atombench", "scipy", "sklearn", "matplotlib"):
    try:
        importlib.import_module(m); print(f"  [x] {m}")
    except Exception as e:
        ok = False; print(f"  [ ] {m}: {e}")
raise SystemExit(0 if ok else 1)
PY

echo
echo "setup complete.  RUN_ID=$RUN_ID   results -> $RESULTS"
echo "next: bash 10_smoke.sh"
