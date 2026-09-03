#!/usr/bin/env bash
# Everything that must be true before phase 4 costs a GPU-hour.
#
#   bash preflight.sh
#
# Checks the two environments, the AtomBench install, the generated
# cluster.env, the data split, the scheduler, every task's unit count and
# blocked-stage status, and the whole post-run analysis chain.  Exits non-zero
# if anything that would waste a queue slot is wrong.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

FAIL=0
ok()   { printf "  [x] %s\n" "$*"; }
bad()  { printf "  [ ] %s\n" "$*"; FAIL=1; }
warn() { printf "  [~] %s\n" "$*"; }
hdr()  { printf "\n=== %s\n" "$*"; }

# ---------------------------------------------------------------------------
hdr "environments"
train_py - <<'PY' && ok "training env: torch, jarvis-tools, alignn.inverse" || bad "training env incomplete"
import torch, jarvis, alignn.inverse.train_csp, alignn.inverse.ablations
PY
score_py - <<'PY' && ok "scoring env: pymatgen, amd, atombench, scipy, sklearn" || bad "scoring env incomplete"
import pymatgen.core, amd, atombench.cli, atombench.tables, atombench.plots
import scipy, sklearn, matplotlib, jarvis, numpy
PY
ver="$(score_py -c 'import importlib.metadata as m;print(m.version("atombench"))' 2>/dev/null || echo NONE)"
case "$ver" in
    2022.*) bad "atombench is the PyPI STUB ($ver) -- no metric code" ;;
    NONE)   bad "atombench not installed in the scoring env" ;;
    *)      ok  "atombench $ver (GitHub package)" ;;
esac
test -x "$SCORE_ENV_PATH/bin/atombench" && ok "atombench CLI on PATH" \
    || bad "atombench console script missing"
test -f "$ATOMBENCH_REPO/scripts/scripts_consolidated/compute_metrics.py" \
    && ok "compute_metrics.py (used by score.sh)" \
    || bad "compute_metrics.py not found at $ATOMBENCH_REPO"

# ---------------------------------------------------------------------------
hdr "site configuration"
CE="$ALIGNN_REPO/task_runners/cluster.env"
if [ -f "$CE" ]; then
    # cluster.env is shared with the sibling line-graph-matrix harness, and the
    # two point CSP_RUNS at different run roots.  Check whose it is before
    # trusting anything else in it.
    if grep -q "^# HARNESS=$HARNESS_ID\$" "$CE"; then
        ok "cluster.env belongs to this harness ($HARNESS_ID)"
    else
        bad "cluster.env was written by a DIFFERENT harness ($(grep -m1 '^# HARNESS=' "$CE" || echo 'unstamped')) -- run: bash env.sh --write"
    fi
    grep -q "CSP_RUNS=\"$CSP_RUNS\"" "$CE" && ok "cluster.env run root: $CSP_RUNS" \
        || bad "cluster.env CSP_RUNS differs -- run: bash env.sh --write"
    grep -q "CSP_GPU_GRES=\"$CSP_GPU_GRES\"" "$CE" && ok "cluster.env GPU pin: $CSP_GPU_GRES" \
        || bad "cluster.env GPU pin differs from env.sh -- run: bash env.sh --write"
    grep -q "CSP_ACCOUNT=\"$CSP_ACCOUNT\"" "$CE" && ok "cluster.env account: $CSP_ACCOUNT" \
        || bad "cluster.env account differs -- run: bash env.sh --write"
    grep -q "CSP_ENV=\"$TRAIN_ENV\"" "$CE" && ok "cluster.env training env" \
        || bad "cluster.env CSP_ENV differs"
    grep -q "CSP_SCORE_ENV=\"$SCORE_ENV_PATH\"" "$CE" && ok "cluster.env scoring env" \
        || bad "cluster.env CSP_SCORE_ENV differs"
else
    bad "no cluster.env -- run: bash env.sh --write"
fi

# ---------------------------------------------------------------------------
hdr "scheduler"
# This cluster runs no accounting associations for this user and every
# partition is AllowAccounts=ALL (see env.sh), so CSP_ACCOUNT is deliberately
# empty here.  Querying sacctmgr for an empty account name always returns
# nothing and always failed this check for a reason that is not a problem --
# skip it rather than report a false negative every time.
if [ -z "${CSP_ACCOUNT:-}" ]; then
    ok "no --account, as this cluster requires (no accounting associations)"
else
    sacctmgr -n show assoc where user="$USER" account="$CSP_ACCOUNT" format=Account 2>/dev/null \
        | grep -q "$CSP_ACCOUNT" && ok "account '$CSP_ACCOUNT' is ours" \
        || bad "no association for account '$CSP_ACCOUNT'"
fi
sinfo -h -p "$PART_FULL" -o %P >/dev/null 2>&1 && ok "partition '$PART_FULL' exists" \
    || bad "partition '$PART_FULL' not found"
gputype="${CSP_GPU_GRES#gpu:}"; gputype="${gputype%:*}"
# -N lists one row per node; without it sinfo collapses nodes by state and
# the count is meaningless.
nnodes=$(sinfo -h -N -p "$PART_FULL" -o "%N|%G" 2>/dev/null | grep -c "$gputype")
ngpus=$(sinfo -h -N -p "$PART_FULL" -o "%G" 2>/dev/null \
        | grep -o "${gputype}:[0-9]*" | cut -d: -f2 \
        | awk '{s+=$1} END{print s+0}')
[ "${nnodes:-0}" -gt 0 ] \
    && ok "$gputype in $PART_FULL: ${nnodes} nodes, ${ngpus:-?} GPUs (per-user cap 20)" \
    || bad "no $gputype nodes in $PART_FULL"
maxsub=$(sacctmgr -n show assoc where user="$USER" format=MaxSubmit 2>/dev/null | tr -d ' ' | head -1)
warn "submit cap ${maxsub:-unknown}; this run peaks at 60 array elements"

# ---------------------------------------------------------------------------
hdr "storage"
avail=$(df -BG --output=avail "$CSP_RUNS" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -gt 50 ] && ok "scratch free: ${avail}G at $CSP_RUNS" \
    || bad "only ${avail:-?}G free at $CSP_RUNS"

# ---------------------------------------------------------------------------
hdr "data"
for s in train val test; do
    f="$CSP_RUNS/data/$SPLIT/$s.json"
    if [ -f "$f" ]; then
        n=$(python3 -c "import json;print(len(json.load(open('$f'))))")
        ok "data/$SPLIT/$s.json  n=$n"
    else
        bad "missing $f -- run: bash 30_full.sh train (submits $DATA_TASK)"
    fi
done

# ---------------------------------------------------------------------------
hdr "offline readiness"
# On THIS cluster the GPU nodes do have outbound network (the environments were
# pip-installed from one), so a lazy first-use download will not simply fail as
# it did on the old site.  It is still warmed ahead of time, because the
# parallel relax workers race on os.makedirs when the cache directory is
# absent -- a concurrency bug, not a connectivity one, and it does not care
# whether the node has internet.
# ALIGNN-FF is the one that bites: generate_benchmark.py uses it for energy
# ranking and relaxation, ff.py fetches it from figshare on a cache miss, and
# the parallel relax workers race on os.makedirs when the directory is absent.
FFDIR="$HOME/.cache/atomgptlab/alignn_ff"
ffmodel=$(find "$FFDIR" -name best_model.pt 2>/dev/null | head -1)
if [ -n "$ffmodel" ] && [ -s "$ffmodel" ]; then
    ok "ALIGNN-FF weights cached: $(du -h "$ffmodel" | cut -f1) $(dirname "$ffmodel" | xargs basename)"
else
    bad "ALIGNN-FF weights NOT cached -- every generate stage would try to"
    echo "      download on a compute node and fail.  Fix on the LOGIN node:"
    echo "      source env.sh && train_py -c \"from alignn.ff.ff import get_figshare_model_ff;"
    echo "        get_figshare_model_ff(model_name='matpes_r2scan')\""
fi
# An empty cache dir is worse than none: it satisfies the exists() check that
# guards makedirs while still forcing a download.
for d in "$FFDIR"/*/; do
    [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ] \
        && bad "empty FF cache dir (remove it): $d"
done
jd="$HOME/.cache/atomgptlab/jarvis_data"
[ -d "$jd" ] && [ -n "$(ls -A "$jd" 2>/dev/null)" ] \
    && ok "jarvis-tools dataset cache present" \
    || warn "no jarvis_data cache; only matters if a split is rebuilt"

# ---------------------------------------------------------------------------
hdr "tasks: units and blocked stages"
for t in "$ABLATION_TASK" "$LINEGRAPH_TASK" symprec-sweep pipeline-ablation; do
    n=$(run_task "$t" --count 2>/dev/null)
    blocked=$(run_task "$t" --dry-run 2>&1 | grep -ci "BLOCKED" || true)
    if [ "${blocked:-0}" -gt 0 ]; then
        bad "$t: $n unit(s), $blocked BLOCKED"
    else
        ok "$t: $n unit(s), no blocked prerequisites"
    fi
done

# ---------------------------------------------------------------------------
hdr "analysis chain"
for s in collect.py stage_benchmarks.py costs.py; do
    python3 -c "import ast;ast.parse(open('$s').read())" \
        && ok "$s parses" || bad "$s does not parse"
done
score_py -c "
import ast; ast.parse(open('analyze.py').read())
import atombench.tables, pymatgen.analysis.structure_matcher, scipy.stats, numpy
from alignn.inverse.ablations import COMPARISONS
assert len(COMPARISONS) >= 6
" && ok "analyze.py imports resolve in the scoring env" \
  || bad "analyze.py dependencies missing in the scoring env"
score_py -c "
from alignn.inverse.ablations import COMPARISONS
print('      contrasts:', len(COMPARISONS), '->', len(set(COMPARISONS.values())), 'distinct arm pairs')"

# ---------------------------------------------------------------------------
hdr "verdict"
if [ "$FAIL" = "0" ]; then
    echo "  READY.  next: bash 10_smoke.sh   (then 20_pilot.sh, then 30_full.sh)"
else
    echo "  NOT READY -- fix the [ ] lines above."
fi
exit "$FAIL"
