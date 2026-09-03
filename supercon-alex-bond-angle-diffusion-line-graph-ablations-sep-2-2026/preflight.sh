#!/usr/bin/env bash
# Everything that must be true before this experiment costs a GPU-hour.
#
#   bash preflight.sh                # DATASET=jarvis
#   DATASET=alex bash preflight.sh
#
# Exits non-zero if anything that would waste a queue slot is wrong.

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
from alignn.inverse.ablations import MATRIX
assert len(MATRIX) == 4, MATRIX
PY
score_py - <<'PY' && ok "scoring env: pymatgen, amd, atombench" || bad "scoring env incomplete"
import pymatgen.core, amd, atombench.cli, atombench.tables, jarvis, numpy
PY

# The new cell has to be *buildable*, not merely named.  Until this commit the
# denoiser raised outright on (angle_diffusion, alignn_layers=0), so this is
# the check that the model change actually landed in the env being used.
train_py - <<'PY' && ok "the no-line-graph angular cell builds and matches on parameters" \
                   || bad "the (angle diffusion, no line graph) cell does not build"
import torch
from alignn.inverse.denoiser import ALIGNNCSPDenoiser
from alignn.inverse.ablations import all_cells, ablation_config
n = {}
for label, c in all_cells().items():
    m = ALIGNNCSPDenoiser(hidden_features=256, knn=12, num_steps=1000,
                          alignn_layers=c["alignn_layers"],
                          gcn_layers=c["gcn_layers"], **ablation_config(c["ablation"]))
    n[label] = sum(p.numel() for p in m.parameters())
# The angular path adds no weights of its own beyond the encoder and head the
# two angular cells share, so within the parameter-matched matrix those two
# must agree exactly.  If they ever stop agreeing, the normalisation claim in
# the report is wrong and this is where it shows up.
assert n["angle diffusion"] == n["both"], n
for k, v in n.items():
    print(f"      {k:26s} {v/1e6:7.4f} M")
PY

# ---------------------------------------------------------------------------
hdr "site configuration"
CE="$ALIGNN_REPO/task_runners/cluster.env"
if [ -f "$CE" ]; then
    # cluster.env is the ONE file this harness shares with its sibling, and the
    # two point CSP_RUNS at different roots.  Submitting with the other one's
    # file in place would put this experiment's checkpoints in that
    # experiment's tree, where two config names already collide.
    if grep -q "^# HARNESS=$HARNESS_ID\$" "$CE"; then
        ok "cluster.env belongs to this harness ($HARNESS_ID)"
    else
        bad "cluster.env was written by a DIFFERENT harness ($(grep -m1 '^# HARNESS=' "$CE" || echo 'unstamped')) -- run: bash env.sh --write"
    fi
    grep -q "^# DATASET=$DATASET\$" "$CE" && ok "cluster.env dataset: $DATASET" \
        || bad "cluster.env is for a different dataset -- run: bash env.sh --write"
    grep -q "CSP_RUNS=\"$CSP_RUNS\"" "$CE" && ok "cluster.env run root: $CSP_RUNS" \
        || bad "cluster.env CSP_RUNS differs -- run: bash env.sh --write"
    grep -q "CSP_ENV=\"$TRAIN_ENV\"" "$CE" && ok "cluster.env training env" \
        || bad "cluster.env CSP_ENV differs"
    grep -q "CSP_SCORE_ENV=\"$SCORE_ENV_PATH\"" "$CE" && ok "cluster.env scoring env" \
        || bad "cluster.env CSP_SCORE_ENV differs"
else
    bad "no cluster.env -- run: bash env.sh --write"
fi

# ---------------------------------------------------------------------------
hdr "scheduler"
# NOTE: no sacctmgr association check.  This cluster runs no accounting
# associations for this user and every partition is AllowAccounts=ALL, so
# CSP_ACCOUNT is deliberately empty and asking sacctmgr about an empty account
# would fail for a reason that is not a problem.
sinfo -h -p "$PART_FULL" -o %P >/dev/null 2>&1 && ok "partition '$PART_FULL' exists" \
    || bad "partition '$PART_FULL' not found"
ngpu=$(sinfo -h -N -p "$PART_FULL" -o "%G" 2>/dev/null | grep -o 'gpu:[0-9]*' | cut -d: -f2 \
       | awk '{s+=$1} END{print s+0}')
[ "${ngpu:-0}" -gt 0 ] && ok "$ngpu GPU(s) in '$PART_FULL' (throttle: $CSP_MAX_CONCURRENT)" \
    || bad "no GPUs in '$PART_FULL'"
[ -z "${CSP_ACCOUNT:-}" ] && ok "no --account, as this cluster requires" \
    || warn "CSP_ACCOUNT=$CSP_ACCOUNT -- this cluster had none when measured"

# ---------------------------------------------------------------------------
hdr "storage"
avail=$(df -BG --output=avail "$CSP_RUNS" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -gt 50 ] && ok "free space: ${avail}G at $CSP_RUNS" \
    || bad "only ${avail:-?}G free at $CSP_RUNS"
if [ -L "$CSP_RUNS/data" ]; then
    ok "data is a symlink -> $(readlink -f "$CSP_RUNS/data")"
else
    warn "$CSP_RUNS/data is not a symlink to the shared store (00_setup.sh makes it)"
fi

# ---------------------------------------------------------------------------
hdr "data ($SPLIT)"
for s in train val test; do
    f="$CSP_RUNS/data/$SPLIT/$s.json"
    if [ -f "$f" ]; then
        n=$(python3 -c "import json;print(len(json.load(open('$f'))))")
        ok "data/$SPLIT/$s.json  n=$n"
    else
        bad "missing $f -- run: bash 30_full.sh train (submits $DATA_TASK)"
    fi
done
# Canonicalisation is the thing this experiment was asked to guarantee: every
# training target and every scored prediction in the primitive Niggli cell.
# The training side is checked here; the prediction side is score.sh's
# canonical_cell, exercised by 10_smoke.sh.
meta="$CSP_RUNS/data/$SPLIT/split_meta.json"
if [ -f "$meta" ]; then
    ok "split provenance: $(tr -d '\n ' < "$meta" | cut -c1-110)"
else
    warn "no split_meta.json -- cannot confirm which partition this is"
fi
if [ -f "$CSP_RUNS/data/$SPLIT/train.json" ]; then
    # `$?` after a && chain reports the chain, not the check, so the probe runs
    # on its own line and its status is captured immediately.
    score_py - "$CSP_RUNS/data/$SPLIT/train.json" <<'CANON'
import json, sys
from pymatgen.core import Lattice, Structure
rows = json.load(open(sys.argv[1]))[:200]
drift = 0
for r in rows:
    s = Structure(Lattice(r["lattice_mat"]), r["elements"], r["frac_coords"])
    c = s.get_primitive_structure().get_reduced_structure(reduction_algo="niggli")
    # Volume and atom count are what a primitive+Niggli reduction would change
    # if the stored cell were not already at its fixed point.  natoms matters
    # twice over: it is also a conditioning input, so a target the metric would
    # shrink is a target the model was conditioned wrongly on.
    if len(c) != len(s) or abs(c.volume - s.volume) > 1e-3 * max(s.volume, 1.0):
        drift += 1
print(f"      {len(rows) - drift}/{len(rows)} sampled training targets already "
      f"sit at the primitive+Niggli fixed point")
raise SystemExit(1 if drift > len(rows) // 20 else 0)
CANON
    canon=$?
    if [ "$canon" -eq 0 ]; then
        ok "training targets are canonicalised (primitive + Niggli)"
    else
        bad "training targets are NOT canonical -- re-run $DATA_TASK with --canonicalize 1"
    fi
fi

# ---------------------------------------------------------------------------
hdr "offline readiness"
# ALIGNN-FF is fetched from figshare on a cache miss, and the parallel relax
# workers race on os.makedirs when the cache directory is absent -- a
# concurrency bug, not a connectivity one.  Warm it before the array runs.
FFDIR="$HOME/.cache/atomgptlab/alignn_ff"
ffmodel=$(find "$FFDIR" -name best_model.pt 2>/dev/null | head -1)
if [ -n "$ffmodel" ] && [ -s "$ffmodel" ]; then
    ok "ALIGNN-FF weights cached ($(du -h "$ffmodel" | cut -f1))"
else
    bad "ALIGNN-FF weights NOT cached -- every generate stage would download on"
    echo "      a compute node and the relax workers would race.  Warm it:"
    echo "      source env.sh && train_py -c \"from alignn.ff.ff import get_figshare_model_ff;"
    echo "        get_figshare_model_ff(model_name='matpes_r2scan')\""
fi
for d in "$FFDIR"/*/; do
    [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ] \
        && bad "empty FF cache dir (remove it): $d"
done

# ---------------------------------------------------------------------------
hdr "the experiment"
ncells=$(python3 -c "
import sys; sys.path.insert(0, '$ALIGNN_REPO')
from alignn.inverse.ablations import all_cells; print(len(all_cells()))")
n=$(run_task "$MATRIX_TASK" --count --seeds "$SEEDS" 2>/dev/null)
nseeds=$(tr ',' '\n' <<<"$SEEDS" | wc -l)
if [ "${n:-0}" = "$((ncells * nseeds))" ]; then
    ok "$MATRIX_TASK: $n unit(s) = $ncells cells x $nseeds seed(s)"
else
    bad "$MATRIX_TASK reports ${n:-?} units, expected $((ncells * nseeds))"
fi
blocked=$(run_task "$MATRIX_TASK" --seeds "$SEEDS" --dry-run 2>&1 | grep -ci "BLOCKED" || true)
[ "${blocked:-0}" -eq 0 ] && ok "no blocked prerequisites" \
    || bad "$blocked stage(s) BLOCKED -- a prerequisite has not run"
# Selection criterion: the whole comparison rests on it.
run_task "$MATRIX_TASK" --seeds "$SEEDS" --dry-run 2>&1 | grep -c -- "--select-on structural" \
    | { read -r c; [ "$c" = "$((ncells * nseeds))" ] \
        && ok "every cell selects its checkpoint on the structural loss" \
        || bad "only $c of $((ncells * nseeds)) cells pass --select-on structural"; }

# ---------------------------------------------------------------------------
hdr "analysis chain"
for s in collect.py stage_benchmarks.py analyze.py costs.py; do
    python3 -c "import ast;ast.parse(open('$s').read())" \
        && ok "$s parses" || bad "$s does not parse"
done
python3 -c "
import sys; sys.path.insert(0, '$ALIGNN_REPO')
from alignn.inverse.ablations import (MATRIX, MATRIX_COMPARISONS, MATRIX_COMPUTE,
                                      MATRIX_COMPUTE_COMPARISONS, all_cells)
assert len(MATRIX) == 4 and len(MATRIX_COMPUTE) == 4
shared = set(MATRIX) & set(MATRIX_COMPUTE)
assert shared == {'line graph', 'both'}, shared
print(f'      {len(all_cells())} distinct cells across two 4-cell matrices '
      f'({len(shared)} shared), '
      f'{len(MATRIX_COMPARISONS) + len(MATRIX_COMPUTE_COMPARISONS)} contrasts')
" && ok "analyze.py runs on the login node (stdlib only)" \
  || bad "analyze.py cannot read the matrix definition"

# ---------------------------------------------------------------------------
hdr "verdict"
if [ "$FAIL" = "0" ]; then
    echo "  READY.  next: bash 10_smoke.sh, then 20_pilot.sh price, then 30_full.sh"
else
    echo "  NOT READY -- fix the [ ] lines above."
fi
exit "$FAIL"
