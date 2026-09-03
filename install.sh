#!/usr/bin/env bash
# Portable installer -- get a fresh clone of this repo to the point where
# either harness's own 00_setup.sh / preflight.sh will succeed, on any Linux
# cluster, whatever its scheduler, GPU, or architecture.
#
#   bash install.sh                              # everything, default workspace
#   bash install.sh --workspace /scratch/me/csp   # a specific workspace root
#   bash install.sh --dry-run                     # print every command, do nothing
#   bash install.sh --cuda-index-url URL           # this site's torch/CUDA wheel index
#   bash install.sh --skip-envs                    # only the submodule + site.env
#   bash install.sh --help
#
# What this does NOT do, on purpose
# ----------------------------------
# It has no SLURM in it anywhere -- no sbatch, no partition names, no account,
# no --gres. Both harnesses already externalise every one of those into
# env.sh via `${VAR:-default}`, which is what makes them portable in the
# first place; the only thing missing on a fresh cluster is *values* for that
# handful of variables, and this script cannot know them for you. What it
# does instead: write `site.env` at the repo root with every path this
# installer actually determined (workspace, envs, AtomBench checkout) filled
# in correctly, and every SLURM/GPU knob left as an explicit placeholder with
# the command that discovers the right value on your site. Both harnesses'
# env.sh source `site.env` automatically if it exists (see the three-line
# hook near their top), so filling in that one file is the entire porting
# step -- neither harness's own env.sh needs editing.
#
# What this DOES do
# -------------------
# 1. Initialise the `alignn` submodule (crhysc/alignn @ angle-diffusion).
# 2. Find or bootstrap conda/mamba, matching the CURRENT machine's
#    architecture (not hard-coded to any one site's aarch64/x86_64 build).
# 3. Create two environments -- training (torch, jarvis-tools, ALIGNN
#    editable) and scoring (pymatgen, amd, AtomBench editable) -- kept apart
#    because AtomBench's dependency tree can pin numpy/pandas versions that
#    would fight a working torch install; see either harness's README for
#    the full rationale.
# 4. Install torch. Portability is the point here: the default is a bare
#    `pip install torch`, which resolves to whatever wheel pip's index
#    offers for the current platform (CPU-only if there is no CUDA wheel for
#    it, a CUDA build otherwise). `--cuda-index-url` overrides this for a
#    site whose GPU needs a specific wheel index (the pattern every site so
#    far has needed is `https://download.pytorch.org/whl/cuXXX`) -- discover
#    the right one for your GPU/driver from pytorch.org's own install matrix;
#    this script does not guess it.
# 5. Clone AtomBench from GitHub (never PyPI -- see the guard below) and
#    install both environments' remaining dependencies.
# 6. Write `site.env` at the repo root.
#
# Idempotent and safe to re-run: every step checks whether its target already
# exists before acting, matching the convention both harnesses' own
# 00_setup.sh already use.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-$(cd "$HERE/.." && pwd)/alignn_csp_workspace}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-}"
ATOMBENCH_URL="${ATOMBENCH_URL:-https://github.com/atomgptlab/atombench}"
DRY_RUN=0
SKIP_ENVS=0
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --workspace) WORKSPACE="$(cd "$(dirname "$2")" 2>/dev/null && pwd)/$(basename "$2")" 2>/dev/null || WORKSPACE="$2"; shift 2 ;;
        --cuda-index-url) CUDA_INDEX_URL="$2"; shift 2 ;;
        --atombench-url) ATOMBENCH_URL="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-envs) SKIP_ENVS=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

say() { printf '\n=== %s\n' "$*"; }
run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '  [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

TRAIN_ENV_PATH="$WORKSPACE/envs/train"
SCORE_ENV_PATH_="$WORKSPACE/envs/score"
ATOMBENCH_REPO_="$WORKSPACE/atombench"
CONDA_PREFIX_DIR="$WORKSPACE/envs/miniforge3"

say "workspace: $WORKSPACE"
[ "$DRY_RUN" = "1" ] && echo "  (--dry-run: printing every command, changing nothing)"

# ---------------------------------------------------------------------------
say "1/6  alignn submodule"
if [ ! -f "$HERE/alignn/setup.py" ]; then
    run git -C "$HERE" submodule update --init --recursive
else
    echo "  already initialised: $HERE/alignn"
fi
ALIGNN_REPO="$HERE/alignn"

# ---------------------------------------------------------------------------
say "2/6  conda / mamba"
# Prefer whatever the machine already has -- most clusters have a module- or
# system-provided conda, and bootstrapping a second one is both slower and a
# second thing that can drift from the site's own toolchain. Miniforge is the
# fallback, never the default, and its installer URL is chosen from `uname
# -m` rather than assumed, so this works on x86_64 and aarch64 alike.
#
# This assumes the node running install.sh matches the architecture of the
# nodes that will actually run jobs. That is true on most clusters and false
# on at least one site this codebase has run on (atomgptlab: x86_64 login
# node, aarch64 GPU nodes) -- there, an env built here would be unusable on
# the GPU nodes, and the fix is site-specific (that site's own HANDOFF.md
# builds a separate aarch64 Miniforge via a submitted GPU job). If your
# login and compute nodes differ in `uname -m`, run this script from an
# interactive session on a compute node instead of the login node.
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="$(command -v mamba)"
    echo "  using mamba: $CONDA_BIN"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    echo "  using conda: $CONDA_BIN"
    echo "  (found on PATH -- verify this is usable on your COMPUTE nodes,"
    echo "   not just here; see the comment above if login/compute arch differ)"
else
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|aarch64) ;;
        *) echo "FATAL: no conda/mamba on PATH, and no Miniforge build for" >&2
           echo "  architecture '$ARCH'. Install conda yourself and re-run." >&2
           exit 1 ;;
    esac
    if [ -x "$CONDA_PREFIX_DIR/bin/conda" ]; then
        echo "  Miniforge already bootstrapped at $CONDA_PREFIX_DIR"
    else
        echo "  no conda/mamba found; bootstrapping Miniforge for $ARCH"
        run mkdir -p "$WORKSPACE/envs"
        run bash -c "curl -sSL -o '$WORKSPACE/envs/miniforge.sh' \
            'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$ARCH.sh'"
        run bash "$WORKSPACE/envs/miniforge.sh" -b -p "$CONDA_PREFIX_DIR"
    fi
    CONDA_BIN="$CONDA_PREFIX_DIR/bin/conda"
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_ENVS" = "1" ]; then
    say "3-5/6  environments (skipped: --skip-envs)"
else
    say "3/6  training environment  ->  $TRAIN_ENV_PATH"
    if [ -x "$TRAIN_ENV_PATH/bin/python" ]; then
        echo "  already exists"
    else
        run "$CONDA_BIN" create -y -p "$TRAIN_ENV_PATH" "python=$PYTHON_VERSION"
    fi

    echo "  torch:"
    if [ -n "$CUDA_INDEX_URL" ]; then
        echo "    --cuda-index-url given: $CUDA_INDEX_URL"
        run "$TRAIN_ENV_PATH/bin/pip" install -q --index-url "$CUDA_INDEX_URL" torch
    elif [ "$DRY_RUN" = "1" ] || ! "$TRAIN_ENV_PATH/bin/python" -c 'import torch' 2>/dev/null; then
        echo "    no --cuda-index-url given; plain 'pip install torch'"
        echo "    (if this resolves to a CPU-only build and you have a GPU,"
        echo "     find your site's wheel index at pytorch.org/get-started"
        echo "     and re-run with --cuda-index-url)"
        run "$TRAIN_ENV_PATH/bin/pip" install -q torch
    else
        echo "    already installed"
    fi

    echo "  alignn.inverse and its declared dependencies (setup.py):"
    run "$TRAIN_ENV_PATH/bin/pip" install -q -e "$ALIGNN_REPO"

    if [ "$DRY_RUN" != "1" ]; then
        "$TRAIN_ENV_PATH/bin/python" - <<'PY'
import torch
import alignn.inverse.train_csp  # noqa: F401
import alignn.inverse.ablations as A
print(f"  [x] torch {torch.__version__}, CUDA available: "
      f"{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"      device: {torch.cuda.get_device_name(0)}")
else:
    print("      no GPU visible from this shell -- expected on a login "
          "node with no allocation; the harnesses request one per job.")
print(f"  [x] alignn.inverse importable, ablations: {sorted(A.ABLATIONS)}, "
      f"{len(A.MATRIX) if hasattr(A, 'MATRIX') else 0}-cell matrix "
      f"{'present' if hasattr(A, 'all_cells') else 'ABSENT (old alignn checkout?)'}")
PY
    fi

    say "4/6  scoring environment  ->  $SCORE_ENV_PATH_"
    if [ -x "$SCORE_ENV_PATH_/bin/python" ]; then
        echo "  already exists"
    else
        run "$CONDA_BIN" create -y -p "$SCORE_ENV_PATH_" "python=$PYTHON_VERSION"
    fi

    echo "  AtomBench (from GitHub -- see the guard below for why never PyPI)"
    if [ -d "$ATOMBENCH_REPO_/.git" ]; then
        echo "    already cloned: $ATOMBENCH_REPO_"
    else
        run git clone "$ATOMBENCH_URL" "$ATOMBENCH_REPO_"
    fi
    run "$SCORE_ENV_PATH_/bin/pip" install -q -e "$ATOMBENCH_REPO_"
    run "$SCORE_ENV_PATH_/bin/pip" install -q \
        "average-minimum-distance" scipy jarvis-tools spglib
    # --no-deps: this environment has no torch and must not gain one. The
    # score env only needs alignn.inverse.ablations (a pure dict/typing
    # module, no torch import at module scope) for its contrast definitions.
    run "$SCORE_ENV_PATH_/bin/pip" install -q -e "$ALIGNN_REPO" --no-deps

    say "5/6  guards and verification"
    if [ "$DRY_RUN" != "1" ]; then
        ver="$("$SCORE_ENV_PATH_/bin/python" -c \
            'import importlib.metadata as m; print(m.version("atombench"))' 2>/dev/null || echo NONE)"
        case "$ver" in
            2022.*)
                echo "FATAL: the PyPI stub ($ver) is installed, not the GitHub" >&2
                echo "  package. It has no metric code and pins numpy==1.19.5." >&2
                echo "  fix: $SCORE_ENV_PATH_/bin/pip uninstall -y atombench" >&2
                echo "       $SCORE_ENV_PATH_/bin/pip install -e $ATOMBENCH_REPO_" >&2
                exit 1 ;;
            NONE)
                echo "FATAL: atombench not installed in $SCORE_ENV_PATH_" >&2
                exit 1 ;;
            *) echo "  [x] atombench $ver from $ATOMBENCH_REPO_" ;;
        esac
        test -x "$SCORE_ENV_PATH_/bin/atombench" \
            && echo "  [x] atombench console script present" \
            || { echo "FATAL: no 'atombench' console script" >&2; exit 1; }

        "$SCORE_ENV_PATH_/bin/python" - <<'PY'
import importlib
ok = True
for m in ("pymatgen", "amd", "atombench", "scipy", "sklearn", "matplotlib",
          "jarvis", "numpy", "pandas"):
    try:
        importlib.import_module(m)
        print(f"  [x] {m}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [ ] {m}: {e}")
import alignn.inverse.ablations as A
print(f"  [x] alignn.inverse.ablations (torch-free): {sorted(A.ABLATIONS)}")
raise SystemExit(0 if ok else 1)
PY
    fi
fi

# ---------------------------------------------------------------------------
say "6/6  site.env"
SITE_ENV="$HERE/site.env"
if [ -f "$SITE_ENV" ]; then
    echo "  $SITE_ENV already exists -- leaving it alone."
    echo "  (delete it and re-run, or edit the paths below by hand, if the"
    echo "   workspace moved: WORKSPACE=$WORKSPACE)"
else
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] would write $SITE_ENV"
    else
        cat > "$SITE_ENV" <<SITEENV
# Site configuration for this cluster, written by install.sh on $(date -Is).
#
# Both harnesses' env.sh source this file automatically if it exists (see
# the hook near their top) -- everything below is picked up as an override
# on the harness's own \${VAR:-default} lines, so nothing in either env.sh
# needs editing. Safe to hand-edit; install.sh will not overwrite it once it
# exists.
#
# ---- filled in by install.sh: paths it actually determined -----------------
# These three are shared by both harnesses -- same conda envs, same AtomBench
# checkout -- so one value each is correct.
export TRAIN_ENV="$TRAIN_ENV_PATH"
export SCORE_ENV_PATH="$SCORE_ENV_PATH_"
export ATOMBENCH_REPO="$ATOMBENCH_REPO_"

# CSP_RUNS_BASE and SMOKE_RUNS, by contrast, must differ per harness: the two
# harnesses share config names (both have an "A0" and an "A3" cell, by
# design -- see the line-graph-matrix harness's PLAN.md section 2), so
# pointing them at the same run root would let one harness's checkpoints
# silently collide with the other's, which is exactly what each harness's
# own env.sh keeps separate on every site this has run on so far. Both
# harnesses export HARNESS_ID (and DATASET) before sourcing this file, so
# the branch below reaches the right one without this file needing to know
# which harness is asking.
case "\${HARNESS_ID:-}" in
    angle-ablation)
        export CSP_RUNS_BASE="\${CSP_RUNS_BASE:-$WORKSPACE/runs/angle-ablation}"
        export SMOKE_RUNS="\${SMOKE_RUNS:-$WORKSPACE/smoke/angle-ablation/\$DATASET}"
        ;;
    lgmatrix)
        export CSP_RUNS_BASE="\${CSP_RUNS_BASE:-$WORKSPACE/runs/lgmatrix}"
        export SMOKE_RUNS="\${SMOKE_RUNS:-$WORKSPACE/smoke/lgmatrix/\$DATASET}"
        # Only the line-graph-matrix harness has this concept: a data store
        # shared across its own datasets but not with the sibling harness's.
        export DATA_STORE="\${DATA_STORE:-$WORKSPACE/data/\$DATASET}"
        ;;
    *)
        echo "site.env: unrecognised HARNESS_ID '\${HARNESS_ID:-<unset>}'" \
             "-- CSP_RUNS_BASE and SMOKE_RUNS left at each env.sh's own" \
             "hard-coded default. If this is a new harness, add a case" \
             "here too." >&2
        ;;
esac

# ---- NOT filled in: this installer has no SLURM in it and cannot know ------
# ---- these for your site. Discover each with the command shown, then ------
# ---- uncomment and set it.                                            ------
#
# The account this job should be billed to, if your site's scheduler
# requires one. Leave unset (as it is here) if it does not -- many small
# clusters run no accounting associations at all, in which case an empty
# CSP_ACCOUNT is correct, not a placeholder.
#   discover with: sacctmgr -n show assoc where user=\$USER format=Account
# export CSP_ACCOUNT=""

# The GPU partition/queue name.
#   discover with: sinfo -o "%P %G"   (look for a partition whose %G lists a gpu)
# export CSP_PARTITION="gpu"

# The --gres string your scheduler expects. Some sites register a typed
# resource (gpu:a100:1), others a bare count (gpu:1); using the wrong form
# is rejected outright rather than silently ignored.
#   discover with: sinfo -o "%P %G" on the partition above
# export CSP_GPU_GRES="gpu:1"

# A substring asserted against \`nvidia-smi --query-gpu=name\`, so a job that
# lands on the wrong GPU model fails loudly instead of silently poisoning a
# timing comparison between arms. Leave it matching nothing in particular
# (a permissive default) if this site has only one GPU model.
# export REQUIRE_GPU_NAME=""

# A pattern that, if it appears in the GPU name, fails the job: guards
# against landing on a partitioned/MIG slice instead of a whole GPU, which
# would silently halve throughput and invalidate any timing comparison.
# export FORBID_GPU_PATTERN="MIG|[0-9]g\\.[0-9]*gb"

# How many array elements may run at once. Set this to (GPUs available) /
# (GPUs per element) -- both harnesses request one GPU per element, so this
# is just the total GPU count unless you are sharing the allocation.
# export CSP_MAX_CONCURRENT="2"

# Cores and memory per array element.
# export CPUS_PER_TASK="16"
# export MEM_PER_TASK="96G"
# export RELAX_WORKERS="16"

# Walltimes. Every harness treats these as placeholders until its own
# 20_pilot.sh measures a real number on this hardware -- do not trust a
# figure carried over from a different site's GPU. See either harness's
# PLAN.md section on cost.
# export PILOT_TIME="08:00:00"
# export PHASE_TIME="48:00:00"     # (the line-graph-matrix harness)
# export PHASE4_TIME="48:00:00"    # (the angular-diffusion harness)
# export MECH_TIME="12:00:00"
SITEENV
        echo "  wrote $SITE_ENV"
    fi
fi

say "done"
cat <<NEXT
next steps:
  1. Fill in the SLURM/GPU section of site.env -- see the comments in it for
     exactly which sinfo/sacctmgr commands answer each one. Nothing else in
     either harness needs editing.
  2. Pick a harness:
       supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/       (8-arm angular-diffusion suite)
       supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/  (line graph x angle diffusion, 2x2)
     and read its README.md and PLAN.md.
  3. cd into it and run its own 00_setup.sh, then preflight.sh, then
     10_smoke.sh -- each refuses to proceed on anything unresolved.
NEXT
