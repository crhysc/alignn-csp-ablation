#!/usr/bin/env bash
# Single source of truth for the line-graph x bond-angle-diffusion 2x2.
#
#   source env.sh                      # get the environment (DATASET=jarvis)
#   DATASET=alex source env.sh         # ... for Alexandria DS-A/B
#   bash env.sh --write                # also regenerate task_runners/cluster.env
#
# Nothing else in this harness hard-codes a path, an account or a partition.
# If a value is wrong, it is wrong here and only here.
#
# Site: atomgptlab (JHU WSE).  The scheduler values below were measured on this
# cluster on 2026-09-02 and are carried over from the sibling harness
# ../supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/env.sh, where they
# were established.  See PLAN.md section "Site".

# ---------------------------------------------------------------------------
# Which benchmark
# ---------------------------------------------------------------------------
# The two benchmarks are kept in completely separate trees -- separate run
# roots, separate results roots, separate config-name prefixes -- so neither
# can overwrite or be confused with the other.
#
#   jarvis   JARVIS Supercon-3D      847 / 105 / 103    3 seeds
#   alex     Alexandria DS-A/DS-B   6603 / 825 / 825    1 seed
#
export DATASET="${DATASET:-jarvis}"

case "$DATASET" in
    jarvis)
        export SPLIT="jarvis"
        export DATA_TASK="data-jarvis"
        # Overridable: MATRIX_TASK=lg-angle-state-matrix selects the
        # angular-STATE 2x2 (B3) over the derived-target one (A3).
        export MATRIX_TASK="${MATRIX_TASK:-lg-angle-matrix}"
        # ONE seed, deliberately traded for cost.  On 103 targets the seed
        # spread is the dominant source of variation -- the ALIGNN README
        # records match rate from 0.437 to 0.524 across fifteen models on
        # this split -- so at one seed, match rate / RMSD / ccRMSD / lattice
        # MAE / KLD on this dataset are single points, not means with a
        # spread, and a difference between cells cannot be distinguished
        # from that seed-to-seed noise. The denoising loss is not subject to
        # this: it is a measured quantity, not a discrete match count, and
        # stays the metric this run can actually resolve on JARVIS. Restore
        # "0,1,2" (or pass --seeds 0,1,2) if the AtomBench-metric columns
        # need to mean something on this split.
        export DEFAULT_SEEDS="0"
        export N_TRAIN=847
        export N_TEST=103
        export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3000}"
        ;;
    alex)
        export SPLIT="alex"
        export DATA_TASK="data-alex"
        # Overridable: MATRIX_TASK=lg-angle-state-matrix-alex selects the
        # angular-STATE 2x2 (B3) over the derived-target one (A3).
        export MATRIX_TASK="${MATRIX_TASK:-lg-angle-matrix-alex}"
        # ONE seed, deliberately.  825 test targets against JARVIS's 103 is an
        # 8x increase in generation cost, and generation -- not training --
        # dominates this benchmark.  The power this split buys comes from the
        # test set, not from seed replication.
        export DEFAULT_SEEDS="0"
        export N_TRAIN=6603
        export N_TEST=825
        # NOT pinned by the manuscript.  Measured at 4.26-4.38 s/epoch on GB10
        # (~1.2 h for 1000 epochs), so training is cheap here and there is no
        # reason to cut it.  Do NOT "fix" the overfitting seen from epoch ~175
        # by early-stopping: the one-cycle schedule is defined over --epochs,
        # so truncating it means the model never sees the anneal.  Shorten
        # TRAIN_EPOCHS if you want a shorter run, so the schedule completes.
        export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
        ;;
    *)
        echo "env.sh: unknown DATASET '$DATASET' (want: jarvis | alex)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HARNESS_ID="lgmatrix"          # stamped into cluster.env, see preflight

# Optional portable-installer output: install.sh (repo root) writes
# <repo-root>/site.env with the paths it determined and a template for the
# SLURM/GPU knobs it cannot know. Sourced here, before any of the
# ${VAR:-default} lines below, so setting a value in site.env is exactly
# equivalent to exporting it before sourcing this file -- neither this file
# nor its sibling harness's env.sh needs editing to move to a new cluster.
SITE_ENV_CANDIDATE="$(cd "$HARNESS/.." && pwd)/site.env"
[ -f "$SITE_ENV_CANDIDATE" ] && source "$SITE_ENV_CANDIDATE"
export ALIGNN_REPO="${ALIGNN_REPO:-$(cd "$HARNESS/.." && pwd)/alignn}"

# There is no /scratch on this cluster.  /data is the 430 TB NFS share and the
# only large filesystem mounted on both the x86_64 CPU nodes and the aarch64
# GPU nodes; /local-fast exists on the CPU nodes ONLY and is invisible from the
# GPU nodes, so nothing durable may live there.
#
# A run root of this experiment's own.  The sibling angular-diffusion suite
# lives under alignn_csp/ and shares two config names with this matrix
# (<split>_A0 and <split>_A3); a shared root would let the two experiments
# resume into each other's checkpoints, and they train with different
# checkpoint-selection criteria, so those checkpoints are NOT interchangeable.
export CSP_RUNS_BASE="${CSP_RUNS_BASE:-/data/ccamp104/alignn_csp_lgmatrix}"
export CSP_RUNS="${CSP_RUNS:-$CSP_RUNS_BASE/$DATASET}"

# The prepared splits, however, ARE shared: they are a pure function of the
# public databases and the preparation script, they are expensive (the
# Alexandria prep is already done and verified at 6603/825/825, 0 fallbacks),
# and re-deriving them per experiment would only create an opportunity for two
# benchmarks to disagree about what the test set is.  So `data` is a symlink
# into one store, created by 00_setup.sh.
export DATA_STORE="${DATA_STORE:-/data/ccamp104/alignn_csp/$DATASET/data}"

export ATOMBENCH_REPO="${ATOMBENCH_REPO:-/data/ccamp104/atombench}"
export SMOKE_RUNS="${SMOKE_RUNS:-/data/ccamp104/csp_smoke_lgmatrix/$DATASET}"

# Both environments are aarch64 builds and are therefore runnable ONLY on the
# GPU nodes (atomgptlab03/04).  Built by the sibling harness's 00_setup.sh and
# reused here rather than rebuilt.
export TRAIN_ENV="${TRAIN_ENV:-/data/ccamp104/envs/alignn2}"
export SCORE_ENV_PATH="${SCORE_ENV_PATH:-/data/ccamp104/envs/csp-score}"
export CONDA_FORGE_PREFIX="${CONDA_FORGE_PREFIX:-/data/ccamp104/envs/miniforge3}"

# Data preparation (prepare_data.py / prepare_alex_data.py) needs pymatgen,
# jarvis-tools, numpy, pandas and tqdm -- pure Python/numpy, no GPU and no
# CUDA anywhere in it.  It was previously routed through SCORE_ENV_PATH
# (aarch64) purely because that is where those packages happened to already
# be installed, which forced it through the on_env_arch srun bridge to reach
# the GPU nodes -- competing for the same congested queue as actual training,
# for work that never touches a GPU.  DATA_ENV is a plain x86_64 build,
# runnable directly on THIS login node, so data prep never touches SLURM at
# all: see run_task_local() below.  SLURM is reserved for what actually needs
# a GPU -- training and generation.
export DATA_ENV="${DATA_ENV:-/data/ccamp104/envs/csp-data-x86}"

# PINNED, not derived from today's date: a full run spans several days, and a
# date-derived RUN_ID would silently split one benchmark across two results
# trees depending on which day each script happened to be invoked.
export RUN_ID="${RUN_ID:-2026-09-02_lgmatrix}"
export RESULTS="$HARNESS/results/$DATASET/$RUN_ID"

# ---------------------------------------------------------------------------
# Scheduler  (measured on atomgptlab, 2026-09-02)
# ---------------------------------------------------------------------------
# This cluster runs no accounting associations for this user and every
# partition is AllowAccounts=ALL / AllowQos=ALL, so --account and --qos are
# omitted entirely.  submit.sh drops empty values rather than passing blanks.
export CSP_ACCOUNT="${CSP_ACCOUNT:-}"
export CSP_QOS=""
export CSP_CONSTRAINT=""          # MUST stay empty: no node features defined
export CSP_RESERVATION=""
export CSP_MAIL_USER="${CSP_MAIL_USER:-}"
export CSP_SBATCH_EXTRA="${CSP_SBATCH_EXTRA:-}"

# Partitions:
#   main   atomgptlab01-02   x86_64, 256 cores, 500 GB, NO GPU
#   gpu    atomgptlab03-04   aarch64, 20 cores, 110 GB, 1x NVIDIA GB10
# Every stage needs the aarch64 environments, so everything -- including the
# CPU-only data preparation -- goes to `gpu`.  The data task declares no --gres
# and so does not hold a GPU while it runs.
export CSP_PARTITION="${CSP_PARTITION:-gpu}"
export PART_DEBUG="gpu"
export PART_PILOT="gpu"
export PART_FULL="$CSP_PARTITION"

# Generic gres: the GPU nodes register plain `gpu:1`, with no type name, so
# `gpu:nvidia_gb10:1`-style typed requests are rejected here.
export CSP_GPU_GRES="${CSP_GPU_GRES:-gpu:1}"

# A tripwire against silently landing on different silicon, not a real choice:
# both GPUs in the cluster are GB10.  GB10 does not support MIG, but the
# pattern is kept so the assertion still reads "a whole GPU".
export REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-GB10}"
export FORBID_GPU_PATTERN="${FORBID_GPU_PATTERN:-MIG|[0-9]g\.[0-9]*gb}"

# THE binding constraint on this cluster: two GPUs, one per node.  A four-arm
# matrix therefore runs two deep -- two rounds on JARVIS's 3 seeds (12
# elements = 6 rounds), one round of four on Alexandria (2 rounds).
export CSP_MAX_CONCURRENT="${CSP_MAX_CONCURRENT:-2}"

# ---------------------------------------------------------------------------
# Resources per array element
# ---------------------------------------------------------------------------
# A GPU node is 20 cores / 110 GB / 1 GPU, so an element that holds the GPU
# holds the node.  16 of 20 cores leaves the node's services room and still
# gives the ALIGNN-FF relaxation plenty of workers.
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEM_PER_TASK="${MEM_PER_TASK:-96G}"
export RELAX_WORKERS="${RELAX_WORKERS:-16}"

# Walltimes.  Every partition here is MaxTime=UNLIMITED, so a generous request
# costs nothing in queue priority -- and the runner resumes at *stage*
# granularity, so a walltime kill restarts a training from zero.
#
# PHASE_TIME is a PLACEHOLDER until 20_pilot.sh price reports a real number.
# Training is measured and cheap (4.3 s/epoch on Alexandria, ~1.2 h for 1000
# epochs).  Generation is neither: 825 targets x 32 candidates x 1000 denoising
# steps plus ALIGNN-FF relaxation, on a memory-bandwidth-bound GB10, has never
# been measured end to end.  It is the number that decides whether this suite
# is a two-day or a two-week job.  Measure it before committing all the arms.
export PILOT_TIME="${PILOT_TIME:-08:00:00}"
export PHASE_TIME="${PHASE_TIME:-48:00:00}"
export MECH_TIME="${MECH_TIME:-12:00:00}"

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
export SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# Checkpoint selection.  The trained objective is
#     L = 1.0*L_lattice + 10.0*L_frac + 1.0*L_angle ,
# and L_angle exists only in two of the four cells.  Selecting best_model.pt on
# that total would select two arms on a different criterion from the other two.
# "structural" is L_lattice + 10*L_frac -- the part all four optimise -- and is
# both the selection criterion and the denoising loss this experiment reports.
# lg-angle-matrix declares it in tasks.py; it is repeated here because the
# hand-written sbatch paths (10_smoke.sh, 20_pilot.sh) do not go through the
# task table.
export SELECT_ON="${SELECT_ON:-structural}"

# Basis-relabelling augmentation.  OFF, and not as a small-data heuristic: the
# prepared splits are stored in AtomBench's primitive+Niggli scoring basis
# (canonical_cell() in scripts/atombench/prepare_*_data.py), so there is
# exactly one correct basis labelling and the 48 signed permutations would
# relabel away from the basis compute_metrics.py measures in.
export AUGMENT="${AUGMENT:-0}"

# Symmetrisation tolerance.  0.1 is the runner default.  It does NOT transfer
# across datasets and it must be chosen on the VALIDATION split -- never on
# test.  30_full.sh symprec/choose does that.
export SYMPREC="${SYMPREC:-0.1}"

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
export ALIGNN_RUNS="$CSP_RUNS"          # what run_task.py reads
export PYTHONUNBUFFERED=1

sbatch_account_line() {
    if [ -n "${CSP_ACCOUNT:-}" ]; then
        echo "#SBATCH --account=$CSP_ACCOUNT"
    else
        echo "# no --account: this cluster runs no accounting associations"
    fi
}
export -f sbatch_account_line 2>/dev/null || true

# --- architecture bridge ---------------------------------------------------
# This cluster is heterogeneous: the login and `main` nodes are x86_64, the GPU
# nodes are aarch64, and both conda environments are aarch64 builds.  Invoking
# an env binary from the login node fails outright ("cannot execute binary
# file"), which would break every preflight check, the symprec choice, the
# pilot measurement and the whole analysis chain.
#
# on_env_arch runs a command where it can actually execute: directly when the
# architecture already matches (inside a job on a GPU node), otherwise through
# a small srun on the GPU partition.  The srun requests NO --gres, so it uses
# spare cores beside a running training rather than queueing behind a GPU.
# stdin is forwarded by srun, so heredoc-driven callers keep working.
export ENV_ARCH="${ENV_ARCH:-aarch64}"
export ENV_SRUN_CPUS="${ENV_SRUN_CPUS:-2}"
export ENV_SRUN_MEM="${ENV_SRUN_MEM:-8G}"
export ENV_SRUN_TIME="${ENV_SRUN_TIME:-01:00:00}"

on_env_arch() {
    if [ "$(uname -m)" = "$ENV_ARCH" ]; then
        "$@"
    else
        srun --quiet --partition="$CSP_PARTITION" --nodes=1 \
             --cpus-per-task="$ENV_SRUN_CPUS" --mem="$ENV_SRUN_MEM" \
             --time="$ENV_SRUN_TIME" "$@"
    fi
}

train_py()  { on_env_arch "$TRAIN_ENV/bin/python" "$@"; }
score_py()  { on_env_arch "$SCORE_ENV_PATH/bin/python" "$@"; }
score_bin() { local b="$1"; shift; on_env_arch "$SCORE_ENV_PATH/bin/$b" "$@"; }
export -f on_env_arch train_py score_py score_bin 2>/dev/null || true

# The submit host is x86_64 and both conda environments are aarch64, so NEITHER
# can run here.  submit.sh only needs to import tasks.py to size the array, and
# tasks.py is stdlib plus alignn.inverse.ablations (a pure typing/dict module),
# so the system python3 does the job given the repo on PYTHONPATH.
export CSP_SUBMIT_PYTHON="${CSP_SUBMIT_PYTHON:-python3}"

# ---------------------------------------------------------------------------
# cluster.env generation
# ---------------------------------------------------------------------------
# task_runners/cluster.env is the one file the ALIGNN repo designates as
# site-local.  It is generated rather than hand-edited, so the configuration is
# reproducible from this directory alone, and collect.py archives a copy.
#
# It is also the one file this harness SHARES with its sibling: both write the
# same path, and the two point CSP_RUNS at different roots.  The HARNESS stamp
# below is what lets preflight.sh notice that the file on disk belongs to the
# other experiment instead of quietly submitting into the wrong tree.
write_cluster_env() {
    local dest="$ALIGNN_REPO/task_runners/cluster.env"
    cat > "$dest" <<CLUSTERENV
# GENERATED by $HARNESS/env.sh on $(date -Is) -- do not hand-edit.
# Edit env.sh and re-run \`bash env.sh --write\`.
# HARNESS=$HARNESS_ID
# DATASET=$DATASET
CSP_ACCOUNT="$CSP_ACCOUNT"
CSP_PARTITION="$CSP_PARTITION"
CSP_QOS=""
CSP_CONSTRAINT=""
CSP_GPU_GRES="$CSP_GPU_GRES"
CSP_RESERVATION=""
CSP_MAIL_USER="$CSP_MAIL_USER"
CSP_MAX_CONCURRENT="$CSP_MAX_CONCURRENT"
CSP_SBATCH_EXTRA="$CSP_SBATCH_EXTRA"

CSP_MODULES=""
CSP_ENV="$TRAIN_ENV"
CSP_SCORE_ENV="$SCORE_ENV_PATH"
CSP_PRE_RUN_HOOK="source $HARNESS/lib/gpu_sampler.sh"

CSP_RUNS="$CSP_RUNS"
CSP_ATOMBENCH_REPO="$ATOMBENCH_REPO"
CLUSTERENV
    echo "wrote $dest  (HARNESS=$HARNESS_ID DATASET=$DATASET)"
}

# Run run_task.py against an environment.
#
# tasks.py builds every stage argv starting with the bare string "python", so
# each stage resolves the interpreter from PATH -- not from whichever python
# launched run_task.py.  Inside a job that is fine, because common.sh has
# already put CSP_ENV on PATH.  Outside one it is not.  Putting the env on PATH
# is what makes the two agree.
#
# NOTE: both environments are aarch64, so these only work from a GPU node --
# from inside a job, or an `srun -p gpu` shell.  From the login node they fail
# with "cannot execute binary file".  Use the sbatch path instead.
run_task_in() {
    local env="$1"; shift
    ( cd "$ALIGNN_REPO" \
      && PATH="$env/bin:$PATH" \
         ALIGNN_RUNS="$CSP_RUNS" \
         SCORE_ENV="$SCORE_ENV_PATH" \
         ATOMBENCH_REPO="$ATOMBENCH_REPO" \
         on_env_arch "$env/bin/python" task_runners/run_task.py "$@" )
}

run_task() { run_task_in "$TRAIN_ENV" "$@"; }

# The data tasks are pure crystallography -- prepare_data.py imports
# jarvis-tools, pymatgen and numpy, and no torch at all -- so they run in the
# scoring environment, where those live.  This is also why `doctor` reports
# pymatgen missing: it probes the training environment, and by design it is not
# there.
#
# NOT used for data-jarvis / data-alex any more -- see run_task_local() below.
# Kept for anything else that still needs the real (aarch64) scoring env from
# a GPU node, e.g. preflight.sh's canonicalisation check.
run_task_data() { run_task_in "$SCORE_ENV_PATH" "$@"; }

# Data preparation, run directly on THIS (x86_64) login node -- no
# on_env_arch, no srun, no SLURM at all.  DATA_ENV is architecture-native
# here, unlike TRAIN_ENV/SCORE_ENV_PATH, so there is nothing to bridge: the
# whole reason run_task_in() exists is that those two aarch64 environments
# cannot execute on this node, and that reason does not apply to a plain
# x86_64 build.  Reserve the scheduler for what actually needs a GPU.
run_task_local() {
    ( cd "$ALIGNN_REPO" \
      && PATH="$DATA_ENV/bin:$PATH" \
         ALIGNN_RUNS="$CSP_RUNS" \
         "$DATA_ENV/bin/python" task_runners/run_task.py "$@" )
}

# Regenerate cluster.env for one phase and submit through the repo's own
# submit.sh.  Partition and walltime have to travel through cluster.env
# (submit.sh sources it *after* the caller's environment, so an exported
# CSP_PARTITION would be overwritten) -- except CSP_SBATCH_EXTRA, which
# submit.sh appends to the sbatch command line, where flags beat the #SBATCH
# directives inside the .sbatch file.
csp_submit() {
    local partition="$1" walltime="$2" task="$3"; shift 3
    CSP_PARTITION="$partition" \
    CSP_SBATCH_EXTRA="--time=$walltime --cpus-per-task=$CPUS_PER_TASK --mem=$MEM_PER_TASK ${CSP_SBATCH_EXTRA:-}" \
        bash "$HARNESS/env.sh" --write >/dev/null
    ( cd "$ALIGNN_REPO" \
      && PYTHONPATH="$ALIGNN_REPO${PYTHONPATH:+:$PYTHONPATH}" \
         CSP_SUBMIT_PYTHON="$CSP_SUBMIT_PYTHON" \
         bash task_runners/submit.sh "$task" "$@" )
}

# Where the GPU sampler drops its traces.  Per (jobid, array index), because
# the hook runs before run_task.py has resolved which unit it is; collect.py
# maps them back through `run_task.py <task> --list`.
export GPU_TRACE_DIR="$RESULTS/_gputrace"

if [ "${1:-}" = "--write" ]; then
    mkdir -p "$RESULTS" "$GPU_TRACE_DIR"
    write_cluster_env
fi
