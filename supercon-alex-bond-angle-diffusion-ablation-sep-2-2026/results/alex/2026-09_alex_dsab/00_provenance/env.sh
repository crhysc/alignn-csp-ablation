#!/usr/bin/env bash
# Single source of truth for the angular-diffusion ablation benchmark.
#
#   source env.sh                      # get the environment (DATASET=alex)
#   DATASET=jarvis source env.sh       # ... for the Supercon-3D benchmark
#   bash env.sh --write                # also regenerate task_runners/cluster.env
#
# Nothing else in this harness hard-codes a path, an account or a partition.
# If a value is wrong, it is wrong here and only here.
#
# Site: atomgptlab (JHU WSE).  Every scheduler value below was measured on
# this cluster on 2026-09-02; see PLAN.md and section "Site" at the bottom.

# ---------------------------------------------------------------------------
# Which benchmark
# ---------------------------------------------------------------------------
# The two benchmarks are kept in completely separate trees -- separate run
# roots, separate results roots, separate config-name prefixes -- so neither
# can overwrite or be confused with the other.  Everything downstream derives
# from this one variable.
#
#   jarvis   JARVIS Supercon-3D      847 / 105 / 103    3 seeds
#   alex     Alexandria DS-A/DS-B   6603 / 825 / 825    1 seed
#
export DATASET="${DATASET:-alex}"

case "$DATASET" in
    jarvis)
        export SPLIT="jarvis"
        export DATA_TASK="data-jarvis"
        export ABLATION_TASK="angle-ablation"
        export LINEGRAPH_TASK="ablation-linegraph"
        # Three seeds was the old budget; it is what the first full run used.
        export DEFAULT_SEEDS="0,1,2"
        export N_TRAIN=847
        export N_TEST=103
        ;;
    alex)
        export SPLIT="alex"
        export DATA_TASK="data-alex"
        export ABLATION_TASK="angle-ablation-alex"
        export LINEGRAPH_TASK="ablation-linegraph-alex"
        # ONE seed, deliberately.  825 test targets against JARVIS's 103 is an
        # 8x increase in generation cost, and generation -- not training --
        # dominates this benchmark.  The power this run buys comes from the
        # test set, not from seed replication: the paired per-target tests
        # (McNemar / CMH / Wilcoxon) are the powered ones and they scale with
        # targets.  See PLAN.md and HANDOFF.md section 2.
        export DEFAULT_SEEDS="0"
        export N_TRAIN=6603
        export N_TEST=825
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
# Stamped into the generated cluster.env.  The sibling harness
# ../supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/ writes
# the SAME file and points CSP_RUNS at a different run root, so without a stamp
# a hand-run `run_task` could put this suite's checkpoints in that experiment's
# tree.  preflight.sh checks it.
export HARNESS_ID="angle-ablation"

# Optional portable-installer output: install.sh (repo root) writes
# <repo-root>/site.env with the paths it determined and a template for the
# SLURM/GPU knobs it cannot know. Sourced here, before any of the
# ${VAR:-default} lines below, so setting a value in site.env is exactly
# equivalent to exporting it before sourcing this file -- neither this file
# nor its sibling harness's env.sh needs editing to move to a new cluster.
SITE_ENV_CANDIDATE="$(cd "$HARNESS/.." && pwd)/site.env"
[ -f "$SITE_ENV_CANDIDATE" ] && source "$SITE_ENV_CANDIDATE"
# Resolved relative to this file, not hard-coded: the harness and the ALIGNN
# checkout are siblings in the workspace repo, so a clone on any cluster finds
# it without editing.  Override ALIGNN_REPO if yours lives elsewhere.
export ALIGNN_REPO="${ALIGNN_REPO:-$(cd "$HARNESS/.." && pwd)/alignn}"

# There is no /scratch on this cluster.  /data is the 430 TB NFS share and is
# the only large filesystem mounted on both the x86_64 CPU nodes and the
# aarch64 GPU nodes; /local-fast exists on the CPU nodes ONLY and is invisible
# from the GPU nodes, so nothing durable may live there.
export CSP_RUNS_BASE="${CSP_RUNS_BASE:-/data/ccamp104/alignn_csp}"
# One run root per benchmark.  The config-name prefixes (jarvis_A0, alex_A0)
# would already keep them apart inside a shared tree; separate roots make it
# impossible to get wrong, and let one benchmark be archived or deleted
# without touching the other.
export CSP_RUNS="${CSP_RUNS:-$CSP_RUNS_BASE/$DATASET}"

export ATOMBENCH_REPO="${ATOMBENCH_REPO:-/data/ccamp104/atombench}"
export SMOKE_RUNS="${SMOKE_RUNS:-/data/ccamp104/csp_smoke/$DATASET}"

# Both environments are aarch64 builds and are therefore runnable ONLY on the
# GPU nodes (atomgptlab03/04).  They live under /data because /home is not
# large enough for two torch installs and, more importantly, because putting
# them beside the data keeps one mount in play.  See "Site" below.
export TRAIN_ENV="${TRAIN_ENV:-/data/ccamp104/envs/alignn2}"
export SCORE_ENV_PATH="${SCORE_ENV_PATH:-/data/ccamp104/envs/csp-score}"
# The aarch64 conda that built them.  Not usable from the login node.
export CONDA_FORGE_PREFIX="${CONDA_FORGE_PREFIX:-/data/ccamp104/envs/miniforge3}"

# RUN_ID namespaces the whole results tree, so a re-run never clobbers a
# previous one.  Same discipline as the runner's own _quick/_smoke suffixes.
#
# PINNED, not derived from today's date: a phase-4 run spans several days, and
# a date-derived RUN_ID would silently split one benchmark across two results
# trees depending on which day each script happened to be invoked.  Change it
# deliberately when starting a genuinely new run.
if [ "$DATASET" = "jarvis" ]; then
    export RUN_ID="${RUN_ID:-2026-08-30_full}"
else
    export RUN_ID="${RUN_ID:-2026-09_alex_dsab}"
fi
# results/<dataset>/<run id>/ -- the separation the two benchmarks need.
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
#   all    all four
# Every stage of this benchmark needs the aarch64 environments, so everything
# -- including the CPU-only data preparation -- goes to `gpu`.  The data task
# declares no --gres and so does not hold a GPU while it runs.
export CSP_PARTITION="${CSP_PARTITION:-gpu}"
export PART_DEBUG="gpu"
export PART_INTER="gpu"
export PART_PILOT="gpu"
export PART_FULL="$CSP_PARTITION"

# Generic gres: the GPU nodes register plain `gpu:1`, with no type name, so
# `gpu:nvidia_a30:1`-style typed requests are rejected here.
export CSP_GPU_GRES="${CSP_GPU_GRES:-gpu:1}"

# The GPU every element must actually receive.  There are only two in the
# cluster and both are GB10, so this is a tripwire against silently landing on
# something else rather than a real choice.  GB10 does not support MIG, but the
# pattern is kept so the assertion still means "a whole GPU".
export REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-GB10}"
export FORBID_GPU_PATTERN="${FORBID_GPU_PATTERN:-MIG|[0-9]g\.[0-9]*gb}"

# THE binding constraint on this cluster: two GPUs, one per node.  The old
# site had 20 A30s and throttled at 8.  Here an 8-arm suite runs four deep.
export CSP_MAX_CONCURRENT="${CSP_MAX_CONCURRENT:-2}"

# ---------------------------------------------------------------------------
# Resources per array element
# ---------------------------------------------------------------------------
# A GPU node is 20 cores / 110 GB / 1 GPU, so an element that holds the GPU
# holds the node.  Taking 16 of 20 cores leaves the node's own services room
# and still gives generation plenty of relaxation workers.
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEM_PER_TASK="${MEM_PER_TASK:-96G}"
export RELAX_WORKERS="${RELAX_WORKERS:-16}"

# Walltimes.  Every partition here is MaxTime=UNLIMITED, so a generous request
# costs nothing in queue priority -- and the runner resumes at *stage*
# granularity, so a walltime kill restarts a training from zero.  Ask for more
# than you need.
#
# PHASE4_TIME is UNMEASURED on this hardware and must stay a placeholder until
# 20_pilot.sh measure reports a real number.  Two things changed at once:
#   * the split is 8x larger on the test side (825 targets vs 103), and
#     generation -- not training -- was already the dominant cost;
#   * the GPU changed from A30 (HBM2, 933 GB/s) to GB10 (LPDDR5X, ~273 GB/s).
#     ALIGNN is a message-passing model and is memory-bandwidth-bound, so the
#     new GPU is NOT safely assumed faster despite being a newer architecture.
# Do not copy the old 12:00:00 forward as if it meant anything here.
export PILOT_TIME="${PILOT_TIME:-08:00:00}"
export PHASE4_TIME="${PHASE4_TIME:-48:00:00}"
export MECH_TIME="${MECH_TIME:-12:00:00}"

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
export SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# The confound arm from the README: smooth *triplet* topology with A0's
# ungated dense pair channel.  Separates "smoothness helps" from "truncating
# the pair range helps".  See PLAN.md section 12.
export RUN_CONFOUND_ARM="${RUN_CONFOUND_ARM:-1}"
export CONFOUND_CONFIG="${SPLIT}_A3_nogate"

# Epochs for the confound arm's hand-written sbatch, which cannot go through
# tasks.py.  Must track EPOCHS[<split>] in task_runners/tasks.py or the
# confound arm is not a control.
if [ "$DATASET" = "jarvis" ]; then
    export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3000}"
else
    # tasks.py flags this as "not pinned by the manuscript".  best_model.pt is
    # selected on validation loss with no early stopping, so this is a search
    # budget rather than a stopping point: overshooting costs GPU hours,
    # undershooting undertrains every arm equally.  Check where the best-val
    # epoch actually lands in history.json after the pilot before trusting it.
    export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
fi

# Basis-relabelling augmentation.  OFF, and not as a small-data heuristic:
# the prepared splits are now stored in AtomBench's primitive+Niggli scoring
# basis (canonical_cell() in scripts/atombench/prepare_*_data.py), so there is
# exactly one correct basis labelling and the 48 signed permutations would
# relabel away from the basis compute_metrics.py measures in.
export AUGMENT="${AUGMENT:-0}"

# Symmetrisation tolerance.  0.1 is the runner default; symprec-sweep chooses
# the real one on the validation split and 30_full.sh reads it back before the
# pipeline runs.  It does NOT transfer across datasets -- rerun the sweep.
export SYMPREC="${SYMPREC:-0.1}"

# Arms that pipeline-ablation is run for, each in its own runs-root (the
# rundir does not contain the checkpoint, so a shared root would collide).
export PIPELINE_ARMS="${PIPELINE_ARMS:-A0 A3}"

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
export ALIGNN_RUNS="$CSP_RUNS"          # what run_task.py reads
export PYTHONUNBUFFERED=1

# Emit a #SBATCH --account directive only when an account is configured.
# This cluster runs no accounting associations for this user, and a bare
# `#SBATCH --account=` is a hard sbatch error rather than a no-op -- so the
# hand-written sbatch scripts have to ask rather than interpolate blindly.
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
# an env binary from the login node therefore fails outright ("cannot execute
# binary file"), which would break every preflight check, the symprec choice,
# the pilot measurement and the whole analysis chain.
#
# on_env_arch runs a command where it can actually execute: directly when the
# architecture already matches (i.e. inside a job on a GPU node), otherwise
# through a small srun on the GPU partition.  The srun requests NO --gres, so
# it uses spare cores beside a running training rather than queueing behind a
# GPU -- with 16 of 20 cores taken by an element there are always a few free.
# stdin is forwarded by srun, so the heredoc-driven callers keep working.
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

# The submit host is x86_64 and the two conda environments are aarch64, so
# NEITHER of them can run here.  submit.sh only needs to import tasks.py to
# size the array, and tasks.py is stdlib plus alignn.inverse.ablations (a pure
# typing/dict module) -- so the system python3 does the job, given the repo on
# PYTHONPATH.  This is why submit.sh takes CSP_SUBMIT_PYTHON.
export CSP_SUBMIT_PYTHON="${CSP_SUBMIT_PYTHON:-python3}"

# ---------------------------------------------------------------------------
# cluster.env generation
# ---------------------------------------------------------------------------
# task_runners/cluster.env is the one file the ALIGNN repo designates as
# site-local.  It is generated from the values above rather than hand-edited,
# so the whole configuration is reproducible from this directory alone, and a
# copy is archived into the results tree by collect.py.
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

# Run run_task.py against the training environment.
#
# tasks.py builds every stage argv starting with the bare string "python", so
# each stage resolves the interpreter from PATH -- not from whichever python
# launched run_task.py.  Inside a job that is fine, because common.sh has
# already put CSP_ENV on PATH.  Outside one it is not.  Putting TRAIN_ENV on
# PATH is what makes the two agree.
#
# NOTE: on this cluster both environments are aarch64, so these helpers only
# work from a GPU node -- i.e. from inside a job, or an `srun -p gpu` shell.
# From the login node they will fail with "cannot execute binary file".  Use
# the sbatch path (csp_submit / the phase scripts) instead.
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
# scoring environment, where those live.  This is also why `doctor` shows
# pymatgen as missing: it probes the training environment, and by design it is
# not there.
run_task_data() { run_task_in "$SCORE_ENV_PATH" "$@"; }

# Regenerate cluster.env for one phase and submit through the repo's own
# submit.sh.  Partition and walltime have to travel through cluster.env
# (submit.sh sources it *after* the caller's environment, so an exported
# CSP_PARTITION would be overwritten) -- except CSP_SBATCH_EXTRA, which
# submit.sh appends to the sbatch command line, where flags beat the
# #SBATCH directives inside the .sbatch file.  That is the documented
# override path, and it is why the shipped headers need no editing.
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
# maps them back through \`run_task.py <task> --list\`.
export GPU_TRACE_DIR="$RESULTS/_gputrace"

if [ "${1:-}" = "--write" ]; then
    mkdir -p "$RESULTS" "$GPU_TRACE_DIR"
    write_cluster_env
fi
