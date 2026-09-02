#!/usr/bin/env bash
# Single source of truth for the angular-diffusion ablation benchmark.
#
#   source env.sh            # get the environment
#   bash env.sh --write      # also regenerate task_runners/cluster.env
#
# Nothing else in this harness hard-codes a path, an account or a partition.
# If a value is wrong, it is wrong here and only here.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolved relative to this file, not hard-coded: the harness and the ALIGNN
# checkout are siblings in the workspace repo, so a clone on any cluster finds
# it without editing.  Override ALIGNN_REPO if yours lives elsewhere.
export ALIGNN_REPO="${ALIGNN_REPO:-$(cd "$HARNESS/.." && pwd)/alignn}"
export CSP_RUNS="${CSP_RUNS:-/scratch/crc00042/alignn_csp}"
export ATOMBENCH_REPO="${ATOMBENCH_REPO:-/scratch/crc00042/atombench}"
export TRAIN_ENV="${TRAIN_ENV:-/scratch/crc00042/envs/alignn2}"
export SCORE_ENV_PATH="${SCORE_ENV_PATH:-/scratch/crc00042/envs/csp-ablation-bench}"
export SMOKE_RUNS="${SMOKE_RUNS:-/scratch/crc00042/csp_smoke}"

# RUN_ID namespaces the whole results tree, so a re-run never clobbers a
# previous one.  Same discipline as the runner's own _quick/_smoke suffixes.
#
# PINNED, not derived from today's date: a phase-4 run spans several days, and
# a date-derived RUN_ID would silently split one benchmark across two results
# trees depending on which day each script happened to be invoked.  Change it
# deliberately when starting a genuinely new run.
export RUN_ID="${RUN_ID:-2026-08-30_full}"
export RESULTS="$HARNESS/results/$RUN_ID"

# ---------------------------------------------------------------------------
# Scheduler  (measured on dollysods, 2026-08-29 -- see PLAN.md section 1)
# ---------------------------------------------------------------------------
export CSP_ACCOUNT="${CSP_ACCOUNT:-alromero}"
export CSP_PARTITION="${CSP_PARTITION:-gpu_7day}"
export CSP_QOS=""                 # only 'normal' exists; passing it is noise
export CSP_CONSTRAINT=""          # MUST stay empty: no node features defined
export CSP_GPU_GRES="${CSP_GPU_GRES:-gpu:nvidia_a30:1}"   # PINNED
export CSP_RESERVATION=""
export CSP_MAIL_USER="${CSP_MAIL_USER:-}"
export CSP_MAX_CONCURRENT="${CSP_MAX_CONCURRENT:-8}"      # cap is 20 A30
export CSP_SBATCH_EXTRA="${CSP_SBATCH_EXTRA:-}"

# Interactive/short partitions used by the earlier phases.
export PART_DEBUG="debug"          # 1 h
export PART_INTER="inter_a30"      # 6 h
export PART_PILOT="gpu_2day"       # 2 d
export PART_FULL="$CSP_PARTITION"  # 7 d

# The GPU every element must actually receive.  A MIG slice (a30_2g.12gb is
# registered in AccountingStorageTRES) would silently halve throughput and
# poison every timing comparison, so this is asserted, not hoped for.
export REQUIRE_GPU_NAME="${REQUIRE_GPU_NAME:-A30}"
export FORBID_GPU_PATTERN="${FORBID_GPU_PATTERN:-MIG|1g\.|2g\.|3g\.|4g\.}"

# ---------------------------------------------------------------------------
# Resources per array element
# ---------------------------------------------------------------------------
# A30/A40 nodes are 32 cores / 257 GB / 4 GPUs, so the fair share per GPU is
# 8 cores and ~60 GB.  The repo ships 16/64G, which halves node packing.
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM_PER_TASK="${MEM_PER_TASK:-56G}"
export RELAX_WORKERS="${RELAX_WORKERS:-8}"

# Walltimes.  PHASE4_TIME is a placeholder until 20_pilot.sh measures it --
# it prints the value to put here and refuses to guess on your behalf.
export PILOT_TIME="${PILOT_TIME:-06:00:00}"
# MEASURED on 2026-08-30 by the debug probe (job 140182), 40 epochs on the
# real split, slope over the second half so startup and warm-up are excluded:
#   A0  0.0321 s/step -> 0.38 h for 3000 epochs
#   A3  0.0393 s/step -> 0.46 h
# Training is therefore NOT the dominant cost; generation (103 targets x 32
# candidates x 1000 denoising steps, plus 412 ALIGNN-FF relaxations) is, and
# that half is not measured at full size.  12 h is deliberately generous: the
# runner resumes at stage granularity, so a walltime kill restarts a training
# from zero, and with 116 A30s against a throttle of 8 the extra request costs
# essentially no queue time.
export PHASE4_TIME="${PHASE4_TIME:-12:00:00}"
export MECH_TIME="${MECH_TIME:-08:00:00}"

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
export SEEDS="${SEEDS:-0,1,2}"

# The confound arm from the README: smooth *triplet* topology with A0's
# ungated dense pair channel.  Separates "smoothness helps" from "truncating
# the pair range helps".  +3 units, ~10% more compute.  See PLAN.md section 12.
export RUN_CONFOUND_ARM="${RUN_CONFOUND_ARM:-1}"
export CONFOUND_CONFIG="jarvis_A3_nogate"

# Symmetrisation tolerance.  0.1 is the runner default; symprec-sweep chooses
# the real one on the validation split and 30_full.sh reads it back from
# 50_costs/../symprec_choice.txt before the pipeline runs.
export SYMPREC="${SYMPREC:-0.1}"

# Arms that pipeline-ablation is run for, each in its own runs-root (the
# rundir does not contain the checkpoint, so a shared root would collide).
export PIPELINE_ARMS="${PIPELINE_ARMS:-A0 A3}"

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
export ALIGNN_RUNS="$CSP_RUNS"          # what run_task.py reads
export PYTHONUNBUFFERED=1

train_py()  { "$TRAIN_ENV/bin/python" "$@"; }
score_py()  { "$SCORE_ENV_PATH/bin/python" "$@"; }
export -f train_py score_py 2>/dev/null || true

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
    echo "wrote $dest"
}

# Run run_task.py against the training environment.
#
# tasks.py builds every stage argv starting with the bare string "python", so
# each stage resolves the interpreter from PATH -- not from whichever python
# launched run_task.py.  Inside a job that is fine, because common.sh has
# already `conda activate`d CSP_ENV.  Outside one it is not: calling
# "$TRAIN_ENV/bin/python task_runners/run_task.py" runs the *runner* in the
# right env and then every stage in whatever python happens to be first on
# PATH.  Putting TRAIN_ENV on PATH is what makes the two agree.
run_task_in() {
    local env="$1"; shift
    ( cd "$ALIGNN_REPO" \
      && PATH="$env/bin:$PATH" \
         ALIGNN_RUNS="$CSP_RUNS" \
         SCORE_ENV="$SCORE_ENV_PATH" \
         ATOMBENCH_REPO="$ATOMBENCH_REPO" \
         "$env/bin/python" task_runners/run_task.py "$@" )
}

run_task() { run_task_in "$TRAIN_ENV" "$@"; }

# The data tasks are pure crystallography -- prepare_data.py imports
# jarvis-tools, pymatgen and numpy, and no torch at all -- so they run in the
# scoring environment, where those live.  Putting pymatgen into the training
# environment just to build a split is the alternative, and it risks the numpy
# and pandas pins of a working torch install for no benefit.  This is also why
# `doctor` shows pymatgen as missing: it probes the training environment, and
# by design it is not there.
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
    ( cd "$ALIGNN_REPO" && bash task_runners/submit.sh "$task" "$@" )
}

# Where the GPU sampler drops its traces.  Per (jobid, array index), because
# the hook runs before run_task.py has resolved which unit it is; collect.py
# maps them back through \`run_task.py <task> --list\`.
export GPU_TRACE_DIR="$RESULTS/_gputrace"

if [ "${1:-}" = "--write" ]; then
    mkdir -p "$RESULTS" "$GPU_TRACE_DIR"
    write_cluster_env
fi
