# Background GPU sampler + device assertion.
#
# Sourced from CSP_PRE_RUN_HOOK, which task_runners/common.sh evaluates inside
# every array element before run_task.py starts.  Two jobs:
#
#   1. assert the GPU we actually received is the one we asked for.  A MIG
#      slice (gres/gpu:a30_2g.12gb is registered in this cluster's accounting
#      TRES) would silently halve throughput and poison every timing number in
#      the cost report, so this hard-fails rather than warning.
#
#   2. sample memory / utilisation / power every 30 s for the life of the
#      element, covering training *and* generation.  Nothing else records GPU
#      memory: train_csp.py does not, and sacct only maybe does (gres/gpumem
#      is registered but unverified -- 10_smoke.sh checks).
#
# The trace is keyed by (job id, array index) rather than by run directory,
# because this hook runs before run_task.py has resolved which unit it is.
# collect.py maps them back through `run_task.py <task> --list`.
#
# common.sh runs under `set -euo pipefail`, so everything here is guarded: a
# machine without nvidia-smi degrades to "no trace", never to a failed job.

_gpu_sampler_start() {
    local dir="${GPU_TRACE_DIR:-}"
    local jid="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}"
    local idx="${SLURM_ARRAY_TASK_ID:-0}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu_sampler: no nvidia-smi, skipping (CPU element?)"
        return 0
    fi

    local name
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" || name=""
    if [ -z "$name" ]; then
        echo "gpu_sampler: nvidia-smi returned no device, skipping"
        return 0
    fi
    echo "gpu_sampler: device '$name'"

    # -- assertion 1: the right GPU model -----------------------------------
    if [ -n "${REQUIRE_GPU_NAME:-}" ] && ! grep -qE "$REQUIRE_GPU_NAME" <<<"$name"; then
        echo "FATAL: got GPU '$name', required '$REQUIRE_GPU_NAME'." >&2
        echo "  The arms must run on identical hardware or the per-step cost" >&2
        echo "  ratio measures the scheduler, not the model." >&2
        return 1
    fi
    # -- assertion 2: a whole GPU, not a MIG slice --------------------------
    if [ -n "${FORBID_GPU_PATTERN:-}" ] && grep -qE "$FORBID_GPU_PATTERN" <<<"$name"; then
        echo "FATAL: '$name' looks like a MIG slice; a partitioned GPU would" >&2
        echo "  halve throughput and invalidate every timing comparison." >&2
        return 1
    fi

    [ -n "$dir" ] || { echo "gpu_sampler: GPU_TRACE_DIR unset, not sampling"; return 0; }
    mkdir -p "$dir" || return 0
    local out="$dir/${SLURM_JOB_NAME:-job}__${jid}__${idx}.csv"

    {
        echo "# device=$name host=$(hostname) job=$jid idx=$idx started=$(date -Is)"
        echo "ts,index,name,mem_used_mib,mem_total_mib,util_gpu_pct,util_mem_pct,temp_c,power_w"
    } > "$out"

    # --format=noheader,nounits keeps the columns numeric; -l 30 matches the
    # 30 s cadence jobacct_gather/cgroup already uses for MaxRSS, so the two
    # records line up.
    ( nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits -l 30 >> "$out" 2>/dev/null ) &
    GPU_SAMPLER_PID=$!
    export GPU_SAMPLER_PID GPU_TRACE_FILE="$out"
    echo "gpu_sampler: pid $GPU_SAMPLER_PID -> $out"

    # SLURM tears down the job step's process group anyway; this is so an
    # interactive `source common.sh` does not leak a sampler.
    trap '[ -n "${GPU_SAMPLER_PID:-}" ] && kill "$GPU_SAMPLER_PID" 2>/dev/null || true' EXIT
}

_gpu_sampler_start
