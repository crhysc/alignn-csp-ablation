#!/usr/bin/env bash
# Phase 1 -- prove the plumbing, and measure GPU throughput, before phase 4.
#
#   bash 10_smoke.sh          # submit both jobs, wait, report
#   bash 10_smoke.sh --probe  # only the GPU probe
#   bash 10_smoke.sh --smoke  # only the CPU smoke
#
# Two SLURM jobs.  Nothing heavy runs on the login node: the CPU smoke trains,
# samples, relaxes and scores, and the probe trains on the real split.
#
#   A. CPU smoke -- every stage of a real task at toy size (2 epochs, 2
#      candidates, 4 targets) in a throwaway root.  Proves data layout,
#      checkpoint format, CSV columns and the scoring environment.  Proves
#      nothing about the science.
#
#   B. GPU probe -- validates four things PLAN.md asserts but has not
#      measured (CUDA with CSP_MODULES empty; a whole GB10 rather than a
#      partitioned one; the sampler writes a usable trace; whether sacct
#      populates gres/gpumem and gres/gpuutil), and then TIMES real training
#      for A0 and A3 so PHASE4_TIME comes from a measurement, not a guess.
#
#      NOTE the projection below covers TRAINING only.  On this benchmark
#      generation dominates and scales with the test set, so treat its
#      number as a lower bound and get the real one from 20_pilot.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
mkdir -p "$RESULTS/00_provenance"
WHICH="${1:---all}"

# ---------------------------------------------------------------------------
submit_smoke() {
    local sb="$RESULTS/00_provenance/smoke.sbatch"
    cat > "$sb" <<SMOKE
#!/usr/bin/env bash
#SBATCH --job-name=csp-smoke
#SBATCH --output=$RESULTS/00_provenance/smoke-%j.out
#SBATCH --error=$RESULTS/00_provenance/smoke-%j.out
#SBATCH --time=00:50:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=32G
$(sbatch_account_line)
#SBATCH --partition=$PART_DEBUG
set -uo pipefail
cd "$HARNESS"; source ./env.sh
echo "=== CPU smoke into $SMOKE_RUNS (throwaway root)"
run_task_data $DATA_TASK --runs-root "$SMOKE_RUNS"
# Unit 0 of the ablation task rather than bench-*: it exercises exactly the
# code path the real run takes, and on Alexandria bench-alex would drag in the
# pretraining checkpoint the ablation deliberately does not use.
run_task $ABLATION_TASK --smoke --device cpu --runs-root "$SMOKE_RUNS" --unit 0
echo
echo "=== CSV header (AtomBench needs id,target,prediction)"
head -1 "$SMOKE_RUNS"/train_smoke/${SPLIT}_A0/seed0/bench/sym/pred.csv
echo "=== metrics.json written?"
ls -l "$SMOKE_RUNS"/train_smoke/${SPLIT}_A0/seed0/bench/sym/metrics.json
SMOKE
    sbatch --parsable --export=ALL "$sb"
}

# ---------------------------------------------------------------------------
submit_probe() {
    local sb="$RESULTS/00_provenance/probe.sbatch"
    cat > "$sb" <<PROBE
#!/usr/bin/env bash
#SBATCH --job-name=csp-probe
#SBATCH --output=$RESULTS/00_provenance/probe-%j.out
#SBATCH --error=$RESULTS/00_provenance/probe-%j.out
#SBATCH --time=00:55:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_DEBUG
set -uo pipefail
cd "$ALIGNN_REPO"
echo "host \$(hostname)  job \${SLURM_JOB_ID}  gres \${SLURM_JOB_GRES:-unset}"

echo; echo "--- nvidia-smi"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || echo "NO nvidia-smi"

echo; echo "--- torch (CSP_MODULES is empty: does it need a module load?)"
"$TRAIN_ENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("FATAL: no CUDA -- set CSP_MODULES in env.sh and re-probe")
print("device:", torch.cuda.get_device_name(0))
PY

echo; echo "--- gpu sampler + MIG assertion"
export GPU_TRACE_DIR="$GPU_TRACE_DIR" REQUIRE_GPU_NAME="$REQUIRE_GPU_NAME"
export FORBID_GPU_PATTERN='$FORBID_GPU_PATTERN'
source "$HARNESS/lib/gpu_sampler.sh" || echo "FATAL: sampler assertion failed"

echo; echo "--- training throughput on the real split (40 epochs, A0 and A3)"
# train_csp prints cumulative seconds at every log point, so a slope over the
# later points gives seconds/epoch with process startup and warm-up excluded.
export PATH="$TRAIN_ENV/bin:\$PATH"
for arm in A0 A3; do
  echo "  == \$arm"
  "$TRAIN_ENV/bin/python" -u -m alignn.inverse.train_csp \\
      --data-dir "$CSP_RUNS/data/$SPLIT" \\
      --output "/tmp/csp_probe_\$arm" \\
      --epochs 40 --seed 0 --ablation "\$arm" \\
      --alignn-layers 3 --gcn-layers 3 --hidden-features 256 --knn 12 \\
      --num-steps 1000 --batch-size 64 --lr 1e-3 --augment $AUGMENT \\
      --device cuda --log-every 5 2>&1 | grep -E "^epoch|^done" | tee "/tmp/probe_\$arm.log"
done

echo; echo "--- projection"
"$TRAIN_ENV/bin/python" - <<'PY'
import re, math, pathlib
N_TRAIN, BATCH, FULL_EPOCHS = $N_TRAIN, 64, $TRAIN_EPOCHS
spe = math.ceil(N_TRAIN / BATCH)
res = {}
for arm in ("A0", "A3"):
    pts = []
    for line in pathlib.Path(f"/tmp/probe_{arm}.log").read_text().splitlines():
        m = re.match(r"epoch\s+(\d+).*?(\d+)s\s*$", line)
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    if len(pts) < 3:
        print(f"  {arm}: not enough log points"); continue
    # slope over the second half: excludes startup and warm-up
    half = pts[len(pts)//2:]
    (e0, t0), (e1, t1) = half[0], half[-1]
    s_per_epoch = (t1 - t0) / max(e1 - e0, 1)
    res[arm] = s_per_epoch
    train_h = s_per_epoch * FULL_EPOCHS / 3600
    print(f"  {arm}: {s_per_epoch:.2f} s/epoch  "
          f"({s_per_epoch/spe:.4f} s/step)  ->  {train_h:.2f} h for {FULL_EPOCHS} epochs")
if "A0" in res and "A3" in res:
    print(f"\n  measured A3/A0 per-step ratio: {res['A3']/res['A0']:.2f}x  "
          f"(README claims 2.4x)")
if res:
    worst_h = max(res.values()) * FULL_EPOCHS / 3600
    total = worst_h + 3.0          # generation+relax+scoring headroom
    pad = total * 1.5
    h = int(pad) + 1
    print(f"\n  slowest arm training      : {worst_h:.1f} h")
    print("  + generation/relax/score  : ~3 h (NOT measured here; on"
          " $N_TEST targets it dominates -- use 20_pilot.sh measure)")
    print(f"  + 50% headroom            : {pad:.1f} h")
    print(f"\n  SET IN env.sh:  PHASE4_TIME=\"{h:02d}:00:00\"")
PY
rm -rf /tmp/csp_probe_A0 /tmp/csp_probe_A3
PROBE
    sbatch --parsable --export=ALL "$sb"
}

# ---------------------------------------------------------------------------
JOBS=()
case "$WHICH" in
    --smoke) S=$(submit_smoke); echo "smoke job $S"; JOBS+=("$S") ;;
    --probe) P=$(submit_probe); echo "probe job $P"; JOBS+=("$P") ;;
    *) S=$(submit_smoke); echo "smoke job $S (CPU, $PART_DEBUG)"
       P=$(submit_probe); echo "probe job $P (A30, $PART_DEBUG)"
       JOBS+=("$S" "$P") ;;
esac
printf 'phase1\tjob\t%s\n' "${JOBS[@]}" >> "$RESULTS/00_provenance/slurm_jobids.txt"
echo
echo "watch:  squeue -j $(IFS=,; echo "${JOBS[*]}")"
echo "logs:   $RESULTS/00_provenance/"
