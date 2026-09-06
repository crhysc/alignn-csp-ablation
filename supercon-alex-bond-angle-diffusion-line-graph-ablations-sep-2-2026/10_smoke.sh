#!/usr/bin/env bash
# Prove the plumbing before anything expensive queues.
#
#   bash 10_smoke.sh          # every stage of every cell, at toy size
#   bash 10_smoke.sh --list   # what would run
#
# Two epochs, two candidates, four targets, one seed, in a throwaway run root.
# It proves data layout, all four denoiser configurations, checkpoint format,
# CSV columns, the symmetrisation step and the scoring environment.  It proves
# nothing whatever about the science, and says so.
#
# It runs the CELLS, not a generic task: the (angle diffusion, no line graph)
# configuration is new code, and a smoke test that skipped it would miss the
# only thing here that has never run before.
#
# GPU, not CPU: the smoke is cheap either way, and the GPU path is the one the
# real run takes.  It requests a whole GB10 for a few minutes.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

if [ "${1:-}" = "--list" ]; then
    run_task "$MATRIX_TASK" --smoke --runs-root "$SMOKE_RUNS" --list
    exit 0
fi

mkdir -p "$RESULTS/00_provenance" "$SMOKE_RUNS"
# The smoke root gets the same shared data store, so the smoke does not spend
# an hour rebuilding a split that already exists and is verified.
mkdir -p "$SMOKE_RUNS"
[ -e "$SMOKE_RUNS/data" ] || ln -s "$DATA_STORE" "$SMOKE_RUNS/data"

SB="$RESULTS/00_provenance/smoke.sbatch"
cat > "$SB" <<SMOKE
#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-smoke
#SBATCH --output=$RESULTS/00_provenance/smoke-%j.out
#SBATCH --error=$RESULTS/00_provenance/smoke-%j.out
#SBATCH --time=01:30:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_DEBUG
set -uo pipefail
cd "$HARNESS"; source ./env.sh

echo "=== GPU"
nvidia-smi --query-gpu=index,name,driver_version --format=csv || echo "NO nvidia-smi"

echo
echo "=== all four cells, --smoke, into $SMOKE_RUNS"
for u in 0 1 2 3; do
    echo
    echo "--- unit \$u"
    run_task $MATRIX_TASK --smoke --runs-root "$SMOKE_RUNS" --unit \$u || {
        echo "SMOKE FAILED at unit \$u" >&2; exit 1; }
done

echo
echo "=== what came out"
for cfg in nolg A0 nolg_ad A3; do
    d="$SMOKE_RUNS/train_smoke/${SPLIT}_\$cfg/seed0"
    [ -d "\$d" ] || { echo "  \$cfg: MISSING"; continue; }
    echo "  \$cfg:"
    python3 - "\$d" <<'CHECK'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
print(f"      {cfg['n_parameters']/1e6:.4f} M params  "
      f"alignn={cfg['alignn_layers']} gcn={cfg['gcn_layers']}  "
      f"angle_diffusion={cfg['angle_diffusion']}  topology={cfg['topology']}  "
      f"select_on={cfg.get('select_on')}")
h = json.loads((d / "history.json").read_text())
v = h[-1]["val"]
print(f"      final val: structural={v.get('loss_structural')}  "
      f"angle={v.get('loss_angle')}  total={v.get('loss')}")
csv = d / "bench" / "sym" / "pred.csv"
if csv.exists():
    head = csv.read_text().splitlines()[0]
    print(f"      CSV header: {head}   (AtomBench needs id,target,prediction)")
mj = d / "bench" / "sym" / "metrics.json"
print(f"      metrics.json: {'yes' if mj.exists() else 'NO'}")
CHECK
done

echo
echo "=== canonicalisation of PREDICTIONS (the other half of the requirement)"
# The training targets are canonicalised at data-prep time and preflight checks
# them.  The prediction side happens inside AtomBench's compute_metrics.py,
# which reduces BOTH columns to a primitive Niggli cell before every lattice
# metric.  This confirms it on a real CSV this smoke just produced.
# score_py, not python3: pymatgen lives in the SCORING environment, which is
# the whole reason there are two.  Running this under the training env's
# python would fail on the import, not on the science.
score_py - "$SMOKE_RUNS/train_smoke/${SPLIT}_A3/seed0/bench/sym/pred.csv" <<'CANON'
import csv, sys
csv.field_size_limit(10**8)
from pymatgen.core import Structure
row = next(csv.DictReader(open(sys.argv[1])))
for col in ("target", "prediction"):
    # generate_benchmark.py escapes newlines as a literal backslash-n so a
    # POSCAR fits one CSV field.  chr(92) rather than a backslash literal:
    # this block is written from an UNQUOTED heredoc, so bash collapses "\\n"
    # to "\n" on the way into the file and the un-escaping silently becomes a
    # no-op -- which reads as a corrupt POSCAR, not as a quoting bug.
    s = Structure.from_str(row[col].replace(chr(92) + "n", chr(10)),
                           fmt="poscar")
    c = s.get_primitive_structure().get_reduced_structure(reduction_algo="niggli")
    same = len(c) == len(s) and abs(c.volume - s.volume) < 1e-3 * max(s.volume, 1)
    print(f"      {col:11s} {len(s):3d} atoms, V={s.volume:8.2f} -> "
          f"primitive+Niggli {len(c):3d} atoms, V={c.volume:8.2f}   "
          f"{'already canonical' if same else 'REDUCED by the metric'}")
CANON
echo
echo "smoke complete.  These numbers mean nothing scientifically -- 2 epochs,"
echo "2 candidates, 4 targets.  They mean the pipeline runs end to end."
SMOKE

if [ "${DRY:-0}" = "1" ]; then echo "would submit: sbatch $SB"; exit 0; fi
JID=$(sbatch --parsable --export=ALL "$SB")
printf 'smoke\tjob\t%s\n' "$JID" >> "$RESULTS/00_provenance/slurm_jobids.txt"
echo "submitted smoke job $JID  ($PART_DEBUG, $CSP_GPU_GRES)"
echo "watch: squeue -j $JID"
echo "log:   $RESULTS/00_provenance/smoke-$JID.out"
