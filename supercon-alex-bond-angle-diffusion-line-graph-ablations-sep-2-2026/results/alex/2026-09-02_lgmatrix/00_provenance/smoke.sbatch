#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-smoke
#SBATCH --output=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/smoke-%j.out
#SBATCH --error=/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/alex/2026-09-02_lgmatrix/00_provenance/smoke-%j.out
#SBATCH --time=01:30:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
# no --account: this cluster runs no accounting associations
#SBATCH --partition=gpu
set -uo pipefail
cd "/home/ccamp104/repositories/alignn-csp-ablation/supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026"; source ./env.sh

echo "=== GPU"
nvidia-smi --query-gpu=index,name,driver_version --format=csv || echo "NO nvidia-smi"

echo
echo "=== all four cells, --smoke, into /data/ccamp104/csp_smoke_lgmatrix/alex"
for u in 0 1 2 3 4 5; do
    echo
    echo "--- unit $u"
    run_task lg-angle-matrix-alex --smoke --runs-root "/data/ccamp104/csp_smoke_lgmatrix/alex" --unit $u || {
        echo "SMOKE FAILED at unit $u" >&2; exit 1; }
done

echo
echo "=== what came out"
for cfg in nolg A0 nolg_ad A3 nolg_d16 nolg_ad_d16; do
    d="/data/ccamp104/csp_smoke_lgmatrix/alex/train_smoke/alex_$cfg/seed0"
    [ -d "$d" ] || { echo "  $cfg: MISSING"; continue; }
    echo "  $cfg:"
    python3 - "$d" <<'CHECK'
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
score_py - "/data/ccamp104/csp_smoke_lgmatrix/alex/train_smoke/alex_A3/seed0/bench/sym/pred.csv" <<'CANON'
import csv, sys
csv.field_size_limit(10**8)
from pymatgen.core import Structure
row = next(csv.DictReader(open(sys.argv[1])))
for col in ("target", "prediction"):
    # generate_benchmark.py escapes newlines as a literal backslash-n so a
    # POSCAR fits one CSV field.  chr(92) rather than a backslash literal:
    # this block is written from an UNQUOTED heredoc, so bash collapses "\n"
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
