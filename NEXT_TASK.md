# NEXT TASK — regenerate the unrelaxed (force-field-free) predictions for every ablation cell

Status of this file: written 2026-09-06 on atomgptlab (JHU). This is the one
open piece of work. Everything else about the project's state is in
`PROJECT_STATE.md`; this file deliberately contains only the problem and the
job to be done.

## The problem, in one paragraph

Every benchmark number we have for the twelve line-graph × angular-tier cells
(`supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/*/2026-09-02_lgmatrix/`)
was scored on structures that had been through ALIGNN-FF: 32 candidates per
target, single-point energy prescreen to 4, 200-step cell relaxation, lowest
energy kept, then (for the `sym` variant) symmetry idealisation. That measures
the diffusion model *and* the force field together. The question the ablation
asks is about the diffusion model alone — in particular whether the angular
channel improves local coordination geometry — and relaxation snaps every sample
into the force field's nearest minimum, which is exactly where such an advantage
would be erased. We need the generator's own output scored directly. **Those raw
structures were never written to disk**: `scripts/atombench/generate_benchmark.py`
holds the sampled crystals in memory, hands them to the relaxation pool, and
serialises only the winner (`--save-candidates` writes energies, not
structures). They must be regenerated from the checkpoints.

## What exists now

| | jarvis (103 test targets) | alex (825 test targets) |
|---|---|---|
| cells trained | 6 of 6 | 6 of 6 |
| `sym` / `nosym` scored (post-relaxation) | 6 / 6 | 6 / 6 |
| `raw` / `rawsym` scored (generator alone) | **0 / 0** | **0 / 0** |
| bond-angle Wasserstein (`angle_eval.json`) | 0 | 0 |

Checkpoints (`best_model.pt`, EMA weights, selected on validation structural
loss) for all twelve cells are DVC-tracked under each results directory's
`10_runs/<arm>/seed0/` and mirrored to Hugging Face (see `PROJECT_STATE.md`
for the namespace). Nothing needs retraining.

On atomgptlab two SLURM arrays were submitted for this on 2026-09-06 and then
**cancelled** the same day (jobs 13199, 13200) so the work could move to the
new cluster. Nothing partial exists; this task starts from the checkpoints.

## The job

Per cell, generation only, checkpoint reused, **no force field anywhere**:

```
generate_benchmark.py --checkpoint <cell>/best_model.pt --data-dir <split dir> --split test \
    --output-csv <cell>/bench/raw/pred.csv --num-candidates 1 --guidance 2.0 \
    --relax none --rank none --seed 0 --device cuda
symmetrize.sh --csv bench/raw/pred.csv --out bench/rawsym/pred.csv --symprec 0.1
score.sh bench/raw/pred.csv ; score.sh bench/rawsym/pred.csv
angle_eval.py bench/rawsym/pred.csv --relax --relax-steps 200 --output bench/rawsym/angle_eval.json
```

That is exactly what `50_unrelaxed.sh` in the line-graph harness does for every
`best_model.pt` it finds under `$CSP_RUNS/train`:

```bash
cd supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026
DATASET=jarvis bash 50_unrelaxed.sh --list     # shows the 6 checkpoints it will use
DATASET=jarvis bash 50_unrelaxed.sh            # submits one array job, 6 elements
DATASET=alex   bash 50_unrelaxed.sh
```

`50_unrelaxed.sh` writes an sbatch with this site's `#SBATCH` lines from
`env.sh`; on a new cluster fill in `site.env` (written by `install.sh`) first,
then `preflight.sh`. The checkpoints must be at `$CSP_RUNS/train/<ds>_<arm>/seed0/best_model.pt`,
which is where `dvc pull` + a copy (or symlink) of `results/<ds>/2026-09-02_lgmatrix/10_runs/<arm>/seed0/`
puts them — note the collected tree drops the `<ds>_` prefix from the arm name
and the run tree needs it back.

### Expected cost (measured on GB10, 2026-09-06)

Sampling cost is per *batch*, and each batch pays the full 1000-step reverse
loop; dropping from 32 candidates to 1 collapses alex from 30 batches to 1 and
jarvis from 4 to 1.

| cell | jarvis | alex |
|---|---|---|
| nolg / nolg_ad / nolg_b3 (no line graph) | 2–3 min | 2–3 min |
| A0 / A3 / B3 (line graph) | 5–5.5 min | 5–5.5 min |
| **six cells** | **~23 min** | **~23 min** |

Scoring adds ~30 s per cell. The `angle_eval.py --relax` step at the end of
each element relaxes every prediction *serially* (it takes no worker count)
at ~4 s per structure: ~7 min per jarvis cell, ~55 min per alex cell. It does
not gate the benchmark table — `raw`/`rawsym` metrics are written before it
runs — so tables can be pulled as soon as scoring finishes.

### Reading the result

- Absolute match rates will be far below the relaxed table (one candidate
  instead of a 32-pool; the pool is the strongest lever on match rate). Only
  the comparison **across the six arms** is meaningful.
- `python analyze.py --variant raw` (and `rawsym`) in the harness reads the
  collected tree; `LGM_MATRIX=state` selects the independent-state 2×2 and
  writes `40_stats/matrix_state.*` and `60_report/REPORT_state.md`.
- The three-tier picture from the relaxed pipeline that this should confirm or
  overturn (seed 0, `sym`):

  | tier | arch | jarvis L_struct | jarvis match | alex L_struct | alex match |
  |---|---|---|---|---|---|
  | none | no lg | 7.1485 | 0.4757 | 5.7178 | 0.5891 |
  | none | lg | 6.2792 | 0.4466 | 4.2778 | 0.5661 |
  | derived | no lg | 7.2223 | 0.4757 | 5.5913 | 0.5721 |
  | derived | lg | 6.2919 | 0.4660 | 4.3665 | 0.5758 |
  | independent | no lg | 6.6815 | 0.4660 | 5.7121 | 0.5903 |
  | independent | lg | 6.1718 | 0.4466 | 5.1182 | 0.5515 |

  The independent channel helped denoising loss on jarvis and did not on alex
  (and hurt with a line graph); on alex the line graph improves the loss by 25%
  while *reducing* match rate. The unrelaxed scores are what decide whether the
  angular channel changes the generator's geometry or only the training loss.

## The root cause is fixed (alignn commit after f8121f4)

`generate_benchmark.py --save-raw-candidates PATH` now writes every sampled
candidate (target id, candidate index, `diverged` flag, POSCAR) before any
prescreen or relaxation, and `task_runners/tasks.py` passes it for every
generate stage (`bench/nosym/raw_candidates.csv`, collected as
`raw_candidates.csv`). `scripts/atombench/raw_from_candidates.py` turns that
file into an AtomBench `pred.csv` from candidate 0 per target, so from now on
the generator-alone benchmark is a scoring step, not a regeneration. The
twelve existing runs predate the flag, which is why this task exists.

## Pitfalls that cost time last week (all documented in `PROJECT_STATE.md`)

- Pending SLURM jobs run the **live working tree** and re-read
  `task_runners/cluster.env` at start; never `bash env.sh --write` for one
  dataset while the other's jobs are queued.
- One trajectory in ~10⁴ leaves the reals during sampling; the sampler now
  quarantines it per crystal and `generate_benchmark.py` drops and counts it.
  If you see `FloatingPointError: non-finite symmetric matrix`, you are running
  a sampler older than alignn commit `f8121f4`.
- The login node is x86_64 and the training/scoring envs are aarch64 on the
  GPU nodes (atomgptlab specific); the new cluster is presumably homogeneous,
  but `install.sh` + `site.env` is still the right way in.
