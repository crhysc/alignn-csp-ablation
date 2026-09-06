---
license: mit
tags: [materials-science, crystal-structure-prediction, diffusion, alignn, ablation]
library_name: alignn
---

# ALIGNN-CSP angular-diffusion ablations — checkpoints

Trained checkpoints for every cell of the line-graph × bond-angle-diffusion
ablation of ALIGNN-CSP (a conditional diffusion model for crystal structure
prediction), plus two partial checkpoints from the earlier A-suite's cancelled
Alexandria port. Source, harness, analysis and full provenance:
https://github.com/crhysc/alignn-csp-ablation (read `PROJECT_STATE.md` there).

## Layout

```
lgmatrix/<dataset>/<arm>_seed0/
  best_model.pt          EMA weights at the epoch of minimum validation structural loss (what was benchmarked)
  config.json            every hyperparameter the run was trained with, incl. angle_mode and n_parameters
  history.json           per-epoch train/val losses (total, lattice, frac, angle, structural)
  metrics_sym.json       AtomBench metrics, symmetrised post-relaxation pipeline
  metrics_nosym.json     AtomBench metrics, unsymmetrised post-relaxation pipeline
  generation_config.json exact generate_benchmark.py arguments
  ABLATION.yaml          the exhaustive per-run record (what it is, how it differs, jobs, hardware, sha256s)
angle-ablation/alex/<arm>_seed0/   partial (cancelled at epoch <=375) A0 and A1, for the record only
```

`<dataset>` is `jarvis` (JARVIS-DFT Supercon-3D, 847/105/103) or `alex`
(Alexandria DS-A/DS-B, 6603/825/825). `<arm>`:

| arm | angular tier | angle_mode | line graph |
|---|---|---|---|
| nolg | none | off | no (9 pair convs) |
| A0 | none | off | yes (3 ALIGNN + 3 pair) |
| nolg_ad | derived (legacy) | derived_aux | no |
| A3 | derived (legacy) | derived_aux | yes |
| nolg_b3 | independent | independent | no |
| B3 | independent | independent | yes |

## Loading

```python
from alignn.inverse.sample import load_model
model, schedule, normalizer, cfg = load_model("lgmatrix/alex/B3_seed0/best_model.pt", device="cuda", use_ema=True)
```

with the `alignn` package from https://github.com/crhysc/alignn at branch
`lg-angle-diffusion-matrix` (commit `f8121f4` or later; the `independent`
cells need that branch).

## What these were trained with

AdamW, lr 1e-3 one-cycle, batch 64, hidden 256, T=1000, σ∈[0.005,0.5],
cosine ᾱ, loss weights (lattice, frac, angle) = (1, 10, 1), EMA 0.999, seed
0, checkpoint selected on validation structural loss. Full protocol and the
reading of the results: `EXPERIMENT_SET.yaml` in the harness directory of the
GitHub repository.

## Caveat

Every benchmark number shipped beside these weights was scored **after**
ALIGNN-FF relaxation of 32 candidates. The generator-alone scores are the
project's open task (see `NEXT_TASK.md` in the repository).
