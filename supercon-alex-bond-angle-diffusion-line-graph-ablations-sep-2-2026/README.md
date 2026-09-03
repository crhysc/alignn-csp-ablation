# Line graph × bond-angle diffusion — a 2×2

Six models, two datasets, three metric families. That is the whole experiment.

The manuscript already ablates the line graph (Table 3, `tab:inverse_ablation`)
and separately proposes bond-angle diffusion. Those two claims were never
crossed. This harness crosses them.

|  | no angle diffusion | angle diffusion |
|---|---|---|
| **no line graph** | `neither` | `angle diffusion` |
| **line graph** | `line graph` | `both` — the proposed model |

`neither` and `line graph` are arms B and A of the published Table 3, unchanged.
So the top row **is** the existing line-graph ablation, and the bottom row
extends it with the angular denoising objective. `both` is `A3`, the proposed
model. Only the bottom-left cell is new, and it needed new code — see §3 of
`PLAN.md`.

## Why six models and not four

Depth cannot level parameters and compute at the same time, so the suite runs
the matrix **twice normalised** and reports both.

A line-graph layer holds the same parameters as two pair convolutions, but runs
its edge update over the triplet set — which on a real 299-atom batch is 8,033
triplets against 1,497 pairs, 5.4 to 1. Measured on a GB10:

| normalisation | no-line-graph row | what is level | what is not |
|---|---|---|---|
| **parameter-matched** | 9 pair convs | parameters, within 1.03% | line-graph cells take **1.50×** the compute |
| **compute-matched** | 16 pair convs | step cost, within 6% | no-line-graph cells carry **~1.6×** the parameters |

Raising pair depth buys the compute back — 18 convs matches to 2% — but costs
+77% parameters. So neither normalisation is free, and each carries the
opposite confound: under parameter matching a line-graph win could be the extra
compute; under compute matching a line-graph win is the *stronger* claim, but a
line-graph loss becomes ambiguous.

**A result that survives both is not a budget artefact in either direction.**

The two matrices share their line-graph cells, so carrying both costs **two**
extra trainings rather than four — six configurations in total. A single depth
of 16 serves both no-line-graph cells: matched individually they would want 18
and 15, but then the two cells of that row would differ in depth and the
angular contrast *within* it would be confounded with a three-layer change.

Read `PLAN.md` before changing anything. It is the design document.

## Run order

```bash
source ./env.sh                  # DATASET=jarvis by default

bash 00_setup.sh                 # link the shared data store, cluster.env, doctor
bash preflight.sh                # refuses to proceed on anything unresolved
bash 10_smoke.sh                 # all six cells, 2 epochs / 4 targets

bash 20_pilot.sh train           # one cell, full settings -> s/epoch
bash 20_pilot.sh price           # generation on 12 targets -> extrapolate
bash 20_pilot.sh report          # -> the PHASE_TIME to put in env.sh

bash 30_full.sh all              # the six cells
bash 30_full.sh symprec          # tolerance sweep, on VALIDATION
bash 30_full.sh choose           # pick it; set SYMPREC in env.sh
bash 30_full.sh train            # re-score (training is skipped by the markers)

bash 40_mechanism.sh             # bond-angle Wasserstein, on the relaxed run
bash 50_unrelaxed.sh             # ... and on the generator's raw output
```

Then, from anywhere:

```bash
python collect.py                # run tree -> results/<dataset>/<run id>/
python stage_benchmarks.py       # -> 20_benchmarks/ (for the atombench CLI)
python analyze.py --variant sym      # the published pipeline
python analyze.py --variant rawsym   # the generator alone
```

`analyze.py` is **stdlib only** and runs on the login node. Everything it
reports is a pure function of files the run already wrote.

Do the whole thing for the second dataset by prefixing `DATASET=alex`.

`DRY=1` in front of any submission script prints the sbatch lines and submits
nothing. Use it.

## What is measured

| family | where it comes from | notes |
|---|---|---|
| **denoising loss** | `history.json` | the **structural** term `w_lat·L_lat + w_frac·L_frac`, never the trained total — see below |
| **AtomBench metrics** | `metrics_*.json` | match rate, RMSD, ccRMSD, lattice MAE (abc and angles), KLD, from AtomBench's own `compute_metrics.py` |
| **bond-angle Wasserstein** | `angle_eval.json` | 1-D earth-mover distance in degrees between the generated and held-out real bond-angle histograms |

**Why the structural loss and not the trained one.** The objective is
`L_lat + 10·L_frac + L_angle`, and `L_angle` exists in only half the
cells. Quoting the total across the angular factor compares two different
objectives. `L_struct` is the part every cell optimises; it is also what
`--select-on structural` selects `best_model.pt` on, so the checkpoint being
scored and the loss being quoted agree. The angular term is reported too, but
only *within* the two cells that have it.

**Percent change makes sense here** — the manuscript quotes 14.5% for the
line-graph arm, and that reading survives, because `L_struct` is the same
quantity on a ratio scale in every cell. What a 2×2 adds is that there are
now *four* such numbers worth quoting: the line-graph effect with and without
the angular objective, the angular effect with and without the line graph, and
the interaction between them. `analyze.py` reports that set once per
normalisation.

**Loss and fidelity are not the same claim.** The published ablation moved
denoising loss by 14.5% with no overlap across twelve runs, and moved match
rate by *exactly nothing* (0.4709 both arms). Expect that pattern again.

## Two environments, on purpose

| env | holds | used for |
|---|---|---|
| `/data/ccamp104/envs/alignn2` | torch, jarvis-tools, ALIGNN (editable) | training, sampling |
| `/data/ccamp104/envs/csp-score` | pymatgen, `amd`, AtomBench (editable) | data prep, scoring |

AtomBench's dependencies pin `numpy==1.19.5` / `pandas==1.2.4` and would wreck
a working torch install. `run_task` and `run_task_data` in `env.sh` dispatch to
the right one. Consequence: `doctor` reporting pymatgen missing is **correct** —
it probes the training env, where pymatgen is deliberately absent.

Both are **aarch64** builds and run only on the GPU nodes. The login node is
x86_64, so `on_env_arch` in `env.sh` bounces commands through a small `srun`.

**AtomBench comes from GitHub, never PyPI.** The PyPI `atombench` is 2022.7.15:
a 2.5 kB wheel, one file, no metric code, no CLI. `00_setup.sh` hard-fails on it.

## What this shares with its sibling, and what it does not

`../supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/` is the eight-arm
angular-diffusion suite. This harness reuses its two conda environments, its
AtomBench checkout and — deliberately — its **prepared data splits**, which are
a pure function of the public databases and are expensive to rebuild. The
Alexandria prep is already done and verified at 6603/825/825 with zero
canonicalisation fallbacks.

It does **not** share the run tree. Two config names collide (`<split>_A0` and
`<split>_A3`), and those checkpoints are not interchangeable: this experiment
selects them on the structural loss and that one selects them on the trained
total. Separate roots make that impossible to get wrong.

The one file both write is `task_runners/cluster.env`, which is generated, not
hand-edited. It carries a `# HARNESS=` stamp and `preflight.sh` refuses to
proceed if the file on disk belongs to the other experiment.

## Files

| file | what it does |
|---|---|
| `PLAN.md` | the design document — read this first |
| `env.sh` | single source of truth; generates `cluster.env`; `csp_submit` |
| `00_setup.sh` | data symlink, editable installs, `cluster.env`, doctor |
| `preflight.sh` | every precondition, including that the new cell builds |
| `10_smoke.sh` | all four cells at toy size; checks both canonicalisations |
| `20_pilot.sh` | **prices generation**, the one unmeasured cost |
| `30_full.sh` | the run, plus the symmetrisation-tolerance sweep |
| `40_mechanism.sh` | `angle_eval.py` over the relaxed predictions |
| `50_unrelaxed.sh` | the generator's raw output, scored and angle-evaluated |
| `collect.py` | run tree → `results/<dataset>/<run id>/`; writes `INDEX.md` |
| `stage_benchmarks.py` | the AtomBench staging tree |
| `analyze.py` | the 2×2: cells, effects, interaction, LaTeX table |
| `costs.py` | cost accounting (carried over; not central here) |

## Canonicalisation

Both halves of the requirement are enforced and checked:

* **training targets** — `scripts/atombench/prepare_{data,alex_data}.py` store
  every structure in the primitive Niggli cell (`canonical_cell()`).
  `preflight.sh` samples 200 of them and confirms they sit at that fixed point.
* **predicted structures** — AtomBench's `compute_metrics.py` reduces *both*
  the target and the prediction the same way before every lattice metric.
  `10_smoke.sh` confirms it on a real prediction CSV.

This is also why `AUGMENT=0`: once the cell is canonical there is one correct
basis labelling, and the 48 signed permutations would relabel away from the
basis the metrics are computed in.
