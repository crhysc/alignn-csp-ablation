# Line graph × bond-angle diffusion — a 2×2

Four models, two datasets, one seed each, four metric families. That is the
whole experiment.

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

## The matrix is parameter-matched, not compute-matched

A line-graph layer holds the same parameters as two pair convolutions, but
runs its edge update over the triplet set — which on a real 299-atom batch is
8,033 triplets against 1,497 pairs, 5.4 to 1. Measured on a GB10, the two
line-graph cells (`line graph`, `both`) cost **1.50×** the wall-clock per
training step of the two no-line-graph cells (`neither`, `angle diffusion`)
at this matched parameter count (within 1.03% — the residual is the angle
encoder).

Depth could buy that compute back for the no-line-graph row (pair depth 16
matches to 6%, depth 18 to 2%), at the cost of 60–77% more parameters — but
that is a second, differently-budgeted set of cells, not a free fix. Rather
than run it, this harness reports the asymmetry directly: `analyze.py` sums
every stage's measured wall-clock into **GPU-hours**, per cell and per
benchmark, so the parameter/compute trade-off sits next to the fidelity
numbers instead of behind a second experiment.

Read `PLAN.md` before changing anything. It is the design document.

## Run order

```bash
source ./env.sh                  # DATASET=jarvis by default

bash 00_setup.sh                 # link the shared data store, cluster.env, doctor
bash preflight.sh                # refuses to proceed on anything unresolved
bash 10_smoke.sh                 # all four cells, 2 epochs / 4 targets

bash 20_pilot.sh train           # one cell, full settings -> s/epoch
bash 20_pilot.sh price           # generation on 12 targets -> extrapolate
bash 20_pilot.sh report          # -> the PHASE_TIME to put in env.sh

bash 30_full.sh all              # the four cells
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

**Which angular arm.** The default task, `lg-angle-matrix`, crosses the line
graph with `A3`, whose angular channel is `angle_mode="derived_aux"`: a
regression of the angular displacement the coordinate noise induced, with no
state of its own. The angular *state* 2x2 -- `B3`, bond angles as a third,
independently diffused channel -- is `lg-angle-state-matrix`. It shares its
two non-angular cells with the default matrix by config name, so on a run
tree that already holds them only the two angular cells train:

```bash
DATASET=jarvis MATRIX_TASK=lg-angle-state-matrix UNITS=2,3 bash 30_full.sh train
LGM_MATRIX=state python analyze.py        # -> 40_stats/matrix_state.*, 60_report/REPORT_state.md
```

`DRY=1` in front of any submission script prints the sbatch lines and submits
nothing. Use it.

## What is measured

| family | where it comes from | notes |
|---|---|---|
| **denoising loss** | `history.json` | the **structural** term `w_lat·L_lat + w_frac·L_frac`, never the trained total — see below |
| **AtomBench metrics** | `metrics_*.json` | match rate, RMSD, ccRMSD, lattice MAE (abc and angles), KLD, from AtomBench's own `compute_metrics.py` |
| **bond-angle Wasserstein** | `angle_eval.json` | 1-D earth-mover distance in degrees between the generated and held-out real bond-angle histograms |
| **GPU-hours** | `stages/*.json` | sum of every stage's measured wall-clock, per cell and per benchmark — see below |

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
the interaction between them. `analyze.py` reports all four, plus the
interaction.

**Loss and fidelity are not the same claim.** The published ablation moved
denoising loss by 14.5% with no overlap across twelve runs, and moved match
rate by *exactly nothing* (0.4709 both arms). Expect that pattern again.

**GPU-hours are not a fidelity metric, and are reported for a different
reason.** The matrix above is parameter-matched, not compute-matched — the
line-graph cells cost 1.50× the measured wall-clock per training step at
that matched parameter count. Rather than a second, compute-matched matrix
to correct for that, `analyze.py` sums every stage's `elapsed_s` (train
through score-sym, plus `angle_eval`/`unrelaxed` if run) into GPU-hours per
cell and a total per benchmark. Every stage of a unit runs inside one SLURM
element holding `--gres` for its whole duration, so this is a measurement of
GPU-hours billed under this cluster's allocation model, not an estimate.

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

The same file is also shared between the two *datasets* of this harness, and
`common.sh` reads it inside every job at start time, not at submission. So a
pending job picks up whatever the file says when it finally runs. The matrix
jobs therefore never take their run root from it: `30_full.sh` passes each
unit and the aggregate an explicit `--runs-root`, and leaves `cluster.env`
alone once it carries this harness's stamp. Both datasets can be queued at
once, and one aggregate can wait on both:

```bash
DATASET=jarvis bash 30_full.sh all
DATASET=alex AGG_JOIN_DATASET=jarvis bash 30_full.sh all   # one aggregate after all 8
```

Do not run `bash env.sh --write` (or the `symprec` step, which goes through
`csp_submit` and rewrites the file) while the other dataset still has matrix
jobs pending.

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
