# csp-ablation-bench

Benchmarking harness for the angular-diffusion ablations (`A0`–`A6`) of
ALIGNN-CSP, on the WVU Dolly Sods cluster.

Lives **outside** the ALIGNN source tree on purpose. The workspace above is its
own repository, which tracks this harness as plain files and `alignn/` as a
submodule, so the two histories stay separate. The one file the harness writes
inside the ALIGNN checkout is `task_runners/cluster.env`, which that repo
designates as site-local — and it is *generated* from `env.sh`, not hand-edited,
so the whole configuration is reproducible from this directory alone.

Moving to a new cluster? Read `../HANDOFF.md` first — it lists every value in
`env.sh` that is specific to Dolly Sods and must be re-measured.

**`PLAN.md` is the design document**: what is being measured, why these six
tasks, what the cluster actually offers, where every number comes from, and
what the known threats to validity are. Read it before changing anything here.

## Run order

```bash
source ./env.sh                # every script does this itself; useful by hand

bash 00_setup.sh               # phase 0: envs, AtomBench, cluster.env, doctor
bash 10_smoke.sh               # phase 1: CPU smoke + one debug-partition probe

bash 20_pilot.sh loss          # phase 2: 12 elements, training only, 6 h
bash 20_pilot.sh quick         # phase 3: full --quick, the decision point
bash 20_pilot.sh measure       # -> the --time to put in env.sh for phase 4

bash 30_full.sh all            # phase 4: the real run on gpu_7day, A30 pinned
bash 30_full.sh choose         # after symprec-sweep: pick the tolerance
bash 30_full.sh pipeline       # after setting SYMPREC in env.sh
bash 40_mechanism.sh           # angle_eval.py over every finished run
```

Then collect and analyse:

```bash
python collect.py                                     # -> results/<RUN_ID>/
python stage_benchmarks.py                            # -> 20_benchmarks/
$SCORE_ENV_PATH/bin/atombench results/<RUN_ID>/20_benchmarks \
                              results/<RUN_ID>/30_atombench
python costs.py --harvest                             # -> 50_costs/
$SCORE_ENV_PATH/bin/python analyze.py                 # -> 40_stats/, 60_report/
```

`analyze.py` needs the **scoring** environment for its paired per-target tests
(pymatgen). Tier 1 runs anywhere and it says so rather than failing obscurely.

Before anything is submitted, `DRY=1` shows exactly what would be:

```bash
DRY=1 bash 30_full.sh train
DRY=1 bash 40_mechanism.sh
```

## Two helpers you will want by hand

`env.sh` defines `run_task` and `run_task_data`. Use them instead of calling
`run_task.py` directly:

```bash
source ./env.sh
run_task angle-ablation --aggregate --latex
run_task_data data-jarvis
```

`tasks.py` builds every stage argv starting with the bare string `python`, so
stages resolve the interpreter from `PATH`, not from whichever python launched
the runner. Inside a job `common.sh` has already activated `CSP_ENV`; outside
one, calling `$TRAIN_ENV/bin/python task_runners/run_task.py` runs the runner
in the right env and every stage in the wrong one. `run_task` fixes `PATH`.
`run_task_data` does the same against the scoring env, because `prepare_data.py`
needs jarvis-tools and pymatgen and no torch.

## Two environments, on purpose

| env | holds | why |
|---|---|---|
| `/scratch/crc00042/envs/alignn2` | torch, jarvis-tools, ALIGNN (editable) | training and sampling |
| `/scratch/crc00042/envs/csp-ablation-bench` | pymatgen, amd, AtomBench (editable) | scoring and analysis |

Keeping them apart is not fussiness: AtomBench pulls pymatgen and
`average-minimum-distance`, and the PyPI package named `atombench` pins
`numpy==1.19.5` / `pandas==1.2.4`, which would wreck a working torch env. The
runner supports the split natively — `score.sh` switches into `CSP_SCORE_ENV`
by itself.

**AtomBench is installed from GitHub, never from PyPI.** PyPI's `atombench` is
version 2022.7.15: a 2.5 kB wheel containing one file, no metric code, no CLI.
`00_setup.sh` hard-fails if it ever displaces the real one.

## Files

| file | what it does |
|---|---|
| `PLAN.md` | the design document — read this first |
| `env.sh` | single source of truth; generates `cluster.env`; `csp_submit` |
| `00_setup.sh` | environments, AtomBench clone + install, doctor |
| `10_smoke.sh` | CPU smoke, then the GPU probe (CUDA / MIG / sampler / sacct) |
| `20_pilot.sh` | phases 2–3 and the walltime measurement |
| `30_full.sh` | phase-4 submissions, the confound arm, per-arm pipeline roots |
| `40_mechanism.sh` | `angle_eval.py --relax` as its own array |
| `lib/gpu_sampler.sh` | GPU trace + the "is this really a whole A30" assertion |
| `collect.py` | run tree → `results/<RUN_ID>/`; writes `INDEX.md` |
| `stage_benchmarks.py` | the AtomBench staging tree |
| `costs.py` | cost accounting, the 2.4× check, the homogeneity audit |
| `analyze.py` | seed statistics, paired per-target tests, Holm adjustment |

`results/<RUN_ID>/INDEX.md` is generated by `collect.py` and explains every
output file and the command that regenerates it. It is not hand-written, so it
cannot drift from what is actually there.

## What this harness does not touch

It does not modify the ALIGNN repository. Three gaps in `task_runners/` are
covered from outside rather than by patching it:

* `aggregate.py:print_cost` surfaces only `train_s`, though every stage records
  `elapsed_s` — `costs.py` harvests all of them.
* `angle_eval.py` is not a stage in any task — `40_mechanism.sh` runs it.
* `claims.py` registers no angle-ablation claims, so `run_task.py verify` does
  not cover this branch — `analyze.py` reports the contrasts instead.

Closing them inside the repo would be cleaner long-term. See `PLAN.md` §12.
