# HANDOFF — angular-diffusion ablation workspace

Context summary for whoever (or whatever) picks this up on a new cluster.
Written 2026-09-01, from the Dolly Sods (WVU) instance where the first full
run was done.

> **2026-09-03 update.** This document predates two things worth knowing
> before reading further: a second harness now exists —
> `supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/`,
> which crosses the line-graph ablation this one describes with the
> bond-angle-diffusion suite itself — and most of what §4–§6 below walk
> through *by hand* for the Dolly Sods → atomgptlab move is now automated by
> `install.sh` at the repo root (see the top-level `README.md`). The
> per-site scheduler knobs in the table below still have to be re-measured
> on any new site; `install.sh` writes a `site.env` with a placeholder and
> the discovery command for each one, rather than a value, precisely because
> it cannot know them for you. This file's account of *why* each trap
> exists is still the best documentation of that — worth reading even though
> the manual steps it narrates are no longer how you'd actually do them.

---

## 1. What this is

A benchmarking harness for an **ablation suite over explicit bond-angle
diffusion** in ALIGNN-CSP. Eight arms — six ablation configurations, a confound
arm and a no-line-graph control — each trained for 3 seeds and scored through
AtomBench on a crystal-structure-prediction benchmark.

| arm | what it is |
|---|---|
| `A0` | baseline: ALIGNN 2.0 diffusion, angles as *features* only |
| `A1` | explicit angular denoising, baseline kNN line-graph topology |
| `A2` | smooth radius topology, no angular denoising objective |
| `A3` | **proposed**: explicit angular denoising + smooth topology |
| `A4` | control: angular objective with the angle→bond coupling removed |
| `A6` | `A3` with a Fourier angular basis instead of ALIGNN's cosine RBF |
| `A3_nogate` | confound arm: smooth triplet topology + `A0`'s ungated dense pair channel |
| `nolg` | no angular channel at all, budget spent on pair-graph depth |

The scientific question is whether making angles a *denoised variable* rather
than a *conditioning feature* improves generation. `alignn/inverse/ablations.py`
holds `ABLATIONS`, `COMPARISONS` and `DESCRIPTIONS` — the arm definitions and
the pre-registered contrasts — and is the authority on all of the above.

> **There is no `A5` configuration.** `A5` in the design brief is a *comparison*
> — hard kNN vs the smooth radius graph — which is `A1` vs `A3` with angles on
> and `A0` vs `A2` with angles off. `ablations.py` says so at its line 26 and
> deliberately has no `"A5"` key. It appears twice in the report's contrast
> headings and is easy to misread as a missing arm.

## 2. Why it is moving clusters

**The first full run came back null, and the honest reading is that the
benchmark is underpowered, not that the effect is absent.**

Every pre-registered contrast on JARVIS Supercon-3D returned Holm-adjusted
p = 1.000. Match rates across all eight arms fall in 0.447–0.489 with per-arm
standard deviations of ±0.015–0.049 — the arms are inside each other's noise:

```
A0        0.4628 ± 0.0148        A3        0.4790 ± 0.0297
A1        0.4628 ± 0.0244        A3_nogate 0.4887 ± 0.0448
A2        0.4628 ± 0.0404        A4        0.4466 ± 0.0257
nolg      0.4595 ± 0.0312        A6        0.4757 ± 0.0485
```

The split's **test set is 103 targets** and there are **3 seeds per arm**. The
report says it plainly (`csp-ablation-bench/results/2026-08-30_full/60_report/REPORT.md`,
"How to read this"): the ALIGNN README records match rate spanning 0.437–0.524
across fifteen independently trained models on this same 103-target split, so
seed-to-seed variance alone covers the entire spread being measured. Three seeds
cannot resolve a few percent. The paired per-target tests (McNemar / CMH over
discordant pairs, Wilcoxon on RMSD) were added precisely because they are the
powered ones, and they too came back flat — n = 126 pairs, mean ΔRMSD
−0.0015 Å, CI [−0.0179, 0.0154].

**So: the intended next move is a larger benchmark split.** Not a re-run of the
same one with more seeds. See §5.

## 3. Layout

```
<workspace>/                 <- this repo
├── HANDOFF.md               <- you are here
├── .gitmodules
├── alignn/                  <- submodule: crhysc/alignn @ angle-diffusion
│   ├── alignn/inverse/      <- the model: diffusion, ablations, angles, sampling
│   └── task_runners/        <- the task DAG, SLURM submission, aggregation
└── csp-ablation-bench/      <- the harness (plain files, this repo's own)
    ├── PLAN.md              <- THE design document, 41 kB, read it
    ├── README.md            <- run order and the two-environment rationale
    ├── env.sh               <- single source of truth for every path and knob
    ├── 00_setup.sh … 60_unrelaxed.sh
    ├── collect.py stage_benchmarks.py costs.py analyze.py
    └── results/2026-08-30_full/   <- PARTIAL, see below
```

Clone with:

```bash
git clone --recurse-submodules <url> ALIGNN
```

If you forgot `--recurse-submodules`: `git submodule update --init`.

### What is deliberately not in the repo

`csp-ablation-bench/.gitignore` keeps the **analysis tiers** and drops the
**bulk tiers** of the results tree. Tracked (~540 kB): `20_benchmarks/`,
`20_benchmarks_rawsym/`, `40_stats/`, `50_costs/`, `60_report/`. Ignored
(~37 MB): `10_runs/`, `30_atombench/`, `30_atombench_raw/`, `00_provenance/`,
`_gputrace/`.

> **`results/2026-08-30_full/INDEX.md` describes a complete tree and this one is
> not complete.** It was generated by `collect.py` against the full run on the
> old cluster. Sections referring to `10_runs/`, `30_atombench*/`,
> `00_provenance/` or `_gputrace/` describe files that were not packaged.
> Nothing is wrong; they were just not worth carrying.

Everything else heavy was never in the tree to begin with — run trees, conda
environments, the AtomBench checkout and the prepared splits all live under
`$CSP_RUNS`, `$TRAIN_ENV`, `$SCORE_ENV_PATH` and `$ATOMBENCH_REPO`, which
`env.sh` points at scratch. `00_setup.sh` rebuilds all of them.

## 4. Before anything runs: what is Dolly-Sods-specific

`csp-ablation-bench/env.sh` is the only file with site knowledge in it, and it
is commented to say where each number came from. **Every value below was
measured on or granted by the old cluster and is wrong until re-established.**

| variable | old value | how it was arrived at |
|---|---|---|
| `CSP_ACCOUNT` | `alromero` | the allocation |
| `CSP_PARTITION` | `gpu_7day` | longest GPU queue available |
| `PART_DEBUG` / `PART_INTER` / `PART_PILOT` | `debug` / `inter_a30` / `gpu_2day` | the 1 h / 6 h / 2 d queues |
| `CSP_GPU_GRES` | `gpu:nvidia_a30:1` | pinned so every arm gets identical silicon |
| `REQUIRE_GPU_NAME` / `FORBID_GPU_PATTERN` | `A30` / MIG slices | asserts a *whole* GPU, see §6 |
| `CSP_MAX_CONCURRENT` | `8` | throttle against a 20×A30 cap |
| `CPUS_PER_TASK` / `MEM_PER_TASK` | `8` / `56G` | fair share of a 32-core / 257 GB / 4-GPU node |
| `PHASE4_TIME` | `12:00:00` | generous; training measured at 0.38–0.46 h, generation dominates and was never measured at full size |
| `CSP_RUNS`, `TRAIN_ENV`, `SCORE_ENV_PATH`, `ATOMBENCH_REPO`, `SMOKE_RUNS` | `/scratch/crc00042/...` | scratch layout |

`ALIGNN_REPO` no longer needs editing — it resolves relative to the harness, so
the sibling `alignn/` submodule is found automatically.

`RUN_ID` is **pinned, not derived from the date**, on purpose: a phase-4 run
spans several days and a date-derived ID would split one benchmark across two
results trees. Set it to a new value for the new run (e.g. `2026-09_<dataset>`)
and the whole results tree namespaces itself.

Then:

```bash
cd csp-ablation-bench
$EDITOR env.sh            # the table above
bash env.sh --write       # regenerates alignn/task_runners/cluster.env
bash 00_setup.sh          # envs, AtomBench from GitHub, doctor
bash preflight.sh         # refuses to proceed on anything unresolved
bash 10_smoke.sh          # CPU smoke + one real GPU probe
```

`cluster.env` inside the submodule is **generated, never hand-edited**. It
arrives in the clone as the pristine upstream template; `bash env.sh --write`
fills it in. (On the old cluster it shows as a permanently dirty submodule file
for exactly this reason — that is expected, not drift.)

`DRY=1` in front of any submission script prints the sbatch lines without
submitting. Use it.

## 5. Switching the benchmark dataset

This is the point of the move. There is already a **second prepared split** in
the task DAG, and it is roughly 8× larger:

| task | split | train/val/test |
|---|---|---|
| `data-jarvis` | JARVIS Supercon-3D | 847 / 105 / **103** |
| `data-alex` | Alexandria DS-A/DS-B | 6603 / 825 / **825** |

825 test targets against 103 is the power increase the null result calls for.
`data-alex` needs the Alexandria DS-A/DS-B pickles
(figshare `10.6084/m9.figshare.31045597`); `_data_alex` in `tasks.py` takes
`--alex-inputs` to point somewhere other than `$ALIGNN_RUNS/data/alexandria/`.

**The ablation arms are currently hard-wired to the JARVIS split.** The levers,
in `alignn/task_runners/tasks.py`:

- `_jarvis_units()` — `data = ctx.data / "jarvis"` (line ~556) and
  `epochs_key="jarvis"` (line ~566). This one function builds every unit for
  both `angle-ablation` and `ablation-linegraph`; it is the main lever.
- `needs=("data-jarvis",)` on the `ablation-linegraph` (~875) and
  `angle-ablation` (~894) task definitions.
- `_pipeline_ablation()` (~686) and `_symprec_sweep()` (~708) also pin
  `ctx.data / "jarvis"`.
- `EPOCHS` (line 50) — `jarvis: 3000`, `alex: 1000`. **The Alexandria and
  pretraining epoch counts are flagged in-file as *not pinned by the
  manuscript***; recover them from a released checkpoint with
  `task_runners/inspect_checkpoint.py` rather than assuming 1000 is right.

Generalising `_jarvis_units` to take the split name as a parameter is the clean
change; renaming it while you are there would stop the next reader assuming it
is JARVIS-only.

On the harness side the same string appears in `10_smoke.sh:45`,
`20_pilot.sh:51`, `30_full.sh:51`, `preflight.sh:92` (as `run_task_data
data-jarvis`) and in `costs.py:107,110` (which reads
`$CSP_RUNS/data/jarvis/{train,…}.json` for the cost model). All five need the
new split name.

Two things that do **not** transfer and must be re-derived on the new split:

- **`SYMPREC`.** Currently `0.1`. It is chosen by `30_full.sh choose` from a
  `symprec-sweep` run **on the validation split, never on test** — rerun it.
- **`PHASE4_TIME`.** 8× the data at the same epoch count is a different
  walltime. `20_pilot.sh measure` exists to tell you the number instead of
  guessing; it refuses to guess on your behalf.

Also re-check `augment=0` in `_jarvis_units`. It is set because "48 basis
relabellings of 847 crystals cost accuracy on a split this small" — that
reasoning is explicitly about small-data, and 6603 training crystals may well
flip it. It is a one-line experiment worth running.

## 6. Traps that already cost time here

- **AtomBench must come from GitHub, never PyPI.** The PyPI `atombench` is
  version 2022.7.15: a 2.5 kB wheel, one file, no metric code, no CLI.
  `00_setup.sh` hard-fails if it ever displaces the real one. Leave that check in.
- **Two environments, on purpose.** Training/sampling (torch, jarvis-tools,
  ALIGNN editable) and scoring/analysis (pymatgen, `amd`, AtomBench editable)
  are separate because AtomBench's deps pin `numpy==1.19.5` / `pandas==1.2.4`
  and would wreck a working torch install. `run_task` and `run_task_data` in
  `env.sh` dispatch to the right one — use them, do not call `run_task.py`
  directly. Consequence: `doctor` reporting pymatgen missing is **correct**; it
  probes the training env, where pymatgen is deliberately absent.
- **MIG slices silently halve throughput and poison every timing comparison.**
  `FORBID_GPU_PATTERN` asserts a whole GPU rather than hoping for one. Whatever
  the new cluster's GPU is, keep an equivalent assertion.
- **`Atoms.from_poscar` takes a path, not POSCAR text.** Handing it text raises
  `OSError: File name too long` once a generated cell exceeds the filename
  limit. Fixed on this branch (`evaluate.py` uses `Poscar.from_string`) — do not
  reintroduce it if you touch benchmark CSV parsing.
- **A 180° spike in generated *and* real bond-angle histograms is not physics.**
  It is back-tracking triplets from the graph builder, identical across arms.
  It does not bias a comparison.
- **`A4` is not a second baseline.** Its trunk is `A3`'s with one aggregation
  zeroed — a different function from `A0`'s. It controls for `A3` only.
- **Do not read across information tiers.** All arms here get identical
  conditioning (per-atom species, `natoms`, Tc) and no ground-truth geometry, so
  the contrasts are clean. The published AtomBench baselines are not comparable
  without saying so: CDVAE sees full structure, AtomGPT formula + Tc, FlowMM
  composition only.

## 7. Where the real documentation is

- `csp-ablation-bench/PLAN.md` — **the design document.** What is measured, why
  these tasks, what the cluster offers, where every number came from, and the
  known threats to validity. §12 covers the confound arm and the three
  `task_runners` gaps the harness works around from outside. Read before
  changing anything.
- `csp-ablation-bench/README.md` — run order, the two-environment rationale, the
  `run_task` / `run_task_data` helpers, per-file index.
- `alignn/alignn/inverse/README.md` — the model side.
- `alignn/task_runners/INSTRUCTIONS.md` — the task runner's own contract.
- `results/2026-08-30_full/60_report/REPORT.md` — the old cluster's numbers and,
  in "How to read this", the statistical caveats that motivated this move.
