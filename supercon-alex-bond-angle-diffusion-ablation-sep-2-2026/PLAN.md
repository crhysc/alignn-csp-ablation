# Benchmarking the angular-diffusion ablations on Dolly Sods

Plan only — nothing here has been run. Everything below was checked against
this machine on 2026-08-29: SLURM metadata from `scontrol`/`sacctmgr`, the
environment inventory from `run_task.py doctor`, the unit counts from
`run_task.py --count`, and the AtomBench findings from the actual PyPI wheel
and a clone of the GitHub repo.

**Harness location:** `/users/crc00042/ALIGNN/csp-ablation-bench/`
`/users/crc00042/ALIGNN` is *not* a git repository (only `ALIGNN/alignn` is),
so this directory sits next to the repo without being inside it.

Three requirements drive the design:

1. **A30 pinned** — every arm on identical hardware, so a timing difference is
   a property of the model and not of the scheduler (§3, §7).
2. **Every computational cost recorded, per ablation** — training, sampling,
   relaxation, scoring, GPU memory, GPU utilisation, queue wait (§7).
3. **Self-contained, organised, documented output** — one archivable directory
   that contains everything needed to re-derive every number without a GPU and
   without the checkpoints (§6).

---

## 0. Two blockers, fix these first

### 0.1 `pip install atombench` installs the wrong package

I checked PyPI. `atombench` there is **version 2022.7.15, a 2.5 kB wheel
containing one file** — `__init__.py` with a version string and nothing else.
Its `Home-page` is `usnistgov/atomvision`. No metric code, no CLI, no
`compute_metrics`. Its pins are actively destructive:

```
numpy==1.19.5   scipy==1.6.3   pandas==1.2.4   jarvis-tools==2021.07.19
```

`alignn2` runs numpy 2.4.6 / pandas 3.0.5 / jarvis-tools 2026.6.12. Installing
it there would try to downgrade all of them.

The real package is on GitHub, properly packaged (`pyproject.toml`,
`name = "atombench"`, `version = "0.1.0"`), with console scripts declared:

```bash
git clone https://github.com/atomgptlab/atombench /scratch/crc00042/atombench
# do NOT --recurse-submodules: .gitmodules points at flowmm, atomgpt, cdvae,
# mattergen — three over SSH, none needed here.

conda create -y -p /scratch/crc00042/envs/atombench python=3.11
conda activate /scratch/crc00042/envs/atombench
pip install -e /scratch/crc00042/atombench
```

Editable-from-clone is deliberate — it gives **both** halves at once: the
`atombench` CLI on `PATH` (cross-arm analysis, §6.3) and
`scripts/scripts_consolidated/compute_metrics.py` at a stable path (what
ALIGNN's `score.sh` shells out to), plus `scripts/harvest_compute_times.py`
whose output schema we match in §7.4. One clone, three needs.

### 0.2 Scoring dependencies are missing

`run_task.py doctor` right now:

```
  [x] torch: 2.5.1+cu121, no CUDA (use --device cpu), one optimiser step ok
  [x] alignn.inverse   [x] jarvis-tools: 2026.6.12
  [ ] pymatgen         [ ] average-minimum-distance (ccRMSD)
  [ ] AtomBench compute_metrics.py not found
```

`no CUDA` is expected — login node, no `nvidia-smi`. The other three are real,
and `pip install -e` on the clone pulls `pymatgen`, `amd`, `scikit-learn`,
`matplotlib`, `click` into the **scoring** env where they belong. `alignn2`
stays the training env and never gets pymatgen. That is what `CSP_SCORE_ENV`
is for. Doctor must be all `[x]` before anything is queued.

---

## 1. Cluster facts (measured)

| | |
|---|---|
| cluster | `dollysods` |
| account | **`alromero`** (the only association) |
| QoS | `normal` (the only one; leave `CSP_QOS` empty) |
| node features | **none** — `AvailableFeatures=(null)` everywhere, so `--constraint` must stay empty; GPU type is pinned through `--gres` |
| max submitted jobs | **100** (association `MaxSubmit`) |
| MaxArraySize | 1001 |
| per-user GPU cap | a30 = **20**, a40 = 8, a100 = 8 |
| QoS-wide GPU cap | a30 = 60, a40 = 12, a100 = 12 |

| partition | walltime | GPUs |
|---|---|---|
| `debug` | 1:00:00 | 4× A30 (dscog001) |
| `inter_a30` | 6:00:00 | 118× A30 |
| `inter_a40` / `inter_a100` | 6:00:00 | 16× A40 / 16× A100 (**both A100 nodes DRAIN today**) |
| `gpu_2day` *(default)* | 2-00:00:00 | all 150 |
| `gpu_7day` | 7-00:00:00 | **29 A30 nodes (116 GPUs)**, 4 A40, 2 A100 |
| `pmm0026`, `dsmebane` | unlimited | **not ours** — `AllowAccounts` excludes `alromero` |

Node shapes: A30/A40 nodes are 32 cores / 257 GB / 4 GPUs → fair share per GPU
is **8 cores, ~60 GB**. A100 nodes are 128 cores / 1007 GB / 8 GPUs.

Storage:

```
/users    8.0T   7.3T used   797G avail   91% FULL
/scratch  379T   236T used   125T avail   66%
```

Accounting (from `scontrol show config`) — this matters for §7:

```
AccountingStorageTRES = cpu,mem,energy,node,billing,fs/disk,vmem,pages,
                        gres/gpu, gres/gpu:a30_2g.12gb, gres/gpu:nvidia_a100,
                        gres/gpu:nvidia_a30, gres/gpu:nvidia_a40,
                        gres/gpumem, gres/gpuutil
AcctGatherEnergyType  = (null)          <-- no energy accounting
JobAcctGatherType     = jobacct_gather/cgroup    (30 s sampling)
```

`gres/gpumem` and `gres/gpuutil` are registered, so `sacct` *may* report them.
`AcctGatherEnergyType = (null)` means **`ConsumedEnergy` will read 0** — we
will not report a fabricated energy number.

---

## 2. Directory layout

```
/users/crc00042/ALIGNN/csp-ablation-bench/     <- outside the git repo
├── PLAN.md  INDEX.md            this file; INDEX explains every output
├── env.sh                       single source of truth: paths, envs, account, RUN_ID
├── 00_setup.sh                  clone+install atombench, score env, doctor, freeze
├── 10_smoke.sh                  phase 1
├── 20_pilot.sh                  phases 2-3 (--quick) incl. walltime measurement
├── 30_full.sh                   phase 4 submissions
├── 40_mechanism.sh              angle_eval.py over every finished run
├── lib/
│   ├── gpu_sampler.sh           background nvidia-smi trace (§7.3)
│   └── sbatch_wrap.sh           wraps every array element: sampler + cost record
├── collect.py                   run tree -> results/<RUN_ID>/ light tier (§6.2)
├── stage_benchmarks.py          AtomBench staging tree (§6.3)
├── costs.py                     the cost report (§7.4)
├── analyze.py                   the statistical analysis (§8)
└── results/<RUN_ID>/            the self-contained deliverable (§6.2)

/scratch/crc00042/alignn_csp/                  <- CSP_RUNS, heavy tier
/scratch/crc00042/atombench/                   <- the clone
/scratch/crc00042/envs/{alignn2,atombench}/    <- the two conda envs
```

The only file inside the ALIGNN repo that gets written is
`task_runners/cluster.env`, which the repo explicitly designates site-local —
and `30_full.sh` **generates** it from `env.sh` rather than having you hand-edit
it, so the whole configuration is reproducible from this directory alone. A
copy of the generated file is archived into the results tree (§6.2).

---

## 3. `task_runners/cluster.env`, generated by `env.sh`

```bash
CSP_ACCOUNT="alromero"
CSP_PARTITION="gpu_7day"
CSP_QOS=""                              # only 'normal' exists; passing it is noise
CSP_CONSTRAINT=""                       # MUST stay empty: no node features defined
CSP_GPU_GRES="gpu:nvidia_a30:1"         # PINNED — see below
CSP_RESERVATION=""
CSP_MAIL_USER=""
CSP_MAX_CONCURRENT="8"                  # cap is 20 A30; 8 is polite and leaves room
CSP_SBATCH_EXTRA=""

CSP_MODULES=""                          # torch 2.5.1 ships its own cu121 runtime
CSP_ENV="/scratch/crc00042/envs/alignn2"
CSP_SCORE_ENV="/scratch/crc00042/envs/atombench"
CSP_PRE_RUN_HOOK="source $HARNESS/lib/gpu_sampler.sh"   # §7.3

CSP_RUNS="/scratch/crc00042/alignn_csp"
CSP_ATOMBENCH_REPO="/scratch/crc00042/atombench"
```

**Why `gpu:nvidia_a30:1` and not `gpu:1`.** `gpu_7day` mixes A30, A40 and A100.
A generic request schedules sooner but scatters the 18 arms across three GPU
generations, and then every wall-time number in §7 — including the 2.4×
per-step claim, which is one of the numbers on the table — becomes a
measurement of the scheduler. Pinning A30 costs some queue latency and buys a
homogeneous comparison. 116 A30 GPUs against a throttle of 8 is not a
bottleneck.

**MIG guard.** `gres/gpu:a30_2g.12gb` is registered in `AccountingStorageTRES`,
so MIG slices exist in this cluster's vocabulary even though no node currently
exposes one. Asking for `nvidia_a30` (not bare `gpu`) already excludes a MIG
slice, and `sbatch_wrap.sh` asserts the device name it actually got. A 12 GB
2g slice would silently halve throughput and poison every timing comparison.

`CSP_ENV` is a **path**, not a name — `alignn2` lives at
`/scratch/crc00042/envs/alignn2` and is not in `envs_dirs`, so `conda activate
alignn2` would fail. `common.sh` passes the value straight to `conda activate`,
which accepts a path.

`CSP_MODULES=""` is a claim to verify in phase 1: if the first GPU job reports
`torch.cuda.is_available() == False`, add whatever `module avail cuda` offers
on a compute node and resubmit.

---

## 4. sbatch header changes

The repo says plainly: *"The walltimes in the sbatch headers are placeholders.
They have not been measured on any particular machine."*

| directive | shipped | Dolly Sods | why |
|---|---|---|---|
| `--cpus-per-task` | 16 | **8** | 32 cores / 4 GPUs = 8 is the fair share. 16 halves node packing and lengthens the queue wait for no gain once `--relax-workers` matches. |
| `--mem` | 64G | **56G** | 257072 MB / 4 = 64.2 GB per GPU share; 64G leaves nothing for overhead and blocks co-scheduling. |
| `--time` | 36:00:00 | **measure in phase 3, then set** | §5 |

Pass `--relax-workers 8` to `run_task.py` so ALIGNN-FF relaxation forks to match
the allocation. `generate_stage` already sets `OMP_NUM_THREADS=1` for that
stage, so BLAS will not oversubscribe on top of it.

---

## 5. The six tasks

All train on the JARVIS Supercon-3D split (847/105/103), which is what the
angular arms are defined against. Unit counts are from `run_task.py --count`
on this machine.

| # | task | units | new GPU units | what it buys |
|---|---|---|---|---|
| 1 | `data-jarvis` | 1 | 0 (CPU) | the split; prerequisite for everything |
| 2 | `angle-ablation` | 18 | **18** | the suite: A0/A1/A2/A3/A4/A6 × seeds 0,1,2 |
| 3 | `bench-jarvis` | 3 | **0** | A0's "first job": reproduce the published 0.524 before any comparison is trusted |
| 4 | `ablation-linegraph` | 6 | **3** | adds `jarvis_nolg` — the no-angles-at-all floor |
| 5 | `symprec-sweep` | 1 | **1** | fixes the symmetrisation tolerance on **validation**, once, for every arm |
| 6 | `pipeline-ablation` | 4 × 2 arms | **8** | does the angular channel reduce how much force-field repair a sample needs? |

**30 new GPU array elements** — inside the 100-job submit cap and the 20-GPU
A30 cap.

- **3 and 4 are nearly free.** `bench-jarvis`, arm A of `ablation-linegraph`
  and A0 of `angle-ablation` all resolve to `train/jarvis_A0/seed{0,1,2}` with
  a byte-identical training command, so the stage markers match and whichever
  runs second skips straight through. Task 3 costs zero training and gives the
  published-number anchor; task 4 costs only the three `jarvis_nolg` runs and
  gives the reference delta — the README records the line-graph deletion as a
  *large* loss gap (2.351 vs 2.011) with *zero* change in match rate, which is
  exactly the calibration needed to read A0→A3.
- **5 runs before 6 and before any test scoring.** The symprec grid is swept on
  validation and the choice applied to every arm. Sweeping per-arm would be
  selecting a hyperparameter on the metric under test.
- **6 runs for A0 and A3.** Pairs with the relaxation-displacement mechanism
  metric: if A3's samples really are locally coherent, its `raw`→`full` gap
  should be *smaller* than A0's.

Excluded: `leakage` (needs `data-pretrain`, a 65k dft_3d download, and its
`alex` half has no Alexandria runs to filter — it would half-fail) and
everything Alexandria (the DS-A/B pickles are not on this machine).

### 5.1 `pipeline-ablation` collides with itself — the one real trap

`_pipeline_ablation` builds run directories as `ctx.out("pipeline", variant)`.
**The checkpoint is not in the path.** Running it twice with different
`--checkpoint` under one `--runs-root` writes both arms into
`runs/pipeline/{raw,rank,relax,full}` and the second silently overwrites the
first. `30_full.sh` gives each arm its own root:

```bash
for arm in A0 A3; do
  root=$CSP_RUNS/pipeline_arms/$arm
  mkdir -p "$root"
  ln -sfn "$CSP_RUNS/data" "$root/data"      # ctx.data is <root>/data
  bash task_runners/submit.sh pipeline-ablation \
      --runs-root "$root" \
      --checkpoint "$CSP_RUNS/train/jarvis_$arm/seed0/best_model.pt" \
      --symprec "$CHOSEN_SYMPREC" --relax-workers 8
done
```

`--checkpoint` is absolute so it reaches into the shared tree; only `ctx.data`
is root-relative, hence the symlink.

---

## 6. Output organisation

### 6.1 Two tiers, and why

| tier | where | holds | size | fate |
|---|---|---|---|---|
| **heavy** | `/scratch/crc00042/alignn_csp/` | checkpoints, `candidates.json`, optimiser state | tens of GB | stays on scratch; regenerable only with a GPU |
| **light** | `csp-ablation-bench/results/<RUN_ID>/` | every CSV, every metrics/history/config/stage JSON, logs, figures, tables, the report | well under 1 GB | the deliverable — archivable, movable, readable anywhere |

The property worth designing for: **the light tier is sufficient to re-derive
every number in the write-up without a GPU and without the checkpoints.** All
AtomBench metrics are pure functions of `pred.csv`, so keeping the CSVs keeps
the science. `/users` has 797 GB free of a 91%-full 8 TB — the light tier fits
comfortably; the heavy tier absolutely does not, which is why `CSP_RUNS` points
at scratch.

**`RUN_ID`** (e.g. `2026-08-30_full`, `2026-08-30_quick`) is set in `env.sh` and
namespaces the whole results tree, so a re-run never clobbers a previous one.
This mirrors the runner's own `_quick`/`_smoke` suffix discipline, which exists
for exactly this reason — the repo notes that without it a 20-minute sanity
check overwrites a checkpoint that cost days.

### 6.2 `results/<RUN_ID>/`

```
00_provenance/
    manifest.json         RUN_ID, UTC timestamps, ALIGNN git rev + dirty flag,
                          atombench git rev, conda env exports (both envs),
                          the generated cluster.env, every sbatch as submitted,
                          every run_task.py argv, sha256 of every collected file
    doctor.txt            run_task.py doctor output at submission time
    slurm_jobids.txt      array job ids -> task, for the sacct harvest (§7.2)
10_runs/<arm>/seed<N>/
    bench_sym.csv  bench_nosym.csv       the predictions (the scientific record)
    metrics_sym.json  metrics_nosym.json AtomBench metrics
    history.json  config.json            training curve, args, n_parameters
    stages/*.json                        per-stage elapsed_s, host, git, argv
    angle_eval.json                      mechanism metrics (§8.4)
    gpu_trace.csv                        30 s GPU mem/util samples (§7.3)
    slurm-<jobid>_<idx>.{out,err}        the logs
20_benchmarks/<arm>_seed<N>/             AtomBench staging tree (§6.3)
30_atombench/
    figures/*.png                        AtomBench's own bar charts
    numerical_calculations/              metrics_table.{json,tex}, epic_metrics.csv
40_stats/
    report.md  stats.json  contrasts.tex the analysis of §8
50_costs/
    computational_costs.{json,tex}       AtomBench-schema cost table (§7.4)
    sacct.tsv  stage_times.csv  gpu.csv  the raw cost record
    cost_report.md
60_report/
    REPORT.md                            the single narrative document
INDEX.md                                 what every file is, how it was made,
                                         and the exact command that regenerates it
```

`collect.py` builds `00_`–`10_` from the run tree, is idempotent, and refuses
to overwrite a `RUN_ID` that already has a `manifest.json` unless given
`--force`. `INDEX.md` is generated, not hand-written, so it cannot drift.

### 6.3 The AtomBench staging tree

Every prediction file in the run tree is called `pred.csv`, and AtomBench names
each benchmark after its containing directory. So `stage_benchmarks.py` builds
`20_benchmarks/<arm>_seed<N>/` with `pred.csv` and `metrics.json` symlinked
back into `10_runs/`. 21 subdirectories (6 arms + `nolg`, × 3 seeds). Those
directory names become the labels in every AtomBench figure and table.

---

## 7. Computational cost accounting

Nothing here is estimated; every number is measured and every gap is stated.

### 7.1 What the runner already records (and what it throws away)

`run_task.py` writes `<rundir>/.stages/<stage>.json` after **every** stage —
train, generate, symmetrize, score-nosym, score-sym — containing:

```json
{"argv": [...], "env": {...}, "finished": "...", "elapsed_s": 1234.5,
 "host": "dscog017", "git": "262ce52"}
```

So per-stage wall time, the node it ran on, and the exact code revision are
already on disk for all five stages. `config.json` adds `n_parameters` and the
full argument namespace.

**But `aggregate.py:print_cost` reads only `train_s` and `params`.** The
sampling, relaxation and scoring times are recorded and then never surfaced.
Given that generation is 103 targets × 32 candidates × 1000 denoising steps
plus 412 ALIGNN-FF relaxations, it is very plausibly the larger half of the
bill — and it is *not* arm-independent, because the smooth-topology arms
rebuild the graph on every forward pass (README, "Deviation 6"), which is a
sampling cost as much as a training one. `costs.py` harvests all five.

`history.json` records `{epoch, train, val}` per epoch with **no timestamp**,
so a per-epoch time series is not recoverable. Per-step time is instead derived
exactly (§7.4), which is what the 2.4× claim needs.

### 7.2 SLURM accounting — harvest it before it ages out

The authoritative record, free, and the one thing that disappears if you wait.
`30_full.sh` writes the array job ids to `00_provenance/slurm_jobids.txt`, and
`costs.py --harvest` runs immediately after each array completes:

```bash
sacct -j "$JOBID" --parsable2 --noheader --units=M --format=\
JobID,JobName,State,ExitCode,Partition,NodeList,ReqTRES,AllocTRES,\
Submit,Start,End,Planned,Elapsed,ElapsedRaw,TotalCPU,CPUTimeRAW,\
MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite,ConsumedEnergy
```

- `Planned` is queue wait — not a compute cost, but it is what determines when
  you actually get results, and it is the number that justifies the A30 pin.
- `TotalCPU` / `CPUTimeRAW` give CPU-seconds, which is how the relaxation
  workers show up.
- `MaxRSS` is host memory, sampled at 30 s by `jobacct_gather/cgroup`.
- `AllocTRES` may carry `gres/gpumem` and `gres/gpuutil` — both are registered
  in `AccountingStorageTRES`. **Verify in phase 1; do not promise them.**
- **`ConsumedEnergy` will read 0.** `AcctGatherEnergyType = (null)`, so this
  cluster does not gather energy. `costs.py` records the field and marks it
  `unavailable` rather than reporting a zero as if it meant zero joules.

### 7.3 GPU memory and utilisation — the one thing nothing records

No component records GPU memory. Rather than patch `train_csp.py` (which would
mean editing the repo, and would miss the sampling stage anyway),
`lib/gpu_sampler.sh` starts a background sampler from `CSP_PRE_RUN_HOOK`, which
`common.sh` evaluates inside every array element:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,\
utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 30 > "$RUNDIR/gpu_trace.csv" &
trap 'kill %1 2>/dev/null' EXIT
```

Fully external, covers training *and* generation, and gives a utilisation trace
rather than a single peak. `costs.py` reduces it to peak memory, mean/median
utilisation, and mean power draw. `power.draw` × elapsed is an *estimate* of
energy and will be labelled as such — it is not SLURM accounting.

The same wrapper asserts the GPU it received is a full `NVIDIA A30` and not a
MIG slice, and fails the element loudly if not (§3).

### 7.4 The cost report

`costs.py` emits `computational_costs.json` in **exactly the schema
AtomBench's own `scripts/harvest_compute_times.py` writes** —
`num_epochs`, `train_s`, `train_h`, `infer_s`, `infer_h`, `total_s`, `total_h`,
`train_s_per_epoch`, `num_test_structures`, `infer_s_per_structure` — plus a
booktabs `.tex`. Matching it means the ALIGNN-CSP arms drop straight into the
same table as the published AtomGPT / CDVAE / FlowMM / MatterGen baselines
instead of needing a second, incompatible cost table.

Per arm and seed, extended beyond that schema:

| field | source | note |
|---|---|---|
| `n_parameters` | `config.json` | the "matched to within 1%" claim |
| `train_s`, `train_s_per_epoch` | `.stages/train.json` | |
| **`train_s_per_step`** | `train_s / (epochs × ⌈847/64⌉)` = `train_s / (3000 × 14)` | epochs and batch size are pinned identically across arms, so the **ratio is exactly the per-step ratio** — this is the 2.4× claim, tested rather than quoted |
| `generate_s`, `infer_s_per_structure` | `.stages/generate.json` ÷ 103 | sampling **and** relaxation; `generate_benchmark.py` runs both in one stage, so the split is not separable without instrumenting it — stated as a limitation |
| `symmetrize_s`, `score_nosym_s`, `score_sym_s` | the remaining markers | CPU-only |
| `angle_eval_s` | `40_mechanism.sh` | GPU (uses `--relax`) |
| `gpu_h` | `elapsed_h × 1` | one GPU per element |
| `peak_gpu_mem_mib`, `mean_gpu_util_pct`, `mean_power_w` | `gpu_trace.csv` | |
| `cpu_s`, `max_rss_mib`, `queue_s` | `sacct` | |
| `node`, `gpu_name` | `.stages/*.json:host`, `gpu_trace.csv` | homogeneity audit |
| `energy_j` | — | **`unavailable`**, `AcctGatherEnergyType = (null)` |

Three derived figures the report leads with:

1. **Per-step cost ratio vs A0**, mean ± sd over seeds, against the README's
   claimed 2.4×. A measured disagreement is a finding, not an error to hide.
2. **Total GPU-hours per arm**, train + generate + mechanism, so the true price
   of the angular channel is visible rather than just its training half.
3. **GPU-seconds per matched structure.** The honest way to ask whether the
   angular channel is worth it: a 2.4× step cost that buys nothing is a
   different result from one that buys three more matches.

**Homogeneity audit.** With A30 pinned, `costs.py` still asserts that every
unit reports the same `gpu_name`, records the `NodeList` per unit, and flags
any unit whose `train_s_per_step` exceeds 1.5× its arm's median — node
contention is real and a contended run makes wall time incomparable even on
identical silicon. Flagged units are excluded from the timing comparison and
reported as excluded, never dropped silently.

---

## 8. Statistical analysis

### 8.1 What AtomBench provides

`atombench <path> <outdir>` takes a **directory**, discovers every CSV with
`id,target,prediction` columns, names each benchmark after its subdirectory,
computes match rate, Cartesian RMSD, ccRMSD (AMD), lattice `abc` and angle MAE,
KLD and per-crystal-system MAE, writes `metrics.json` beside each CSV, then
emits cross-benchmark bar charts and
`{metrics_table.json, metrics_table.tex, epic_metrics.csv}`.

Two facts shape the design:

1. **The schema already matches `score.sh`'s output.** The packaged
   `compute_metrics` returns the same nested dict as
   `scripts_consolidated/compute_metrics.py`, differing only in `ccRMSD` vs
   `ccRMSE` as the key — and ALIGNN's `collect_results.extract` already reads
   `raw.get("ccRMSD", raw.get("ccRMSE", {}))`. The two paths interoperate.
2. **`atombench.tables.collect_metrics(path)` reads `metrics.json` from beside
   each CSV — it does not recompute.** So the cross-arm analysis is free once
   the array has scored, and `analyze.py` imports it rather than reimplementing
   any metric.

### 8.2 What AtomBench does not provide

I grepped the package for `ttest`, `bootstrap`, `std(`, `sem(`, `wilcoxon`,
`confidence`: **no matches.** AtomBench has no notion of repeated seeds. It
treats each CSV as one independent benchmark and reports point estimates.

That is a specific problem here, because the README is emphatic: match rate
across fifteen independently trained models spanned **0.437–0.524** on a
103-target split. A bar chart of 21 point estimates would be actively
misleading.

### 8.3 What `analyze.py` adds

**Tier 1 — arm-level over seeds.** Mean ± sd per arm for every metric, plus a
Welch two-sided p for each contrast declared in `COMPARISONS`. This duplicates
`run_task.py --aggregate`, and the two are cross-checked against each other as
a consistency test. n = 3 per arm — every p here is descriptive.

**Tier 2 — per-target paired tests, where the power actually is.** Every arm is
scored on the *same* 103 targets; comparing three seed-means throws that
pairing away. Joining `pred.csv` on `id`:

- **match rate** is paired *binary* → **McNemar exact** on the discordant
  pairs, per contrast, plus a pooled Cochran–Mantel–Haenszel across seeds.
- **RMSD, ccRMSD, lattice MAE** are paired *continuous* per target →
  **Wilcoxon signed-rank** plus a paired **BCa bootstrap CI** on the mean
  difference (10,000 resamples over targets).
- **Hedges' g** with the small-sample correction, so effect size sits next to
  the p-value rather than instead of it.

This is a more powerful test of the *same* pre-registered metrics — not a new
metric and not a fishing expedition. The write-up should say so explicitly,
because the README pre-registered the suite precisely so a favourable metric
could not be chosen afterwards.

**Tier 3 — multiplicity.** Seven contrasts over seven metrics.
Holm–Bonferroni across the seven contrasts within each metric family, with the
three primary questions (A0↔A1, A0↔A2, A4↔A3) marked pre-registered and
everything else labelled exploratory.

### 8.4 The mechanism metrics are not wired up

`scripts/atombench/angle_eval.py` computes the two metrics the hypothesis
actually turns on — the bond-angle distribution distance (KL / JS / 1-D
Wasserstein in degrees, FoldingDiff's own diagnostic) and the relaxation
displacement (distance to the nearest ALIGNN-FF minimum, plus volume change and
energy drop).

I grepped `task_runners/` for `angle_eval`: **no references.** It is not a
stage in any task, and `claims.py` registers no angle-ablation claims. The
runner will never call it. `40_mechanism.sh` runs it as its own A30 array over
every finished run (it uses `--relax`, so it is a GPU job), writing
`angle_eval.json` into each run directory and its own cost record.

---

## 9. Execution ladder

### Phase 1 — plumbing (`debug`, 1 h cap, minutes of work)

```bash
bash 00_setup.sh                        # clone, install, doctor, freeze envs
bash 10_smoke.sh
```

CPU smoke first (`bench-jarvis --smoke --device cpu --runs-root
/scratch/crc00042/csp_smoke`): two epochs, two candidates, four targets, in a
throwaway root. Exercises train → sample → relax → symmetrise → score, so a
missing dependency or an unreachable AtomBench install surfaces in minutes
instead of after a queue wait.

Then **one A30 element on `debug`**, which is the phase that validates four
claims at once: `torch.cuda.is_available()` without a `module load`; the
allocated device really is a full A30 and not a MIG slice; `gpu_sampler.sh`
writes a usable trace; and `sacct` on that job id tells us whether `AllocTRES`
actually carries `gres/gpumem` and `gres/gpuutil`. Record the answer in
`INDEX.md` either way.

### Phase 2 — loss-only filter (`inter_a30`, 6 h cap)

```bash
bash task_runners/submit.sh angle-ablation --quick --loss-only --relax-workers 8
python task_runners/run_task.py angle-ablation --quick --aggregate
```

12 elements (6 arms × 2 seeds), 300 epochs, training only — no scoring env, no
sampling. The README calls the denoising validation loss the most reproducible
arm-vs-arm signal there is (it repeated to three decimals across two machines).
It also tells you almost nothing about match rate, by the repo's own account:
a filter for a broken arm, not a verdict on a real one.

### Phase 3 — the decision point (`gpu_2day`)

```bash
bash 20_pilot.sh
```

Full `--quick`: 300 epochs, 8 candidates, 2 seeds, **the whole test split, the
same pipeline, the same scoring code**, landing in `runs/train_quick/` where it
cannot touch a full checkpoint. Roughly a tenth of the cost and a genuine
arm-vs-arm comparison. Run `collect.py`, `costs.py` and `analyze.py` on it end
to end — this debugs the entire analysis and cost pipeline on real data before
the expensive run exists.

**This is where the walltime gets measured.** `20_pilot.sh` runs the §7.2 sacct
harvest, scales by 10 (epochs) and 4 (candidates), and prints the `--time` to
put in the phase-4 headers — replacing the placeholder rather than trusting it.

### Phase 4 — the real run (`gpu_7day`, A30 pinned)

```bash
python task_runners/run_task.py data-jarvis                        # CPU, minutes
bash 30_full.sh                # angle-ablation, ablation-linegraph, symprec-sweep,
                               # then pipeline-ablation per arm (§5.1)
bash 40_mechanism.sh           # angle_eval over every finished run
```

`gpu_7day` rather than `gpu_2day` because a walltime kill costs a fresh queue
wait and the runner resumes at *stage* granularity — a training killed at 90%
restarts from zero, not from a checkpoint.

Then the collection and analysis:

```bash
python collect.py                              # -> results/<RUN_ID>/{00,10}_*
python stage_benchmarks.py                     # -> 20_benchmarks/
atombench results/<RUN_ID>/20_benchmarks results/<RUN_ID>/30_atombench
python costs.py --harvest                      # -> 50_costs/
python analyze.py                              # -> 40_stats/, 60_report/REPORT.md
python task_runners/run_task.py angle-ablation --aggregate --latex   # cross-check
```

---

## 10. Risks and things to hold onto

**`angle_weight` is unswept.** Fixed at 1.0, and the README says so. It must be
identical across arms or the comparison silently becomes a hyperparameter
search. If A1 and A3 come out flat, sweeping it is the next step, not a
conclusion — and the sweep has to be reported.

**Validation loss will fall in the angular arms by construction.** `L_ang` is a
new term being optimised. The `loss` column is **not** comparable between arms
with and without the angular objective, and `analyze.py` will print it with
that warning attached rather than in the same table as the shared metrics.

**Interaction range is confounded with topology smoothness in A2/A3/A4.** With
`gate_pair_messages=True` and `r_c = 5 Å` those arms differ from A0 in two ways
at once. The README recommends running `--ablation A3 --gate-pair-messages 0`
before writing anything up: it is the difference between "smoothness helps" and
"truncating the pair range helps". **Still open** (see §12) — +3 units, ~10%
more compute, taking the total to 33, still well inside every cap. I'd run it
up front; deciding after A3 means a second queue wait on a 7-day partition.

**Information level is identical across arms, and is NOT identical against
the published baselines.** AtomBench groups models "by the information each
accesses at inference" and warns that comparing across those groups confounds
architecture with information. Within A0-A6 this is a non-issue: every arm
calls the same `make_generation_batch`, a "conditioning-only batch (no
ground-truth geometry)" carrying per-atom species, `natoms` and Tc, and the
ablation switches never touch conditioning. Against the baselines it is a real
confound -- CDVAE gets the full structure through a latent, AtomGPT gets
formula + Tc (our tier), FlowMM gets composition alone. So a win over FlowMM is
partly an information advantage and a loss to CDVAE is partly an information
disadvantage. `aggregate.py:print_baselines` prints all of them side by side
without tier annotation; the cross-model table must carry it. Two open items:
whether AtomGPT is given Z (we are, via `natoms`, so if it is not then even
that row is not level), and whether the MatterGen rows in `collect_results.py`
`BASELINES` are actually in the paper -- two readings of it disagreed.

**Seed spread can invert any conclusion.** Three seeds is the runner default
and the floor. If a contrast lands close, `--seeds 0,1,2,3,4` resizes the array
automatically — `submit.sh` sizes it from the arguments you actually pass.

**Back-tracking triplets put a large spike at 180°** in both generated and real
angle histograms. Inherited from the graph builder and identical across arms,
so it does not bias a comparison — but do not present it as physics.

**A4 is not a second baseline.** Its structural trunk is A3's with one
aggregation zeroed, a different function from A0's. It controls for A3 only,
and the analysis output labels it that way.

**Harvest `sacct` promptly.** SLURM's accounting database is the only source
for queue wait, CPU-seconds and MaxRSS, and it ages out. `costs.py --harvest`
runs immediately after each array, not at the end of the week.

**`/users` is 91% full (797 GB free of 8 TB).** The heavy tier goes to
`/scratch`; the light tier is designed to fit comfortably on `/users`.

---

## 11. What was built

Built and tested against a 21-run synthetic fixture on 2026-08-29. `README.md`
in this directory is the operating manual; this section records what exists and
where the implementation diverged from the plan above.

| file | lines | status |
|---|---|---|
| `env.sh` | 152 | paths, scheduler, resources, `RUN_ID`; generates `cluster.env`; `csp_submit` |
| `00_setup.sh` | 105 | environments, AtomBench clone + install, stub guard, doctor |
| `10_smoke.sh` | 127 | CPU smoke + the `debug` GPU probe |
| `20_pilot.sh` | 98 | phases 2–3 and the walltime measurement |
| `30_full.sh` | 204 | phase-4 submissions, confound arm, per-arm pipeline roots |
| `40_mechanism.sh` | 77 | `angle_eval.py --relax` as its own A30 array |
| `lib/gpu_sampler.sh` | 79 | GPU trace + whole-A30 assertion |
| `collect.py` | 380 | run tree → `results/<RUN_ID>/`, `INDEX.md`, `--verify` |
| `stage_benchmarks.py` | 95 | the AtomBench staging tree |
| `costs.py` | 475 | cost accounting, the 2.4× check, homogeneity audit |
| `analyze.py` | 720 | tiers 1–3 |
| `README.md` | 97 | run order and file map |

### Verified

* **Full chain** — collect → stage → `atombench` CLI → costs → analyze →
  verify, clean on 21 runs. The AtomBench CLI produced all eight figures plus
  `metrics_table.{json,tex}` and `epic_metrics.csv` from our staging tree.
* **Statistics, against hand calculations.** McNemar matches `scipy.binomtest`
  to 1e-12; CMH pooling increases evidence with strata and gives p ≈ 1 on
  symmetric input; Hedges' g recovers a known d = 0.8 and the correction
  shrinks it; BCa covers the true mean, excludes 0 for a real effect, covers 0
  under the null, and refuses n < 8; Holm matches by hand and is monotone.
* **The tier-2 argument, demonstrated.** On a fixture with a *planted* A0→A3
  improvement, the seed-level Welch test gives p = 0.070 (Holm 0.35) — it would
  be written off as noise — while the paired McNemar pooled across seeds gives
  **p = 0.002**. That gap is the entire reason tier 2 exists.
* **`costs.py` recovers an injected 2.4× per-step ratio** and emits valid
  booktabs LaTeX in AtomBench's cost schema.
* **`per_target` reproduces the fixture's planted match counts exactly**, so it
  agrees with AtomBench's StructureMatcher settings.

### Divergences from the plan above

**`lib/sbatch_wrap.sh` was not written.** On inspection it had no job left:
the GPU sampler enters through `CSP_PRE_RUN_HOOK`, which `common.sh` already
evaluates inside every element, and job ids come from `submit.sh`'s own stdout.
One less layer.

**`metrics.json` is copied into the staging tree, not symlinked.** Found the
hard way: the `atombench` CLI recomputes metrics and writes `metrics.json`
beside each CSV, and Python's `open(path, "w")` follows a symlink — so the
first design let the CLI silently overwrite the collected record and invalidate
its manifest checksum (6/6 files mutated in the test). `pred.csv` stays a
symlink because nothing writes to it. `10_runs/` is now immutable, and the
recomputed values can be diffed against it as a real cross-check.

**`collect.py --verify` was added** for exactly that failure class: it re-checks
every collected file against the manifest's sha256 and reports anything that
mutated the archival tier after collection.

**Holm adjusts over distinct arm pairs, not question names.** `COMPARISONS`
asks `("A0","A2")` twice — as *"does smooth topology alone help"* and as
*"A5: hard kNN vs smooth radius, with angles off"*. Two names, one test.
Counting it twice inflated the correction (7 tests instead of 6); the
adjustment now runs over the six unique pairs and maps back, and the report
notes the aliasing on both questions.

**The stub guard uses distribution metadata, not `__version__`.** The GitHub
package has no `__version__` attribute — only the PyPI stub does, since that
attribute is literally its entire contents. `importlib.metadata.version` gives
`0.1.0` for the real one and `2022.7.15` for the stub, and the absence of the
`atombench` console script is a second independent signal.

**The scoring environment is `/scratch/crc00042/envs/csp-ablation-bench`**, not
`.../atombench` — named for the project. ALIGNN is installed into it with
`--no-deps` as well, so `analyze.py` reads the declared contrasts from
`alignn.inverse.ablations` rather than a literal copy; `ablations.py` imports
only `typing`, so this pulls no torch into the scoring env.

**`analyze.py` tier 1 no longer requires pymatgen.** It reads `metrics.json`
and needs nothing else; only tier 2's per-target matching does. It now runs in
either environment and skips tier 2 with one clear line naming the python to
re-run with, instead of emitting one warning per run and a misleading summary.

**Stages resolve `python` from `PATH`, so the harness sets it.** `tasks.py`
builds every stage argv starting with the bare string `"python"`. Inside a job
that is correct, because `common.sh` has already `conda activate`d `CSP_ENV`.
Outside one it is a trap: running `"$TRAIN_ENV/bin/python" run_task.py ...`
launches the *runner* in the right environment and then every stage in whatever
python happens to be first on `PATH`. It failed exactly that way here
(`ModuleNotFoundError: numpy`). `env.sh` now provides `run_task`, which puts
the environment on `PATH` as well, and every direct call goes through it.

**The data tasks run in the scoring environment.** `prepare_data.py` imports
jarvis-tools, pymatgen, numpy and tqdm — and no torch at all. Rather than put
pymatgen into the training environment just to build a split, risking the numpy
and pandas pins of a working torch install, `env.sh` provides `run_task_data`,
which runs those stages where the crystallography already lives. This is also
the reason `doctor` reports pymatgen as missing and that is *not* a fault: it
probes the training environment, and by design pymatgen is not there.

**`DRY=1` was added to `30_full.sh` and `40_mechanism.sh`**, which print every
`sbatch` they would issue and submit nothing — the last checkpoint before the
run costs anything.

### Fixed on this machine

**`alignn2`'s editable install predated `alignn/inverse`** — `import alignn`
worked while `import alignn.inverse` raised `ModuleNotFoundError`, exactly what
`INSTRUCTIONS.md` warns about. `00_setup.sh` now repairs it with
`pip install -e --no-deps` rather than only reporting it, and the fix has been
applied.

**`task_runners/cluster.env` is generated and in place** with the A30 pin, the
`alromero` account and both environment paths. It is the only modified file in
the ALIGNN repository.

## 12. Still open

**The `--gate-pair-messages 0` confound arm — in phase 4 up front, or held
until A3 shows something?** (§10). It costs 3 units / ~10% more compute.
`env.sh` carries it as `RUN_CONFOUND_ARM`, **currently defaulting to 1** (my
recommendation: the README asks for it before write-up, and deciding later
means a second queue wait on a 7-day partition). Set it to 0 before
`30_full.sh` runs if you would rather wait.

**Should the three `task_runners` gaps be closed inside the ALIGNN repo
instead?** I found three: `aggregate.py:print_cost` surfaces only `train_s`
though all five stages record `elapsed_s`; `angle_eval.py` is not a stage in
any task; and `claims.py` registers no angle-ablation claims, so
`run_task.py verify` does not cover this branch at all. This plan covers all
three from outside. Adding a `cost` stage, an `angle_eval` stage and the claim
entries inside `task_runners/` is the cleaner long-term answer — but it is a
change to a repo you may want to keep clean for the manuscript, so I have not
assumed it.
