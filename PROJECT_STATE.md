# PROJECT STATE — alignn-csp-ablation

**Read this first.** It is written for someone (or some model) who has never
seen this project and has just cloned it on a new machine. Last updated
2026-09-06 on atomgptlab (JHU WSE) by Claude Code with crhysc.

`NEXT_TASK.md` holds the one open job. This file holds everything else.

---

## 1. What this project is, in five sentences

ALIGNN-CSP is a conditional diffusion model for crystal structure prediction
(given composition and a target property, generate the lattice and fractional
coordinates). The `alignn/` submodule (branch `lg-angle-diffusion-matrix`)
adds a third diffusion channel: the crystal's **bond angles**, either as a
legacy auxiliary regression target (`angle_mode=derived_aux`) or, as of
2026-09-04, as an **independently diffused redundant state** on a persistent
triplet index (`angle_mode=independent`). Two experiment sets test whether
that channel helps: an eight-arm angular suite on JARVIS (done, null,
underpowered) and a crossed **3 angular tiers × 2 line-graph levels**
factorial on JARVIS and Alexandria (done, twelve cells, post-relaxation
benchmarks scored). The headline so far: the independent channel improves the
denoising loss on the small dataset and not on the large one, the line graph
improves the denoising loss on both while never improving reconstruction,
and nothing moves match rate outside noise. The open task is to score the
generator's own output without the force field, which was never saved.

## 2. Repository map

```
PROJECT_STATE.md              <- this file: the state of everything
NEXT_TASK.md                  <- the one open job (regenerate unrelaxed predictions)
README.md                     <- original layout/setup notes (install.sh, site.env)
HANDOFF.md, HANDOFF2.md       <- the two design briefs that produced the independent angular state
methods_inverse_rewrite.tex   <- (untracked, like current_manuscript: manuscript prose) axiomatic methods rewrite, 2026-09-04
install.sh                    <- portable env/submodule/AtomBench installer; writes site.env
tools/
  write_metadata.py           <- regenerates every ablations/*.yaml, INDEX.md and the hf/ staging from the files
  hf_publish.sh               <- creates/updates the three Hugging Face repos
  hf_sync.sh                  <- pull|push the DVC remote through Hugging Face
datasets/                     <- DVC-tracked (datasets.dvc): the two prepared splits + raw Alexandria inputs
alignn/                       <- git submodule: the model, the task runner, the AtomBench scripts
supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/               <- experiment set "angle-ablation" (A-suite)
supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/   <- experiment set "lgmatrix" (the 3x2)
  EXPERIMENT_SET.yaml         <- exhaustive context for the set (question, design, protocol, timeline, results, incidents)
  results/<dataset>/<run_id>/
    ablations/<arm>_seed<N>.yaml   <- one exhaustive record per ablation run (git)
    ablations/INDEX.md             <- table of the runs in this results dir (git)
    00_provenance/                 <- sbatch files, SLURM job ids, manifest with sha256s, job logs (git)
    10_runs/  (.dvc)               <- weights, predictions, metrics, histories, stage timings, GPU traces (DVC)
    _gputrace/ (.dvc)              <- raw nvidia-smi traces per job (DVC)
    20_benchmarks*, 40_stats, 50_costs, 60_report   <- analysis tiers (git)
hf/cards/                     <- README cards for the Hugging Face repos (git); hf/model, hf/datasets are staging (ignored)
```

Two harness directories, two experiment sets, both self-contained: each has
its own `env.sh` (single source of truth for paths and scheduler values),
numbered run scripts, `PLAN.md`, `README.md`, and now `EXPERIMENT_SET.yaml`.

## 3. The vocabulary, so the names line up

| you see | it means |
|---|---|
| `A0` | ablation key: no angular objective, hard kNN topology (`angle_mode=off`) |
| `A3` | ablation key: derived angular target + smooth radius topology + gated pair messages (`angle_mode=derived_aux`); = `B1` |
| `B3` | ablation key: **independent** angular diffusion state + smooth topology (`angle_mode=independent`) |
| `A1, A2, A4, A6, A3_nogate` | the A-suite's other arms (kNN+angles; topology alone; coupling cut; Fourier basis; gate off) |
| `nolg` | config dir: no line graph, 0 ALIGNN / 9 pair convs, ablation A0 |
| `nolg_ad`, `nolg_b3` | no line graph with A3 / with B3 |
| `A0`, `A3`, `B3` as config dirs | line graph present (3 ALIGNN / 3 pair convs) with that ablation |
| `jarvis_<x>`, `alex_<x>` | the same config dir on each dataset (run trees never collide) |
| `MATRIX` | the 2x2 {nolg, A0, nolg_ad, A3} — the *derived* tier; task `lg-angle-matrix[-alex]` |
| `MATRIX_STATE` | the 2x2 {nolg, A0, nolg_b3, B3} — the *independent* tier; task `lg-angle-state-matrix[-alex]`; shares nolg/A0 with MATRIX by config name |
| angular tier | `none` / `derived` / `independent` — the three levels of the crossed design |
| `sym` / `nosym` | scored after / without spglib symmetry idealisation (symprec 0.1); both post ALIGNN-FF relaxation |
| `raw` / `rawsym` | the generator alone, no force field (one candidate, no relax, no rank) — **not yet produced** |
| `L_struct` | validation lattice loss + 10 × fractional loss; the checkpoint-selection criterion and the denoising metric of record |

All of these are defined in `alignn/alignn/inverse/ablations.py` (the dicts
`ABLATIONS`, `ANGLE_STATE`, `MATRIX`, `MATRIX_STATE`) and realised by
`alignn/task_runners/tasks.py`.

## 4. Status of every experiment set

| set | dataset | run_id | cells/arms | trained | sym+nosym scored | raw scored | where the weights are |
|---|---|---|---|---|---|---|---|
| angle-ablation | jarvis | 2026-08-30_full | 8 arms × 3 seeds | 24/24 | 24/24 (+rawsym) | rawsym only | **not in this repo** (WVU Dolly Sods `/scratch/crc00042/alignn_csp`) |
| angle-ablation | alex | 2026-09_alex_dsab | 8 arms × 1 seed | 2 partial (cancelled) | 0 | 0 | DVC (partial A0, A1) |
| lgmatrix | jarvis | 2026-09-02_lgmatrix | 6 cells × 1 seed | 6/6 | 6/6 | **0/6** | DVC + HF |
| lgmatrix | alex | 2026-09-02_lgmatrix | 6 cells × 1 seed | 6/6 | 6/6 | **0/6** | DVC + HF |

Per-run detail (every hyperparameter, selected epoch, metrics, stage times,
SLURM job ids, hosts, GPU, sampler version, sha256 of every artifact) is in
each `results/<ds>/<run_id>/ablations/<arm>_seed<N>.yaml`. The set-level
narrative (question, design, protocol, timeline, results reading, incidents,
decisions) is in each harness's `EXPERIMENT_SET.yaml`.

## 5. The results in one table (lgmatrix, seed 0, `sym`)

| tier | arch | jarvis L_struct | jarvis match | alex L_struct | alex match |
|---|---|---|---|---|---|
| none | no line graph | 7.1485 | 0.4757 | 5.7178 | 0.5891 |
| none | line graph | 6.2792 | 0.4466 | 4.2778 | 0.5661 |
| derived | no line graph | 7.2223 | 0.4757 | 5.5913 | 0.5721 |
| derived | line graph | 6.2919 | 0.4660 | 4.3665 | 0.5758 |
| independent | no line graph | 6.6815 | 0.4660 | 5.7121 | 0.5903 |
| independent | line graph | 6.1718 | 0.4466 | 5.1182 | 0.5515 |

How to read it, and what is and is not supported, is in
`supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/EXPERIMENT_SET.yaml`
§5. The one-line version: nothing here is evidence *for* the angular channel
on the powered dataset; the generator-alone scores (NEXT_TASK.md) are what
decide it.

## 6. Restoring the full state on a new machine

```bash
git clone --recurse-submodules https://github.com/crhysc/alignn-csp-ablation.git
cd alignn-csp-ablation
python -m venv .tools && .tools/bin/pip install dvc huggingface_hub     # or any env with both
export HF_NAMESPACE=<HF_NAMESPACE>                                      # see §7
bash tools/hf_sync.sh pull        # downloads the DVC remote from the Hub, points DVC at it, `dvc pull`
bash install.sh                   # training/scoring envs, AtomBench clone, site.env with this site's paths
```

After `dvc pull`, every `10_runs/`, `_gputrace/` and `datasets/` directory is
populated and `dvc status -c` reports the cache in sync. To put a cell's
checkpoint where the harness expects it for generation:

```bash
# collected tree drops the dataset prefix; the run tree needs it
mkdir -p $CSP_RUNS/train/jarvis_B3/seed0
cp results/jarvis/2026-09-02_lgmatrix/10_runs/B3/seed0/{best_model.pt,config.json} $CSP_RUNS/train/jarvis_B3/seed0/
```

`install.sh` writes `site.env`; fill in the scheduler placeholders it leaves,
then `preflight.sh` in a harness refuses to proceed until every requirement is
met. Both harnesses' `env.sh` source `site.env` automatically.

## 7. Where things live off-repo

| thing | location |
|---|---|
| this repo | https://github.com/crhysc/alignn-csp-ablation (public) — branch `main` |
| model code | https://github.com/crhysc/alignn (public) — branch `lg-angle-diffusion-matrix`, pinned by the submodule at `f8121f4` |
| DVC remote (what `dvc pull` reads) | Hugging Face dataset repo `<HF_NAMESPACE>/alignn-csp-ablation-dvc` — content-addressed; `tools/hf_sync.sh pull` |
| checkpoints, browsable | Hugging Face model repo `<HF_NAMESPACE>/alignn-csp-angular-ablations` — one folder per (set, dataset, cell) with config, history, metrics, `ABLATION.yaml` |
| datasets, browsable | Hugging Face dataset repo `<HF_NAMESPACE>/alignn-csp-ablation-datasets` |
| on atomgptlab only | run trees `/data/ccamp104/alignn_csp_lgmatrix/{jarvis,alex}`, data store `/data/ccamp104/alignn_csp/*/data`, envs `/data/ccamp104/envs/*`, DVC remote dir `/data/ccamp104/dvc-remote/alignn-csp-ablation` |
| on WVU Dolly Sods only | the A-suite jarvis checkpoints (`/scratch/crc00042/alignn_csp`) |

`<HF_NAMESPACE>` is a placeholder until the first `tools/hf_publish.sh` run
records it (the script prints the `sed` that does so).

DVC note: DVC has no native Hugging Face remote (dvc 3.67, checked
2026-09-06), so the remote is a plain directory mirrored to a Hub dataset repo
by `tools/hf_sync.sh`. Everything DVC-shaped stays DVC-shaped; the Hub is
transport. `.dvc/config` names the remote `hfmirror`.

## 8. Pitfalls that cost real time (each is also in the relevant EXPERIMENT_SET.yaml)

1. **Pending SLURM jobs run the live working tree** and re-read
   `task_runners/cluster.env` when they start. Every shared-code edit made
   with jobs queued must be proven behaviour-identical first (unit lists,
   argv, and for the sampler a bitwise comparison). Never `bash env.sh --write`
   for one dataset while the other's jobs are pending.
2. **"Angle diffusion" means `angle_mode=independent` (B3).** `A3` is the legacy
   derived target. The first 2x2 was accidentally the legacy tier; it is kept
   as the middle level.
3. **The sampler quarantines diverged crystals** (alignn `f8121f4`). If you
   see `FloatingPointError: non-finite symmetric matrix`, the code is older
   than that commit.
4. **The generator's raw candidates are not saved** by `generate_benchmark.py`.
   `NEXT_TASK.md` explains what to do about it.
5. **Checkpoint selection is on the structural loss**, not the trained total,
   because only four of six arms have an angular term. The two rules pick
   different epochs in the independent-mode cells.
6. **atomgptlab is heterogeneous**: x86_64 login node, aarch64 GPU nodes,
   aarch64 conda envs. Tests run in `/data/ccamp104/envs/csp-test-x86`.
7. The `symprec` tolerance (0.1) was never swept on validation; the `sym`
   columns are at the default.

## 9. What is deliberately *not* in the repository

- `current_manuscript` (the full LaTeX draft of the ALIGNN 2.0 paper) — untracked at the root; it is not this project's and the repo is public.
- The A-suite's jarvis checkpoints, histories and configs — never left Dolly Sods.
- `task_runners/cluster.env` in the submodule — site-local and generated; the working copy here is atomgptlab's and is left modified on purpose.
- `hf/model`, `hf/datasets` — staging hardlinks regenerated by `tools/write_metadata.py`; only `hf/cards/` is tracked.

## 10. Pending on atomgptlab when this was written

SLURM arrays `13199` (jarvis) and `13200` (alex), `csp-lgm-unrelaxed`, six
elements each, PENDING behind another user's queue. They are the generation in
`NEXT_TASK.md`. If that task is done elsewhere, `scancel 13199 13200`.

## 11. Tooling on this machine

`/data/ccamp104/envs/repo-tools-x86/bin/{dvc,hf,python}` (dvc 3.67.1,
huggingface_hub 1.30.0, pyyaml). `gh` is authenticated as `crhysc` and set up
as the git credential helper for https pushes.
