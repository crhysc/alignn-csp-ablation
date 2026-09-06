# Results index — `2026-09_alex_dsab`

Collected 2026-09-06T15:04:24.575460+00:00 from `/data/ccamp104/alignn_csp/alex`
(`train/` tree). ALIGNN `f8121f4e` on `lg-angle-diffusion-matrix`  **working tree dirty**, AtomBench `324ed9d1`.

2 runs over 2 arms: A0, A1.

This directory is self-contained: every metric here is a pure function
of the prediction CSVs, so the whole analysis re-runs with no GPU and
no checkpoints. The checkpoints stay in the run tree above.

## What is where

| path | contents | regenerate with |
|---|---|---|
| `00_provenance/manifest.json` | git revisions, env freezes, per-file sha256 | `python collect.py` |
| `00_provenance/cluster.env` | the generated site config, as submitted | `bash env.sh --write` |
| `00_provenance/doctor.txt` | dependency check at submission time | `bash 00_setup.sh` |
| `00_provenance/phase1_probe.txt` | CUDA / MIG / sampler / sacct-TRES probe | `bash 10_smoke.sh` |
| `00_provenance/slurm_jobids.txt` | array job ids, for the sacct harvest | `bash 30_full.sh` |
| `10_runs/<arm>/seed<N>/` | predictions, metrics, history, config, stage costs, GPU trace | `python collect.py` |
| `20_benchmarks/<arm>_seed<N>/` | AtomBench staging tree (symlinks) | `python stage_benchmarks.py` |
| `30_atombench/` | AtomBench's own figures and metrics table | `atombench 20_benchmarks 30_atombench` |
| `40_stats/` | seed statistics, paired tests, contrasts | `python analyze.py` |
| `50_costs/` | cost table, sacct record, GPU traces reduced | `python costs.py --harvest` |
| `60_report/REPORT.md` | the narrative document | `python analyze.py` |

## Per-run files

| file | what it is |
|---|---|
| `bench_sym.csv` / `bench_nosym.csv` | predictions, symmetrised and not. Both are kept because they answer different questions: the manuscript's lattice columns are measured after symmetrisation, the pipeline ablation before it. |
| `metrics_sym.json` / `metrics_nosym.json` | AtomBench metrics for each |
| `history.json` | per-epoch train/val denoising loss (no timestamps — per-step cost is derived in `50_costs/`) |
| `config.json` | the full argument namespace plus `n_parameters` |
| `stages/*.json` | per-stage `elapsed_s`, `host`, git revision, exact argv |
| `angle_eval.json` | bond-angle distribution distance and relaxation displacement |
| `gpu_trace.csv` | 30 s samples of GPU memory, utilisation and power |

## Incomplete

20 expected file(s) were absent at collection time — unfinished or failed stages:

```
A0/seed0: bench/nosym/pred.csv
A0/seed0: bench/nosym/metrics.json
A0/seed0: bench/sym/pred.csv
A0/seed0: bench/sym/metrics.json
A0/seed0: bench/sym/angle_eval.json
A0/seed0: bench/raw/pred.csv
A0/seed0: bench/raw/metrics.json
A0/seed0: bench/rawsym/pred.csv
A0/seed0: bench/rawsym/metrics.json
A0/seed0: bench/rawsym/angle_eval.json
A1/seed0: bench/nosym/pred.csv
A1/seed0: bench/nosym/metrics.json
A1/seed0: bench/sym/pred.csv
A1/seed0: bench/sym/metrics.json
A1/seed0: bench/sym/angle_eval.json
A1/seed0: bench/raw/pred.csv
A1/seed0: bench/raw/metrics.json
A1/seed0: bench/rawsym/pred.csv
A1/seed0: bench/rawsym/metrics.json
A1/seed0: bench/rawsym/angle_eval.json
```
