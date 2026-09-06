# INSTALL — building this project on a new cluster and running the generation

Written 2026-09-06. Follow it top to bottom. Every command has been run on
atomgptlab (JHU) unless it says otherwise. If you are a model reading this:
do the steps in order, check each "you should see" line, and do not skip
`preflight.sh`.

## 0. What you are building

| environment | what runs in it | Python | key pins we ran with (see `envs/observed-*.txt` for the full lists) |
|---|---|---|---|
| training (`TRAIN_ENV`) | `train_csp.py`, `generate_benchmark.py`, `run_task.py` | 3.11 | torch 2.14.0+cu130, jarvis-tools 2026.6.12, numpy 2.4.6, spglib 2.7.0, ase 3.29.0, alignn editable |
| scoring (`SCORE_ENV_PATH`) | `score.sh`, `symmetrize.sh`, `angle_eval.py` | 3.11 | pymatgen 2026.5.4, AMD 2.6, atombench 0.1.0 (from GitHub, not PyPI), jarvis-tools, spglib |
| data prep (`DATA_ENV`) | `prepare_*_data.py`, only if you rebuild the splits (you do not need to) | 3.11 | pymatgen, jarvis-tools, numpy, pandas, tqdm |
| repo tools | `dvc`, `hf` | 3.12 | dvc 3.67.1, huggingface_hub 1.30.0 |

Training and scoring are two environments on purpose: AtomBench's
dependency pins can break a working torch install. `install.sh` builds the
first two and clones AtomBench; the third is optional; the fourth is a plain
venv.

The GPU we used is a single NVIDIA GB10 per node (CUDA 13.0 wheels). Your
cluster will differ; the only thing that has to change is the torch wheel
index (`--cuda-index-url`) and the SLURM values in `site.env`.

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/crhysc/alignn-csp-ablation.git
cd alignn-csp-ablation
git -C alignn log --oneline -1        # you should see the commit PROJECT_STATE.md names for the submodule
```

## 2. Repo tools and the data

```bash
python3 -m venv .tools && .tools/bin/pip install -q "dvc>=3" "huggingface_hub>=0.30"
export PATH="$PWD/.tools/bin:$PATH"
export HF_NAMESPACE=<the namespace in PROJECT_STATE.md §7>
export HF_TOKEN=<a read token, only needed if the repos are private>
bash tools/hf_sync.sh pull
```

You should see `hf download` fetch the DVC remote into `.dvc-remote/`, then
`dvc pull` populate, and `dvc status -c` say the cache is in sync. Check:

```bash
ls supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/results/jarvis/2026-09-02_lgmatrix/10_runs/B3/seed0/
# best_model.pt  config.json  history.json  metrics_nosym.json  metrics_sym.json  bench_*.csv  stages/ ...
ls datasets/jarvis_supercon3d datasets/alexandria_dsab
# train.json val.json test.json split_meta.json (+ raw/ for alexandria)
```

## 3. The compute environments

```bash
bash install.sh --workspace /scratch/$USER/csp --cuda-index-url https://download.pytorch.org/whl/cu<XYZ>
```

Pick `cu<XYZ>` from pytorch.org's install matrix for your driver. Without
`--cuda-index-url` you get whatever `pip install torch` resolves to, which
may be CPU-only. `install.sh --dry-run` prints every command first;
`install.sh --help` explains each step. It:

1. initialises the `alignn` submodule (already done by the clone),
2. finds or bootstraps conda/mamba for this machine's architecture,
3. creates the training env (torch + jarvis-tools + alignn editable) and the scoring env (pymatgen + `average-minimum-distance` + AtomBench editable, cloned from GitHub),
4. writes `site.env` at the repo root with the paths it determined and every SLURM knob as a commented placeholder.

You should see `[x] torch <version>, CUDA available: True` and
`[x] alignn.inverse.ablations (torch-free): ['A0', 'A1', ...]`.

To reproduce our exact package set instead of the installer's latest
resolution, install from `envs/observed-train.txt` and
`envs/observed-score.txt` (they are the dist-info names from the envs the
results were made with; torch's `+cu130` suffix is site-specific).

## 4. site.env — the only file you edit

Open `site.env` and fill in the placeholders. For each one the file shows
the `sinfo`/`sacctmgr` command that answers it on your site. The ones that
matter for generation:

```bash
export CSP_PARTITION="<your GPU partition>"
export CSP_GPU_GRES="gpu:1"            # or gpu:<type>:1 if your site types them
export CSP_ACCOUNT=""                  # only if your scheduler bills to an account
export REQUIRE_GPU_NAME="<substring of nvidia-smi's name, e.g. A100>"   # the sampler hook asserts it
export CSP_MAX_CONCURRENT="<how many GPUs you may hold at once>"
export CPUS_PER_TASK=16; export MEM_PER_TASK=96G; export RELAX_WORKERS=16
export MECH_TIME="12:00:00"            # walltime 50_unrelaxed.sh requests per element
```

Both harnesses' `env.sh` read `site.env` automatically. Do not edit `env.sh`.

## 5. Warm the force-field cache once (login node, training env)

`angle_eval.py --relax` in the unrelaxed pipeline uses ALIGNN-FF, fetched
from figshare on first use; parallel workers race on that download.

```bash
source supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026/env.sh
$TRAIN_ENV/bin/python -c "from alignn.ff.ff import get_figshare_model_ff; get_figshare_model_ff(model_name='matpes_r2scan')"
ls ~/.cache/atomgptlab/alignn_ff/    # should contain matpes_r2scan/
```

## 6. Put the checkpoints and data where the harness looks

The harness expects `$CSP_RUNS/train/<dataset>_<arm>/seed0/best_model.pt`
and `$CSP_RUNS/data/<dataset>/{train,val,test}.json`. The collected tree
drops the `<dataset>_` prefix, so restore it:

```bash
cd supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026
for ds in jarvis alex; do
  ( DATASET=$ds; source ./env.sh
    mkdir -p "$CSP_RUNS/train" "$CSP_RUNS/data"
    ln -sfn "$PWD/../datasets/$( [ $ds = jarvis ] && echo jarvis_supercon3d || echo alexandria_dsab )" "$CSP_RUNS/data/$ds"
    for arm in nolg A0 nolg_ad A3 nolg_b3 B3; do
      mkdir -p "$CSP_RUNS/train/${ds}_$arm/seed0"
      cp -p "results/$ds/2026-09-02_lgmatrix/10_runs/$arm/seed0/"{best_model.pt,config.json,history.json} "$CSP_RUNS/train/${ds}_$arm/seed0/"
    done )
done
DATASET=jarvis bash 00_setup.sh      # links the data store, checks envs, writes cluster.env, runs doctor
DATASET=jarvis bash preflight.sh     # must end with READY
DATASET=alex   bash preflight.sh
```

`preflight.sh` checks the envs import, the split hashes match
`split_meta.json`, the FF cache is warm, the partition exists, and that the
four matrix cells build with matching parameter counts. Fix every `[ ]` line
before submitting anything.

## 7. Run the generation (this is NEXT_TASK.md)

```bash
DATASET=jarvis bash 50_unrelaxed.sh --list    # 6 checkpoints
DATASET=jarvis bash 50_unrelaxed.sh           # one array job, 6 elements, %CSP_MAX_CONCURRENT at a time
DATASET=alex   bash 50_unrelaxed.sh
```

Each element: sample one candidate per test target with no force field,
symmetrise, score both, then `angle_eval.py`. Cost per cell is 2–6 minutes
of sampling plus a serial ~4 s/structure relaxation in `angle_eval` (~1 h per
alex cell) that does not gate the scores. Outputs land in
`$CSP_RUNS/train/<ds>_<arm>/seed0/bench/{raw,rawsym}/`.

If your scheduler rejects the generated sbatch, the header comes from
`site.env`; `DRY=1 DATASET=jarvis bash 50_unrelaxed.sh` prints it.

## 8. Bring the results back into the repo

```bash
DATASET=jarvis python3 collect.py --force     # copies bench_raw.csv, metrics_raw.json, ... into results/jarvis/2026-09-02_lgmatrix/10_runs/
DATASET=alex   python3 collect.py --force
cd ..
.tools/bin/python tools/write_metadata.py     # refreshes ablations/*.yaml (they will now show variant_raw)
dvc add supercon-*/results/*/2026-09-02_lgmatrix/10_runs
bash tools/hf_sync.sh push                    # dvc push + mirror the remote to the Hub
git add -A -- . ':!current_manuscript' && git commit -m "raw (force-field-free) benchmarks for all 12 cells" && git push
```

Then, in the harness, `LGM_MATRIX=state python analyze.py --variant raw` and
`--variant rawsym` write the tables and report.

## 9. If something breaks

- `cannot execute binary file` → you are running an env built for another
  architecture (atomgptlab's are aarch64). Rebuild with `install.sh` here.
- `FloatingPointError: non-finite symmetric matrix` → the submodule is older
  than alignn `f8121f4`; `git submodule update --init`.
- `preflight` says the split hash differs → you are not using
  `datasets/…` from DVC; re-run `tools/hf_sync.sh pull`.
- The sampler hook says `FATAL: got GPU '<name>', required '<other>'` →
  `REQUIRE_GPU_NAME` in `site.env` does not match `nvidia-smi` on your GPU
  nodes.
