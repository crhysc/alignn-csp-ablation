# PROGRESS — `alex/2026-09_alex_dsab` (Alexandria DS-A/DS-B)

**Status: CANCELLED mid-run on 2026-09-02 at the user's request.**
Nothing was lost that cannot be re-derived; no results were produced.

This run is the port of the angular-diffusion ablation suite from JARVIS
Supercon-3D (`results/jarvis/2026-08-30_full/`) to the **Alexandria DS-A/DS-B**
split used by AtomBench. Read `../../../../HANDOFF.md` §2 for *why* the benchmark
moved: the JARVIS run returned every pre-registered contrast at Holm-adjusted
p = 1.000 on a 103-target test set, and the honest reading was underpowered,
not null. DS-A/B gives an **825-target** test set — 8× larger.

Per the user's instruction this run uses **1 seed per arm**, not 3, because
DS-A/B is a much larger dataset and the 3-seed budget does not fit.

---

## 1. What was completed and verified

| step | status |
|---|---|
| Located DS-A/DS-B via the AtomBench data-availability statement | done |
| Downloaded + staged AtomBench repo and data (`/data/ccamp104/atombench`) | done |
| Cluster port: aarch64, GB10, 2 GPUs, no `/scratch`, no SLURM account | done |
| Split `results/` into separate `jarvis/` and `alex/` trees | done |
| Niggli canonicalisation of training targets *and* predicted structures | done, verified |
| Data prep — **6603 / 825 / 825**, 0 canonicalisation fallbacks | done, verified |
| 8 arms × 1 seed submitted to SLURM, chained on the data job | done |
| Training | **cancelled at epoch 375 (A0) / 350 (A1) of 1000; 6 arms never started** |
| Generation, relaxation, scoring | never reached |
| Stats, contrasts, report | not started |

### Split provenance (`/data/ccamp104/alignn_csp/alex/data/alex/split_meta.json`)

```json
{"dataset":"alexandria_DS-A_DS-B","target_key":"Tc","seed":123,
 "max_size":8253,"n_train":6603,"n_val":825,"n_test":825,
 "hash10_ids":"5703564835"}
```

Matches the partition AtomGPT / CDVAE / FlowMM / MatterGen were evaluated on.
`scripts/atombench/prepare_alex_data.py` mirrors AtomBench's own
`alexandria_preprocess.py` — DS-A then DS-B concatenated in that order, rows
skipped on missing `Tc` or unparseable `structure`, truncate at 8253, then
`random.seed(123); shuffle` and 80/10/10. No duplicate/leakage drops (unlike
the JARVIS pipeline — AtomBench does not do them here).

### Canonicalisation check — the thing that was actually broken before

Training targets and the AtomBench metric now live in **one basis**:

| check | result |
|---|---|
| targets already at AtomBench's canonical fixed point | **820 / 825** |
| worst drift for the remaining 5 | 1.5×10⁻⁴ Å (numerical noise) |
| targets whose atom count the metric would change | **0** — `natoms` conditioning is stable |
| `target_poscar` agrees with `lattice_mat` | yes |

The 39 cells that fail `a≤b≤c` were a **false alarm** raised and then retracted
during the run: pymatgen's Niggli output is not length-sorted, but those cells
*are* the canonical cell the metric produces. Do not "fix" this.

---

## 2. Where the run stopped

Cancelled with `scancel 12824 12825 12826 12827 12828`. Queue is empty and no
stray `train_csp` / `generate_benchmark` processes remain.

```
12824_0  alex_A0  seed0  RUNNING   → CANCELLED after 00:26:57  (epoch 375/1000)
12824_1  alex_A1  seed0  RUNNING   → CANCELLED after 00:26:57  (epoch 350/1000)
12824_[2-5]  A2,A3,A4,A6  PENDING  → CANCELLED (JobArrayTaskLimit, never ran)
12826_[0-1]  alex_nolg           PENDING → CANCELLED (never ran)
12828_[0]    alex_A3_nogate      PENDING → CANCELLED (never ran)
12825, 12827 csp-aggregate       PENDING → CANCELLED (dependency)
```

**Surviving artefacts** (not deleted — resume or discard as you choose):

```
/data/ccamp104/alignn_csp/alex/data/alex/{train,val,test}.json   <- reusable, verified
/data/ccamp104/alignn_csp/alex/train/alex_A0/seed0/{best_model.pt,history.json,config.json}
/data/ccamp104/alignn_csp/alex/train/alex_A1/seed0/{best_model.pt,history.json,config.json}
results/alex/2026-09_alex_dsab/00_provenance/{slurm_jobids.txt,confound.sbatch}
results/alex/2026-09_alex_dsab/_gputrace/*.csv
```

The data prep is the expensive verified step and **it does not need to be
re-run**. `best_model.pt` for A0/A1 is the epoch-176/174 checkpoint (see §3),
not the final one, so it is usable if you want a cheap generation smoke test.

---

## 3. Two findings worth keeping

**(a) Measured training cost — training is not the bottleneck.**

```
alex_A0-seed0:  4.26 s/epoch  ->  ~1.2 h for 1000 epochs
alex_A1-seed0:  4.38 s/epoch  ->  ~1.2 h for 1000 epochs
```

An earlier claim in the session that A1 ran >2× slower than A0 was **wrong** —
it was read off a single early log point. Measured over 300+ epochs the two are
within 3%. The README's 2.4× figure applies to A3-style arms, and had not been
measured here. The epoch rate did drift upward early (2.9 → 4.5 s/epoch over
epochs 1→100) and then flattened at ~4.3, consistent with warm-up/clock
settling rather than thermal or memory-pressure degradation.

**(b) 1000 epochs is the wrong budget for this split — both arms overfit at ~175.**

| arm | best val loss | at epoch | val loss at cancel |
|---|---|---|---|
| A0 | 4.2961 | **176** | 4.7798 (epoch 375) |
| A1 | 4.5449 | **174** | 5.0054 (epoch 350) |

Train loss kept falling (A0: 3.15 by epoch 350) while val loss rose steadily
from epoch ~175 onward. Both arms turned over at essentially the same epoch, so
this is a property of the split, not of an arm. Before resubmitting, consider
early stopping or a ~300-epoch budget — that alone cuts training wall-clock by
~3×, and the checkpoint selection already takes the best-val model so the
*results* would be nearly unchanged.

**(c) The unknown that actually governs the schedule: generation.**
825 targets × 32 candidates × 1000 denoising steps + ALIGNN-FF relaxation, on a
memory-bandwidth-limited GB10. This number was never measured — no arm reached
the generate stage. It is 8× the JARVIS target count. **Measure it on one arm
before committing to all eight.** With 8 arms ÷ 2 GPUs = 4 sequential rounds,
generation cost is what decides whether the suite is a 2-day or a 2-week job.

---

## 4. Code changes made for the port — all UNCOMMITTED

Both the superproject and the `alignn` submodule have dirty working trees. The
submodule is on a **detached HEAD** at `67c2b82`, so commit to a branch there
before anything reclaims it.

**`csp-ablation-bench/` (superproject), 367 insertions / 141 deletions:**
`env.sh` (+297/-? — the bulk: `DATASET` switch, `results/$DATASET/$RUN_ID`
layout, GB10/aarch64 paths under `/data/ccamp104`, empty `CSP_CONSTRAINT`,
no-account SLURM), `30_full.sh`, `preflight.sh`, `10_smoke.sh`, `20_pilot.sh`,
`60_unrelaxed.sh`, `collect.py`, `costs.py`, `.gitignore`.
131 staged renames move the old run to `results/jarvis/2026-08-30_full/`.

**`alignn/` submodule, 272 insertions / 85 deletions:**
`scripts/atombench/prepare_alex_data.py` (new DS-A/B prep),
`prepare_data.py`, `score.sh` (canonicalise predictions before scoring),
`task_runners/{tasks.py,common.sh,submit.sh,cluster.env}`,
`sbatch/data-{alex,jarvis,pretrain}.sbatch`, plus two **untracked** files:
`sbatch/angle-ablation-alex.sbatch`, `sbatch/ablation-linegraph-alex.sbatch`.

### Two bugs found and fixed during the run — keep these

1. **Data task ran in the training env** and died on `ModuleNotFoundError:
   pymatgen`, which lives in the scoring env by design. `common.sh` now honours
   a `CSP_WANT_SCORE_ENV` flag set by the three `data-*.sbatch` files, and the
   switch happens *after* `cluster.env` is sourced so it is not overwritten.
2. **Training arms were not chained to the data task.** Making data prep an
   asynchronous submitted job silently dropped the ordering the old synchronous
   call got for free. All three arm submissions now carry
   `--dependency=afterok:<data job id>`.

---

## 5. How to resume

Data prep is done and verified; you are resubmitting the arms only.

```bash
cd /home/ccamp104/repositories/alignn-csp-ablation/csp-ablation-bench
DATASET=alex bash preflight.sh          # re-check env, paths, GPUs
DATASET=alex bash 30_full.sh            # resubmits the 8 arms, 1 seed
```

`RUN_ID` must stay `2026-09_alex_dsab` to land back in this directory.
Before you do, decide the two open questions from §3:

- **epoch budget** — 1000 is demonstrably too many; ~300 with early stopping
  reproduces the same best-val checkpoint at a third of the cost;
- **arm count** — if generation turns out expensive, `A4` (the `A3` control)
  and `A3_nogate` (the confound) are the two that can be cut without losing
  the main `A0 → A3` contrast. Cutting them takes 8 arms to 6, i.e. 4 rounds
  to 3 on 2 GPUs.

Consider running **one arm end-to-end first** to price generation, rather than
resubmitting all eight blind. That is the number nobody has yet.

---

*Written 2026-09-02 after cancelling the run. Companion documents:
`../../../../HANDOFF.md` (why the benchmark moved clusters),
`../../../PLAN.md` (the design document),
`../../jarvis/2026-08-30_full/60_report/REPORT.md` (the null JARVIS result).*
