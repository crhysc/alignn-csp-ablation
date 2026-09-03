# Angular-diffusion ablation — results

Variant: `sym`. Contrasts from `alignn.inverse.ablations`.

## Arms

| arm | n | match rate | RMSD | ccRMSD | MAE abc | MAE ang | KLD |
|---|---|---|---|---|---|---|---|
| A0 | 3 | 0.4628 ± 0.0148 | 0.0426 ± 0.0075 | 0.4934 ± 0.0053 | 0.4848 ± 0.0216 | 10.3874 ± 1.4806 | 0.0221 ± 0.0024 |
| A1 | 3 | 0.4628 ± 0.0244 | 0.0403 ± 0.0108 | 0.4979 ± 0.0220 | 0.5016 ± 0.0083 | 10.0898 ± 0.5808 | 0.0226 ± 0.0008 |
| A2 | 3 | 0.4628 ± 0.0404 | 0.0385 ± 0.0179 | 0.5096 ± 0.0122 | 0.5323 ± 0.0439 | 11.2693 ± 0.2660 | 0.0251 ± 0.0014 |
| A3 | 3 | 0.4790 ± 0.0297 | 0.0376 ± 0.0202 | 0.4963 ± 0.0094 | 0.5032 ± 0.0017 | 9.9277 ± 0.5354 | 0.0222 ± 0.0002 |
| A3_nogate | 3 | 0.4887 ± 0.0448 | 0.0353 ± 0.0052 | 0.4995 ± 0.0120 | 0.4898 ± 0.0568 | 10.2105 ± 0.7062 | 0.0224 ± 0.0031 |
| A4 | 3 | 0.4466 ± 0.0257 | 0.0340 ± 0.0165 | 0.5243 ± 0.0192 | 0.5306 ± 0.0614 | 10.6132 ± 0.1179 | 0.0227 ± 0.0014 |
| A6 | 3 | 0.4757 ± 0.0485 | 0.0255 ± 0.0116 | 0.5089 ± 0.0434 | 0.5212 ± 0.0148 | 10.7712 ± 0.4565 | 0.0242 ± 0.0013 |
| nolg | 3 | 0.4595 ± 0.0312 | 0.0375 ± 0.0189 | 0.5107 ± 0.0129 | 0.5598 ± 0.0112 | 9.9415 ± 1.2490 | 0.0230 ± 0.0020 |

Arm descriptions:

- **A0** — baseline: current ALIGNN 2.0 diffusion, angles as features only
- **A1** — explicit angular denoising, baseline kNN line-graph topology
- **A2** — smooth radius topology, no angular denoising objective
- **A3** — proposed: explicit angular denoising + smooth topology
- **A4** — control: angular objective with the angle->bond coupling removed
- **A6** — A3 with the Fourier angular basis instead of ALIGNN's cosine RBF

> **Denoising validation loss is not in this table on purpose.**
> `L_ang` is a new term being optimised in the angular arms, so it
> falls there *by construction*. It is reported separately below and
> is not comparable between arms with and without the objective.

## Denoising validation loss (within-family only)

| arm | best val loss |
|---|---|
| A0 | 6.3041 ± 0.0733 |
| A1 | 6.5929 ± 0.0275 |
| A2 | 6.2556 ± 0.0449 |
| A3 | 6.5955 ± 0.0443 |
| A3_nogate | 6.5424 ± 0.0365 |
| A4 | 7.4611 ± 0.0283 |
| A6 | 6.6194 ± 0.0463 |
| nolg | 7.1780 ± 0.0349 |

## Contrasts

### does explicit angular denoising help

`A0` → `A1` (**pre-registered**)

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4628 | 0.4628 | 0.0000 | 1.000 | 0.00 | 1.000 |
| coordinate RMSD | 0.0426 | 0.0403 | -0.0023 | 0.779 | -0.20 | 1.000 |
| ccRMSD | 0.4934 | 0.4979 | 0.0045 | 0.762 | 0.22 | 1.000 |
| lattice MAE abc | 0.4848 | 0.5016 | 0.0168 | 0.310 | 0.82 | 1.000 |
| lattice MAE angles | 10.3874 | 10.0898 | -0.2976 | 0.770 | -0.21 | 1.000 |
| KLD | 0.0221 | 0.0226 | 0.0005 | 0.763 | 0.22 | 1.000 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 40 | 6 | 8 | 49 | 0.791 |
| 1 | 41 | 7 | 9 | 46 | 0.804 |
| 2 | 37 | 12 | 8 | 46 | 0.503 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.02, p = 0.888

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 118
- mean difference (test − ref): -0.0009 Å
- Wilcoxon p: 0.9432
- 95% CI (BCa): [-0.0132, 0.0113]

### does smooth topology alone help

`A0` → `A2` (**pre-registered**)

> Same arms, and therefore the same statistical test, as: *A5: hard kNN vs smooth radius, with angles off*. Counted once in the Holm adjustment.

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4628 | 0.4628 | 0.0000 | 1.000 | 0.00 | 1.000 |
| coordinate RMSD | 0.0426 | 0.0385 | -0.0041 | 0.741 | -0.24 | 1.000 |
| ccRMSD | 0.4934 | 0.5096 | 0.0162 | 0.135 | 1.38 | 0.674 |
| lattice MAE abc | 0.4848 | 0.5323 | 0.0475 | 0.194 | 1.10 | 1.000 |
| lattice MAE angles | 10.3874 | 11.2693 | 0.8819 | 0.411 | 0.66 | 1.000 |
| KLD | 0.0221 | 0.0251 | 0.0030 | 0.148 | 1.24 | 0.739 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 37 | 9 | 6 | 51 | 0.607 |
| 1 | 42 | 6 | 9 | 46 | 0.607 |
| 2 | 40 | 9 | 9 | 45 | 1.000 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.02, p = 0.885

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 119
- mean difference (test − ref): 0.0017 Å
- Wilcoxon p: 0.7519
- 95% CI (BCa): [-0.0038, 0.0153]

### do the two together help

`A0` → `A3` (exploratory)

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4628 | 0.4790 | 0.0162 | 0.461 | 0.55 | 1.000 |
| coordinate RMSD | 0.0426 | 0.0376 | -0.0050 | 0.718 | -0.26 | 1.000 |
| ccRMSD | 0.4934 | 0.4963 | 0.0029 | 0.677 | 0.30 | 1.000 |
| lattice MAE abc | 0.4848 | 0.5032 | 0.0184 | 0.277 | 0.96 | 1.000 |
| lattice MAE angles | 10.3874 | 9.9277 | -0.4597 | 0.654 | -0.33 | 1.000 |
| KLD | 0.0221 | 0.0222 | 0.0001 | 0.965 | 0.03 | 1.000 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 39 | 7 | 7 | 50 | 1.000 |
| 1 | 39 | 9 | 13 | 42 | 0.523 |
| 2 | 38 | 11 | 12 | 42 | 1.000 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.27, p = 0.603

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 116
- mean difference (test − ref): -0.0028 Å
- Wilcoxon p: 0.9886
- 95% CI (BCa): [-0.0163, 0.0068]

### is the coupling doing the work (not just auxiliary loss)

`A4` → `A3` (**pre-registered**)

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4466 | 0.4790 | 0.0324 | 0.228 | 0.93 | 1.000 |
| coordinate RMSD | 0.0340 | 0.0376 | 0.0036 | 0.824 | 0.16 | 1.000 |
| ccRMSD | 0.5243 | 0.4963 | -0.0280 | 0.111 | -1.48 | 0.668 |
| lattice MAE abc | 0.5306 | 0.5032 | -0.0274 | 0.520 | -0.50 | 1.000 |
| lattice MAE angles | 10.6132 | 9.9277 | -0.6856 | 0.151 | -1.41 | 0.757 |
| KLD | 0.0227 | 0.0222 | -0.0006 | 0.536 | -0.48 | 1.000 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 40 | 9 | 6 | 48 | 0.607 |
| 1 | 41 | 4 | 11 | 47 | 0.118 |
| 2 | 39 | 5 | 11 | 48 | 0.210 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 1.76, p = 0.185

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 120
- mean difference (test − ref): 0.0007 Å
- Wilcoxon p: 0.5136
- 95% CI (BCa): [-0.0110, 0.0145]

### A5: hard kNN vs smooth radius, with angles on

`A1` → `A3` (exploratory)

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4628 | 0.4790 | 0.0162 | 0.508 | 0.48 | 1.000 |
| coordinate RMSD | 0.0403 | 0.0376 | -0.0027 | 0.849 | -0.13 | 1.000 |
| ccRMSD | 0.4979 | 0.4963 | -0.0016 | 0.915 | -0.08 | 1.000 |
| lattice MAE abc | 0.5016 | 0.5032 | 0.0016 | 0.769 | 0.22 | 1.000 |
| lattice MAE angles | 10.0898 | 9.9277 | -0.1621 | 0.740 | -0.23 | 1.000 |
| KLD | 0.0226 | 0.0222 | -0.0004 | 0.447 | -0.60 | 1.000 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 39 | 9 | 7 | 48 | 0.804 |
| 1 | 40 | 10 | 12 | 41 | 0.832 |
| 2 | 39 | 6 | 11 | 47 | 0.332 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.29, p = 0.590

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 118
- mean difference (test − ref): 0.0028 Å
- Wilcoxon p: 0.5687
- 95% CI (BCa): [-0.0114, 0.0180]

### A5: hard kNN vs smooth radius, with angles off

`A0` → `A2` (**pre-registered**)

> Same arms, and therefore the same statistical test, as: *does smooth topology alone help*. Counted once in the Holm adjustment.

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4628 | 0.4628 | 0.0000 | 1.000 | 0.00 | 1.000 |
| coordinate RMSD | 0.0426 | 0.0385 | -0.0041 | 0.741 | -0.24 | 1.000 |
| ccRMSD | 0.4934 | 0.5096 | 0.0162 | 0.135 | 1.38 | 0.674 |
| lattice MAE abc | 0.4848 | 0.5323 | 0.0475 | 0.194 | 1.10 | 1.000 |
| lattice MAE angles | 10.3874 | 11.2693 | 0.8819 | 0.411 | 0.66 | 1.000 |
| KLD | 0.0221 | 0.0251 | 0.0030 | 0.148 | 1.24 | 0.739 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 37 | 9 | 6 | 51 | 0.607 |
| 1 | 42 | 6 | 9 | 46 | 0.607 |
| 2 | 40 | 9 | 9 | 45 | 1.000 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.02, p = 0.885

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 119
- mean difference (test − ref): 0.0017 Å
- Wilcoxon p: 0.7519
- 95% CI (BCa): [-0.0038, 0.0153]

### A6: does the angular basis matter

`A3` → `A6` (exploratory)

| metric | ref | test | Δ | Welch p | Hedges g | Holm |
|---|---|---|---|---|---|---|
| match rate | 0.4790 | 0.4757 | -0.0032 | 0.927 | -0.06 | 1.000 |
| coordinate RMSD | 0.0376 | 0.0255 | -0.0121 | 0.430 | -0.59 | 1.000 |
| ccRMSD | 0.4963 | 0.5089 | 0.0126 | 0.667 | 0.32 | 1.000 |
| lattice MAE abc | 0.5032 | 0.5212 | 0.0180 | 0.168 | 1.37 | 1.000 |
| lattice MAE angles | 9.9277 | 10.7712 | 0.8435 | 0.108 | 1.36 | 0.649 |
| KLD | 0.0222 | 0.0242 | 0.0021 | 0.107 | 1.80 | 0.642 |

**Paired match-rate test.** Both arms are scored on the same targets, so the informative quantity is the discordant pairs, not the difference of two rates.

| seed | both | only ref | only test | neither | McNemar p |
|---|---|---|---|---|---|
| 0 | 43 | 3 | 6 | 51 | 0.508 |
| 1 | 45 | 7 | 9 | 42 | 0.804 |
| 2 | 38 | 12 | 6 | 47 | 0.238 |

Pooled (Cochran–Mantel–Haenszel over 3 seeds): χ² = 0.00, p = 1.000

**Paired RMSD.** Wilcoxon signed-rank over per-target differences, with a BCa bootstrap CI resampled over targets.

- n pairs: 126
- mean difference (test − ref): -0.0015 Å
- Wilcoxon p: 0.3753
- 95% CI (BCa): [-0.0179, 0.0154]

## How to read this

- Tier-1 Welch p-values have n = 3 per arm and are **descriptive**. The README records match rate spanning 0.437–0.524 across fifteen independently trained models on a 103-target split; three seeds cannot resolve a few percent.
- The paired tests are the powered ones. They test the *same* pre-registered metrics, using the fact that every arm sees the same targets — not a new metric, and not a post-hoc selection.
- Holm columns adjust across the **distinct arm pairs** within each metric family. `COMPARISONS` names one pair twice, and two names for one test is still one test. Only the three marked pre-registered were fixed in advance; the rest are exploratory.
- **Information level.** All arms receive the same conditioning (per-atom species, `natoms`, Tc) and no ground-truth geometry, so these contrasts are not confounded by what the model was shown. That is *not* true of any comparison against the published AtomBench baselines, which sit at different information tiers (CDVAE: full structure; AtomGPT: formula + Tc; FlowMM: composition only). Do not read across tiers without saying so.
- **A4 is not a second baseline.** Its structural trunk is A3's with one aggregation zeroed, a different function from A0's. It controls for A3 only.
- A generated-vs-real bond-angle histogram will show a large spike at 180° in *both* distributions: back-tracking triplets, inherited from the graph builder and identical across arms. It does not bias a comparison and is not physics.
