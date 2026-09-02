# Computational cost

Baseline arm: `A0`. Mean ± s.d. over seeds.

## The 2.4× per-step claim

The README records the angular channel as costing **2.4× per training
step**. Epochs and batch size are pinned identically across arms, so
the wall-time ratio below *is* the per-step ratio — this is a test of
that number, not a restatement of it.

| arm | s/step | × baseline |
|---|---|---|
| A0 | 0.0319 ± 0.0008 | 1.00 |
| A1 | 0.0359 ± 0.0017 | 1.13 |
| A2 | 0.0309 ± 0.0002 | 0.97 |
| A3 | 0.0352 ± 0.0002 | 1.10 |
| A3_nogate | 0.0359 ± 0.0001 | 1.13 |
| A4 | 0.0357 ± 0.0015 | 1.12 |
| A6 | 0.0357 ± 0.0005 | 1.12 |
| nolg | 0.0293 ± 0.0006 | 0.92 |

## Total cost per arm

| arm | params | train | inference | total GPU-h | GPU-s per match |
|---|---|---|---|---|---|
| A0 | 3.79 M | 22m18s | 22m27s | 0.75 | 56 |
| A1 | 3.86 M | 25m06s | 21m14s | 0.77 | 58 |
| A2 | 3.79 M | 21m37s | 21m05s | 0.71 | 54 |
| A3 | 3.86 M | 24m36s | 22m19s | 0.78 | 57 |
| A3_nogate | 3.86 M | 25m08s | 20m04s | 0.75 | 54 |
| A4 | 3.86 M | 25m00s | 21m39s | 0.78 | 61 |
| A6 | 3.86 M | 24m58s | 21m11s | 0.77 | 57 |
| nolg | 3.75 M | 20m31s | 13m14s | 0.56 | 43 |

`inference` is sampling + relaxation + symmetrisation + scoring.
`generate_benchmark.py` runs sampling and ALIGNN-FF relaxation in one
stage, so the split between them is not separable without
instrumenting that script — a stated limitation, not an omission.

## GPU

| arm | peak memory | mean utilisation |
|---|---|---|
| A0 | 3.9 GiB | 53% |
| A1 | 7.2 GiB | 51% |
| A2 | 8.2 GiB | 51% |
| A3 | 7.3 GiB | 50% |
| A3_nogate | — GiB | —% |
| A4 | 8.2 GiB | 50% |
| A6 | 5.9 GiB | 52% |
| nolg | 2.1 GiB | 42% |

Energy: **not available**. `AcctGatherEnergyType = (null)` on this
cluster, so `sacct ConsumedEnergy` reads 0 and is recorded as
`unavailable` rather than as zero joules. The `est_energy_kwh` column
in `gpu.csv` integrates `power.draw` from the sampler and is an
estimate, not accounting.

## Audit

- no GPU device recorded (sampler did not run); hardware homogeneity is unverified.
- jarvis_A0/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A0/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A0/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A1/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A1/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A1/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A2/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A2/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A2/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3_nogate/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3_nogate/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A3_nogate/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A4/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A4/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A4/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A6/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A6/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_A6/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_nolg/seed0: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_nolg/seed1: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
- jarvis_nolg/seed2: 0/412 null energies, 4 worker errors -- relaxation degraded, treat these metrics as unreliable
