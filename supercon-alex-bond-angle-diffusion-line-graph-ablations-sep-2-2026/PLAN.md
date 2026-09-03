# PLAN — line graph × bond-angle diffusion

Design document. Written 2026-09-02, atomgptlab (JHU WSE).

---

## 1. The question

The manuscript makes two separate claims about three-body information in the
ALIGNN-CSP denoiser.

1. **The line graph earns its place.** Table 3 (`tab:inverse_ablation`) deletes
   the angular channel, spends the same budget on pair-graph depth, and reports
   denoising loss 14.5% lower with it (2.011 ± 0.018 vs 2.351 ± 0.007, no
   overlap across twelve runs) — while match rate does not move at all,
   agreeing to four decimal places between the arms.
2. **Bond-angle diffusion is a good inductive bias.** Angles become a *denoised
   variable* rather than a conditioning feature. This was added after Table 3
   was written, and the eight-arm suite that tested it
   (`../supercon-alex-bond-angle-diffusion-ablation-sep-2-2026/`) returned every
   pre-registered contrast at Holm-adjusted p = 1.000 on 103 targets.

**These were never crossed.** Claim 1 was measured with the angular objective
off in both arms; claim 2 was measured with the line graph on in every arm. So
nobody has asked the question that decides whether they are the same mechanism:

> Does the angular denoising objective still buy anything once the architecture
> that carries angles is gone — and does the line graph still buy anything once
> angles are supervised?

A 2×2 answers both, plus the interaction, which is the part neither one-factor
experiment can reach.

## 2. The cells

Defined in `alignn/inverse/ablations.py:MATRIX`. Nothing new is *configured*
there — each cell names an existing switch set plus the depth that realises it.

| cell | config dir | ablation | ALIGNN / pair | topology | params |
|---|---|---|---|---|---|
| `neither` | `<split>_nolg` | A0 | 0 / 9 | kNN | 3,489,478 |
| `line graph` | `<split>_A0` | A0 | 3 / 3 | kNN | 3,528,518 |
| `angle diffusion` | `<split>_nolg_ad` | A3 | 0 / 9 | radius + gate | 3,594,567 |
| `both` | `<split>_A3` | A3 | 3 / 3 | radius + gate | 3,594,567 |

and its compute-matched twin (§2.3), which reuses the two line-graph cells:

| cell | config dir | ablation | ALIGNN / pair | params |
|---|---|---|---|---|
| `neither (compute)` | `<split>_nolg_d16` | A0 | 0 / 16 | 5,799,398 |
| `angle diffusion (compute)` | `<split>_nolg_ad_d16` | A3 | 0 / 16 | 5,904,487 |

Six distinct configurations in total (parameter counts here are denoiser-only;
`config.json` records the full model, which adds a constant 0.26 M conditioner
— the manuscript's "3.79 M versus 3.75 M" are the full-model figures).

Everything not under test is fixed by the shared builders in `tasks.py`: hidden
size 256, kNN 12, 1000 denoising steps, batch 64, lr 1e-3, one-cycle schedule,
EMA, guidance 2.0, 32 candidates with a 4-way relaxed prescreen, the same
symmetrisation tolerance, the same seeds.

### 2.1 Why the budget is matched at nine convolution blocks, not at parameters

Three ALIGNN layers are six edge-gated convolutions (a node update and an edge
update each) plus three pair convolutions. The no-line-graph cells spend the
same nine on the pair graph. This is the manuscript's own control and it is
kept unchanged.

It matches *depth* exactly and *parameters* to 1.106% — the residual is the
angle encoder (39,040 parameters: an RBF expansion, which is a buffer, plus two
`MLPLayer`s). The manuscript rounds this to "within 1%".

The angular cells match each other **exactly**, to the parameter, because the
angular path introduces no weights of its own beyond the encoder and head that
both share (§3).

### 2.2 Why continuity rides with the angular objective and is not a third factor

`A3` differs from `A0` in two switches: `angle_diffusion` and
`topology="radius"` with `gate_pair_messages`. It would look more careful to
separate them. It is not.

Under the hard kNN rule a bond joins a triplet if it is among the twelve
shortest at its destination atom. During reverse diffusion, when coordinates
are near-uniform, neighbour ranks swap constantly — so the triplet set jumps,
θ is a **discontinuous** function of the coordinates, and a denoising target on
θ is not well defined. `alignn/inverse/angles.py:39-46` makes exactly this
argument. `A1` — angle diffusion on hard kNN — is the incoherent configuration,
not the clean one.

The radius topology is what makes θ continuous, and it is architecturally free.
Measured, not assumed:

* `A0` and `A2` (identical but for the topology) have **the same parameter
  count**, 3,528,518, and an identical named-parameter tree.
* `CutoffPolynomial` holds 0 parameters and 0 buffers — three floats computed
  in `__init__`.
* `WeightedALIGNNConv` / `WeightedEdgeGatedGraphConv` take the weights as
  optional arguments defaulting to `None`, with parameter names and shapes
  deliberately identical to the stock layers so checkpoints interchange.

What *does* change is the graph and the message weights, both derived from
geometry. On a random 8-atom cell: in a 4 Å cube kNN and radius both keep all
64 bonds and 512 triplets; in an 8 Å cube kNN keeps 64 bonds and 512 triplets
while radius keeps 48 and 294, of which 27% carry `s_ijk < 0.01` and are fading
out. So it is a real change to the computed function, and a non-change to the
architecture. That is the right thing to bundle with "angles are a continuous
variable now".

`gate_pair_messages` additionally scales the atom-graph messages and the
per-edge terms of the coordinate score by the same envelope. Also parameter-
free; also not a no-op. It is kept on so that the `both` corner is literally
`A3`, the model the manuscript proposes.

### 2.3 Normalisation: parameters or compute, not both

Depth is the compensation knob, and it can level one axis at a time. Measured
on a GB10 over a real 64-crystal Alexandria batch (299 atoms, 1,497 pairs), as
a full forward + backward + optimiser step:

| cell | depth | params | ms/step |
|---|---|---|---|
| `neither` | 0/9 | 3.7509 M | 13.4 |
| `line graph` | 3/3 | 3.7899 M | 20.1 |
| `angle diffusion` | 0/9 | 3.8559 M | 15.8 |
| `both` | 3/3 | 3.8559 M | 21.4 |

**The line graph costs 1.50×, not 2.4×.** The `inverse/README.md` records
2.4× per training step and the manuscript quotes it. That figure does not hold
on this hardware and split, and should be corrected rather than carried
forward.

The cost comes from the graph, not the weights: kNN builds **8,033 triplets
against 1,497 pairs**, 5.4 to 1, and the line-graph layer runs its edge update
over all of them. A line-graph layer holds the same parameters as two pair
convolutions, so parameters and compute move together under depth but apart
under the line graph. Buying the compute back with pair depth:

| pair convs | params | ms/step | vs `line graph` params | vs its compute |
|---|---|---|---|---|
| 9 | 3.7509 M | 13.6 | −1.03% | 0.68× |
| 12 | 4.7408 M | 15.8 | +25.1% | 0.79× |
| 15 | 5.7308 M | 17.8 | +51.2% | 0.88× |
| 16 | 6.0607 M | 18.6 | +59.9% | 0.94× |
| 18 | 6.7207 M | 20.4 | +77.3% | 1.01× |
| 21 | 7.7107 M | 23.1 | +103.5% | 1.15× |

So the suite carries **both** normalisations rather than choosing one:

* **parameter-matched** — 9 pair convs. Parameters level within 1.03%; the
  line-graph cells take 1.50× the compute. A line-graph win could be the extra
  compute.
* **compute-matched** — 16 pair convs. Step cost level within 6%; the
  no-line-graph cells carry ~1.6× the parameters. A line-graph win is then the
  stronger claim, but a line-graph *loss* becomes ambiguous.

A result that survives both is not a budget artefact in either direction.

**Why one depth of 16 and not two.** Matched individually the two
no-line-graph cells want different depths — 18 without the angular objective
(20.4 ms vs 20.1) and 15 with it (21.0 vs 21.3), because the objective carries
its own cost. But then the two cells of that row would differ by three layers,
and the angular contrast *within* the row would be confounded with a depth
change, which is exactly what a factorial exists to avoid. One depth keeps the
row clean at the price of matching compute to 6% (18.6 vs 20.1, 21.9 vs 21.4)
instead of 2%.

The two matrices share their line-graph cells, so both cost **two** extra
trainings rather than four.

## 3. The new cell, and the code it needed

`(angle diffusion, no line graph)` could not be built before this work.
`denoiser.py` raised outright:

```
angle_diffusion needs alignn_layers > 0: the angular channel lives on the
line graph, which is not built when there are no ALIGNN layers
```

and `use_line_graph = alignn_layers > 0` gated both the ALIGNN convolutions
*and* the construction of triplets at all.

### 3.1 What changed

`use_line_graph` (are there ALIGNN layers?) is now separate from `use_triplets`
(does the triplet set get built?). The angular objective needs the second
without the first: it has to evaluate θ somewhere even when no message ever
travels along a triplet.

With no line graph there is no evolved triplet representation for the angle
head to read, so the triplet feature is assembled from what the pair trunk does
have:

```
z_ijk = angle_embedding(cos θ_ijk) + y_ij + y_jk
```

taken **after** the pair convolutions. Three properties make this the right
construction rather than a convenient one:

* **It introduces no parameters.** So `angle diffusion` and `both` match
  exactly on parameter count — 3,594,567 each.
* **It is symmetric under swapping the two bonds**, as a bond angle is.
* **It is the combination rule the ALIGNN convolution already uses** — its own
  message is `e_src[src] + e_dst[dst] + edge_gate(y)`, a sum of features.

Reading it after the pair trunk rather than before is what lets the angular
loss shape the *whole* trunk. Read before, the gradient would reach only the
edge embedding.

The angle head gets `cos θ_t` for the same reason `A3` does: in `A3`, `z` is
initialised from it. Without it the head cannot see the quantity whose
displacement it is predicting.

### 3.2 What was verified, and how

Everything below was measured on the GPU node, not reasoned about:

| check | result |
|---|---|
| `neither`, `line graph`, `both` parameter counts | unchanged from before the edit |
| `angle diffusion` vs `both` | 0 parameters apart |
| angular loss is a real objective | `L_ang` = 2.44 (no LG) / 0.63 (LG) on a noised cell |
| the objective reaches the trunk | ‖grad‖ from `L_ang` alone: pair trunk 10.6 (no LG), ALIGNN stack 4.87 (LG); edge embedding and species embedding non-zero in both |
| forward coupling in `both` | perturbing `angle_embedding` moves `eps_frac` by 1.4e-3 |
| forward coupling in `angle diffusion` | perturbing `angle_embedding` moves `eps_frac` by **exactly 0** |

The last two rows *are* the line-graph factor: the same objective, shaping
everything upstream of the head in both cells, with a forward path to the
coordinate score in one and none in the other.

> **A trap for whoever tests this next.** The output heads are zero-initialised
> on purpose (`score_combine`, `lattice_head[-1]`, `angle_head[-1]`), so at
> initialisation `eps_frac` is identically zero and any probe of it reads 0 for
> reasons that have nothing to do with the architecture. Perturb those heads
> first. Likewise, the angular target is `wrap(θ_t − θ_0)`: pass the *same*
> structure as both and the target is identically zero and every gradient
> vanishes.

### 3.3 The honest limitation

In `angle diffusion` the angular channel couples to coordinates **only** through
the shared trunk's gradient, never through a forward path. Architecturally that
places it closer to the old suite's `A4` (auxiliary supervision on a shared
trunk) than to `A3`. This is not a defect in the design — the absence of a
forward path *is* what "no line graph" means — but it should be stated in the
manuscript rather than left for a reader to infer. A useful consequence: this
2×2 subsumes the old `A4`-vs-`A3` coupling question in a cleaner form, since
`A4` kept the line-graph convolutions and only zeroed one aggregation.

## 4. Checkpoint selection

The trained objective is `w_lat·L_lat + w_frac·L_frac + w_ang·L_ang` with
weights 1.0 / 10.0 / 1.0. `L_ang` exists in two cells.

Selecting `best_model.pt` on that total would select two cells on a different
criterion from the other two. So `train_csp.py` gained `--select-on
{total,structural}`; the tasks declare `structural`, which is
`w_lat·L_lat + w_frac·L_frac` — the part every cell optimises. `preflight.sh`
asserts that every unit passes it.

This is a deliberate departure from how the eight-arm suite trained `A0` and
`A3`, and it is why this experiment gets its own run root: the checkpoints are
not interchangeable with that one's.

`train_csp.py` also gained `--early-stop-patience`, **off by default and left
off**. The one-cycle schedule is defined over `--epochs`, so stopping early
means the model never sees the anneal. Shorten `--epochs` instead if a run has
to be cheaper — that keeps the schedule complete.

## 5. What is measured, and what each measurement is worth

### Denoising loss — the structural term

Cheap, reproducible to three decimals across machines, and the only one of the
three measured without any sampling, force field or matcher in the loop. This
is where a real effect will show up first if there is one.

It is also the one where the manuscript's percent-change framing transfers
directly. A 2×2 gives four such numbers: the line-graph effect at each level of
the angular objective, the angular effect at each level of the line graph, and
the interaction.

**It is not a fidelity claim.** The published ablation moved this by 14.5% and
moved match rate by exactly nothing. `analyze.py`'s report says so in "How to
read this", because the temptation to read a loss improvement as a generation
improvement is exactly the error this benchmark has already made once.

### AtomBench metrics

Match rate, coordinate RMSD, ccRMSD, lattice MAE (lengths and angles), KLD —
read from AtomBench's own `compute_metrics.py` output, never recomputed.

Underpowered and known to be. On 103 JARVIS targets the ALIGNN README records
match rate spanning 0.437–0.524 across fifteen independently trained models: the
seed spread alone covers any effect this experiment could plausibly produce.
The 825-target Alexandria split exists for exactly this reason and is why its
budget goes to targets rather than to seeds.

### Bond-angle Wasserstein

The 1-D earth-mover distance in degrees between the pooled bond-angle histogram
of the generated structures and of the held-out real ones, on 180 bins over
[0°, 180°]. Exact for a 1-D histogram (the integral of the absolute CDF
difference), and in degrees, so it reads directly as "the generated angles are
off by this much on average". FoldingDiff's own diagnostic, and the metric that
speaks most directly to the hypothesis.

**Report it on the unrelaxed predictions.** The headline pipeline ranks 32
candidates by ALIGNN-FF energy and relaxes the survivors, which snaps every
sample into the force field's nearest local minimum. That is precisely where a
local-geometry advantage would be erased, and local geometry is what the
angular channel claims to fix; the relaxed histogram is substantially a
measurement of ALIGNN-FF's preferred angles. `50_unrelaxed.sh` regenerates one
sample per target with no ranking and no relaxation for this purpose; it is
cheap, because it is one candidate instead of thirty-two.

> **A 180° spike appears in the generated *and* the real histograms.** It is
> back-tracking triplets (`i → j → i`, cos = −1 exactly) inherited from the
> graph builder, identical across cells. It is not physics and does not bias a
> comparison.

## 6. No p-values

The manuscript note asks for none, and the data would not support them. Four
cells at one to three seeds cannot resolve a few percent, and the eight-arm
suite at three seeds returned every pre-registered contrast at Holm-adjusted
p = 1.000 on this exact benchmark.

What is reported instead: the mean, the seed spread, the percent change, and an
explicit `~` flag whenever a change is smaller than the larger of the two
cells' spreads. A change inside the noise is marked as such rather than
discussed.

## 7. Cost, and the one number nobody has

Measured on GB10 (Alexandria split): **4.26–4.38 s/epoch**, so ~1.2 h for 1000
epochs. Training is not the bottleneck and there is no reason to economise on
it. An earlier claim in the session logs that the angular arm ran >2× slower
was **wrong** — read off a single early log point; over 300+ epochs the two are
within 3%. The README's 2.4× figure is for `A3`-style arms and has not been
measured on this hardware.

**Generation has never been measured end to end on this cluster.** The previous
attempt was cancelled during training. It is 825 targets × 32 candidates × 1000
denoising steps plus an ALIGNN-FF prescreen and four relaxations per target, on
a memory-bandwidth-bound GB10 — and it is what decides whether this suite is a
two-day or a two-week job. `20_pilot.sh price` measures it at full per-target
settings on twelve targets and extrapolates. Every walltime in `env.sh` is a
placeholder until it has run. **Do not copy a number forward from the old
cluster; the GPU changed from A30 (HBM2, 933 GB/s) to GB10 (LPDDR5X, ~273 GB/s)
and ALIGNN is bandwidth-bound, so the newer part is not safely assumed faster.**

Scheduling: two GPUs, one per node, `CSP_MAX_CONCURRENT=2`. JARVIS is 18
elements — nine sequential rounds. Alexandria is 6 elements — three rounds.

## 8. Threats to validity

* **Neither normalisation levels both budgets.** §2.3. Read the two matrices
  together; a contrast is only meaningful between two cells normalised the same
  way, which is why `analyze.py` never puts a parameter-matched cell in the
  same table as a compute-matched one.
* **Underpowered on JARVIS.** Stated above; it is why both datasets are run.
* **One seed on Alexandria.** No spread on that split at all. The percent
  changes there are single-run differences and must be labelled as such. The
  power comes from 825 paired targets, not from seed replication — but this
  harness does not run the paired per-target tests that would cash that in.
  If the Alexandria numbers turn out interesting, the sibling harness's
  `analyze.py` has McNemar / CMH / Wilcoxon / BCa machinery that applies
  unchanged to these CSVs.
* **The `angle diffusion` cell's coupling is gradient-only.** §3.3.
* **`select_on=structural` departs from how `A0`/`A3` were trained elsewhere.**
  Intentional and documented; it is why the run root is separate.
* **The symmetrisation tolerance does not transfer between splits** and is
  swept per dataset, on validation, with reduced candidates (the tolerance is a
  property of the predicted cell distribution, not of the candidate count).
* **Do not read across information tiers.** All four cells get identical
  conditioning (per-atom species, `natoms`, Tc) and no ground-truth geometry,
  so the contrasts are clean. The published AtomBench baselines are not
  comparable without saying so: CDVAE sees full structure, AtomGPT formula +
  Tc, FlowMM composition only.
* **`Atoms.from_poscar` takes a path, not POSCAR text.** Handing it text raises
  `OSError: File name too long` once a generated cell exceeds the filename
  limit. Fixed on this branch (`evaluate.py` uses `Poscar.from_string`); do not
  reintroduce it if you touch benchmark CSV parsing.

## 9. Site

atomgptlab (JHU WSE), measured 2026-09-02. Carried over from the sibling
harness, where it was established.

* `main` = atomgptlab01-02, x86_64, 256 cores, 500 GB, **no GPU**.
* `gpu` = atomgptlab03-04, **aarch64**, 20 cores, 110 GB, 1× NVIDIA GB10 each.
* Both conda environments are aarch64 and run **only** on the GPU nodes. The
  login node is x86_64; `on_env_arch` bounces commands through a small `srun`.
* No `/scratch`. `/data` is the 430 TB NFS share and the only large filesystem
  mounted on both architectures. `/local-fast` is CPU-nodes-only and invisible
  from the GPU nodes, so nothing durable may live there.
* No accounting associations for this user; every partition is
  `AllowAccounts=ALL` / `AllowQos=ALL`. `--account` and `--qos` are omitted
  entirely, and a bare `#SBATCH --account=` is a hard error rather than a
  no-op — hence `sbatch_account_line`.
* No node features defined, so `CSP_CONSTRAINT` must stay empty.
* GPUs register as plain `gpu:1` with no type name, so typed gres requests are
  rejected.
* Every partition is `MaxTime=UNLIMITED`, so a generous walltime costs nothing
  in queue priority — and the runner resumes at *stage* granularity, so a
  walltime kill restarts a training from zero. Ask for more than you need.
