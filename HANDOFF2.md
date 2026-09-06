# Claude Code handoff: migrate independent triplet-angle diffusion to a finite redundant angular moment field

## Objective

The repository has already been modified according to the previous handoff so that the generative state can contain an independently diffused persistent triplet-angle field,

$$
X_t=(x_t,F_t,\Theta_t),
$$

with joint reverse prediction

$$
D_\phi(x_t,F_t,\Theta_t,t,c)
\rightarrow
(\hat\epsilon_L,\hat s_F,\hat s_\Theta).
$$

Do **not** discard that implementation. Preserve it as an experimental baseline.

The new proposed architecture replaces the explicit persistent triplet state \(\Theta_t\) with a finite-dimensional, per-atom redundant angular moment field

$$
P_t=\{P_{i,t}\}_{i=1}^{N},
\qquad
P_{i,t}\in\mathbb R^{D_\theta}.
$$

The primary state becomes

$$
X_t=(x_t,F_t,P_t).
$$

The clean angular state \(P_0\) is a deterministic finite expansion of the local three-body geometry of the clean crystal. It is then corrupted independently of \(x_t\) and \(F_t\). At noisy timesteps,

$$
P_t\neq P(F_t,L_t)
$$

in general, and this inconsistency is intentional.

The shared reverse model must satisfy

$$
(\hat\epsilon_L,\hat s_F,\hat\epsilon_P)
=
D_\phi(x_t,F_t,P_t,t,c).
$$

All three current states must influence the common denoiser.

At the final readout, discard \(P_0\). The generated crystal remains defined entirely by

$$
(L_0,F_0).
$$

The motivation for this migration is computational. The explicit persistent triplet implementation gives each \((j,i,k)\) combination its own stochastic state. The finite angular field instead compresses the same class of local angular correlations into a systematically truncatable basis whose size is independent of the number of triplets.

---

# 1. Preserve the implementation that already exists

The just-completed persistent-triplet implementation is scientifically valuable and must remain runnable.

Do not silently redefine

```text
angle_mode="independent"
```

if that is the name currently used by the explicit persistent-triplet model.

Recommended semantics after this migration:

```text
angle_mode="off"
    original F,L model

angle_mode="derived_aux"
    original geometry-derived angular auxiliary target

angle_mode="independent"
    independently diffused explicit persistent theta_ijk state
    implemented by the previous handoff

angle_mode="spectral"
    NEW proposed finite angular moment field
```

Existing checkpoints with `angle_mode="independent"` must continue to instantiate the explicit triplet implementation.

The new proposed model is `angle_mode="spectral"`.

Do not delete:

```text
theta_state
triplet_index
angle_sigmas
noise_angle()
explicit angular reverse sampler
```

because those are required to reproduce the previous experiment.

They simply must not execute when `angle_mode == "spectral"`.

---

# 2. Mathematical definition of the new angular field

The new state should encode a finite set of moments of the local bond-angle distribution.

For atom \(i\), let \(\mathbf r_{ij}\) be the current minimum-image Cartesian vector from atom \(i\) to neighbor \(j\),

$$
r_{ij}=\|\mathbf r_{ij}\|,
\qquad
\hat{\mathbf r}_{ij}=
\frac{\mathbf r_{ij}}{r_{ij}}.
$$

Use the same smooth pair relevance already implemented for the radius topology,

$$
s_{ij}=u(r_{ij};r_c),
$$

where \(u\) is the current DimeNet-style cutoff envelope.

Let \(R_\mu(r)\), \(\mu=1,\ldots,n_r\), be a small radial basis. Reuse ALIGNN's existing `RBFExpansion`; do not introduce a separate SOAP radial implementation.

Define the weighted radial coefficient

$$
a_{ij\mu}
=
s_{ij}R_\mu(r_{ij}).
$$

For angular tensor rank \(\nu\), define the local moment tensor

$$
M^{(i)}_{\mu,\nu}
=
\sum_j
a_{ij\mu}
\,
\hat{\mathbf r}_{ij}^{\otimes\nu}.
$$

Then form rotational invariants by complete contraction:

$$
P^{(i)}_{\mu\mu'\nu}
=
M^{(i)}_{\mu,\nu}
:
M^{(i)}_{\mu',\nu}.
$$

Expanding the contraction gives

$$
P^{(i)}_{\mu\mu'\nu}
=
\sum_{j,k}
a_{ij\mu}
a_{ik\mu'}
\left(
\hat{\mathbf r}_{ij}
\cdot
\hat{\mathbf r}_{ik}
\right)^\nu.
$$

Therefore every component explicitly contains moments of

$$
\cos\theta_{jik},
$$

but the code never constructs the \((j,k)\) pair enumeration.

This is the central density-trick identity.

Because the powers

$$
1,x,x^2,\ldots,x^{\nu_{\max}}
$$

span the same finite polynomial space as Legendre polynomials up to that order, this can be regarded as a Cartesian finite angular spectrum rather than a literal spherical-harmonic SOAP spectrum.

Use only the unique radial-channel pairs

$$
\mu\leq\mu'
$$

because the contracted matrix is symmetric.

---

# 3. Default truncation

Use these defaults for the first implementation:

```text
spectrum_n_radial = 4
spectrum_rank_min = 1
spectrum_rank_max = 4
spectrum_cutoff = 5.0 Å
spectrum_envelope_exponent = 5
```

For four radial channels,

$$
\frac{4(4+1)}{2}=10
$$

unique radial pairs exist.

Using ranks

$$
\nu=1,2,3,4
$$

therefore gives

$$
D_\theta
=
10\times4
=
40.
$$

This is a particularly useful default:

```text
P_i has exactly 40 scalars per atom.
```

It is compact, comparable in dimensionality to the existing 40-bin ALIGNN angular expansion, and the truncation parameters have a clear systematic interpretation.

Do not include rank \(\nu=0\) in the default field. Rank zero is predominantly a radial/coordination-density correlation and the pair channel already represents that information. It may be added in an ablation later.

Finite truncation is intentional. Do not attempt to make the descriptor complete in the first implementation.

---

# 4. Complexity target

The new implementation must never explicitly enumerate neighbor pairs for the spectral state.

For fixed \(n_r\) and \(\nu_{\max}\), descriptor construction should consist of:

```text
edge -> local moment accumulation
local moment -> scalar contractions
```

The angular-state storage must be

$$
O(ND_\theta),
$$

which for the default is simply

$$
40N.
$$

The descriptor-construction work should scale linearly with the number of pair edges for fixed basis size.

The ordinary ALIGNN line graph may still have its usual local triplets. That is separate from the stochastic angular state.

The new spectral channel must **not** inherit the persistent dense triplet index from `angle_mode="independent"`.

---

# 5. New file: `alignn/inverse/angular_spectrum.py`

Create this module.

Do not put the descriptor implementation inside `denoiser.py`.

## 5.1 Public API

Implement approximately:

```python
class AngularMomentSpectrum(nn.Module):
    def __init__(
        self,
        n_radial: int = 4,
        rank_min: int = 1,
        rank_max: int = 4,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        ...

    @property
    def out_features(self) -> int:
        ...

    def forward(
        self,
        r: torch.Tensor,
        dist: torch.Tensor,
        center: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        ...
```

Inputs:

```text
r:
    (E,3) minimum-image Cartesian vectors

dist:
    (E,) corresponding distances

center:
    (E,) central atom index for each vector

num_nodes:
    total atoms in flattened batch
```

Output:

```text
(N,D_theta)
```

Use `center = src` if the existing minimum-image helper defines

```text
r = position[dst] - position[src]
```

as it does in the public branch.

## 5.2 Reuse existing primitives

Use:

```python
RBFExpansion
CutoffPolynomial
scatter_sum
```

from the repository.

Do not add a new package dependency.

Do not add `e3nn`.

Do not add `sphericart`.

Do not call SciPy from the training path.

## 5.3 Tensor accumulation

Let:

```python
unit = r / dist.clamp_min(1e-8).unsqueeze(-1)
radial = rbf(dist) * envelope(dist).unsqueeze(-1)
```

with shape

```text
unit   : (E,3)
radial : (E,n_radial)
```

Construct tensor powers recursively.

For rank one:

```python
tensor_power = unit
```

For the next rank:

```python
tensor_power = (
    tensor_power.unsqueeze(-1)
    * unit.unsqueeze(-2)
).flatten(start_dim=1)
```

so its last dimension is \(3^\nu\).

For each rank:

```python
edge_moment = (
    radial.unsqueeze(-1)
    * tensor_power.unsqueeze(1)
)
```

with shape

```text
(E,n_radial,3**rank)
```

then

```python
moment = scatter_sum(
    edge_moment,
    center,
    num_nodes,
)
```

with shape

```text
(N,n_radial,3**rank)
```

and contract:

```python
power = torch.einsum(
    "ind,ijd->nij",
    moment,
    moment,
)
```

giving

```text
(N,n_radial,n_radial)
```

Take only the upper-triangular radial pairs using a cached

```python
torch.triu_indices(n_radial, n_radial)
```

and concatenate the result from every retained rank.

For the default configuration the returned shape must be

```text
(N,40)
```

exactly.

## 5.4 No self-term subtraction

Do not subtract the \(j=k\) terms.

The redundancy is intentional and SOAP/ACE-style power spectra also contain lower-body contributions through self-correlations.

The purpose here is a useful finite redundant field, not an orthogonal body-order decomposition.

## 5.5 No species channels

The first implementation should be geometric only.

Do not split the spectrum by chemical element.

Species information is already supplied to ALIGNN through node embeddings and composition conditioning.

Species-resolved SOAP would increase the field dimension dramatically and confound the question being tested: whether an independently corrupted redundant **geometric** representation helps CSP.

A species-resolved field can be a later ablation.

---

# 6. Critical descriptor tests before touching diffusion

Add a dedicated

```text
alignn/tests/test_inverse_angular_spectrum.py
```

before integrating the state into the diffusion model.

## 6.1 Explicit-sum equivalence

For a tiny random local environment, compute

$$
P_{\mu\mu'\nu}
$$

two ways:

1. with `AngularMomentSpectrum`,
2. explicitly with the double sum over \(j,k\),

$$
\sum_{j,k}
a_{j\mu}a_{k\mu'}
(\hat r_j\cdot\hat r_k)^\nu.
$$

Assert equality to numerical tolerance for every rank and radial pair.

This is the most important mathematical unit test.

It proves that the implementation captures the intended angular correlations without triplet enumeration.

## 6.2 Rotation and reflection invariance

Generate a random orthogonal matrix \(Q\).

Verify

```python
spectrum(r) == spectrum(r @ Q)
```

for both:

```text
det(Q)=+1
det(Q)=-1
```

within numerical tolerance.

## 6.3 Neighbor permutation invariance

Randomly permute the edges incident on every center.

The spectrum must remain unchanged.

## 6.4 Smooth cutoff

Move one neighbor through `spectrum_cutoff`.

Verify the descriptor approaches the same value continuously from inside and outside the cutoff.

## 6.5 Dimension

For defaults:

```python
assert spectrum.out_features == 40
```

## 6.6 Basis-relabelling invariance

Use the existing signed-permutation lattice augmentation.

The physical Cartesian environment is unchanged, so the spectrum must remain unchanged.

This is important because the fractional-coordinate and lattice channels explicitly train over these basis relabellings.

---

# 7. `alignn/inverse/diffusion.py`

Public pre-migration landmark: `DiffusionSchedule` begins around lines 122--160 and currently contains lattice VP plus fractional VE schedules.

The local post-handoff version also contains the wrapped angular schedule and `noise_angle()`.

Keep those for `angle_mode="independent"`.

## 7.1 Add a Euclidean spectral forward process

The standardized moment field lives in ordinary Euclidean space, not on a circle.

Do **not** use wrapped-normal diffusion.

Use the same variance-preserving process as the lattice:

$$
P_t
=
\sqrt{\bar\alpha_t}P_0
+
\sqrt{1-\bar\alpha_t}\epsilon_P,
\qquad
\epsilon_P\sim\mathcal N(0,I).
$$

Add:

```python
def noise_spectrum(
    self,
    p0: torch.Tensor,
    t_node: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ab = self.alpha_bar[t_node].unsqueeze(-1)
    eps = torch.randn_like(p0)
    p_t = ab.sqrt() * p0 + (1.0 - ab).sqrt() * eps
    return p_t, eps
```

The draw of `eps` must be independent of lattice and coordinate noise.

Do not add a new schedule yet.

Use the same `alpha_bar` as the lattice by default.

This gives the spectral field a proper independent stochastic state without introducing an extra schedule hyperparameter in the first experiment.

---

# 8. `alignn/inverse/data.py`

Public landmarks:

```text
Normalizer: lines ~20--56
compute_normalizer: lines ~140--177
```

Extend `Normalizer` backward-compatibly.

## 8.1 Add optional spectrum statistics

Add:

```python
spectrum_mean: Optional[torch.Tensor] = None
spectrum_std: Optional[torch.Tensor] = None
```

and:

```python
def norm_spectrum(self, p):
    ...

def denorm_spectrum(self, p):
    ...
```

`to_dict()` should omit or serialize them cleanly.

`from_dict()` must accept old checkpoints where the keys do not exist.

`to(device)` must move them if present.

## 8.2 Fit statistics only on the training split

Compute clean \(P_0\) for every atom in the unaugmented training structures.

Concatenate across atoms:

```text
(total_training_atoms,D_theta)
```

and compute component-wise mean/std.

Clamp standard deviations in the same spirit as the lattice normalizer.

Because the spectrum is invariant to the existing signed-permutation augmentation, statistics need only be computed from one physical representation of every training crystal.

## 8.3 Avoid import cycles

If `dense_pair_index()` and the minimum-image helper still live in `denoiser.py`, do not make `data.py` import the whole denoiser.

Either:

1. add a small `alignn/inverse/geometry.py` containing shared pair-index/minimum-image utilities, or
2. compute spectrum statistics from `train_csp.py` after the ordinary normalizer is constructed.

Prefer option 1 if the previous handoff has already partially factored minimum-image geometry out of the denoiser.

Do not duplicate minimum-image mathematics.

---

# 9. `alignn/inverse/denoiser.py`

Public landmarks before the local migration:

```text
constructor: ~168 onward
node initialization: ~492 onward
ALIGNN path: ~538 onward
structural heads: ~578 onward
```

The local version now additionally contains:

```text
angle_mode
theta_state
triplet_index
dense_triplet_index
independent angular-state embedding
```

Preserve those for explicit-triplet mode.

## 9.1 Configuration

Add spectral arguments:

```python
spectrum_n_radial: int = 4
spectrum_rank_min: int = 1
spectrum_rank_max: int = 4
spectrum_cutoff: float = 5.0
spectrum_envelope_exponent: int = 5
spectrum_feedback: bool = True
```

When

```python
angle_mode == "spectral"
```

instantiate:

```python
self.angular_spectrum = AngularMomentSpectrum(...)
self.spectrum_dim = self.angular_spectrum.out_features
```

The descriptor itself has no learned parameters.

## 9.2 New state input

Extend `forward()` with:

```python
spectrum_state: Optional[torch.Tensor] = None
```

For spectral mode require:

```text
spectrum_state.shape == (num_nodes, spectrum_dim)
```

Fail loudly otherwise.

Do not accept a triplet index as part of the spectral state.

## 9.3 Embed \(P_t\) at the atom level

Add:

```python
self.spectrum_embedding = nn.Sequential(
    nn.Linear(self.spectrum_dim, hidden_features),
    nn.SiLU(),
    nn.Linear(hidden_features, hidden_features),
)
```

Compute:

```python
p_emb = self.spectrum_embedding(spectrum_state)
```

For the fully coupled model:

```python
h = h + p_emb
```

before the ALIGNN message-passing stack.

This is the key `P -> structure` coupling.

Because node states participate in ALIGNN edge gating and subsequent bond/atom updates, the spectral state can influence both coordinate and lattice predictions without requiring a special cross-attention module.

## 9.4 Keep the ordinary ALIGNN line graph

Do **not** remove ALIGNN's geometry-derived angle features.

The model should contain both:

```text
ordinary ALIGNN line graph:
    theta(F_t,L_t) used as a current geometric feature

independent spectral state:
    P_t independently diffused from P_0
```

These serve different purposes.

The line graph remains a local architectural inductive bias.

The spectral field is a member of the stochastic generative state.

For spectral mode, use the normal local `knn` or smooth-radius line graph. Do not build the persistent dense triplet graph introduced for explicit independent-angle mode.

## 9.5 Spectral score head

The spectral prediction must use both:

```text
the supplied noisy state P_t
the final geometry-informed node representation h
```

Implement approximately:

```python
self.spectrum_head = nn.Sequential(
    nn.Linear(2 * hidden_features, hidden_features),
    nn.SiLU(),
    nn.Linear(hidden_features, self.spectrum_dim),
)
```

and:

```python
eps_spectrum = self.spectrum_head(
    torch.cat([h, p_emb], dim=-1)
)
```

Return:

```python
out["eps_spectrum"] = eps_spectrum
```

Do not predict one scalar per triplet.

## 9.6 Coupling-off control

`spectrum_feedback=False` must mean:

```text
P_t is still provided
P_t is still embedded
P_t is still denoised
F_t,L_t can still affect eps_spectrum
P_t cannot affect eps_frac or eps_lattice
```

Implement this by skipping:

```python
h = h + p_emb
```

when feedback is disabled.

Still give both `h` and `p_emb` to `spectrum_head`.

This produces a particularly clean mechanistic ablation.

## 9.7 Desired dependency graph

Full spectral mode must have:

$$
P_t
\rightarrow
h
\rightarrow
(\hat s_F,\hat\epsilon_L),
$$

and:

$$
(F_t,L_t)
\rightarrow
h
\rightarrow
\hat\epsilon_P.
$$

Therefore:

$$
D_\phi(F_t,L_t,P_t)
\rightarrow
(\hat s_F,\hat\epsilon_L,\hat\epsilon_P)
$$

is genuinely non-factorized.

---

# 10. `alignn/inverse/model.py`

Public wrapper landmark: `ALIGNNCSP.forward`, roughly lines 56--82.

The local post-handoff wrapper already forwards `theta_state` and `triplet_index`.

Add:

```python
spectrum_state=None
```

and pass it directly to `self.denoiser`.

Do not put descriptor construction or diffusion logic in `model.py`.

---

# 11. `alignn/inverse/train_csp.py`

Public `diffusion_loss()` landmark begins around line 79.

The local post-handoff version now contains explicit clean-angle construction and `schedule.noise_angle()`.

Keep that branch for `angle_mode="independent"`.

Add a separate spectral path.

## 11.1 Build the clean spectrum

Using the clean

```text
batch["frac"]
batch["lattice"]
```

construct the ordinary dense pair index and resolve the same minimum-image Cartesian pair vectors used elsewhere.

Then:

```python
p0_raw = model.denoiser.angular_spectrum(
    r=r0,
    dist=dist0,
    center=src,
    num_nodes=batch["frac"].shape[0],
)
```

Normalize:

```python
p0 = normalizer.norm_spectrum(p0_raw)
```

Do this from the clean structure only.

Do not compute the target from `F_t,L_t`.

## 11.2 Independently corrupt it

Use graph timestep \(t\), broadcast to atoms:

```python
t_node = t[batch["node_graph_id"]]
```

then:

```python
p_t, eps_p = schedule.noise_spectrum(
    p0,
    t_node,
)
```

The resulting three forward states are:

```text
x_t
f_t
p_t
```

with independent random draws.

## 11.3 One common denoiser call

Call:

```python
out = model(
    frac=f_t,
    lattice=lattice_t,
    lattice_vec6=x_t,
    spectrum_state=p_t,
    ...
)
```

once.

Do not calculate the spectral output in a separate network call.

## 11.4 Loss

Use ordinary MSE:

```python
loss_spectrum = F.mse_loss(
    out["eps_spectrum"],
    eps_p,
)
```

and:

$$
\mathcal L
=
w_L\mathcal L_L
+
w_F\mathcal L_F
+
w_P\mathcal L_P.
$$

Add CLI/config:

```text
--spectrum-weight
```

with initial default:

```text
1.0
```

but keep it tunable.

“Coequal diffusion channel” refers to its status in the state and joint denoiser, not necessarily equal raw loss coefficients.

## 11.5 No relevance-weighted spectral loss

Do not multiply spectral MSE by current noisy-geometry cutoff weights.

The smooth cutoff was already used when defining \(P_0\).

Once \(P_t\) has been independently corrupted, it is an autonomous state.

Weighting its denoising loss by \(F_t,L_t\) would partially subordinate it to the coordinate channel again.

## 11.6 CLI additions

Add:

```text
--spectrum-n-radial
--spectrum-rank-min
--spectrum-rank-max
--spectrum-cutoff
--spectrum-envelope-exponent
--spectrum-feedback
--spectrum-weight
```

All values must be serialized in `config.json` and checkpoints.

## 11.7 Diagnostics

Extend the per-noise-level report with:

```text
spectral zero-prediction baseline MSE
spectral model MSE
```

The spectral target is ordinary Gaussian noise, so the predict-zero baseline should be near one in standardized coordinates.

---

# 12. `alignn/inverse/sample.py`

The local post-handoff sampler now explicitly carries:

```text
frac
x_lat
theta
```

and independently integrates the angular wrapped-normal state.

Preserve that behavior for `angle_mode="independent"`.

For `angle_mode="spectral"`, the state must instead be:

```text
frac
x_lat
p
```

## 12.1 Prior

Initialize:

```python
p = torch.randn(
    n_nodes,
    model.denoiser.spectrum_dim,
    device=device,
)
```

because \(P\) is standardized and follows a VP Gaussian process.

Do not initialize it uniformly.

## 12.2 CFG duplication

Because \(P_t\) is per-node, CFG duplication is trivial:

```python
p_in = torch.cat([p, p])
```

No triplet-index duplication is required.

This removes one of the most brittle parts of the persistent-triplet implementation.

## 12.3 Joint denoiser

Spectral `_denoise` must receive:

```python
_denoise(frac_t, x_lat_t, p_t, t_idx)
```

and return:

```text
eps_frac
eps_lattice
eps_spectrum
```

All three predictions for one predictor transition must come from the same current joint state.

## 12.4 Coordinate corrector

Keep the existing coordinate Langevin corrector.

During each corrector iteration, evaluate the joint model using the current:

```text
frac
x_lat
p
```

but update only `frac`.

After that update, the next denoiser call sees the new coordinate state together with the unchanged lattice and spectral states.

## 12.5 Predictor

At the predictor stage evaluate once:

```python
eps_f, eps_l, eps_p = _denoise(
    frac,
    x_lat,
    p,
    t,
)
```

Then compute all three next states from the old joint state.

Coordinates retain the existing VE predictor.

Lattice retains the existing VP/DDPM posterior-mean update.

Spectrum uses the **same VP posterior mathematics as the lattice**, applied elementwise to `(N,D_theta)`.

Refactor the current lattice posterior update into a helper such as:

```python
def _vp_ancestral_step(
    x_t,
    eps,
    t,
    schedule,
    x0_clip,
):
    ...
```

and use it for both:

```text
x_lat
p
```

This avoids duplicating the numerically stabilized posterior-mean implementation.

Use a separate argument:

```text
spectrum_x0_clip = 4.0
```

initially.

## 12.6 Do not impose realizability

The sampled \(P_t\) need not correspond to any actual atomic configuration at intermediate times.

Do not:

```text
project P_t onto P(F_t,L_t)
force its moment matrices positive semidefinite
clip it to clean-data physical constraints beyond standardized x0 clipping
```

The whole point is to allow independently corrupted redundant representations to disagree and let the learned reverse model exploit their correlations.

## 12.7 Final readout

Default output remains:

```python
{
    "frac": ...,
    "lattice": ...,
    "natoms": ...,
}
```

Discard `p`.

Under diagnostic `return_latents=True`, optionally expose:

```python
"spectrum_state": p
```

but downstream `Atoms` construction must never use it.

---

# 13. `alignn/inverse/generate.py`

Public schedule-reconstruction landmark is around lines 188--202.

Spectral mode uses the ordinary VP `alpha_bar`, so no extra schedule reconstruction is required when `num_steps` changes.

However:

* propagate all `spectrum_*` architecture fields through checkpoint reconstruction;
* load spectrum normalization statistics;
* continue preserving the separate `angle_sigma_*` fields for explicit-triplet mode.

No change to `GeneratedStructure` is required.

---

# 14. `alignn/inverse/angles.py`

Do not repurpose this file for the spectral descriptor.

It should remain responsible for:

```text
explicit triplet angular geometry
derived auxiliary target
wrapped angle process utilities
smooth pair/triplet relevance
```

The new finite field belongs in:

```text
angular_spectrum.py
```

This separation is important because the two approaches now represent distinct experiments.

---

# 15. `alignn/inverse/ablations.py`

Do not overwrite the existing A-series results.

Add a second family for the representation question.

Suggested names:

```text
R0
    F,L baseline

R1
    derived angular auxiliary model
    preserve the already-run weaker angular-inductive-bias result

R2
    explicit independently diffused persistent triplet state
    the architecture from the previous handoff

R3
    finite angular spectrum, feedback OFF

R4
    finite angular spectrum, fully coupled
    PROPOSED MODEL

R5
    finite angular spectrum derived from current geometry but NOT independently
    diffused, if a representation-only control is desired
```

The most informative comparisons are:

```text
R0 vs R1
    does angular supervision help at all?

R1 vs R2
    does promoting individual bond angles to independent stochastic state help?

R2 vs R4
    does the finite factorized representation preserve/improve the effect while
    reducing computational cost?

R3 vs R4
    does P_t -> structural feedback matter?

R0 vs R4
    final proposed architecture vs baseline
```

For the finite-truncation study, do not create dozens of architectural variants.

Use a compact sweep:

```text
rank_max = 2, 4, 6
n_radial = 2, 4, 6
```

with the primary point at:

```text
n_radial=4
rank_max=4
```

Record both accuracy and wall-clock/memory because truncation is explicitly a cost--resolution parameter.

---

# 16. New benchmark script

Add:

```text
scripts/benchmark_angular_state.py
```

or the repository-equivalent location.

Compare at least:

```text
derived_aux
independent explicit triplet
spectral
```

Measure:

```text
forward-pass milliseconds
training-step milliseconds
peak CUDA memory
number of stochastic angular state scalars
number of ordinary ALIGNN triplets
```

Use synthetic or real batches with increasing atom count, e.g.

```text
N = 8, 16, 32, 64
```

when feasible.

For spectral mode explicitly report:

$$
N D_\theta
$$

as the angular stochastic-state size.

For default \(D_\theta=40\),

```text
N=64 -> 2560 angular state scalars
```

rather than a combinatorial triplet state.

This benchmark should be kept with the paper ablation outputs.

---

# 17. Required integration tests

In addition to the descriptor tests in Section 6:

## 17.1 Independent corruption

Hold the lattice and coordinate noise draws fixed.

Change only the RNG draw for `noise_spectrum()`.

Assert:

```text
x_t identical
F_t identical
P_t different
```

## 17.2 P affects coordinate score

In full spectral mode,

$$
\frac{\partial\hat s_F}{\partial P_t}\neq0.
$$

## 17.3 P affects lattice noise

$$
\frac{\partial\hat\epsilon_L}{\partial P_t}\neq0.
$$

## 17.4 Geometry affects P prediction

With \(P_t\) fixed, perturb \(F_t\) or \(L_t\).

Require:

$$
\hat\epsilon_P
$$

to change.

Equivalently verify a generic nonzero gradient such as

$$
\frac{\partial\hat\epsilon_P}{\partial F_t}\neq0.
$$

## 17.5 Coupling-off control

With `spectrum_feedback=False`:

```text
changing P_t does NOT change eps_frac
changing P_t does NOT change eps_lattice
changing F_t/L_t DOES change eps_spectrum
```

## 17.6 No projection

Construct an intentionally inconsistent `P_t`.

Verify the denoiser consumes it exactly as supplied.

There must be no call equivalent to:

```python
spectrum_state = angular_spectrum(F_t, L_t)
```

inside the model forward path.

## 17.7 Sampler carries P

Instrument the sampler and verify every predictor model call receives:

```text
F_t
L_t
P_t
```

and emits all three reverse predictions.

## 17.8 Final readout discards P

Default generated structure output remains unchanged.

## 17.9 Legacy explicit-triplet mode still works

Run the core previous-handoff tests unchanged.

A checkpoint/config using:

```text
angle_mode="independent"
```

must still instantiate the explicit persistent-triplet model.

## 17.10 Old original checkpoints still work

Configurations without any new angular mode or spectrum fields must continue to reproduce the original F,L model.

---

# 18. Optional but highly useful consistency diagnostic

Do not train on a consistency loss.

But during validation or diagnostic sampling, compute

$$
P_{\rm geom,t}
=
P(F_t,L_t)
$$

from the current geometry and compare it with the independently evolving \(P_t\).

Because `P_t` is stored standardized, compare in a common representation.

For example,

$$
C_t
=
\frac{1}{ND_\theta}
\|
P_t^{\rm denorm}
-
P_{\rm geom,t}
\|_F^2.
$$

Record \(C_t\) versus reverse timestep.

This directly tests the hypothesis that independently corrupted redundant representations become more mutually compatible as reverse diffusion approaches clean structures.

Do not feed \(C_t\) into the loss.

Do not project either representation toward the other.

---

# 19. Architectural interpretation that comments/docs should preserve

The spectral field is not merely an auxiliary descriptor.

It is an independent stochastic state.

The clean crystal generates three redundant views:

$$
L_0:
\text{global cell geometry},
$$

$$
F_0:
\text{periodic atomic geometry},
$$

$$
P_0:
\text{finite local three-body angular spectrum}.
$$

Their forward corruptions are independent:

$$
q(X_t|X_0)
=
q_L(x_t|x_0)
q_F(F_t|F_0)
q_P(P_t|P_0).
$$

Their reverse dynamics are coupled:

$$
p_\phi(X_{t-1}|X_t)
\not=
p_L\,p_F\,p_P.
$$

The denoiser uses all three representations jointly:

$$
(\hat\epsilon_L,\hat s_F,\hat\epsilon_P)
=
D_\phi(x_t,F_t,P_t,t,c).
$$

The spectral state is discarded only after the reverse trajectory is complete.

---

# 20. Why the finite field is still a bond-angle representation

Do not describe \(P_i\) as an arbitrary learned latent.

It has an explicit geometric meaning:

$$
P^{(i)}_{\mu\mu'\nu}
=
\sum_{j,k}
a_{ij\mu}a_{ik\mu'}
\cos^\nu\theta_{jik}.
$$

Therefore each component is a finite radial/angular moment of the local environment.

Increasing:

```text
n_radial
rank_max
```

systematically increases the radial/angular resolution.

The finite truncation is a deliberate inductive bias and computational control.

The implementation should call this something like:

```text
redundant angular moment field
```

or:

```text
finite angular spectrum
```

rather than claiming to implement canonical SOAP.

The conceptual precedent is the density-trick family of atomistic representations—SOAP, moment tensor descriptors, ACE—while the independently diffused redundant state is the new ALIGNN-CSP construction.

---

# 21. Expected code simplification relative to the previous handoff

For `angle_mode="spectral"` the following primary-path machinery is no longer needed:

```text
persistent dense triplet index
per-triplet stochastic state
CFG triplet-index duplication
wrapped angular score
angular VE corrector
angular VE predictor
dynamic P-dimensionality tied to triplet count
```

Do not globally delete these because explicit-triplet mode must remain reproducible.

But the spectral path should not touch them.

Its only new per-sample stochastic tensor is:

```text
p: (N,40)
```

under the default truncation.

---

# 22. Definition of done

The migration is complete when all of the following are true.

```text
1. angle_mode="independent" still reproduces the explicit persistent-triplet
   implementation from the previous handoff.

2. angle_mode="spectral" carries an independent per-atom P_t tensor through
   every diffusion timestep.

3. P_0 is computed from a finite tensor-moment expansion of the clean local
   geometry.

4. Descriptor construction agrees numerically with the explicit j,k angular
   double sum in a unit test.

5. No explicit j,k enumeration occurs in the production spectrum builder.

6. Default P dimension is 40 per atom.

7. P_t is independently Gaussian-corrupted with a VP process.

8. One shared denoiser maps
       (L_t,F_t,P_t) -> (eps_L,score_F,eps_P).

9. Full coupling tests show
       P_t -> coordinate prediction,
       P_t -> lattice prediction,
       F_t/L_t -> spectral prediction.

10. The spectral sampler has no persistent-triplet state or triplet CFG
    duplication.

11. The ordinary local ALIGNN line graph remains available as the backbone's
    geometry-derived three-body inductive bias.

12. Final structures are read exclusively from L_0 and F_0.

13. Existing baseline, derived-angle, and explicit-triplet checkpoints remain
    loadable.

14. Timing/memory benchmarks compare explicit triplets against the finite
    spectrum.

15. No consistency projection or consistency training loss has been added.
```

The scientific hypothesis tested by the final model is:

> Crystal structure diffusion can benefit from jointly denoising independently corrupted, redundant representations of global, pairwise, and finite three-body geometry; a truncated angular moment field supplies the three-body state without explicitly carrying the combinatorial set of bond-angle variables.

