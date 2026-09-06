# Claude Code handoff: promote bond angles to a coequal redundant diffusion field

## Goal

Modify the `lg-angle-diffusion-matrix` branch of ALIGNN-CSP so that bond angles become a genuine third diffusion state, coequal with the lattice and fractional-coordinate states.

The current branch does **not** yet do this. It derives the angular corruption from the currently noised \((F_t,L_t)\) geometry and applies an auxiliary angular denoising objective. That implementation is useful and should be preserved as a legacy ablation, especially because preliminary reconstruction results indicate that even this weaker angular inductive bias improves nearly every reconstruction benchmark.

The new proposed model must instead have the state

$$
X_t=(x_t,F_t,\Theta_t),
$$

where

* \(x_t\) is the existing six-dimensional diffused lattice representation,
* \(F_t\) is the existing diffused fractional-coordinate representation,
* \(\Theta_t\) is an independently diffused redundant bond-angle field.

At clean data,

$$
\Theta_0=g(F_0,L_0),
$$

but after independent forward corruption there is no requirement that

$$
\Theta_t=g(F_t,L_t).
$$

Intermediate inconsistency is intentional.

The reverse model must be joint:

$$
(\hat\epsilon_L,\hat s_F,\hat s_\Theta)
=
D_\phi(x_t,F_t,\Theta_t,t,c).
$$

All three current states must influence the shared denoiser before all three reverse predictions are made. Do not implement three independent denoisers.

At the end of generation, discard \(\Theta_0\). The physical crystal is defined only by

$$
(L_0,F_0).
$$

Do not project \(F_0,L_0\) to agree with the generated \(\Theta_0\), and do not use \(\Theta_0\) to reconstruct the final structure.

---

# 1. Non-negotiable architectural contract

The implementation is correct only if all of the following are true.

### 1.1 Independent forward corruption

The forward kernel must factorize:

$$
q(X_t|X_0)
=
q_L(x_t|x_0)
q_F(F_t|F_0)
q_\Theta(\Theta_t|\Theta_0).
$$

The three channels use the same graph-level timestep \(t\), but independent random noise.

This is a statement about the corruption kernel only. It does **not** imply independent reverse denoising.

### 1.2 Joint reverse prediction

At each denoising evaluation, one shared forward pass must receive

```text
x_t
F_t
Theta_t
t
conditioning
```

and return

```text
eps_lattice
eps_frac
eps_angle
```

where `eps_frac` and `eps_angle` denote the current code's sigma-scaled score convention for the two wrapped-normal VE channels.

The following computational dependencies must exist:

```text
Theta_t -> triplet z -> pair y -> atom h -> eps_frac
Theta_t -> triplet z -> pair y -> atom h -> eps_lattice

F_t,L_t -> pair y -> triplet z -> eps_angle
```

The existing ALIGNN hierarchy is already almost sufficient for this. Do not add cross-attention, a second GNN, or a separate angular denoiser.

### 1.3 Persistent angular degrees of freedom

The identity and number of angular state variables must not change as coordinates denoise.

Do **not** create and destroy entries of `Theta_t` using the current kNN/radius triplet set.

Build a broad persistent triplet index once for the batch and carry the corresponding angular state through the complete forward or reverse trajectory.

The simplest primary implementation is the full line graph of the existing dense ordered pair graph:

```python
pair_index = dense_pair_index(natoms)
lg_src, lg_dst = _line_graph_edges(
    src, dst, num_nodes, allowed=None
)
```

The state index is therefore the persistent ordered pair of parent-edge indices `(lg_src, lg_dst)`.

With a dense \(N^2\) pair graph this gives \(O(N^3)\) triplets. For the crystal sizes currently used by ALIGNN-CSP this should be benchmarked before introducing additional pruning. Do not silently reintroduce geometry-dependent triplet membership merely to save memory.

### 1.4 Smooth relevance remains, but changes meaning

Retain the existing continuous pair relevance

$$
s_{ij,t}=u(r_{ij,t};r_c)
$$

and triplet relevance

$$
s_{ijk,t}=s_{ij,t}s_{jk,t}.
$$

These are deterministic functions of the current \((F_t,L_t)\).

Their new role is to gate **interaction strength**, not define whether the angular state exists.

Thus:

```text
Theta_ijk,t exists for the complete persistent triplet set.
s_ijk,t controls how strongly that state participates in ALIGNN messages.
```

A triplet with `s_ijk == 0` remains present in `Theta_t`; its message contribution is simply zero.

This retains the current DimeNet-style smooth cutoff / ReaxFF-style product-gating idea without subordinating the angular diffusion state to the coordinate topology.

### 1.5 No consistency loss in the primary model

Do not add

$$
\|\Theta_t-g(F_t,L_t)\|^2
$$

or an equivalent consistency penalty.

Do not replace `Theta_t` by the geometry-derived angle during sampling.

Do not average the two.

The clean training distribution already establishes their relationship through

$$
\Theta_0=g(F_0,L_0).
$$

The purpose of this experiment is to determine whether the joint reverse model can exploit and reconstruct correlations among independently corrupted redundant representations.

A consistency penalty can be introduced later as a separate ablation if desired.

### 1.6 Discard the angular state at readout

`sample()` should continue to return the physical result

```python
{
    "frac": ...,
    "lattice": ...,
    "natoms": ...,
}
```

by default.

It may optionally expose the final angular latent under a diagnostic flag such as `return_latents=True`, but no downstream structure-building code should depend on it.

---

# 2. Angular stochastic process

Use a wrapped-normal score process, following the angular-torus construction used by Torsional Diffusion and the existing wrapped-normal coordinate process.

FoldingDiff is precedent for directly diffusing bond/internal angles. Torsional Diffusion is the cleaner precedent for wrapped-normal score matching on an angular torus.

Do not retain the current geometry-induced angular-displacement target as the primary new process.

## 2.1 Clean angular state

For every persistent line-graph edge `(lg_src[p], lg_dst[p])`, compute the clean interior angle from the clean minimum-image pair vectors:

$$
\theta_{p,0}
=
\arccos
\frac{-r_{a,0}\cdot r_{b,0}}
{|r_{a,0}||r_{b,0}|}.
$$

Use the existing ALIGNN convention in `bond_angle()` / `torch_bond_cosines()`.

For numerical convenience, represent the diffused angular state internally in **turns**

$$
\phi_{p,0}=\frac{\theta_{p,0}}{2\pi},
$$

so the angular state has unit period exactly like each fractional-coordinate component.

Clean physical bond angles then occupy

$$
\phi_0\in[0,1/2].
$$

No requirement is imposed that noisy \(\phi_t\) remain in that half interval.

## 2.2 Forward angular corruption

Use

$$
\phi_t
=
w\!\left(
\phi_0+\sigma^\Theta_t z_\Theta
\right),
\qquad
z_\Theta\sim\mathcal N(0,1),
$$

where `w` wraps into `[0,1)`.

The target is the sigma-scaled wrapped-normal score, exactly analogous to the fractional-coordinate process:

$$
s^\Theta_t
=
\sigma^\Theta_t
\nabla_{\phi_t}
\log q(\phi_t|\phi_0).
$$

Start by using the same **dimensionless** geometric sigma ladder as the fractional-coordinate channel:

```text
0.005 -> 0.5
```

because both `F` and normalized angular state `phi` have unit periodicity. This avoids choosing an angular noise scale merely because radians have different units.

Implement separate `angle_sigma_min` / `angle_sigma_max` configuration fields anyway, defaulting to the coordinate values. This lets the angular schedule be tuned or ablated later without architectural changes.

Independent channel means independent random noise, not necessarily a different schedule.

## 2.3 Reverse angular process

Use the same wrapped VE predictor-corrector mathematics as the fractional-coordinate channel, but with `schedule.angle_sigmas`.

Initialize

```python
phi = torch.rand(n_triplets, device=device)
```

from the uniform distribution on its periodic domain.

At every corrector call and predictor call, evaluate the model on the **joint current state**

```python
_denoise(frac, x_lat, phi, t)
```

and obtain all three predictions.

Do not call an angle-only network.

Do not recompute `phi` from `frac` or `lattice`.

---

# 3. Required file changes

Line numbers below refer to the current `lg-angle-diffusion-matrix` branch before these edits. After the first patch, function/class names become the authoritative anchors.

---

## A. `alignn/inverse/diffusion.py`

### Current anchors

* lines 0–18: docstring describes only two processes.
* lines 94–100: unit-period wrapping helpers.
* lines 103–120: `wrapped_normal_score`.
* lines 122–160: `DiffusionSchedule`.
* lines 162–178: `noise_lattice()` and `noise_frac()`.

### Required changes

#### A1. Generalize or reuse the wrapped-normal machinery

Because the proposed angular state is normalized to unit-period turns, the existing `wrapped_normal_score()` can be reused without inventing a second formula.

Add an explicit angular forward method:

```python
def noise_angle(
    self,
    theta0_turns: torch.Tensor,
    t_triplet: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigma = self.angle_sigmas[t_triplet]
    z = torch.randn_like(theta0_turns)
    theta_t = wrap_frac(theta0_turns + sigma * z)
    target = wrapped_normal_score(
        theta_t - theta0_turns,
        sigma,
    )
    return theta_t, target
```

Shapes:

```text
theta0_turns: (P,)
t_triplet:    (P,)
theta_t:      (P,)
target:       (P,)
```

where `P` is the number of persistent triplets in the batch.

#### A2. Add angular sigma schedule

Extend `DiffusionSchedule` around lines 122–160:

```python
angle_sigma_min: Optional[float] = None
angle_sigma_max: Optional[float] = None
```

Interpret `None` as:

```python
angle_sigma_min = sigma_min
angle_sigma_max = sigma_max
```

Construct

```python
self.angle_sigmas
```

with the same geometric-ladder logic as `self.sigmas`.

Update `.to()` at lines 157–159 to move it.

#### A3. Rewrite module documentation

The module docstring must say there are now three forward processes:

```text
lattice: VP/DDPM
fractional coordinates: wrapped-normal VE
redundant bond-angle field: independently corrupted wrapped-normal VE
```

State explicitly that the forward corruptions share `t` but draw independent noise.

---

## B. `alignn/inverse/angles.py`

### Current anchors

* lines 0–56: documentation says the angle is a derived denoising target and explicitly says no persistent angular state exists.
* lines 67–77: exports.
* lines 86–104: `bond_angle`, `wrap_angle`.
* lines 106–120: relevance helpers.
* lines 123–139: edge-vector helper.
* lines 141–158: `angular_denoising_target`.
* lines 160–181: current wrapped smooth-L1 objective.

### Required changes

#### B1. Rewrite the conceptual documentation

Delete the current claim that a crystal cannot have a persistent angular state because its triplet set changes.

That was true only under the old geometry-defined sparse triplet representation.

The new implementation resolves the issue by defining a persistent broad triplet field and using the smooth topology only as a relevance gate.

#### B2. Keep these helpers

Retain:

```python
bond_angle
wrap_angle
pair_relevance
triplet_relevance
edge_vectors
CutoffPolynomial
```

#### B3. Preserve the old target for the legacy ablation

Do **not** simply delete:

```python
angular_denoising_target
angle_denoising_loss
```

Rename or clearly mark them as legacy, e.g.

```python
derived_angular_target(...)
derived_angle_loss(...)
```

because the current weaker angle-supervision experiment is scientifically useful and should remain reproducible.

Old checkpoints/configurations need a route back to this behavior.

#### B4. Add a clean-state constructor

Add a helper whose responsibility is only to convert clean geometry and persistent indices into the clean redundant field.

Suggested API:

```python
def clean_angle_state(
    r: torch.Tensor,
    lg_src: torch.Tensor,
    lg_dst: torch.Tensor,
) -> torch.Tensor:
    theta = bond_angle(r[lg_src], r[lg_dst])
    return torch.remainder(theta / TWO_PI, 1.0)
```

Do not perform forward noising here; that belongs in `DiffusionSchedule`.

---

## C. `alignn/inverse/denoiser.py`

This is the central architectural change.

### Current anchors

* lines 94–118: `dense_pair_index`.
* lines 147–166: angle embedding builder.
* lines 168–359: `ALIGNNCSPDenoiser.__init__`.
* lines 369–417: `_edge_geometry`.
* lines 427–455: `_angle_output`.
* lines 457–478: `forward()` signature/docs.
* lines 479–511: current pair geometry/features.
* lines 515–537: dynamically builds triplets and derives angle features from current geometry.
* lines 538–562: ALIGNN angle→bond→atom coupling.
* lines 563–577: no-line-graph angular control.
* lines 578–597: coordinate/lattice heads and output dict.

### Required changes

#### C1. Add persistent triplet-index helper near `dense_pair_index`

Implement:

```python
def dense_triplet_index(
    pair_index,
    num_nodes: int,
):
    src, dst, edge_graph_id = pair_index

    lg_src, lg_dst = _line_graph_edges(
        src,
        dst,
        num_nodes,
        allowed=None,
    )

    triplet_graph_id = edge_graph_id[lg_src]

    if not torch.equal(
        triplet_graph_id,
        edge_graph_id[lg_dst],
    ):
        raise RuntimeError("line-graph edge crossed crystal boundary")

    return lg_src, lg_dst, triplet_graph_id
```

This index must depend only on atom count / dense pair connectivity, never on current distances.

#### C2. Make minimum-image geometry reusable

Training needs clean pair vectors before the denoiser call.

Refactor the current `_edge_geometry` implementation at lines 369–417 into a module-level helper, e.g.

```python
minimum_image_geometry(...)
```

and have the model method call that helper if retaining the method wrapper.

Do not duplicate minimum-image logic in `train_csp.py`.

#### C3. Introduce a mode instead of silently changing `angle_diffusion`

Preserve reproducibility of the existing angular-objective branch.

Recommended configuration:

```python
angle_mode: str = "off"
```

with:

```text
"off"          baseline / angles only as ordinary geometry features
"derived_aux"  current branch behavior
"independent"  new persistent independently diffused angular field
```

Retain old `angle_diffusion` only as a deprecated checkpoint/config compatibility field if needed.

An old config with

```python
angle_diffusion=True
```

and no `angle_mode` must resolve to

```python
angle_mode="derived_aux"
```

rather than silently becoming the new model.

#### C4. Add explicit angular-state input

Change `forward()` around lines 457–468 to accept:

```python
theta_state: Optional[torch.Tensor] = None
triplet_index: Optional[Tuple] = None
```

For `angle_mode == "independent"` both are mandatory.

Recommended shape:

```text
theta_state: (P,) normalized turns in [0,1)
triplet_index:
    lg_src            (P,)
    lg_dst            (P,)
    triplet_graph_id  (P,)
```

Fail loudly if the sizes disagree.

#### C5. Do not derive the independent state from current geometry

The current lines 515–537 do this:

```python
allowed = ...
lg_src, lg_dst = _line_graph_edges(...)
cos_theta = torch_bond_cosines(...)
z = self.angle_embedding(cos_theta)
```

That must remain only for `"off"` / `"derived_aux"` legacy behavior.

For `"independent"`:

```python
lg_src, lg_dst, triplet_graph_id = triplet_index
```

must be used directly.

Do not apply `allowed=`.

Do not change `lg_src`/`lg_dst` with timestep.

Do not calculate the state value from `r`.

#### C6. Embed the independent angular state periodically

The current `cosine_rbf` encoding loses the distinction between different points of a full circular noisy state.

For the independently diffused field, encode `theta_state` periodically using sine/cosine harmonics.

Since `theta_state` is stored in turns:

$$
[\sin(2\pi k\phi),\cos(2\pi k\phi)].
$$

Add a dedicated `angle_state_embedding`.

Do not route the new independent state through the current function that first computes `acos(cos_theta)`.

The existing geometric `cosine_rbf` and `FourierAngular` encoders can remain for the legacy/basis ablations.

A reasonable API distinction is:

```python
angle_feature_basis       # old geometry-derived ALIGNN angle feature
angle_state_basis         # new independently diffused state
```

For the proposed model, use a periodic Fourier state embedding.

#### C7. Keep geometry-derived relevance gates

At lines 486–491 continue computing:

```python
s_edge = envelope(dist)
```

from the current `F_t,L_t`.

For persistent triplets compute:

```python
tri_w = triplet_relevance(
    s_edge,
    lg_src,
    lg_dst,
)
```

for **all** persistent triplets.

Never use:

```python
allowed = s_edge > 0
```

to delete them in independent mode.

#### C8. Preserve the existing coupling path

The existing `WeightedALIGNNConv` stack around lines 538–558 is useful.

For the proposed model:

```text
z starts from theta_state
y starts from F_t,L_t pair geometry
h starts from species/t/lattice/conditioning
```

Then ALIGNN message passing mixes them.

Keep:

```python
h, y, z = layer.forward_tensors(...)
```

with `tri_w` as the line-graph message gate.

This already creates the crucial `Theta -> y -> h` path.

The corresponding `F,L -> z` dependence also exists because the line-graph convolution updates `z` using pair states produced from the coordinate/lattice geometry.

Do not add another network unless tests demonstrate that this dependency is absent.

#### C9. Angular head predicts the angular score

Replace the independent-mode `_angle_output` payload.

For the new mode it should be approximately:

```python
angle_out = {
    "eps": self.angle_head(z).squeeze(-1),
    "weight": tri_w,
    "lg_src": lg_src,
    "lg_dst": lg_dst,
    "triplet_graph_id": triplet_graph_id,
}
```

Do not return a geometry-derived `"theta_t"` as the training target.

The model was given the actual stochastic `theta_state`; it does not need to reconstruct it from geometry.

#### C10. Structural outputs must depend on theta state

The final coordinate and lattice heads at lines 578–593 can remain structurally similar, because `y` and `h` have already received angular information.

Do not detach `z`, `y`, or `h` anywhere in this path.

---

## D. `alignn/inverse/layers.py`

### Current anchor

`WeightedALIGNNConv`, lines 76–116.

### Expected code change

Probably none.

Inspect rather than rewrite.

The existing layer already performs the required hierarchical mixing:

```text
atom graph update
line-graph update
```

and accepts both pair and triplet relevance gates.

The new tests must verify that, with nonzero weights:

```text
theta_state changes -> structural outputs change
F/L changes -> angular output changes
```

If those tests pass, do not make the convolution more complicated.

---

## E. `alignn/inverse/model.py`

### Current anchor

`ALIGNNCSP.forward`, lines 56–82.

### Required change

Add:

```python
theta_state=None
triplet_index=None
```

and forward both to `self.denoiser`.

No other architectural logic belongs in this wrapper.

---

## F. `alignn/inverse/train_csp.py`

### Current anchors

* lines 14–17: imports old angular target/loss.
* lines 56–77: `angle_loss_from_output`.
* lines 79–133: `diffusion_loss`.
* lines 136–184: noise-bucket diagnostic.
* lines 198–231: diffusion/angular CLI options.
* lines 305–329: denoiser configuration.
* lines 367–371: schedule construction.
* lines 391 onward: loss aggregation/checkpointing.

### Required changes

#### F1. Build persistent pair/triplet indices before noising

Inside `diffusion_loss`, after obtaining `natoms`:

```python
pair_index = dense_pair_index(natoms)
triplet_index = dense_triplet_index(
    pair_index,
    num_nodes=batch["frac"].shape[0],
)
```

Use these same indices for clean state construction and the model call.

#### F2. Construct `Theta_0` from clean geometry

Resolve the clean pair geometry once using the shared minimum-image helper:

```python
_, _, r0, _, _ = minimum_image_geometry(
    batch["frac"],
    batch["lattice"],
    *pair_index,
)
```

Then:

```python
lg_src, lg_dst, triplet_graph_id = triplet_index
theta0 = clean_angle_state(
    r0,
    lg_src,
    lg_dst,
)
```

#### F3. Independently corrupt all three channels

Current lines 93–99 already do lattice and coordinates.

Extend to:

```python
x_t, target_lat = schedule.noise_lattice(x0, t)

t_node = t[batch["node_graph_id"]]
f_t, target_frac = schedule.noise_frac(
    batch["frac"],
    t_node,
)

t_triplet = t[triplet_graph_id]
theta_t, target_angle = schedule.noise_angle(
    theta0,
    t_triplet,
)
```

The random draws inside those three calls must be independent.

#### F4. Pass all three current states into one denoiser call

```python
out = model(
    frac=f_t,
    lattice=lattice_t,
    lattice_vec6=x_t,
    theta_state=theta_t,
    triplet_index=triplet_index,
    ...
    pair_index=pair_index,
)
```

The call must occur once for the three predictions.

#### F5. Replace the old independent-mode loss

For the new mode:

```python
loss_ang = F.mse_loss(
    out["angle"]["eps"],
    target_angle,
)
```

because `target_angle` is now a sigma-scaled wrapped-normal score, analogous to `target_frac`.

Do not call `angular_denoising_target()` for independent mode.

The old `angle_loss_from_output()` should survive only for `"derived_aux"`.

#### F6. Keep relevance weighting as a separate scientific factor

Current code automatically relevance-weights the angular auxiliary loss.

For the new independent field, make this explicit:

```python
angle_loss_weighting: Literal["uniform", "relevance"]
```

Implement both.

For `"uniform"`:

```python
loss_ang = mean(error**2)
```

For `"relevance"`:

```python
loss_ang = sum(tri_w * error**2) / sum(tri_w)
```

The triplet weights must gate messages in both cases.

Do not let the choice of loss weighting change the angular state set.

This ablation is required because message relevance and supervision relevance are distinct design choices.

#### F7. Loss weights

Keep

```python
lattice_weight
frac_weight
angle_weight
```

as explicit hyperparameters.

“Coequal diffusion state” does not mean the raw numerical MSEs must each receive coefficient 1. Their scales differ.

Do not silently change the current `frac_weight=10.0` code default merely because the manuscript currently contains another value. Record the actual experimental configuration.

#### F8. CLI/config additions

Around lines 198–231 add at minimum:

```text
--angle-mode {off,derived_aux,independent}
--angle-sigma-min
--angle-sigma-max
--angle-loss-weighting {uniform,relevance}
--angle-state-basis {fourier}
```

Keep backward-compatible old flags only if needed for existing runs.

All new fields must be written into `config.json` and checkpoint metadata.

#### F9. Schedule construction

Around lines 367–371 pass:

```python
angle_sigma_min=args.angle_sigma_min
angle_sigma_max=args.angle_sigma_max
```

#### F10. Diagnostics

Extend `sigma_bucket_report()` or add an analogous report so training logs show both:

```text
fractional score baseline -> model loss
angular score baseline -> model loss
```

This is useful because the new angular channel is now a genuine score problem rather than an auxiliary regression.

---

## G. `alignn/inverse/sample.py`

This is the other major required change.

### Current anchors

* lines 20–25: imports.
* lines 28–35: CFG pair-index duplication.
* lines 65–90: fixed pair graph and priors.
* lines 91–124: `_denoise`.
* lines 125–153: coordinate corrector/predictor.
* lines 154–173: lattice reverse update.
* lines 174–175: final physical output.
* lines 209–255: checkpoint config and schedule reconstruction.

### Required changes

#### G1. Build persistent triplets once

Immediately after line 67:

```python
pair_index = dense_pair_index(natoms)
triplet_index = dense_triplet_index(
    pair_index,
    num_nodes=n_nodes,
)
```

This exact `triplet_index` must survive the entire reverse trajectory.

#### G2. Add CFG duplication for triplets

Create a helper analogous to `_double_pair_index`.

Because `lg_src` / `lg_dst` index parent edges, when duplicating the batch they must be offset by the number of pair edges in the original batch.

`triplet_graph_id` must be offset by `n_graphs`.

Be careful here: node offsets, edge offsets, and graph offsets are different quantities.

Add a test specifically for this.

#### G3. Add angular prior

Current lines 88–90 initialize:

```python
frac = uniform
x_lat = normal
```

Add:

```python
theta = torch.rand(n_triplets, device=device)
```

representing a uniform prior over angular turns.

#### G4. `_denoise` takes the complete state

Change:

```python
def _denoise(frac_t, x_lat_t, t_idx):
```

to:

```python
def _denoise(frac_t, x_lat_t, theta_t, t_idx):
```

Pass `theta_t` and persistent `triplet_index` through CFG duplication and into the model.

Return all three guided predictions:

```python
eps_frac
eps_lattice
eps_angle
```

CFG must be applied to the angular prediction exactly as it is to the structural predictions.

#### G5. Every reverse prediction must use one common state

At a given substep, call:

```python
eps_f, eps_l, eps_theta = _denoise(
    frac,
    x_lat,
    theta,
    t,
)
```

Do not calculate the angle prediction before mutating `frac` and then calculate the coordinate prediction after mutating `theta`.

All predictions used for one predictor transition should correspond to the same current joint state.

For corrector iterations, after a stochastic update the next corrector iteration may of course evaluate the newly updated joint state.

#### G6. Add angular Langevin corrector

Use the same VE corrector structure as coordinates:

```python
score_theta = eps_theta / angle_sigma_t
```

and update the wrapped angular state.

Because both fractional coordinates and angular turns use unit periodic domains, their numerical form can be shared.

Prefer extracting a small wrapped-VE update helper rather than duplicating formulas.

#### G7. Add angular predictor

After the joint denoiser call:

```python
d_angle_sigma2 = (
    angle_sigma_t**2
    - angle_sigma_prev**2
)
```

apply the same wrapped VE reverse predictor to `theta`.

Wrap back into `[0,1)` afterward.

#### G8. Leave lattice DDPM logic intact

Lines 154–173 can remain except that `eps_l` now came from the same three-state denoiser evaluation.

#### G9. Discard theta by default

Keep:

```python
return {
    "frac": wrap_frac(frac),
    "lattice": lattice,
    "natoms": natoms,
}
```

Optionally:

```python
if return_latents:
    out["theta_state"] = theta
    out["triplet_index"] = triplet_index
```

for mechanism diagnostics only.

#### G10. Checkpoint loading

Around lines 209–255:

* add new denoiser configuration fields,
* reconstruct `angle_sigmas`,
* map old `angle_diffusion=True` checkpoints to `angle_mode="derived_aux"`.

Do not interpret an old auxiliary-angle checkpoint as an independent-angle checkpoint.

---

## H. `alignn/inverse/generate.py`

### Current anchor

`ALIGNNGenerator.__init__`, lines 188–202.

### Required change

When a user overrides `num_steps`, this file reconstructs `DiffusionSchedule`.

Propagate:

```python
angle_sigma_min=self.config.get(
    "angle_sigma_min",
    self.config["sigma_min"],
)
angle_sigma_max=self.config.get(
    "angle_sigma_max",
    self.config["sigma_max"],
)
```

Otherwise an independently diffused checkpoint will use a different angular schedule when sampled with a shortened step count.

No public `GeneratedStructure` change is needed because `Theta` is intentionally discarded.

---

## I. `alignn/inverse/ablations.py`

### Current anchors

* lines 0–28: current scientific questions.
* lines 45–83: existing A0/A1/A2/A3/A4/A6 configurations.
* lines 84–100: descriptions/comparisons.
* lines 102–175: current line-graph × angular-objective matrix.

### Required conceptual rewrite

Do not delete the current derived-angular experiment. It is now an important intermediate baseline.

The primary scientific comparison should distinguish:

### B0 — ordinary structural diffusion

```text
state: F,L
angles: ordinary geometry-derived ALIGNN features
independent theta state: no
```

This is the existing baseline.

### B1 — derived angular auxiliary objective

```text
state: F,L
angles: geometry-derived
angular head/loss: yes
independent theta state: no
```

This is essentially the current A3 behavior and preserves the preliminary result that the weaker angular inductive bias already improves reconstruction metrics.

### B2 — independent angular state, coupling severed

```text
state: F,L,Theta
Theta independently noised: yes
angular score loss: yes
Theta -> pair/atom structural path: severed
F,L -> angular prediction: retain if possible
```

This tests whether the additional diffusion task is merely auxiliary supervision.

### B3 — independent angular state, fully coupled

```text
state: F,L,Theta
Theta independently noised: yes
persistent triplets: yes
smooth relevance: yes
joint shared denoiser: yes
```

This is the proposed model.

### Required secondary controls

Keep/add:

```text
smooth topology only
uniform vs relevance-weighted angular score loss
direct theta-state vs cosine/angular-feature representation, if that
    previously planned representation ablation is still being run
```

Do not use one ablation switch to change both message gating and loss weighting.

The comparison of primary interest is:

```text
B1 vs B3
```

because it asks whether promoting angles from a derived auxiliary target to an actual stochastic state provides additional value.

The mechanistic comparison is:

```text
B2 vs B3
```

because it asks whether cross-representation coupling is doing the work.

---

## J. `alignn/tests/test_inverse_angle_diffusion.py`

The current test file is heavily tied to the derived-target interpretation and needs substantial revision.

### Current anchors

* lines 0–11: test-suite contract.
* lines 73–82: `_forward`.
* lines 83–160: angular wrapping/derived-target tests.
* lines 162–273: smooth-cutoff/topology tests.
* lines 299–330: baseline compatibility.
* lines 332 onward: feedback/coupling and end-to-end tests.
* lines 407 onward: angular-gradient test.

### Required tests

#### J1. Independent forward corruption

```python
def test_angle_noise_is_independent_of_frac_noise():
```

Fix the clean structure and timestep.

Control RNG so lattice/F noise are identical between two runs but angular noise differs.

Assert:

```text
x_t identical
F_t identical
Theta_t different
```

This directly tests forward-factorized corruption.

#### J2. Angular wrapped-normal score

Test `noise_angle()` against the unit-period wrapped-normal score helper.

Assert finite scores at low and high sigma.

Assert periodicity under:

```python
theta0 + integer
```

in normalized-turn representation.

#### J3. Persistent triplet identity

Replace the current test that expects triplets to be inserted/deleted across a cutoff.

New requirement:

```python
def test_triplet_index_is_invariant_under_geometry_changes():
```

Generate two radically different `F,L` states with the same atom counts.

Assert exact equality of:

```text
lg_src
lg_dst
triplet_graph_id
```

Then separately assert that `tri_w` changes smoothly and can reach zero.

#### J4. Smooth relevance survives

Keep the existing cutoff-envelope tests.

The expected behavior is now:

```text
state remains
message weight -> 0
```

rather than:

```text
triplet disappears after its contribution reaches 0
```

#### J5. Theta affects structural predictions

For the fully coupled model:

```python
out1 = model(F, L, Theta1, ...)
out2 = model(F, L, Theta2, ...)
```

with every other input identical.

Initialize zero-initialized final heads to known nonzero test weights if necessary.

Assert at least one of:

```text
eps_frac changes
eps_lattice changes
```

Prefer testing both.

This proves the angular state is truly an input to structural denoising.

#### J6. F/L affect angular prediction

Hold `Theta_t` fixed and perturb `F_t` or `L_t`.

Assert:

```text
eps_angle changes
```

This proves the angular reverse model is not an isolated angular denoiser.

#### J7. Coupling-off control

For the B2/feedback-off model:

```text
changing Theta_t does not change eps_frac/eps_lattice
changing F/L can still change eps_angle
```

The test should distinguish architectural coupling from shared-loss effects.

#### J8. No geometry projection

Supply a deliberately inconsistent pair:

```text
Theta_t != g(F_t,L_t)
```

and verify that the model consumes the supplied `Theta_t` unchanged.

Do not allow any forward path to overwrite it with a geometry-derived angle.

#### J9. Joint sampler call

Instrument or monkeypatch the model so the test can verify that every predictor evaluation receives:

```text
F_t
L_t
Theta_t
```

and returns all three heads.

#### J10. Angular prior

Assert sampling initializes the angular state inside `[0,1)` and independently of the coordinate prior.

#### J11. Final readout discards theta

Default `sample()` output must remain:

```text
frac
lattice
natoms
```

and no physical structure-construction code should consume the angular state.

With `return_latents=True`, diagnostics may expose it.

#### J12. Backward compatibility

Keep a test that the default model with angle mode off has the exact original parameter/output contract.

Also test:

```text
old angle_diffusion=True config
    -> derived_aux behavior
```

rather than independent behavior.

#### J13. Gradient connectivity

For the fully coupled model, explicitly test nonzero gradients corresponding to the desired dependency graph:

$$
\frac{\partial \hat s_F}{\partial\Theta_t}\neq0,
\qquad
\frac{\partial \hat\epsilon_L}{\partial\Theta_t}\neq0,
$$

and

$$
\frac{\partial \hat s_\Theta}{\partial F_t}\neq0
$$

for a generic nondegenerate test input.

Because output heads are zero-initialized, set their final-layer weights to deterministic nonzero values inside this architecture test.

This is the strongest unit test for the phrase “all three representations talk to each other.”

---

# 4. `README.md` documentation update

The explicit bond-angle-diffusion reference section beginning around the current README line 296 is now conceptually obsolete in one crucial respect.

It currently says that there is no persistent `Theta_t` and that the angular objective is induced by `(F_t,L_t)`.

Preserve that description under a clearly labeled **legacy derived-angular auxiliary model**.

Add a new section for the proposed model:

```text
persistent redundant bond-angle field
independent forward corruption
persistent broad triplet index
smooth geometric relevance gates
joint ALIGNN reverse denoiser
Theta discarded at final crystal readout
```

Do not call the new channel an additional physical degree of freedom.

It is a redundant stochastic representation of the same crystal geometry.

---

# 5. Files that should not require semantic changes

## `alignn/inverse/evaluate.py`

Existing physical bond-angle distribution and relaxation-displacement metrics remain valid because final crystals are still represented by `F,L`.

Optionally add a **diagnostic only** consistency measurement:

$$
C_t
=
\frac{\sum_p s_{p,t}
d_{\mathrm{circ}}
\left(
\Theta_{p,t},
g_p(F_t,L_t)
\right)}
{\sum_p s_{p,t}},
$$

recorded as a function of reverse timestep.

Do not train against this quantity.

It would be scientifically useful because it lets us observe whether independently corrupted redundant representations spontaneously become mutually consistent during denoising.

## `alignn/inverse/layers.py`

No semantic changes expected unless the dependency tests fail.

## `alignn/inverse/relax_rank.py`

No change. The angular latent has already been discarded before relaxation.

---

# 6. Implementation order

Implement in this order so failures remain localized.

1. Add persistent triplet-index utilities.
2. Add `clean_angle_state()`.
3. Add `angle_sigmas` and `noise_angle()` to `DiffusionSchedule`.
4. Add unit tests for the angular forward kernel.
5. Modify denoiser API to accept `theta_state` and `triplet_index`.
6. Make independent-mode `z` originate from `theta_state`, not `g(F_t,L_t)`.
7. Preserve smooth pair/triplet relevance only as message gates.
8. Add angular score head output.
9. Update training to produce independent `Theta_t` and score targets.
10. Add dependency/gradient tests proving joint reverse coupling.
11. Add the angular prior and reverse integrator to `sample.py`.
12. Add CFG duplication for the persistent angle field.
13. Update checkpoint/config schedule restoration.
14. Update `generate.py` shortened-schedule reconstruction.
15. Rewrite ablation definitions while preserving the old derived-angular arm.
16. Rewrite tests that assumed dynamic triplet insertion/deletion.
17. Update README documentation.
18. Run the complete inverse test suite and then the repository test suite.

---

# 7. Things not to do

Do **not**:

```text
recompute Theta_t from F_t,L_t at each reverse step
project Theta_t onto geometry
project F_t,L_t onto Theta_t
add a consistency penalty to the primary model
give Theta its own standalone GNN
use hard kNN changes to create/delete angular state
delete a Theta entry when its relevance reaches zero
let angular loss weighting implicitly determine topology
use three separate denoiser calls for L, F, Theta at the same timestep
use an old angle_diffusion=True checkpoint as if it contained an
    independently diffused angular state
read the final physical crystal from Theta
```

---

# 8. Acceptance criteria

The implementation is complete only when all of these statements are true.

### State

```text
The sampler explicitly carries (x_lat, frac, theta) through every timestep.
```

### Forward process

```text
Theta_t is sampled independently from F_t and L_t given the clean state.
```

### Persistent field

```text
The number and identity of theta variables remain fixed for an entire
trajectory.
```

### Continuous topology

```text
Pair/triplet relevance varies with F_t,L_t, but zero relevance suppresses a
message rather than destroying the corresponding theta state.
```

### Reverse coupling

A forward pass has the contract

```text
D(x_t, F_t, Theta_t, t, c)
    -> (eps_lattice, eps_frac, eps_angle)
```

and architecture tests demonstrate:

```text
Theta_t affects eps_frac
Theta_t affects eps_lattice
F_t/L_t affect eps_angle
```

### Sampling

```text
All three reverse predictions at a predictor step are evaluated from the same
current joint state.
```

### Readout

```text
Only F_0 and L_0 define the generated crystal.
Theta_0 is discarded.
```

### Compatibility

```text
The original F,L baseline still works unchanged.
The current derived-angular auxiliary implementation remains reproducible as
a legacy ablation.
```

### Scientific controls

At minimum the experiment suite can distinguish:

```text
F,L baseline
derived angular auxiliary objective
independently diffused theta, coupling cut
independently diffused theta, fully coupled
smooth topology alone
angular-loss relevance weighting
```

---

# 9. Architectural intent

The intended contribution is not merely “predict bond angles too.”

The clean crystal is represented redundantly at three geometric levels:

$$
L:
\text{global cell geometry},
$$

$$
F:
\text{periodic atomic geometry},
$$

$$
\Theta:
\text{local three-body geometry}.
$$

Each representation is independently corrupted in the forward process. Their statistical dependence survives that corruption because all three originate from the same clean crystal.

The reverse model should exploit that dependence.

The central hypothesis is therefore:

> An ALIGNN denoiser can improve crystal reconstruction by jointly denoising multiple independently corrupted, redundant geometric representations and allowing information to propagate among them during every reverse step.

The current derived-angle branch tests a weaker statement: whether explicit angular supervision helps when angles remain secondary to the coordinate/lattice process. Preserve that result. The new model is specifically meant to test whether promoting the bond-angle field to an actual member of the stochastic state produces an additional gain.

