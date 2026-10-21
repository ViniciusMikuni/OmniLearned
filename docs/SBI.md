# Likelihood-based inference with OmniLearned generators (`omnilearned.sbi`)

Route A (neural likelihood estimation): a conditional flow-matching generator
`O` trained across parameter settings defines a density `q_O(G | theta)` for
a particle cloud `G`. Observed clouds constrain `theta` through the summed
per-event log densities. This package is generic; physics simulators,
training studies and closure reports belong to the application workspace.

## Conventions

* Flow matching uses `z_t = (1 - t) x + t eps` with data at `t = 0` and base
  noise at `t = 1`; the network predicts `v = dz/dt = eps - x`.
  Generation integrates `t: 1 -> 0`. Density evaluation integrates the data
  point forward `t: 0 -> 1` and uses
  `log p_data(x) = log p_base(z_1) + integral_0^1 div v(z_t, t) dt`
  (sign verified by analytic tests). A flow-matching regression loss is not
  a log likelihood.
* `GaussianBase` (diagonal Gaussian per particle feature) is the single base
  specification. The upstream `get_eps` presets (`jetnet`, `lhco`, standard)
  are produced by `GaussianBase.from_dataset`, so training, generation and
  evaluation agree. Checkpoints written by `omnilearned.sbi.flow` store it.
* Validity masks are explicit `[N, P]` booleans. Padded particles are not
  random coordinates: they are held at zero and excluded from velocity,
  divergence and base density. `PET_body`/`PET_generator` accept an optional
  explicit mask (legacy value-derived masks are unchanged when it is absent).
* Densities are per event and in the physical coordinates of the input when
  a `FeatureTransform` is supplied (bounded `atanh` + affine, with its
  log-Jacobian). The measure is the conditional density of the constituent
  features given the multiplicity and whatever selection the training data
  had; a multiplicity or yield factor is not included and must be added
  explicitly before calling a result a joint likelihood.

## Modules

| module | purpose |
| --- | --- |
| `device` | `resolve_device("auto"/"cpu"/"cuda"/"mps", allow_fallback)` returns a `DeviceInfo` record; float32 model tensors; `to_cpu_float64` (see MPS note) |
| `base` | `GaussianBase` sampling and per-event log density |
| `transforms` | `FeatureTransform` (bounded atanh + standardization, fitted on training data only) |
| `parameters` | `ConditionLayout`: named unknown parameters vs known context, fixed normalization; candidates fill unknown columns, observations never do |
| `likelihood` | `log_prob(velocity, x, mask, base, cond, options, event_ids)` -> per-event float64 log densities + diagnostics; `sample()`; `frozen_for_inference()` |
| `scan` | `Grid`, `joint_scan`, `ScanResult` (log L grid, posterior mass with trapezoid cell volumes, slices, profiles, marginals, intervals, boundary flags), `local_variable_loglik`, `refine_axes`, `support_scan` |
| `observations` | truth-free observation files (`x`, `mask`, `event_id`, optional `context`); the reader refuses truth/diagnostic fields |
| `velocity` | `SetFlowVelocity` (smooth, permutation-equivariant, global masked attention, FiLM conditioning) and `PETVelocity` (upstream PET2 with explicit mask) |
| `flow` | `train_flow` (single process, explicit device, EMA, validation-only selection), `FlowSurrogate` (model + base + transform + layout; `log_prob`, `log_prob_paired`, `sample`, `save`/`load` with sha256) |
| `fit` | `run_fit(config)`; CLI `omnilearned sbi-fit --config fit.toml` |
| `smoothness` | reconstruction, step refinement, finite-difference divergence, kNN switch counting, permutation and normalization checks |

### Likelihood options

`LikelihoodOptions(solver, nsteps, rtol, atol, divergence, num_probes, probe,
probe_seed, probe_schedule, max_nfe)`. Fixed-step solvers `euler`, `midpoint`, `rk4`;
adaptive `dopri5`, `bosh3`, `adaptive_heun` through torchdiffeq (adaptive
steps are shared across the rows of a batch). `divergence="exact"` loops over
real coordinates (reference tests, small clouds); `"hutchinson"` uses
`num_probes` Rademacher or Gaussian probes per trajectory. Probes are derived
from `(probe_seed, event_id)` only, per particle index, so they do not depend
on batching, chunking or padding, and the same event reuses the same probes
for every candidate (reducing noise in likelihood differences).
`probe_schedule="per_event"` (default) reuses one probe set at every solver
evaluation, so the probe error is correlated along the whole path;
`"per_step"` (fixed-step solvers only) draws independent probes for each
divergence evaluation from `(probe_seed, event_id, evaluation index)`. It has
the same expectation and the same cost, and the variance is averaged over the
path. The midpoint solver evaluates no divergence at its discarded first stage.
Integration
runs in float32 on the model device; per-event results are returned as
float64 on CPU and ensembles are accumulated in float64.

### Fit modes

* Fixed vector: `log L(theta) = sum_i log q(G_i | theta)`
  (`joint_scan`). `log_norm_fn(theta)` may supply `log Pr_q(S | theta)` of
  a surrogate whose support extends beyond the training selection; then
  `-N log Pr_q(S | theta)` is added (selected-distribution normalization).
* Shared + local: `log L(phi) = sum_i log sum_j w_j g(b_j | phi) q(G_i | b_j, phi)`
  with normalized quadrature weights (`normalized_log_weights`) and per-event
  log-sum-exp (`local_variable_loglik`), which also returns event-local
  posteriors of `b`.

Posterior grids are normalized with trapezoid cell volumes over the declared
domain; `marginal` integrates, `profile` maximizes, `slice` conditions. The
full joint grid (and per-event values) is saved. A refined grid
(`refine_axes`) uses only the scan result, never truth.

For posteriors much narrower than the domain (thin, curved degeneracies),
`support_scan` (fit config `[scan].support = {seed_stride, delta_logpost}`)
evaluates a seed sub-lattice of a fine `[grid]`, then repeatedly every
neighbour (3^d − 1) of an evaluated node with log posterior ≥ max − delta,
until none is left (`support_closed`). Unevaluated nodes have `loglik = nan`
and zero mass; the `evaluated` mask is saved. It reproduces the dense grid up
to the mass below max − delta, provided the seed lattice intersects every
relevant supported region.

### Fit configuration (`omnilearned sbi-fit`)

See the docstring of `omnilearned/sbi/fit.py`. The configuration may contain
only observations, checkpoint, candidate domain, prior and numerical options.
Keys containing `truth`, `holdout`, `theta_star` or `test_settings` are
rejected. File hashes are verified when given.

## Device support (Apple MPS)

`omnilearned train/evaluate --device {auto,cuda,mps,cpu}`. `auto` keeps the
upstream DDP behaviour for CUDA (and for CPU when no MPS exists); `mps` or
`cpu` run single process without a process group (`SingleDeviceModule`
exposes `.module`). Checkpoints load to CPU and are moved to the selected
device; autocast uses the device type. The `omnilearned.sbi` trainer and
evaluator always take an explicit device.

MPS notes (torch 2.14.1, verified by `tests/sbi/test_device_mps.py`):

* MPS has no float64. Convert with `t.cpu().double()` (`to_cpu_float64`).
  The fused `t.to("cpu", torch.float64)` of an MPS float32 tensor silently
  returns zeros in this torch build.
* Likelihoods, input derivatives, training steps, sampling and checkpoint
  round trips agree with CPU references to float32 roundoff.

## Smoothness

Upstream `PET_body` selects local neighbours with a hard `topk`; along a
density trajectory neighbour sets change, so the velocity field is only
piecewise smooth. `omnilearned.sbi.smoothness` provides diagnostics. The
smooth `SetFlowVelocity` is the density-compatible variant; the PET2
architecture is preserved and usable through `PETVelocity` for comparisons.
Freezing neighbour graphs for evaluation is not offered because it would
change the field used for generation.

## Legacy

`estimate_max_logp` (`evaluate --sbi`) remains for legacy outputs and emits a
`DeprecationWarning`: it scans coordinates independently around defaults,
sums the likelihood over the minibatch and fixes a JetNet base.

## Tests

`pytest tests/sbi` (no GPU or network required; MPS tests are skipped
without MPS): analytic zero/translation/linear/time-dependent fields with
nonunit bases, solver refinement orders, Hutchinson unbiasedness, padding,
batch and permutation invariance, model-state restoration, physical Jacobian,
correlated joint optimum where coordinate-wise scans fail, chunk invariance,
nonuniform grids and priors, local-variable integration against a closed
form, selection normalization term, measured-input fixture without truth,
`sbi-fit` CLI, MPS vs CPU, upstream CLI on single-process CPU/MPS, smoothness
diagnostics.
