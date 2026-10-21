# NLE jet-quenching inference at scale: plan for the `dev/nle-perlmutter` branch

This file is the complete brief for the agent continuing this work on Perlmutter. Read it
fully before acting; it replaces the earlier hand-off documents. Branch: `dev/nle-perlmutter`
of `matplo/omnilearned-sbi`, created 2026-10-21 from `dev/agent-a` (commit `fe19020`).

## 1. Goal

Infer (b [fm], T_ref [GeV], τ_init [fm/c]) from sets of jets with a **neural likelihood
(NLE, "Route A")**:
- a conditional flow-matching generator q(cloud | θ) of the jet constituent cloud;
- its exact per-jet log density (ODE + divergence);
- summed over N jets and scanned jointly over θ.

The bar is **direct-simulator closure**: pseudo data drawn directly from the simulator at
reserved, off-grid settings give unbiased posteriors with correct coverage at N = 250 and
N = 1000 jets. The toy `toy-quenching-v1` comes first, JEWEL later.

What is new on this branch:
- the toy is part of the repository;
- training data are much larger, in a JEST-like HDF5 layout;
- **generator fidelity is established before any inference**;
- everything runs at Perlmutter scale (A100 GPUs, many CPU nodes).

We keep **our NLE approach**. The JEST project (github.com/sadekr/JEST) trains an OmniLearned
PET2 generator and then an amortized posterior (NPE). We borrow its data scale and
conditioning ideas, not its inference.

## 2. Ground rules (non-negotiable)

1. **Pseudo data:** only from the simulator at reserved settings, never from the generator.
2. **Training data:** no training θ equals a reserved setting. The main test setting lies
   strictly between design values on every axis (continuous designs: keep a margin, §5).
3. **Truth:** test truth is opened only by a report step after each fit's seal (hashes of
   outputs) verifies. Fit configurations contain observations, checkpoint, domain, prior and
   numerics only; `omnilearned.sbi.fit` rejects keys named truth/holdout/theta_star/test_settings.
4. **Choices on development or validation data only:** every model, numerical and scan
   choice is made there and recorded in config before any test truth is compared. Use
   the development setting (7.5, 0.30, 0.95) freely; its truth is open by design.
5. **Failed checks stay failed:** report them; never re-define a criterion after seeing a result.
6. **Numerics:** Δlog L between nearby candidates must converge (steps and probes) at the
   event count actually used, for each new model.
7. **Repository hygiene:**
   - do not commit data, checkpoints, run trees, figures or binaries;
   - never force-push;
   - add tests for every new generator, record format, driver and numerical method.
8. **JEWEL:** no production without a budget approved by the user.
9. **Asking:** ask the user only for decisions, budgets or external blockers. Keep a running
   log `FINDINGS_NLE.md` at the repository root: findings, numbers, commands, budgets, open
   items. Commit it with the code.

## 3. First day

- **Environment:** the user prepares an `henv` environment on Perlmutter with torch (CUDA),
  numpy, scipy, h5py, torchdiffeq and hepyy (FastJet 3.5 through cppyy). Do not create or
  modify environments.
- **Checkout and install:**
  - `git checkout dev/nle-perlmutter`, then `pip install -e .` only if the user's environment
    expects it (ask);
  - `python -c "import omnilearned.sbi.toy, omnilearned.sbi.likelihood"`.
- **FastJet:** cppyy's precompiled header must not be rebuilt in a shared environment. Set
  `CLING_STANDARD_PCH=$TMPDIR/cling-$USER/allDict.cxx.pch` per node. Check with
  `python -c "from omnilearned.sbi.toy import reco; print(reco.fastjet_version())"`.
- **Tests:** `pytest tests/sbi`. The cross-host check is `test_toy.py::test_cross_host_reproducibility`:
  θ = (6.0, 0.40, 0.65), 300 jets, seed 424242, array sha256 `63a9cf13…d24d4`, 7760 attempts.
  **If it fails, stop and report**: a non-reproducing toy invalidates everything downstream.
- **Smoke chain** before scaling up:
  1. tiny data (e.g. 27 settings × 500 jets);
  2. a 1k-step train;
  3. one sample-and-select;
  4. one 2-candidate fit.

## 4. What exists in this branch

**Framework** (`src/omnilearned/sbi/`, documented in `docs/SBI.md`; tests in `tests/sbi`):
- `flow.py`:
  - `FlowSurrogate` (sample, `log_prob`, `log_prob_paired`, load/save);
  - `train_flow`: flow matching with AdamW, cosine schedule, EMA, and selection by
    validation loss of the EMA weights.
- `velocity.py`: `SetFlowVelocity`, the smooth permutation-equivariant set transformer (FiLM
  time and condition, optional condition MLP into every token). It passed the smoothness
  checks: reversibility, finite-difference divergence, permutation invariance (1.5e-8, 3e-14).
  Upstream PET2 with hard kNN failed them (divergence |Δ| up to 32, permutation gap 1.33);
  see `smoothness.py`.
- `likelihood.py`, `LikelihoodOptions`:
  - solvers euler/midpoint/rk4 with fixed steps;
  - divergence exact or Hutchinson (Rademacher/Gaussian probes keyed by stable event id);
  - `probe_schedule = "per_event"` or `"per_step"` (independent probes per evaluation;
    production choice);
  - float64 accumulation.
- `scan.py`:
  - `Grid`, `joint_scan` (dense), `refine_axes`;
  - `support_scan`: evaluates only the supported region of a fine grid by flood-fill from a
    seed sub-lattice;
  - `ScanResult` (trapezoid cell volumes, marginals, profiles, slices, intervals, boundary
    flags, `evaluated` mask);
  - `local_variable_loglik` for per-event b integration.
- `fit.py`: configuration-driven fit (`omnilearned sbi-fit`, `[scan].support` or `refine`),
  with truth-key rejection.
- `observations.py`: observation file format (features, mask, event ids, sha256).
- `base.py`, `transforms.py`, `parameters.py`: base distributions, the atanh feature
  transform with log-Jacobian, condition layout.
- `fidelity.py`: the statistics-aware fidelity gate (§7, M3).

**Toy** (`src/omnilearned/sbi/toy/`, config `configs/toy-quenching-v1.toml`; ported unchanged
from the omni-sbi-mc application):
- `simulator.sample_toy(theta, count, rng, constants, selection, stream, index)` returns
  records plus a manifest (attempted / accepted / acceptance / rejections);
- `reco` (FastJet anti-kt R = 0.4 through cppyy);
- `selection.select_clouds` (the same selection applied to generated clouds);
- `selection.acceptance_table` (Pr_q(S | θ) of a surrogate);
- `observables.jet_observables` (jet pT, η, width, soft-8 fraction, outer fraction, pTD,
  leading fraction);
- `design` (event-id streams, `rng_for(seed, *keys)`, reserved settings, `check_reservation`);
- `records` (HDF5 writers);
- `config.sample_setting` and `config.array_sha256`.

## 5. The toy and the inference problem

**Toy physics** (`toy-quenching-v1`; constants in `[toy]`):
- **Medium:** overlap geometry with R_geom = 6.5 fm. Production density ∝ F+F−, uniform
  direction, T(ℓ) = T_ref g_b^(1/3) (τ/ℓ)^(1/3) for ℓ ≥ τ_init, interactions for T > 0.17 GeV.
- **Path moments:** I_m by a 64-step midpoint rule on [τ_init, L_exit].
- **Hard spectrum:** P0 ~ P^−5 on [40, 500] GeV, lognormal loss exp(−0.12 I_3 Z).
- **Constituents:** 24 core + 8 soft, with Dirichlet momentum weights; soft fraction f_soft(I_1).
- **Widths:** σ_vac(P0), σ_core(I_2), σ_soft = 2.2 σ_core.
- **Jet position:** jet centre η0 ∈ [−0.5, 0.5], φ0 ∈ [−0.6, 0.6], random order of constituents.

**Selection** (`[selection]`):
- all 32 particles have |η| < 2.5 and |φ| < 1.2;
- exactly one anti-kt R = 0.4 jet, containing all 32;
- 60 ≤ pT_jet ≤ 200 GeV and |η_jet| < 1;
- acceptance 1–9% per setting.

The steep spectrum makes the selected jet-pT shape nearly θ-invariant. The information is in
broadening and the soft fraction.

**Box and reserved settings:**
- box: b ∈ [0, 11], T_ref ∈ [0.25, 0.55], τ_init ∈ [0.2, 1.2];
- reserved (`[holdout]`): primary test (4.5, 0.40, 0.65); additional tests (1.5, 0.50, 0.35)
  and (10.0, 0.30, 1.0); development (7.5, 0.30, 0.95).

**Per-jet likelihood:**
- The selected density is q_sel(x | θ) = q(x | θ) 1_S(x) / Pr_q(S | θ). The generator is
  trained on selected jets, but its support leaks.
- −N log Pr_q(S | θ) comes from an acceptance table. It is small: Pr ≈ 0.88–0.92, spanning
  about 9 units of log L at N = 250.
- Fixed-vector mode: log L(θ) = Σ_i log q_sel(x_i | θ).
- Later, the shared-medium mode: b is event-local, integrated per event against the selected
  geometry law g_S(b | φ). Never multiply likelihoods before integrating b.

## 6. Lessons already learned (GPU node, TITAN RTX, 2026-10-21; development data unless noted)

1. **The posterior is a thin curved tube across the box.** b, T_ref and τ_init trade off.
   - Conditional widths at N = 250: σ_b ≈ 0.5 fm, σ_T ≈ 0.012 GeV, σ_τ ≈ 0.2 fm/c.
   - Marginal widths: about 1.4–2.4 fm, 0.08 GeV, 0.23 fm/c.
   - b–T correlation: 0.86.
2. **Coarse grids are artifacts.** A 5³ + 2 × 6³ refinement grid gave marginals 0.71σ off a fine
   reference, because the refinement box never shrinks. `support_scan` on fine grids fixes it.
   Nested-grid comparison (subsampling the fine reference every other node) is free and reliable.
   - At N = 250 the cheapest grid within 0.1σ was 23 × 14 × 11, with Δlog post 10 and seed
     stride (6, 3, 4), and ≈ 1300–1650 evaluations.
   - The error is not monotone in per-axis spacing.
   - N = 1000 needs ≈ 4× the nodes.
3. **Numerics** (convergence vs rk4-64 with 16 per-step probes over 12 probe seeds; criteria:
   single-seed shift, seed-averaged shift and probe std, each < 0.5 at the N used):
   - midpoint-64 with 8 per-step Rademacher probes passes at N = 250 and 1000 (shift 0.04 /
     0.03, std 0.17 / 0.20);
   - midpoint-32 with 4 per-event probes fails (solver bias grows as √N);
   - the CUDA vs CPU device check passes (Δlog L 1.9e-4).
4. **Cost:**
   - one N = 250 candidate takes ≈ 2.1 s on a TITAN RTX, and an N = 250 fit about 30–55 min;
   - one fit saturates a GPU: two concurrent fits gave 0.89× total throughput. Parallelize
     across GPUs and nodes, one fit per GPU.
5. **Fidelity was the real problem.** The 1 M-parameter model (80 grid settings × 2000 jets,
   20k steps) gives smooth likelihoods but biased physics:
   - at the development setting the surrogate's jet width is low by 5σ and the soft fraction
     by 5.7σ (4000 jets);
   - jet-pT χ² is 17–42 per 8 degrees of freedom at grid settings, the 60 GeV edge is smeared,
     and the jet-pT response is 1.44×;
   - constituent η and φ are tilted by ±5%.

   The resulting **development closure fails**: 7 ensembles at N = 250 (12 × 14 × 6 support grid).
   - Mean pulls: T_ref **+0.82 ± 0.17**, τ_init **−1.14 ± 0.28**, b −0.04 ± 0.22.
   - Truth inside the joint 68% (and 95%) region in only 1 of 7 ensembles; joint levels
     0.56–1.00 (uniform if calibrated).
   - The fidelity response curves explain the sign. Near the development point the
     surrogate's broadening (width, soft-8, outer fraction, pTD) lags the toy's as T_ref rises,
     so fits overshoot T_ref.
   - The gate's equivalent τ_init shift (+0.34 ± 0.13 fm/c with 128-step samples) matches the
     τ_init undershoot.
   - Validation log-density and flow-matching loss did not reveal any of this.
   - A permutation-calibrated gate on 22 settings × 4000 jets flags 5 settings, with
     7 of 22 p-values below 0.01.
   - Record of all this: the omni-sbi-mc application's FINDINGS.md.
5b. **Sample with a converged sampler.** 32-step samples hid part of the model error (128-step
   samples fail the gate at the development point; 32-step ones do not). The likelihood
   integrates the model's density, so fidelity must use samples whose distribution no longer
   changes with more sampler steps: check 64 vs 128 vs 256 and use the converged setting.
6. **Selection normalization** matters little at this toy's acceptance, but keep it.
7. **Earlier scale-up test.** On the GPU node, 1430 settings × 1500 jets (2.1 M) with the same
   and with a wider model; outcome recorded in the omni-sbi-mc FINDINGS, "Surrogate fidelity".
   Use it as a baseline, not as a result to reproduce.

## 7. Work plan

Each milestone ends with tests, a `FINDINGS_NLE.md` entry (numbers, commands, wall time, GPU
hours) and a commit.

### M1: Data at scale (CPU)
- **Design:** continuous θ, e.g. scrambled Sobol in the box.
  - ≥ 1000 training points and ≥ 2000 selected jets per point;
  - plus ≥ 100 separate validation points at ≥ 1000 jets each;
  - plus a held-out set of development scans: 1D lines through (7.5, 0.30, 0.95) along each
    axis and points along the tube, as in lesson 1.
  - Exclude a margin around the reserved settings: no design point within 0.02 of the box size
    in every coordinate simultaneously. `check_reservation` covers exact equality; add the
    margin check.
- **Generation:** parallel over points with Slurm job arrays. Seeds via
  `rng_for(seed, STREAMS[split], point_index)`. One shard per job, then a merged index with
  sha256 per shard.
- **Layout** (HDF5, one file per shard, JEST-like):
  - `data` [n, 32, 3] float32: (η, φ, log pT/GeV), absolute coordinates as now, plus `mask` [n, 32];
  - `global` [n, k]: θ (3), jet pT, jet η, jet φ, N_const, with `attrs["global_names"]`;
  - `event_id` int64 (stream-tagged); `point_id`;
  - `p4` and `membership` optional;
  - hidden generator diagnostics in a separate file, never read by training or fitting.

  Keep 32 constituents (`toy-quenching-v1` unchanged). Write a reader that serves batches
  without loading everything into memory if the data exceed host RAM.
- **Acceptance:** shard hashes reproducible; the reproducibility test passes on several node
  types; per-point acceptance recorded.

### M2: Generators (GPU)
Train **both** backbones at equal data and compute budget; the fidelity gate (M3) decides.
1. **Smooth set transformer** (`SetFlowVelocity`), scaled: dim 256–384, depth 6–8, heads 8,
   with the condition MLP into tokens.
2. **PET2 backbone** (OmniLearned `network.PET2` body: interaction features, class tokens)
   with the **hard kNN replaced by a smooth local attention**. For example, a Gaussian
   distance bias exp(−ΔR²/2s²) on attention logits, or soft top-k. It must pass the
   `smoothness.py` checks (reversibility, finite-difference divergence, permutation
   invariance) before any likelihood use: NLE needs an everywhere-differentiable velocity
   for the divergence.

Common to both:
- DDP across GPUs;
- bf16 only if the likelihood check in M4 passes at float32 evaluation (train mixed, evaluate fp32);
- EMA, cosine schedule, checkpoint by validation flow-matching loss of the EMA weights;
- log GPU hours.

Conditioning: θ mapped to [−1, 1] over the box. Jet pT is **not** a condition in the baseline
(see §9 for the factorized variant).

### M3: Fidelity gate (CPU + light GPU sampling): must pass before any inference
- **Samples:**
  - at every validation point, ≥ 20k selected direct jets and ≥ 20k generated jets, with
    generated clouds selected by `select_clouds`;
  - generation with a converged sampler (lesson 5b): at least 128 steps, verified against 256
    at a few points.
- **Observables:**
  - jet level: jet pT, width, soft-8 fraction, outer fraction (ΔR > 0.15), pTD, leading fraction;
  - constituent level: η, φ, log pT, ΔR to the jet axis, log z.
- **Significance (calibrated null):** per point, a permutation test.
  1. Pool direct and generated jets, re-split at random (jets as units), R ≥ 200 times.
  2. Statistic: the largest KS distance over observables, each standardized by its null median.
  3. This gives a family-wise p-value per point that is exact under "same distribution".
  4. Significant if p < α / S (α = 0.05, S points).
- **Relevance:** for each jet observable, |Δmean| √N / σ_direct at the analysis N (250 and 1000).
  Relevant if the 2-standard-error lower bound of |Δmean| exceeds 0.5 of a standard error at N.
- **Verdict per point:** FAIL only if significant **and** relevant, so statistical jitter never
  fails a model.
- **Global check:** the per-point family-wise p-values should be uniform. Report a KS test
  against uniformity and the number below 0.01 against Binomial(S, 0.01). It is a warning, not
  a failure, unless points also fail.
- **Equivalent parameter shift**, at the development setting and along the tube:
  δθ = (GᵀC⁻¹G)⁻¹GᵀC⁻¹Δmean, with direct-toy response gradients G (fits to the scans) and
  per-jet covariance C. Requirement: |δθ| ≤ 0.2 posterior σ at N = 1000, within 2 standard errors.
- **End-to-end, which decides:** development closure (M5).

The significance and relevance parts are implemented in `src/omnilearned/sbi/fidelity.py`
(ported from the omni-sbi-mc application, with `tests/sbi/test_fidelity.py`, including uniform
p-values under the null). Extend it with the δθ step at scale; the application's
`a/scripts/fidelity_report.py` and `fidelity_html.py` show one way to drive it and build the
page.

Produce a self-contained HTML report:
- verdict per point;
- particle- and jet-level histograms with ratio panels;
- response curves along the development scans;
- tables.

### M4: Numerics for the chosen generator (development data)
- Convergence at N = 250 and 1000: candidates around the development fit maximum along the
  tube. Reference rk4-64 with 16 per-step probes, 12 probe seeds. Criteria as in lesson 3.
- Exact-divergence subset: 24 jets.
- Device check: CUDA vs CPU log p.
- Record `[likelihood]` settings in config before M5.

### M5: Development closure (decides the fidelity question)
- ≥ 10 independent development ensembles each at N = 250 and N = 1000.
- Support scans: choose the grid with the nested subsample check against a finer reference,
  as in lesson 2.
- Per ensemble: pulls, marginal 68/95% coverage, and the **joint level of truth** (posterior
  mass at higher density than at the truth; evaluate log L at the truth as one extra candidate).
- **Pass:** mean pulls consistent with 0 within 2 standard errors, or |mean pull| < 0.3; and
  joint levels consistent with uniform (KS p > 0.01).
- If M5 fails while M3 passes, the gate missed something relevant: add it to M3 and iterate
  M2/M3. Never tune on test data.

### M6: Test closure
- Reserved settings: primary test 20 ensembles at N = 250 and ≥ 5 at N = 1000; additional
  tests 5 each.
- Fits are sealed, then the report opens truth: bias, Clopper–Pearson 68/95% coverage,
  predictive checks (direct toy at posterior draws vs observed).
- Parallelize: one fit per GPU, many nodes.

### M7: Shared medium with event-local b
`local_variable_loglik`: the φ = (T_ref, τ_init) grid needs the same resolution study; g_S from
an independent direct acceptance table.

### M8: JEWEL
Only with an approved budget. JEWEL 2.6.0-custom adapter from the omni-sbi-mc application.
- Record requested vs reported parameters, seeds and attempted vs selected counts.
- Peripheral limit from its actual geometry.

## 8. Compute notes (Perlmutter)

- **Toy:** 1000 selected jets in about 1.4 s on one CPU core, so 2 M jets is about 45
  core-minutes. FastJet is called in a compiled cppyy loop.
- **Surrogate sampling:** 5200 jets with 32 steps take about 45 s on 6 CPU threads; use GPUs
  for the fidelity samples.
- **Fits:** a candidate's cost scales with N × (steps × NFE) × probes. Budget support scans
  at roughly 1–2 k candidates per fit at N = 250, more at 1000.
- **Request budgets from the user** for:
  - M1 core-hours;
  - M2 GPU-hours per backbone;
  - M5–M6 GPU-hours.

## 9. To-do and forks (not in the baseline)

1. **Factorized jet kinematics**, as in JEST: q(x | θ) = p(pT_J, η_J, φ_J | θ) × q(rel | pT_J, θ).
   - The first factor is an explicit 1D/3D density, e.g. a mixture density on
     log(pT_J − 60) with uniform η_J, φ_J.
   - The second factor is a flow over constituent coordinates relative to the jet.
   - The change of variables must be full rank for an exact density: the jet axis is a
     function of the constituents. Drop one constituent's relative (η, φ) (e.g. the leading
     one) or use another bijection, and include the Jacobian.
   - Fork from this branch once the baseline gate results are known.
2. **JEST NPE** as an independent comparison on the same data.
3. **A variable-multiplicity toy v2** (new backend id), closer to JEWEL.
4. A **classifier two-sample test** per point as an additional fidelity diagnostic (never a
   replacement for the calibrated gate).

## 10. Questions to settle with the user at start

- the Perlmutter account and queue;
- the per-milestone GPU and CPU budgets;
- whether `pip install -e .` into the prepared environment is expected;
- where large data live (`$PSCRATCH` path) and the retention policy.
