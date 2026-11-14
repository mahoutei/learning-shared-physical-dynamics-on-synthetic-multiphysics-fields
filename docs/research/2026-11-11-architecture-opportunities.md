# Beyond GPhyT reproduction

Architecture research brief · 11 November 2026

## Recommendation

Build a forecaster that infers a **small, structured evolution operator from observed transitions and adapts it only in directions the observations support**. Keep GPhyT's four-frame residual backbone as a control and as the source of a learned correction. The research question is whether the model can distinguish *what state it is in* from *what dynamics produced it*, and whether that distinction improves transfer to unseen coefficients, combinations of mechanisms, and sampling intervals.

The strongest experiment holds the query initial state fixed and changes only the support observations. Both forecasts have known counterfactual targets. A second experiment removes precisely the observed modes that identify a coefficient. The model should then show increased uncertainty or shrink toward its learned prior along that coefficient direction. These tests are stronger than a gain in mean NMSE on a familiar dataset. The proposed novelty is the combination of evidence-dependent operator adaptation, a constrained numerical update, and paired interventions that test whether the inferred update is actually used. This is a research hypothesis from a targeted literature search, with substantial nearby prior art.

## What our evidence supports

| Observation | Architectural implication | Limitation |
|---|---|---|
| Both Heun variants improved held-out h1 in all three seeds. Each improved pooled h24 in two seeds and regressed in a different seed. | Evaluate propagation over many steps alongside one-step accuracy. | NMSE alone cannot distinguish phase drift, amplitude drift, and errors in the inferred dynamics. |
| Both Heun variants worsened wave and pooled training-system h24 at every seed. Last-frame Heun reduced the wave penalty relative to whole-window Heun. | Preserve the observed history during intermediate updates and test wave-specific diagnostics. | This experiment does not identify the cause of the wave penalty. |
| Eight frames led aligned seed-0 held-out h24 at 0.8067 versus 0.8258 for two frames. Four frames led the pooled training systems. | Separate information about the dynamics from information about the current state. | One training seed and four test trajectories per system leave the context ranking uncertain. |

The original wave generator fixes wave speed at 0.2 and initializes its time derivative to zero. Its displacement and time derivative occupy slots that represent pressure and spatial velocity in other systems. Thermal convection uses another recording cadence and stores a buoyancy-driven velocity field. The corpus can expose rollout failures but cannot establish that the model inferred unseen coefficients. A model could exploit empty channels, initial-condition spectra, or cadence as shortcuts. A universal divergence penalty or kinetic-energy decay constraint would be inappropriate for these mixed field semantics. Use typed fields, an observation mask, and timestamps. For the new benchmark, apply a shared training-split normalization within each field schema and retain raw units for diagnostics. [Paired integrator scores](data/paired-integrator-nmse.csv) · [Synthetic generators](https://github.com/mahoutei/learning-shared-physical-dynamics-on-synthetic-multiphysics-fields/blob/59ea8bf3046d8a560ab9275bbc186e503ddb7199/gphyt/repro/systems.py) · [GPhyT paper](https://arxiv.org/abs/2509.13805)

There is a diagnostic reason to investigate wave amplitudes. Applied to an actual harmonic generator with eigenvalue $i\omega$, explicit Heun has amplification magnitude squared $1+(\omega\Delta t)^4/4$. Our Heun network learned slopes through endpoint losses, so the textbook amplification factor cannot explain the measured error without additional evidence. Replay the saved checkpoints and measure phase lag, energy drift, and mode amplitude on identical targets. A periodic wave reference can use the exact Fourier evolution and should handle the spatial mean separately.

The corpus writes frame indices to its time dimension, even though the physical snapshot intervals differ by system. Its evaluation also uses system-specific normalization statistics computed from each system's synthetic training partition, including the two held-out families. These are documented parts of the old protocol. For a new equation-blind benchmark, preserve physical timestamps, use the same train-only normalization across regimes with the same field schema, and balance cadence within each regime. Use periodic spatial stencils and physical grid spacing for any derivative features on the periodic pilot. Compare forecasts at matched elapsed times when sampling intervals differ.

## Prior art that narrows the claim

| Upgrade | Closest work | Implication |
|---|---|---|
| Infer dynamics from a few observed transitions | ICON and VICON infer operators from examples. PIANO encodes invariant dynamics from observed histories. DISCO infers the weights of a small evolution network with a hypernetwork. GEPS adapts low-rank context parameters. | Context inference, caching, and variable-rate prompts already have strong baselines. Compare against a matched model that sees the same support and timestamps. [ICON](https://arxiv.org/abs/2304.07993) · [VICON](https://arxiv.org/abs/2411.16063) · [PIANO](https://academic.oup.com/nsr/article/11/4/nwad336/7503933) · [DISCO](https://arxiv.org/html/2504.19496) · [GEPS](https://arxiv.org/abs/2410.23889) |
| Select or compose physical mechanisms | HotPOT uses family-specific neural–numerical experts. Origo separates global spectral evolution from local mechanisms. Neural Operator Splitting searches a library for a composition that fits test observations. CompNO and LegONet reuse structured blocks across equations. | A generic expert router or operator library would compete in an already crowded area. Test whether support quality determines which operator directions can be adapted, including cases with weak or conflicting evidence. [HotPOT](https://arxiv.org/abs/2610.38218) · [Origo](https://proceedings.mlr.press/v306/sun26aa.html) · [Neural Operator Splitting](https://arxiv.org/abs/2602.00884) · [CompNO](https://arxiv.org/abs/2605.11691) · [LegONet](https://arxiv.org/abs/2603.07882) |
| Stabilize a spectral update | SGNO constrains a learned generator. SSP evolves latent modes with linear propagation and nonlinear closure. Deep-OSG trains for time-composition consistency. Transolver-σ jointly models physical and spectral subspaces. | Exponential stepping, a spectral branch, and time consistency each have precedent. Check phase and damping separately, and isolate the structured component from any neural correction. [SGNO](https://arxiv.org/abs/2602.18801) · [SSP](https://arxiv.org/abs/2605.10154) · [Deep-OSG](https://arxiv.org/abs/2302.03358) · [Transolver-σ](https://arxiv.org/abs/2609.37279) |
| Infer uncertainty or audit an inferred operator | Bayesian operator inference represents operator uncertainty. Posterior-First learns a task-sufficient posterior from ambiguous observations. Spectral Audit differentiates a fixed-prompt forecast with respect to its query and measures Fourier gain, phase, and coupling. WSINDy fits candidate mechanisms from observations using weak forms. | Use spectral audit as an attributed diagnostic. Uncertainty by itself is also established. The contribution must improve adaptation under deliberately insufficient observations, at competitive cost, against both learned and classical inference. [Bayesian operator inference](https://arxiv.org/abs/2204.10829) · [Posterior-First](https://arxiv.org/html/2605.03247) · [Spectral Audit](https://arxiv.org/abs/2606.02427) · [WSINDy](https://arxiv.org/abs/2007.02848) |
| Adapt tokens, boundaries, and resolution | Physics Transformer projects fields onto adaptive local bases and decodes at query points. Walrus adapts tokenization and stabilizes rollouts. Unisolver conditions explicitly on PDE components. | Start with timestamp and grid metadata as controls. A stronger spatial contribution would test whether the same model maintains physically meaningful dynamics when observations arrive on different grids. [Physics Transformer](https://arxiv.org/abs/2607.24513) · [Walrus](https://arxiv.org/abs/2511.15684) · [Unisolver](https://arxiv.org/abs/2405.17527) |

This was a targeted search of primary papers through November 2026, including preprints. Reported gains use different splits, information at inference, and compute budgets. The closest papers define baselines and constrain the contribution claim.

## First architecture to investigate

1. **Separate support from query state.** Supply a small set of observed transitions with physical elapsed times. Support comes from a different initial condition under the same dynamics. The query provides its own observed state or history. Fit the dynamics once for an autonomous episode. Forecasts generated by the model are excluded from the evidence used to refit it.
2. **Constrain the support update.** Begin with a small library of interpretable generator terms, such as translation, diffusion, and wave propagation. Estimate coefficients from observed transitions with regularization toward a prior. Inspect singular values and residuals of the local, noise-scaled sensitivity matrix of the observation fit. Adapt directions whose sensitivity clears a threshold selected on validation. Shrink poorly observed directions toward the prior and evaluate calibration under noise and missing modes. A well-conditioned fit can still select a misspecified model, so measure held-out transition residuals as well.
3. **Apply the fitted operator and one learned correction.** On the periodic linear pilot, use closed-form Fourier evolution for supported modes. A scalar transport–diffusion mode with wavevector $k$ has multiplier $\exp[(-\kappa |k|^2-i a\cdot k)\Delta t]$. A wave mode with state $(\omega_k\hat u_k,\hat v_k)$ evolves by a rotation for the undamped system. Retain a one-call GPhyT residual for unresolved behavior. Test a projected correction on the linear pilot to prevent the neural branch from overriding selected modes. For nonlinear systems, resolved modes can be driven by cross-mode transfers, so an unconditional projection would remove legitimate dynamics. Add explicitly modeled forcing or coupling only after that diagnostic.
4. **Separate the claims.** With fixed coefficients, the exact linear component composes across time intervals. The full learned forecast has no such guarantee once a correction is added. Wave energy is meaningful in physical coordinates and the correct energy norm. Contractivity of a diffusion term makes no claim about the total energy of driven thermal convection. A learned prior and correction may improve performance beyond classical identification when the mechanism library is incomplete. That claim requires its own ablation.

## A pilot that can rule it out

**Stage 0 · Diagnose the existing results.** Replay the residual and Heun checkpoints on aligned wave and thermal targets. On wave, compare amplitude by mode, phase on sufficiently excited modes, and energy drift. Compare open-loop forecasts with a diagnostic that periodically restores genuine observations to distinguish local errors from feedback accumulation. On thermal, inspect temperature variance, divergence of spatial velocity, and h24 error. Plot per-trajectory results before attributing a failure to the update rule. [Spectral Audit](https://arxiv.org/abs/2606.02427) · [APEBench](https://arxiv.org/abs/2411.00180)

**Stage 1 · Build a controlled benchmark before scaling training.** Generate periodic 64×32 scalar advection, diffusion, and combined advection–diffusion from independently sampled transport and diffusivity. Use the same scalar channel, normalization, initial-field distribution, and cadence distribution across mechanisms. Use exact Fourier propagation for this pilot. Train on a subset of coefficient combinations and reserve new combinations and coefficient ranges. Draw paired queries with the same initial state under different coefficients, use an independent support trajectory for each coefficient, and swap only the support. Run this counterfactual assay with a single query state and the same validity mask for every arm. The ordinary four-frame benchmark remains a separate transfer check. Add paired cases with support varying only along one axis, which makes the orthogonal transport coefficient unobservable. Hold the query fixed while adding a support transition with energy along that axis. Also vary observation times with fixed physical evaluation endpoints. Treat the existing wave and thermal splits as development diagnostics and reserve fresh systems or coefficient ranges for final claims.

**Stage 2 · Compare a small set of models with equal evidence.** Run a query-only residual control, a support-conditioned residual control with the same backbone budget, the structured operator without a neural correction, and the full structured model. Compare the full model with and without its sensitivity-based shrinkage, holding all other components fixed. An exact coefficient oracle bounds recoverable gains. Classical regularized spectral fitting or WSINDy supplies a serious non-neural control. Correct support, shuffled support, rank-deficient support, and restored informative support diagnose whether the model uses evidence or exploits dataset cues. Fit thresholds on validation and report per-seed paired results. Use matched forecast times and report wall-clock and calls. A training-horizon change can serve as a strong control but should be held fixed across the architecture ablations.

**Decision criteria.** Continue only if the model improves held-out h24 over the same-information conditioned residual control across seeds and moves the same-state counterfactual forecasts in the correct direction. Report the error of the *difference* between the two counterfactual forecasts, as well as ordinary per-system NMSE. Test whether the added evidence reduces coefficient error or improves forecast calibration in the previously unobservable direction. Report phase and amplitude errors, worst-system behavior, and setup plus rollout cost. If classical fitting matches the full model on the analytic pilot, treat that as the expected ceiling for a correctly specified library. Test the learned correction next on one nonlinear held-out combination with known reference trajectories and a deliberately incomplete library. Stop pursuing it if it fails to improve on classical fitting and the equal-information residual control there. Use matched parameter counts or explicitly report the difference, and resample by trajectory or regime rather than treating frames as independent evidence.

## Two independent architecture bets

### Cross-resolution forecasting with causal memory

Generate a single high-resolution trajectory and derive lower-resolution observations from it using a fixed restriction $R$. Choose a lift $P$ with $RP=I$. A fine forecast can be represented as $\hat x_f = P x_c + (I-PR)r$, where $x_c$ is the shared coarse prediction and the fine detail has zero restriction. The identity $R\hat x_f=x_c$ follows directly. Nonlinear evolution can still transfer energy between resolved and unresolved scales. First measure the discrepancy between the restricted fine solution and a coarse simulation initialized from the restricted state. Use linear wave with complete low-mode state as a negative control and Burgers or Navier–Stokes as a nonlinear test. A negligible discrepancy leaves little closure for the model to learn.

Add a small recurrent closure state that remembers fine-scale evidence while it is available and continues after the input stream switches to a coarse grid. Score matched sequences with and without genuine fine observations before the switch. Compare coarse-only training, exact cross-grid reconstruction without memory, memory without the cross-grid constraint, and an oracle using continuing fine observations. This has a plausible contribution if the model preserves coarse forecast skill after a grid switch under genuinely missing fine-scale information. Hard coarse-average consistency and in-context multifidelity downscaling already exist. [Harder et al.](https://www.jmlr.org/beta/papers/v24/23-0158.html) · [MemNO](https://arxiv.org/abs/2409.02313) · [In-context multifidelity closure](https://arxiv.org/abs/2610.38179)

### Evidence-gated constraints on a residual forecast

Keep the residual model and infer which observable-specific constraints should apply from genuine observations. For example, a periodic scalar mean is conserved in unforced transport–diffusion but can change with a source. Hold initial fields and channel layouts fixed and test different source regimes. Compare the plain residual model, a fixed projection, an oracle that knows when to project, and an evidence-dependent projection. Measure false activation on sources and missed activation on conservative episodes in addition to drift and forecast accuracy. This is a smaller first build if structured operator fitting proves too restrictive. FINDE already learns and preserves invariants, and context-conditioned Flux NO already integrates inferred dynamics within a conservative discretization. The distinguishing experiment is detecting when an otherwise reasonable constraint should be withheld across regimes. [FINDE](https://arxiv.org/abs/2210.00272) · [Context-conditioned Flux NO](https://arxiv.org/abs/2605.05488)

The first build should fit and forecast the analytic counterfactual pairs with a tiny classical model and a support-conditioned neural control. This makes the data and evaluation auditable before investing in a large architecture. Once the test distinguishes a correct operator from a shortcut, add the structured head to GPhyT and test it on a nonlinear holdout. The research deliverable is a demonstrated ability to infer and reuse dynamics under changing evidence, alongside the forecast scores.

## Analytic pilot results

The first pilot uses 48 test pairs across seeds 0–2. Each pair shares a query initial field and differs in north–south velocity. The test split combines transport and diffusion, which the train-only prior saw separately, and uses higher east–west speeds and diffusivities. The fitter sees only support transitions and their elapsed times. Test endpoints are identical across methods and h24 is 0.96 time units.

| Support | Fit rank | Median h24 NMSE | Median h24 counterfactual difference error |
|---|---:|---:|---:|
| Train-only prior | 0 | 7.379 | 1.000 |
| East–west-only, clean | 2 | 1.661 | 1.000 |
| Add one informative transition, clean | 3 | < 1e-29 | < 1e-29 |
| Swap the informative support, clean | 3 | 2.454 | 4.000 |
| East–west-only, 3% noise, unguarded | 3 | 33.268 | 1.000 |
| East–west-only, 3% noise, noise floor | 2 | 1.664 | 1.000 |
| Add one informative transition, 3% noise, noise floor | 3 | 0.000319 | 0.000335 |

The pilot uses relative squared L2 error over each field. Its counterfactual metric applies the same normalization to the difference between the two forecasts and their two targets. An error of 1 corresponds to predicting no change in this assay. An error of 4 corresponds to predicting the exact change with the wrong sign. The noise floor uses the simulator's known noise standard deviation. It is a controlled baseline for selective adaptation, with calibration under unknown or spatially correlated noise still open.

Because the benchmark uses exact linear dynamics contained in the fitted model, the clean result verifies the protocol and sets a classical baseline. The next comparison should add a nonlinear transport term to held-out simulations while keeping it out of the spectral fitter. Give the structured model and a support-conditioned residual control the same observations, and compare both with the incomplete spectral fit before spending compute on larger models. [Pilot implementation](../../gphyt/counterfactual_transport.py) · [Per-pair results](counterfactual-transport-results.json)

```bash
python -m gphyt.counterfactual_transport --output docs/research/counterfactual-transport-results.json
```

## Nonlinear Burgers pilot

The nonlinear reference adds a conservative scalar Burgers flux in the east–west direction. It uses the same periodic 64×32 grid, a dealiased Fourier discretization, and Strang splitting with a maximum substep of 0.004. Each test pair varies either the north–south velocity or the nonlinear flux strength. An axis-only support trajectory hides the varied coefficient. A second support trajectory, with variation in both directions, makes its effect observable. The training and test splits use independent initial fields and parameter draws, with a shift toward stronger nonlinearities and crosswind on test. The ranges overlap, so this is a shifted held-out test, not a strict extrapolation test.

On 48 held-out pairs, giving the incomplete spectral model the true linear coefficients leaves h24 NMSE of 0.1525 on crosswind interventions and 0.0805 on nonlinear-strength interventions. Estimating those coefficients from restored nonlinear support raises error to 3.1820 and 0.1305, respectively. Even with rank three, the model can fit the wrong operator because nonlinear mode coupling violates its assumption that Fourier modes evolve independently. On crosswind pairs, restored support improves the fitted forecast's field error compared with axis-only support, while its error on the paired difference rises from 1.000 to 2.844. This is a case where more information helps one metric and hurts the other. [Spectral baseline results](counterfactual-burgers-results.json)

The learned comparison uses the same compact two-layer PhysicsTransformer in both arms. Each receives the current field, identical support transitions with physical elapsed times, and the same fitted linear coefficients. The residual control adds the prediction to the current field. The hybrid adds it to one fitted spectral step. Both subtract the spatial mean from the learned increment to preserve scalar mean. Each arm trained from scratch on CPU for 500 updates with 32 training pairs per seed. Evaluation used 16 test pairs per seed across seeds 0–2.

| Forecast with informative support | Crosswind h24 NMSE | Crosswind difference error | Nonlinear strength h24 NMSE | Nonlinear strength difference error |
|---|---:|---:|---:|---:|
| Support-conditioned residual | 5.1737 | 1.0676 | 4.6481 | 1.1309 |
| Spectral fit, all support | 3.1820 | 2.8436 | 0.1305 | 1.1440 |
| Spectral fit, informative transition only | 0.2447 | 0.2276 | 0.1423 | 1.1931 |
| Spectral fit with east–west averaging | 0.3371 | 0.2585 | 0.1212 | 1.0465 |
| Spectral hybrid | 0.7202 | 0.6894 | 0.1502 | 1.1099 |
| Spectral forecast, true linear coefficients | 0.1525 | 0.1213 | 0.0805 | 1.0000 |
| Spectral hybrid, true linear coefficients at evaluation | 0.1564 | 0.1200 | 0.0946 | 1.0000 |

The averaging diagnostic estimates north–south velocity and diffusion from modes constant in the east–west direction, where the omitted flux integrates to zero. Its east–west velocity still comes from the ordinary fit. This control uses knowledge of the direction of the omitted flux. The informative-only fit needs no such assumption and beat the hybrid on crosswind h24 on all three seeds.

Correct support helped the hybrid distinguish the two crosswind futures. Its h24 difference error was 0.6894 with correct support, 1.0000 with restricted support, and 3.4486 after the swap. On nonlinear strength, even supplying true linear coefficients at evaluation left the difference error near 1. The correction did not recover the effect of the hidden nonlinear coefficient in this test. The oracle evaluation also changes the coefficients the model was trained to receive, so it is a diagnostic for the current checkpoint.

On the original clean linear benchmark with informative support, the nonlinear-trained hybrid scored 0.0643 at h24. The classical fit had previously reached numerical precision. The learned correction therefore misses the linear-retention criterion too. Keep this hybrid as a baseline and defer larger training runs. The next candidate is an explicit fit of the nonlinear flux strength from observed transitions, followed by a learned correction only where the fitted model still has measurable error. An oracle nonlinear solve and the protected spectral fit give references for distinguishing parameter inference from forecast error. [Matched learned runs](data/burgers-learned-results.json) · [Linear retention scores](data/burgers-linear-retention.json)

```bash
python -m gphyt.counterfactual_burgers --output docs/research/counterfactual-burgers-results.json
python -m gphyt.counterfactual_burgers_models --output results/burgers-pilot
```

## Fit the known nonlinear flux

The next classical control fits east–west velocity, north–south velocity, diffusivity, and Burgers flux strength jointly. It uses only the observed transition endpoints and elapsed times, with the train-split mean coefficients as a starting prior. A finite-difference sensitivity matrix measures how each parameter changes the predicted endpoint. The least-squares update drops directions with a relative singular value below $10^{-6}$, and diffusivity stays nonnegative. The rank diagnoses local observability in this clean synthetic assay. It does not provide a calibrated uncertainty estimate under noise.

The fit and the synthetic data use the same known equation and numerical solver. This is a deliberately matched control for parameter inference. Success here bounds what a learned correction can add when the equation is known. It leaves open how a method performs when the equation library omits a mechanism or the support is noisy.

Across 48 held-out pairs and three seeds, adding one informative transition raised the fit to rank four in every pair. With restricted support, the hidden coefficient stayed at its prior and the predicted counterfactual difference was zero. Swapping the informative support reversed the forecast difference, producing a difference error of 4 in both interventions.

| Intervention | Best incomplete spectral h24 NMSE | Restricted Burgers fit | Informative Burgers fit |
|---|---:|---:|---:|
| Crosswind | 0.2447 | 1.3192 | $3.74\times10^{-26}$ |
| Nonlinear flux strength | 0.1212 | 0.7068 | $1.05\times10^{-23}$ |

The informative fit's median absolute strength error was below $4\times10^{-13}$ for both interventions. Fitting both arms of a pair took about 2.7–2.9 seconds on CPU. Near-zero forecast errors reflect the clean support and matched simulator. The result sets a reference for future tests with a missing mechanism, noisy observations, or a lower fitting budget. [Per-pair fit results](counterfactual-burgers-fit-results.json)

```bash
python -m gphyt.counterfactual_burgers_fit --output docs/research/counterfactual-burgers-fit-results.json
```

## Missing transverse flux test

The simulator adds a north–south derivative of the quadratic flux. The base fitter still uses the east–west-only Burgers equation. Each of 24 pairs across three seeds holds the query field and base coefficients fixed and changes only the sign of the transverse flux. The correction specifies the candidate flux in advance and estimates its signed coefficient from observed transition endpoints. It holds the fitted base coefficients fixed, skips an already accurate support fit or a negligible sensitivity, and accepts a correction only if it reduces support squared error by at least 5%. No neural network is trained in this test.

| Informative-support method | Median h24 NMSE | Median h24 difference error |
|---|---:|---:|
| Misspecified fit | 0.12235 | 1.27872 |
| Fit plus transverse correction | 0.01678 | 0.15258 |
| Fit plus correction with swapped support | 0.29306 | 3.86850 |
| True base coefficients, no transverse correction | 0.11554 | 1.00000 |
| True base coefficients plus fitted correction | 0.000082 | 0.000392 |

The correction improved median h24 error on each seed. With support invariant in the north–south direction, it stayed inactive on all pairs and left median h24 error at 0.39831. It also stayed inactive on all matched-equation cases. The support conditions have different sampled base coefficients and cadences due to how their support fields are generated. Compare the methods within each condition. The matched and restricted controls establish abstention for this clean simulator. With the true base coefficients, the correction is much more accurate, consistent with substantial bias in the base fit when the missing term is omitted.

This is evidence for choosing when to apply an explicit correction from support. The omitted operator was supplied to the correction in advance. The next useful comparison is a joint fit with this operator versus the staged fit, with noise and an additional mechanism absent from both. That gives a learned operator a stronger control. [Per-pair results](counterfactual-missing-flux-results.json)

```bash
python -m gphyt.counterfactual_missing_flux --output docs/research/counterfactual-missing-flux-results.json
```

## Joint fit under measurement noise and model mismatch

The staged fit first estimates four Burgers coefficients, freezes them, and then fits the transverse coefficient. The joint fit estimates all five together from the same observed endpoints. The benchmark crosses clean versus additive measurement noise (standard deviation 0.03) with presence versus absence of a cubic east–west flux. Both fitting methods omit the cubic flux, which the simulator applies on its inner half-band to avoid aliasing. The paired queries hold all parameters fixed except the sign of the transverse flux. The same noise realization is used in both arms of each pair, and query initial states and targets remain clean. The oracle uses the true five quadratic-model coefficients and also omits the cubic term. It measures how well those coefficients transfer when the forecast equation is incomplete. Fit time excludes the query rollout.

| Condition | Staged median h24 NMSE | Joint median h24 NMSE | Oracle quadratic median h24 NMSE | Joint pair wins |
|---|---:|---:|---:|---:|
| Clean, correct library | 0.01276 | Near numerical zero | 0 | 12/12 |
| Noisy, correct library | 0.01306 | 0.0000442 | 0 | 12/12 |
| Clean, cubic flux omitted | 0.11705 | 0.08189 | 0.01021 | 8/12 |
| Noisy, cubic flux omitted | 0.11747 | 0.08441 | 0.01021 | 8/12 |

The comparison uses twelve paired queries per condition across seeds 0–2. With noise and the omitted flux, joint fitting reduced h24 difference error from 0.30555 to 0.07966 and took a median 1.99 seconds per pair on CPU versus 1.71 seconds for staged fitting. Its support RMSE was lower than staged fitting, while its h24 error was over eight times the oracle quadratic reference. The omitted flux can therefore contaminate parameter estimates, even if joint fitting improves the support fit. This pilot makes joint fitting the stronger classical baseline and gives a concrete next comparison: train a correction and test whether it can improve on the joint fit and the oracle quadratic reference while preserving matched-library accuracy. [Per-pair results](counterfactual-joint-flux-results.json)

```bash
python -m gphyt.counterfactual_joint_flux --output docs/research/counterfactual-joint-flux-results.json
```

## Learned calibration of the missing flux

The quadratic joint fit provides five base coefficients. A local endpoint sensitivity calculation adds a sixth candidate direction for the cubic flux and estimates one update to all six coefficients. The learned variant calibrates that update using supervised ridge regression on separate simulated trajectories with known coefficients. At test time, both variants depend only on observation endpoints. The same gate controls both updates, using the largest apparent cubic strength observed on independent matched-equation validation examples. Below the threshold, the forecast uses the quadratic joint fit unchanged.

The four test conditions and query pairs match the joint-fit comparison. The calibrator sees the cubic mechanism during training and uses its known mathematical form at inference. This tests supervised calibration of a candidate mechanism under model mismatch. It does not test discovery of an unseen equation or a neural correction that operates directly on fields.

| Condition | Quadratic joint fit h24 NMSE | One local update | Learned calibration | Oracle quadratic |
|---|---:|---:|---:|---:|
| Clean, matched equation | Near zero | Near zero | Near zero | 0 |
| Noisy, matched equation | 0.0000442 | 0.0000442 | 0.0000442 | 0 |
| Clean, cubic flux present | 0.08189 | 0.0000166 | 0.000382 | 0.01021 |
| Noisy, cubic flux present | 0.08441 | 0.000127 | 0.000383 | 0.01021 |

The learned calibration beat the quadratic references under model mismatch, but the untrained local update gave lower median h24 error on every seed under both omitted-flux conditions. It also gave lower pooled counterfactual difference error. The shared gate activated on all 24 arms in each cubic condition, none of the clean matched arms, and one of 24 noisy matched arms. Matched-equation medians were preserved. The control to carry forward is the gated local update. A future learned model should compete with it using a genuinely incomplete operator library. [Per-pair calibration results](counterfactual-flux-calibration-results.json)

```bash
python -m gphyt.counterfactual_flux_calibration --output docs/research/counterfactual-flux-calibration-results.json
```

## Infer the missing flux function from support

This variant retains the joint quadratic fit and represents an additional east–west conservative flux as a weighted sum of nine Gaussian basis functions of the local scalar state. Its weights and the change to the base coefficients come from a regularized endpoint fit on the first two support transitions. The third observed endpoint must improve by at least 10% in squared error relative to a quadratic fit on the same two transitions. If the check fails, forecasting uses the quadratic fit to all available support. The fit runs separately for each episode and uses no coefficient labels or neural training.

The functional form is learned from support. The method assumes the correction is a local scalar flux in the east–west direction and uses the simulator's half-band convention for that correction. This leaves the direction, conservation form, and spectral band fixed while withholding the polynomial. The earlier cubic-specific update is a stronger-information comparison. The query field and its future are excluded from the coefficient fit and gate.

| Observations with cubic flux present | Quadratic joint fit h24 NMSE | Basis fit | Cubic-specific update | Pairs improved by basis fit |
|---|---:|---:|---:|---:|
| Clean | 0.08189 | 0.001125 | 0.0000166 | 12/12 |
| Noisy | 0.08441 | 0.001517 | 0.000127 | 10/12 |

The basis fit improved the median at every seed under both conditions. It activated in 24/24 clean cubic arms and 19/24 noisy cubic arms. It stayed inactive in every matched-equation arm, preserving the clean and noisy baseline scores exactly. In the noisy cubic condition, h24 counterfactual difference error fell from 0.07966 to 0.00307, and swapping the support raised it to 3.948. The median cost of fitting both arms and selecting corrections was 4.25 CPU seconds, excluding forecast time. Keep the cubic-specific result as a reference for known mechanisms and use the basis fit as the control for unknown functional forms. The next discriminating test is to vary the direction or spatial structure of the omitted flux to test which assumptions are carrying the result. [Per-pair basis-fit results](counterfactual-learned-flux-results.json)

```bash
python -m gphyt.counterfactual_learned_flux --output docs/research/counterfactual-learned-flux-results.json
```

## Withhold the direction of the missing flux

The simulator now applies the cubic flux along a hidden oblique direction. Opposite arms share a query state, support initial condition, observation cadence, and all five base coefficients. Only the sign of the omitted flux changes. The fixed-axis fit uses its existing nine basis functions in the east–west direction. The two-axis fit uses the same basis in both directions and jointly fits the coefficients from the same observed transitions. Both use the same quadratic baseline and support check. The two-axis fit has nine additional coefficients, so this comparison tests a larger candidate space as well as removal of the fixed direction assumption.

The direction diagnostic projects each fitted flux function onto the third Legendre polynomial over field values from −1.5 to 1.5. This removes constant, linear, and quadratic components that do not identify the missing cubic flux. The signed cosine compares the resulting vector with the simulator's hidden direction. An inactive correction contributes zero alignment. The simulator's direction enters this diagnostic only. Both fits still assume local conservative fluxes and share the inner half-band convention.

The restricted control supplies east–west-only observations with a purely north–south omitted flux. The matched noisy control has no omitted term, and varies a known quadratic coefficient between pair arms to keep their forecast difference defined.

| Support condition | Quadratic h24 NMSE | Fixed-axis h24 NMSE | Two-axis h24 NMSE | Median signed direction cosine |
|---|---:|---:|---:|---:|
| Oblique flux, clean | 0.100631 | 0.103727 | 0.000838 | 0.99989 |
| Oblique flux, noisy | 0.100625 | 0.100625 | 0.001448 | 0.99940 |
| Transverse flux hidden | 0.473281 | 0.473281 | 0.473281 | Inactive |
| Matched equation, noisy | 0.000043 | 0.000043 | 0.000043 | Inactive |

Each condition has 12 paired queries across three seeds. Two-axis fitting beat fixed-axis fitting on 12/12 clean and 11/12 noisy oblique pairs. It activated on 24/24 clean and 21/24 noisy oblique arms, and on none of the hidden or matched arms. For noisy oblique support, h24 difference error fell from 6.235 for the quadratic fit to 0.069 for the two-axis fit, then rose to 3.408 when the support was swapped. The test shows recovery of a spatially uniform direction. It leaves open how the fit handles a correction that varies across the domain. [Per-pair results](counterfactual-flux-direction-results.json)

```bash
python -m gphyt.counterfactual_flux_direction --output docs/research/counterfactual-flux-direction-results.json
```

## Spatially varying flux with partially observed endpoints

The simulator rotates the omitted flux direction through one period across the grid, with a random phase in each episode. The two arms share that phase and all base parameters, and have opposite omitted fluxes. The uniform model retains two flux components that depend on the scalar field. The spatial model multiplies each component's nine state basis functions by a constant and two first-harmonic spatial modes. This increases the correction from 18 to 54 weights and gives it the true spatial frequency and coordinate axis, while leaving the phase and dependence on the scalar field unknown.

Each support transition starts from a fully observed independent field. In the partial conditions the last field of each transition reveals only 75% of the rows. Both the parameter fit and the gate discard the hidden values, and the gate uses a separate transition. Query initial fields are fully observed. The hidden-row metric measures how well each forecast transfers into that strip. The noisy conditions add independent Gaussian noise with a standard deviation of 0.03 to observed endpoints, paired across arms. This protocol tests partial endpoints and an unseen query. It does not test forecasts from a partially observed initial state.

| Condition | Uniform h24 NMSE | Spatial h24 NMSE | Uniform h24 NMSE in strip | Spatial h24 NMSE in strip |
|---|---:|---:|---:|---:|
| Fully observed, clean | 0.04058 | 0.01620 | 0.05594 | 0.01262 |
| Partially observed, clean | 0.06504 | 0.02380 | 0.09863 | 0.01811 |
| Partially observed, noisy | 0.06560 | 0.06159 | 0.12642 | 0.06859 |
| Matched equation, partially observed, noisy | 0.000080 | 0.000080 | 0.000132 | 0.000132 |

The spatial correction won 7/9 pairs in each omitted-flux condition. With noisy partial observations it improved median full-grid error in two of three seeds. It reduced difference error in the hidden strip from 7.145 to 1.514, which remains above the score of 1 for predicting no difference. Swapping the support raised that score to 3.811. The gate activated on every omitted-flux arm and no matched-equation arms. The strip results support using spatial context, while the counterfactual errors and small noisy full-grid gain show that a good field forecast can still misestimate the effect of the changed flux. [Per-pair results](counterfactual-spatial-flux-results.json)

```bash
python -m gphyt.counterfactual_spatial_flux --output docs/research/counterfactual-spatial-flux-results.json
```

## Final milestone: equal-budget observation coverage

The frozen benchmark uses eight fresh paired queries for each of seeds 31, 32, and 33 on the 64×32 grid. Each support transition observes 48 of 64 endpoint rows, with independent Gaussian endpoint noise of standard deviation 0.03. Starting fields remain fully observed. All methods use identical fields, observation times, and noise within each pair. The first two endpoints fit the correction and the third gates it.

The fixed schedule hides rows 0–15 at every endpoint. The varied schedule hides rows 16–31 at the second fitting endpoint and uses the fixed mask at the first and third endpoints. Both spend 4,608 observed scalar values per arm. The evaluation strip stays at rows 0–15 for every method. The quadratic and uniform controls use fixed masks. The spatial model is evaluated under both schedules with the same capacity and regularization. Matched-equation controls use the same query fields and noise seeds.

The criteria were set before running the fresh pairs. To pass, a spatial schedule must have median h24 paired-difference NMSE below 1.0 on the full grid and in the strip, beat the uniform control on median strip forecast NMSE for all three seeds, and activate on at most 5% of matched-equation arms. Changing hidden endpoint values must leave the fit unchanged, and forecasts must preserve the mean. The benchmark has a single-run, two-hour budget.

| Method | Full-grid forecast NMSE | Strip forecast NMSE | Full-grid difference NMSE | Strip difference NMSE |
|---|---:|---:|---:|---:|
| Quadratic fit | 0.07828 | 0.10554 | 5.1334 | 5.9913 |
| Uniform correction | 0.05927 | 0.07433 | 2.5171 | 4.9071 |
| Spatial correction, fixed masks | 0.02332 | 0.02415 | 1.1992 | 1.1993 |
| Spatial correction, varied masks | 0.02522 | 0.02192 | 1.2093 | 0.8204 |

Neither spatial schedule passed all criteria. Varying coverage brought the strip difference error below the no-change reference of 1.0 while the full-grid difference error remained above it. Both spatial schedules beat the uniform correction in strip forecast error on all three seeds and had zero false activations among 48 matched-equation arms per schedule. Maximum scalar mean drift was below $5\times10^{-17}$ across both conditions. The mask test replaced hidden endpoint values with NaNs and confirmed identical coefficients for an active fit.

The single run took 21 minutes 18 seconds wall time and about 21 minutes 13 seconds of CPU time. The milestone closes with localized evidence for the value of varying observations, and a negative result for the full-grid counterfactual claim. [Per-pair results, mask schedules, and criteria](counterfactual-flux-coverage-results.json)

```bash
python -m gphyt.counterfactual_flux_coverage --output docs/research/counterfactual-flux-coverage-results.json
```
