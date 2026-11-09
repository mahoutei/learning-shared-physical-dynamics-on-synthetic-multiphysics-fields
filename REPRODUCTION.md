# Reduced-scale reproduction notes

This is an independent reproduction study of **Towards a Physics Foundation
Model** (GPhyT, arXiv:2509.13805). It is not the official implementation. The
goal is to test the paper's core architectural claims under a budget that fits a
single CPU machine: a small synthetic multi-physics corpus, a 7.9M-parameter
GPhyT-S, and controlled ablations. It does **not** reproduce the paper's 1.8 TB
corpus, multi-GPU training, or full-scale benchmark numbers.

Original paper: <https://arxiv.org/abs/2509.13805>  
Original code: <https://github.com/FloWsnr/General-Physics-Transformer>  
Released weights: <https://huggingface.co/flwi/Physics-Foundation-Model>

## Corpus

`gphyt/repro/systems.py` generates six 2D systems with a spectral solver:

| Split | System | Dynamics |
|---|---|---|
| train | `advection` | Steady cellular flow, passive scalar advection-diffusion |
| train | `burgers` | 2D viscous Burgers equation |
| train | `incompressible_ns` | Vorticity-streamfunction Navier-Stokes with Kolmogorov forcing |
| train | `sine_gordon` | Nonlinear sine-Gordon wave equation |
| held-out | `wave` | Linear wave equation (exact Fourier propagator) |
| held-out | `thermal_convection` | Boussinesq thermal convection |

Each system has 24 train / 4 validation / 4 test trajectories, 49 frames per
trajectory, on a 64x32 grid. Every frame carries five channels in The Well
layout: pressure, density, temperature, and two velocity components. Fields that
a system does not use are stored as zeros. `gphyt/repro/corpus.py` writes the
HDF5 files and computes z-score statistics.

The corpus is deterministic and generated on demand:

```bash
uv run --locked python -m gphyt.repro.corpus --root ~/.cache/gphyt_repro/corpus
```

## Training recipe

- GPhyT-S: 7,900,613 parameters, absolute positional encoding, full attention,
  linear tokenizer/detokenizer, finite-difference derivative channels, Euler
  residual update.
- 4 input frames, 2-step autoregressive training loss, batch size 8.
- 2,000 optimizer steps, AdamW, lr 3e-4 with 100-step warmup, weight decay 0.01,
  gradient clipping 1.0.
- CPU (8 threads), mixed precision disabled.
- Evaluation: single-step NMSE over all test windows, plus 24-step
  autoregressive rollout from each test trajectory. Metrics below are the
  **median per-sample NMSE** in normalized space.

```bash
uv run --locked python -m gphyt.repro.run
```

The command writes `metrics.json` and `model.pt` under
`results/repro/<run_name>/`.

## Results

24-step autoregressive rollout, median NMSE (lower is better):

| Experiment | Train systems h1 | Train systems h8 | Train systems h24 | Held-out h1 | Held-out h8 | Held-out h24 | Train loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPhyT-S baseline (4 frames) | 0.0001 | 0.0144 | 0.1267 | 0.0013 | 0.2832 | 1.1466 | 0.00094 |
| Context length 1 | 0.0019 | 0.1209 | 0.6138 | 0.0021 | 0.1169 | 0.9680 | 0.00225 |
| No derivative channels | 0.0005 | 0.0390 | 0.6011 | 0.0013 | 0.1196 | 1.6380 | 0.00196 |
| UNet-S direct predictor | 0.0074 | 0.9360 | 2.0547 | 1.0716 | 2.2363 | 1.6636 | 0.00161 |
| Direct prediction (no residual) | 0.0036 | 0.2030 | 25,793.6 | 0.5448 | 4.2107 | 119,326.9 | 0.00345 |

Key observations:

- **The residual/differentiator path is essential.** Removing the identity
  residual makes short-horizon predictions much worse and long rollouts diverge
  by orders of magnitude.
- **Finite-difference derivative channels help.** Removing them roughly
  doubles training loss and degrades 24-step train-system NMSE by about 4.7x;
  held-out rollout also worsens.
- **The transformer beats a same-budget UNet-S.** UNet-S is unstable from the
  first rollout step and reaches much higher NMSE at every horizon.
- **One frame is surprisingly competitive zero-shot but weaker in-distribution.**
  Context length 1 improves held-out h24 (0.97 vs 1.15) yet degrades
  train-system h24 (0.61 vs 0.13). Context length 2 and 8 are configured as
  further nodes in the experiment tree but were not completed at the time of
  writing.

## Limitations

- Synthetic 64x32 data and a 2,000-step budget are far below the paper's scale;
  absolute NMSE values are not comparable to the paper's benchmarks.
- Only one seed per configuration; differences of a few percent should not be
  over-interpreted.
- The corpus systems are simplified and low-resolution. Long-horizon failure on
  `advection` and `thermal_convection` may reflect data difficulty rather than
  architecture alone.
- The released GPhyT-S checkpoint was verified to load strictly and match the
  expected positional-encoding shape, but full released-checkpoint evaluation is
  left for a GPU/Colab run.

## Repository layout

- `gphyt/repro/` — corpus generators, Well writer, config, training/eval entry.
- Experiment branches `orx/*` — one branch per node in the experiment tree;
  each records the exact code that produced its run.
- `repro` (this branch) — publication branch with these notes.
