# Reduced-scale reproduction notes

This is an independent reproduction study of **Towards a Physics Foundation
Model** (GPhyT, arXiv:2509.13805). It is not the official implementation. The
goal is to test the paper's core architectural claims under a budget that fits a
single T4 GPU on Google Colab (with a CPU fallback): a small synthetic
multi-physics corpus, a 7.9M-parameter GPhyT-S, and controlled ablations. All
seven reported training runs were executed on Colab T4 with the official Colab
CLI (two T4 sessions). A third T4 session handled the released-checkpoint
evaluation. This does **not** reproduce the paper's 1.8 TB corpus, multi-GPU
training, or full-scale benchmark numbers. As a separate transfer check, we also
evaluate the released GPhyT-S/M/L/XL checkpoints zero-shot on a 256x128 version
of the synthetic corpus; paper-scale benchmarks remain out of scope.

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
- Baseline: 4 input frames, 2-step autoregressive training loss, batch size 8.
- 2,000 optimizer steps, AdamW, lr 3e-4 with 100-step warmup, weight decay 0.01,
  gradient clipping 1.0.
- T4 GPU via the official Colab CLI, full precision. The CPU fallback uses 8
  threads and no mixed precision. The `training.amp` config key is not read by
  `gphyt/repro/run.py`; no reported run used AMP.
- Evaluation: single-step NMSE over all test windows, plus 24-step
  autoregressive rollout from each test trajectory. Metrics below are the
  **median per-sample NMSE** in normalized space.

Run one variant locally:

```bash
uv run --locked python -m gphyt.repro.run
```

Run one variant on Colab:

```bash
colab new -s gphyt --gpu T4
colab exec -s gphyt -f scripts/colab_train.py --timeout 7200 \
  --env GPHYT_BRANCH=repro \
  --env GPHYT_DEVICE=cuda \
  --env GPHYT_STEPS=2000 \
  --env GPHYT_TAG=gphyt_s_baseline_colab
colab download -s gphyt \
  /content/gphyt-repro/results/repro/gphyt_s_baseline_colab/metrics.json \
  ./metrics.json
colab stop -s gphyt
```

Run the full batch on one T4 session:

```bash
colab new -s gphyt-all --gpu T4
bash scripts/colab_all.sh gphyt-all
colab stop -s gphyt-all
```

The first run on a fresh VM takes about 8 minutes including environment setup
and corpus generation; later runs reuse the 5.7 GB `uv` environment and the
368 MB corpus and take about 3-5 minutes.

The command writes `metrics.json` and `model.pt` under
`results/repro/<run_name>/`.

## Results

24-step autoregressive rollout on Colab T4, median NMSE (lower is better):

| Experiment | Context | Params | Train loss | Train h1 | Train h8 | Train h24 | Held-out h1 | Held-out h8 | Held-out h24 | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPhyT-S baseline | 4 | 7,900,613 | 0.000939 | 0.000138 | 0.014386 | 0.128697 | 0.001320 | 0.300507 | 1.169014 | 301.4 s |
| Context length 1 | 1 | 7,896,005 | 0.002247 | 0.001939 | 0.120811 | 0.613885 | 0.002032 | 0.137947 | 0.974039 | 268.1 s |
| Context length 2 | 2 | 7,897,541 | 0.001061 | 0.000144 | 0.013432 | 0.164155 | 0.001008 | 0.253456 | 1.022413 | 275.3 s |
| Context length 8 | 8 | 7,906,757 | 0.000676 | 0.000145 | 0.016188 | 0.143084 | 0.001083 | 0.171519 | **0.811642** | 327.3 s |
| No derivative channels | 4 | 7,163,333 | 0.001963 | 0.000510 | 0.038556 | 0.594297 | 0.001194 | **0.113368** | 1.620801 | 290.8 s |
| UNet-S direct predictor | 4 | 7,720,101 | 0.001605 | 0.007558 | 0.919175 | 2.049422 | 1.087328 | 2.234564 | 1.662289 | 209.1 s |
| Direct prediction | 4 | 7,900,613 | 0.003450 | 0.003635 | 0.202984 | 25,792.6 | 0.544798 | 4.207137 | 119,636.1 | 288.1 s |

Per-system held-out medians (four test trajectories per system):

| Experiment | Wave h8 | Wave h24 | Thermal h8 | Thermal h24 |
|---|---:|---:|---:|---:|
| GPhyT-S baseline | 0.009971 | 0.080510 | 0.599894 | 2.326553 |
| Context length 1 | 0.156384 | 0.741876 | **0.133074** | 1.359496 |
| Context length 2 | 0.013679 | 0.175330 | 0.487037 | 1.956366 |
| Context length 8 | **0.007602** | **0.050953** | 0.372554 | 1.603582 |
| No derivative channels | 0.045338 | 2.334184 | 0.138929 | 1.561314 |
| UNet-S direct predictor | 1.095968 | 1.897055 | 3.250047 | **1.324035** |
| Direct prediction | 1.009101 | 32,984.237305 | 7.309248 | 222,188.148438 |

Best per column in bold. Direct prediction is omitted from the per-system figure
because its h24 NMSE exceeds 10^4. The data is in
`docs/data/rollout_nmse_t4_per_system.csv`;
`docs/figures/build_data.py` builds it from the result JSONs, and
`docs/figures/heldout_systems.py` draws the per-system figure.

The earlier CPU baseline reproduced the T4 baseline closely: train h24 0.1267
vs 0.1287, held-out h24 1.1466 vs 1.1690. The full CPU data is preserved in
`docs/data/rollout_nmse_cpu.csv`.

Key observations:

- **The residual/differentiator path is essential.** Removing the identity
  residual makes short-horizon predictions much worse and long rollouts diverge
  by orders of magnitude (train h24 25,792.6; held-out h24 119,636.1; per-system
  wave 32,984.2 and thermal 222,188.1).
- **Derivative channels are system-dependent at long horizons.** Removing them
  roughly doubles training loss and raises training h24 from 0.129 to 0.594. On
  held-out data it helps `thermal_convection` (2.327 -> 1.561) but destroys
  `wave` (0.081 -> 2.334), so the pooled held-out h24 rises from 1.169 to 1.621.
- **The transformer is more robust across systems, but UNet-S is best on thermal
  h24.** UNet-S is unstable early in the rollout on `wave` (1.897 at h24) and on
  training systems (2.049 at h24), but on `thermal_convection` it reaches
  1.324, narrowly ahead of Context 1 (1.359) and Context 8 (1.604).
- **Context length is a trade-off, and its ranking is system-specific.** Eight
  frames is best on `wave` (0.051 vs 0.081 for four frames) and on pooled
  held-out h24 (0.812 vs 1.169). On `thermal_convection` it improves the
  four-frame baseline (2.327 -> 1.604) but Context 1 (1.359) and UNet-S (1.324)
  are lower. One frame is best on thermal h8 (0.133); Context 8 is best on wave
  h8 (0.0076). No variant wins both systems.
- **Pooled held-out h8 also hides the per-system ranking.** No derivatives has
  the lowest pooled h8 (0.113) but wins neither system: Context 8 is best on
  `wave` (0.0076) and Context 1 is best on `thermal_convection` (0.133).

### Released-checkpoint transfer at 256x128

The released checkpoints were loaded strictly after stripping the
`module._orig_mod.` prefixes and evaluated at native 256x128 (512 tokens). The
evaluation corpus has 4 train / 1 validation / 4 test trajectories and 29
frames; z-score statistics come from its four training trajectories. Evaluation
uses batch 1, four input frames, and a 24-step rollout over all six systems.
This is zero-shot synthetic-domain transfer, not a held-out split of the
released models' training data.

| Model | Params | State tensors | h1 | h8 | h24 | Eval time | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPhyT-S | 7,992,773 | 165 | **0.001652** | 0.168915 | 1.000584 | 9.6 s | 0.12 GB |
| GPhyT-M | 111,597,317 | 165 | 0.004207 | 0.276754 | 0.874560 | 32.6 s | 0.57 GB |
| GPhyT-L | 384,885,765 | 321 | 0.001796 | 0.220459 | 0.846636 | 85.3 s | 1.67 GB |
| GPhyT-XL | 795,813,125 | 425 | 0.001880 | **0.156684** | **0.679890** | 156.5 s | 3.39 GB |

Per-system h24 medians:

| Model | Advection | Burgers | Incompressible NS | Sine-Gordon | Wave | Thermal |
|---|---:|---:|---:|---:|---:|---:|
| GPhyT-S | 1.609438 | **1.543170** | 0.093940 | 0.708727 | 0.888914 | **1.055033** |
| GPhyT-M | 1.705814 | 1.988563 | 0.047564 | 0.784912 | 0.826753 | 1.170085 |
| GPhyT-L | 1.490042 | 1.796777 | 0.058418 | 0.719075 | 0.779728 | 1.234888 |
| GPhyT-XL | **1.475180** | 1.826200 | **0.025760** | **0.545785** | **0.537446** | 1.346042 |

**Scaling helps long rollouts overall, but not uniformly.** Overall h24 falls
from S (1.001) to M (0.875) to L (0.847) to XL (0.680), and XL is also best at
h8; S remains best at h1. XL wins four of six systems at h24 (advection,
incompressible NS, sine-Gordon, and wave), while S wins burgers and thermal
convection. Thermal h24 degrades monotonically with scale
(1.055 -> 1.170 -> 1.235 -> 1.346). M has a thermal h8 spike (8.745) that largely
resolves by h24 (1.170). These are medians over four test trajectories per
system. Data: `docs/data/released_scaling_256.csv`.

## Limitations

- Synthetic 64x32 data and a 2,000-step budget are far below the paper's scale;
  absolute NMSE values are not comparable to the paper's benchmarks.
- Only one seed per configuration; differences of a few percent should not be
  over-interpreted. No confidence intervals are reported. Per-system medians
  use four test trajectories, so system-level rankings are especially noisy.
- The free-tier T4 session is a single point of failure: availability varies,
  and a session can be interrupted or reclaimed.
- The corpus systems are simplified and low-resolution. Long-horizon failure on
  `advection` and `thermal_convection` may reflect data difficulty rather than
  architecture alone.
- The released-checkpoint evaluation uses a separate 256x128, 29-frame, 4/1/4
  corpus with normalization statistics from its four training trajectories. It
  is zero-shot synthetic-domain transfer, not a paper-scale benchmark.
- `training.amp` is present in the config but unused by the runner; all reported
  numbers are full-precision.

## Repository layout

- `gphyt/repro/` — corpus generators, Well writer, config, training/eval entry.
- `scripts/colab_train.py` — one-variant runner for the official Colab CLI.
- `scripts/colab_all.sh` — sequential all-variant launcher for one T4 session.
- `scripts/eval_released.py` — strict released-checkpoint loader and eval entry.
- `scripts/eval_corpus.py`, `scripts/build_eval_corpus.py` — 256x128 corpus
  generator and CLI.
- Experiment branches `orx/*` — one branch per node in the experiment tree;
  each records the exact code that produced its run.
- `repro` (this branch) — publication branch with these notes.
- `docs/data/rollout_nmse_t4.csv` — T4 rollout measurements used by the figure.
- `docs/data/rollout_nmse_cpu.csv` — earlier CPU measurements.
- `docs/data/released_scaling_256.csv` — released S/M/L/XL 256x128 measurements.
