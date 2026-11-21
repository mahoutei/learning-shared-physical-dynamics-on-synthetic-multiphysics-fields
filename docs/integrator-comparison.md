# Four-frame integrator comparison

Nine runs on one Colab Tesla T4 compare residual, whole-window Heun, and last-frame-only Heun at seeds 0–2. Each uses 2,000 optimizer updates and the same hashed synthetic corpus. The 24-step evaluation uses identical forecast targets for all runs.

Both Heun variants improved the first held-out forecast across all seeds. Each improved pooled held-out h24 on two seeds, while both worsened wave h24 and pooled training-system h24 on every seed. Residual remains the default for this benchmark.

The [paired results](data/integrator-comparison/paired_nmse.csv) and per-run artifacts include normalized squared error, model configuration, runtime, and target hashes. To verify the saved result:

```sh
python scripts/verify_integrator_sweep.py docs/data/integrator-comparison
```
