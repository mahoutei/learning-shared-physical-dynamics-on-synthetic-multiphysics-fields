"""Run the frozen equal-budget endpoint-coverage benchmark."""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from gphyt.counterfactual_burgers import score_pair, summarize
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_joint_flux import NOISE
from gphyt.counterfactual_learned_flux import fit_flux, forecast_flux
from gphyt.counterfactual_missing_flux import PRIOR
from gphyt.counterfactual_spatial_flux import make_pair, observation_mask
from gphyt.counterfactual_transport import FORECAST_TIMES, GRID, Transition

SEEDS = (31, 32, 33)
PAIRS_PER_SEED = 8
AMPLITUDE = 0.16
MAX_SECONDS = 7200


def coverage_masks(shape):
    fixed = observation_mask(shape)
    shifted = np.roll(fixed, shape[0] // 4, axis=0)
    return (fixed, fixed, fixed), (fixed, shifted, fixed)


def noisy_support(support, seed, index):
    rng = np.random.default_rng([seed, index, 12])
    return tuple(
        Transition(t.initial, t.final + rng.normal(scale=NOISE, size=t.final.shape), t.elapsed)
        for t in support
    )


def fit_options(support, fixed, varied):
    for name, masks in (("fixed", fixed), ("varied", varied)):
        started = perf_counter()
        base = fit_burgers(support, PRIOR, transverse_prior=0.0, endpoint_mask=masks)
        base_seconds = perf_counter() - started
        started = perf_counter()
        prefix = fit_burgers(support[:-1], PRIOR, transverse_prior=0.0, endpoint_mask=masks[:-1])
        prep_seconds = base_seconds + perf_counter() - started
        if name == "fixed":
            yield "quadratic", base.coefficients, False, base_seconds
            started = perf_counter()
            uniform = fit_flux(support, bidirectional=True, endpoint_mask=masks,
                               base_fit=base, prefix_fit=prefix)
            yield "uniform", uniform.coefficients, uniform.active, prep_seconds + perf_counter() - started
        started = perf_counter()
        spatial = fit_flux(support, bidirectional=True, spatial=True, endpoint_mask=masks,
                           base_fit=base, prefix_fit=prefix)
        yield f"spatial_{name}", spatial.coefficients, spatial.active, prep_seconds + perf_counter() - started


def evaluate_pair(seed, index, amplitude, fixed, varied):
    query, cases = make_pair(seed, index, amplitude)
    all_targets = [target for _, target in cases]
    strip = ~fixed[0]
    strip_targets = [target[:, strip] for target in all_targets]
    forecasts, details = {}, {}
    for support, _ in cases:
        support = noisy_support(support, seed, index)
        for name, coefficients, active, seconds in fit_options(support, fixed, varied):
            prediction = forecast_flux(query, coefficients, FORECAST_TIMES)
            drift = float(np.max(np.abs(prediction.mean(axis=(-2, -1)) - query.mean())))
            if not np.isfinite(prediction).all() or drift > 1e-12:
                raise ValueError(f"Nonfinite or nonconservative forecast for {name}/{seed}/{index}")
            forecasts.setdefault(name, []).append(prediction)
            details.setdefault(name, []).append((active, seconds, drift))
    for name, frames in forecasts.items():
        gates, times, drifts = zip(*details[name], strict=True)
        yield name, {
            **score_pair(frames, all_targets),
            **{f"strip_{key}": value for key, value in score_pair(
                [frame[:, strip] for frame in frames], strip_targets,
            ).items()},
            "negative_gate": gates[0],
            "positive_gate": gates[1],
            "fit_seconds_per_pair": sum(times),
            "max_mean_drift": max(drifts),
        }


def criteria(missing, matched):
    result = {}
    for method in ("spatial_fixed", "spatial_varied"):
        rows = missing[method]
        full = float(np.median([row["pair_nmse_h24"] for row in rows]))
        strip = float(np.median([row["strip_pair_nmse_h24"] for row in rows]))
        matched_rows = matched[method]
        false_activations = sum(row["negative_gate"] + row["positive_gate"] for row in matched_rows)
        rate = false_activations / (2 * len(matched_rows))
        seed_improvements = {
            seed: bool(
                np.median([r["strip_nmse_h24"] for r in rows if r["seed"] == seed])
                < np.median([r["strip_nmse_h24"] for r in missing["uniform"] if r["seed"] == seed])
            )
            for seed in SEEDS
        }
        checks = {
            "full_difference_below_one": full < 1.0,
            "strip_difference_below_one": strip < 1.0,
            "strip_beats_uniform_on_all_seeds": all(seed_improvements.values()),
            "false_activation_at_most_five_percent": rate <= 0.05,
        }
        result[method] = {"checks": checks, "passed": all(checks.values()),
                          "strip_beats_uniform_by_seed": seed_improvements,
                          "false_activations": false_activations,
                          "false_activation_rate": rate}
    return result


def run_benchmark():
    started, cpu_started = perf_counter(), process_time()
    fixed, varied = coverage_masks(GRID)
    conditions = {"missing_flux": AMPLITUDE, "matched_equation": 0.0}
    records = {condition: {} for condition in conditions}
    for seed in SEEDS:
        for index in range(PAIRS_PER_SEED):
            for condition, amplitude in conditions.items():
                for name, row in evaluate_pair(seed, index, amplitude, fixed, varied):
                    records[condition].setdefault(name, []).append(
                        {"seed": seed, "pair": index, **row},
                    )
                if max(perf_counter() - started, process_time() - cpu_started) >= MAX_SECONDS:
                    raise TimeoutError("Frozen benchmark exceeded the two-hour budget")
        print(f"Scored seed {seed} in {perf_counter() - started:.1f}s total", flush=True)
    return {
        "protocol": {
            "grid": GRID, "seeds": SEEDS, "pairs_per_seed": PAIRS_PER_SEED,
            "conditions": conditions, "noise_std": NOISE, "max_seconds": MAX_SECONDS,
            "forecast_times": FORECAST_TIMES.tolist(),
            "fixed_hidden_rows": [np.flatnonzero(~mask[:, 0]).tolist() for mask in fixed],
            "varied_hidden_rows": [np.flatnonzero(~mask[:, 0]).tolist() for mask in varied],
        },
        "elapsed_seconds": perf_counter() - started,
        "cpu_seconds": process_time() - cpu_started,
        "criteria": criteria(records["missing_flux"], records["matched_equation"]),
        **{name: {"summary": summarize(rows), "records": rows} for name, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Result file exists. Preserve the frozen benchmark result.")
    report = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["criteria"], indent=2))


if __name__ == "__main__":
    main()
