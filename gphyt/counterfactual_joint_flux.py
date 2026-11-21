"""Compare staged and joint fits when both can miss a flux in the simulator."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast, score_pair, summarize
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_missing_flux import PRIOR, make_case, support_correction
from gphyt.counterfactual_transport import FORECAST_TIMES, add_noise

NOISE = 0.03
CONDITIONS = {
    "clean": (0.0, 0.0),
    "noisy": (NOISE, 0.0),
    "cubic_omitted": (0.0, 0.12),
    "noisy_cubic_omitted": (NOISE, 0.12),
}


def fit_pair(cases, seed, index, noise):
    fitted = {"staged": [], "joint": []}
    for case in cases:
        support = case.support
        if noise:
            # Use the same measurement perturbation on each arm of the pair.
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 9]))
            support = add_noise(support, rng, noise)
        started = perf_counter()
        base = fit_burgers(support, PRIOR)
        correction = support_correction(support, base.coefficients)
        fitted["staged"].append((
            np.append(base.coefficients, correction.strength),
            correction.support_rmse, perf_counter() - started,
        ))
        started = perf_counter()
        joint = fit_burgers(support, PRIOR, transverse_prior=0.0)
        fitted["joint"].append((joint.coefficients, joint.support_rmse, perf_counter() - started))
    return fitted


def evaluate_pair(seed, index, strength, noise, cubic_strength):
    cases = [
        make_case(seed, index, flux, cubic_strength=cubic_strength)
        for flux in (-strength, strength)
    ]
    for name, fits in fit_pair(cases, seed, index, noise).items():
        coefficients, residuals, timings = zip(*fits, strict=True)
        predictions = [
            nonlinear_forecast(case.query, c[:4], FORECAST_TIMES, transverse_strength=c[4])
            for case, c in zip(cases, coefficients, strict=True)
        ]
        yield name, {
            **score_pair(predictions, [case.target for case in cases]),
            "support_rmse": float(np.mean(residuals)),
            "flux_mae": float(np.mean(np.abs(np.asarray(coefficients)[:, 4] - (-strength, strength)))),
            "fit_seconds_per_pair": sum(timings),
        }
    # Use the real five coefficients but leave the cubic term outside the forecast.
    predictions = [
        nonlinear_forecast(case.query, case.coefficients, FORECAST_TIMES,
                           transverse_strength=flux)
        for case, flux in zip(cases, (-strength, strength), strict=True)
    ]
    yield "oracle_quadratic", score_pair(predictions, [case.target for case in cases])


def run_benchmark(seeds, pairs):
    seeds = list(seeds)
    records = {name: {} for name in CONDITIONS}
    for seed in seeds:
        for index in range(pairs):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 4]))
            strength = float(rng.uniform(0.2, 0.4))
            for condition, (noise, cubic_strength) in CONDITIONS.items():
                for name, row in evaluate_pair(seed, index, strength, noise, cubic_strength):
                    records[condition].setdefault(name, []).append(
                        {"seed": seed, "pair": index, **row},
                    )
        print(f"Scored seed {seed}", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs,
                     "forecast_times": FORECAST_TIMES.tolist(), "conditions": CONDITIONS},
        **{name: {"summary": summarize(rows), "records": rows} for name, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or min(args.seeds) < 0:
        parser.error("provide a positive pair count and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
