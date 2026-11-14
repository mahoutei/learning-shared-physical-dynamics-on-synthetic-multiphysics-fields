"""Assess a support-fitted correction for a missing transverse Burgers flux."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast, score_pair, summarize
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_transport import FORECAST_TIMES, Transition, initial_field

PRIOR = (0.0, 0.0, 0.011, 0.25)
PERTURBATION = 1e-3
MIN_RESPONSE = 1e-6
MIN_REDUCTION = 0.05


@dataclass(frozen=True)
class Case:
    query: np.ndarray
    coefficients: tuple[float, float, float, float]
    support: tuple[Transition, ...]
    target: np.ndarray


@dataclass(frozen=True)
class Correction:
    strength: float
    support_rmse: float
    fractional_improvement: float


def make_case(seed, index, transverse_strength, *, x_only=False, cubic_strength=0.0,
              cubic_transverse_strength=0.0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 3]))
    query = initial_field(rng)
    support_initial = initial_field(rng, x_only=x_only)
    coefficients = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                    rng.uniform(0.008, 0.014), rng.uniform(0.15, 0.35))
    times = np.r_[0.0, np.cumsum(rng.uniform(0.025, 0.045, size=3))]
    support = nonlinear_forecast(
        support_initial, coefficients, times, transverse_strength=transverse_strength,
        cubic_strength=cubic_strength,
        cubic_transverse_strength=cubic_transverse_strength,
    )
    transitions = tuple(
        Transition(a, b, dt) for a, b, dt in zip(
            support[:-1], support[1:], np.diff(times), strict=True,
        )
    )
    target = nonlinear_forecast(
        query, coefficients, FORECAST_TIMES, transverse_strength=transverse_strength,
        cubic_strength=cubic_strength,
        cubic_transverse_strength=cubic_transverse_strength,
    )
    return Case(query, coefficients, transitions, target)


def support_correction(transitions, coefficients):
    """Fit one extra flux coefficient against observed endpoints after freezing the base."""
    if not transitions:
        raise ValueError("At least one support transition is required")

    def endpoints(strength):
        return np.stack([
            nonlinear_forecast(t.initial, coefficients, [t.elapsed], transverse_strength=strength)[0]
            for t in transitions
        ])

    observations = np.stack([t.final for t in transitions])
    errors = observations - endpoints(0.0)
    baseline_mse = float(np.mean(errors**2))
    if baseline_mse < 1e-12:
        return Correction(0.0, np.sqrt(baseline_mse), 0.0)
    sensitivity = (endpoints(PERTURBATION) - endpoints(-PERTURBATION)) / (2 * PERTURBATION)
    power = float(np.mean(sensitivity**2))
    if power < MIN_RESPONSE**2:
        return Correction(0.0, np.sqrt(baseline_mse), 0.0)
    strength = float(np.mean(sensitivity * errors) / power)
    corrected_mse = float(np.mean((observations - endpoints(strength))**2))
    improvement = 1.0 - corrected_mse / baseline_mse
    if not np.isfinite(improvement) or improvement < MIN_REDUCTION:
        return Correction(0.0, np.sqrt(baseline_mse), 0.0)
    return Correction(strength, np.sqrt(corrected_mse), improvement)


def evaluate_pair(seed, index, strength, *, x_only):
    cases = [make_case(seed, index, signed, x_only=x_only) for signed in (-strength, strength)]
    fits = [fit_burgers(case.support, PRIOR) for case in cases]
    corrections = [
        support_correction(case.support, fit.coefficients)
        for case, fit in zip(cases, fits, strict=True)
    ]
    oracle_corrections = [support_correction(case.support, case.coefficients) for case in cases]
    base_coefficients = [fit.coefficients for fit in fits]
    corrected_strengths = [c.strength for c in corrections]
    true_coefficients = [case.coefficients for case in cases]
    for name, coefficients, fluxes in (
        ("misspecified", base_coefficients, (0.0, 0.0)),
        ("corrected", base_coefficients, corrected_strengths),
        ("swapped", base_coefficients[::-1], corrected_strengths[::-1]),
        ("oracle_base", true_coefficients, (0.0, 0.0)),
        ("oracle_base_corrected", true_coefficients, [c.strength for c in oracle_corrections]),
    ):
        predicted = [nonlinear_forecast(case.query, c, FORECAST_TIMES, transverse_strength=flux)
                     for case, c, flux in zip(cases, coefficients, fluxes, strict=True)]
        yield name, {
            **score_pair(predicted, [case.target for case in cases]),
            "flux_mae": float(np.mean(np.abs(np.array(fluxes) - (-strength, strength)))),
            "mean_abs_flux": float(np.mean(np.abs(fluxes))),
        }


def run_benchmark(seeds, pairs):
    seeds = list(seeds)
    results = {"informative": {}, "restricted": {}, "matched": {}}
    for seed in seeds:
        started = perf_counter()
        for index in range(pairs):
            strength = float(np.random.default_rng(
                np.random.SeedSequence([seed, index, 4]),
            ).uniform(0.2, 0.4))
            for subset, x_only in (("informative", False), ("restricted", True)):
                for method, row in evaluate_pair(seed, index, strength, x_only=x_only):
                    results[subset].setdefault(method, []).append({"seed": seed, "pair": index, **row})
            case = make_case(seed, index, 0.0)
            fit = fit_burgers(case.support, PRIOR)
            correction = support_correction(case.support, fit.coefficients)
            for method, flux in (("misspecified", 0.0), ("corrected", correction.strength)):
                prediction = nonlinear_forecast(
                    case.query, fit.coefficients, FORECAST_TIMES, transverse_strength=flux,
                )
                target = case.target[-1]
                results["matched"].setdefault(method, []).append({
                    "seed": seed, "pair": index,
                    "nmse_h24": float(np.mean((prediction[-1] - target)**2) / np.mean(target**2)),
                    "mean_abs_flux": abs(flux),
                })
        print(f"Scored seed {seed} in {perf_counter() - started:.1f}s", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs,
                     "forecast_times": FORECAST_TIMES.tolist()},
        **{subset: {"summary": summarize(records), "records": records}
           for subset, records in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or min(args.seeds) < 0:
        parser.error("provide a positive pair count and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
