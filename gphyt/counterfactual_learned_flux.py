"""Infer a conservative flux function from observed transitions."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast, score_pair, summarize
from gphyt.counterfactual_burgers_fit import (
    PARAMETER_SCALE,
    endpoint_error,
    endpoint_masks,
    fit_burgers,
)
from gphyt.counterfactual_joint_flux import CONDITIONS
from gphyt.counterfactual_missing_flux import PRIOR, make_case
from gphyt.counterfactual_transport import FORECAST_TIMES, add_noise

KNOTS = np.linspace(-3.0, 3.0, 9)
BANDWIDTH = 0.85
DIFFERENCE_STEP = 1e-4
RIDGE = 1e-5


@dataclass(frozen=True)
class FluxFit:
    base: np.ndarray
    coefficients: np.ndarray
    active: bool
    support_rmse: float


def basis_values(field):
    distances = (field[..., None] - KNOTS) / BANDWIDTH
    return np.exp(-0.5 * distances**2)


def spatial_modes(height):
    phase = 2 * np.pi * np.arange(height) / height
    return np.column_stack((np.ones(height), np.cos(phase), np.sin(phase)))


def flux_function(weights, *, modes=None):
    if not weights.any():
        return None
    if modes is not None:
        weights = modes @ weights.reshape(-1, len(KNOTS))
        weights = weights[:, None, :]

    def flux(field):
        return np.sum(basis_values(field) * weights, axis=-1)

    return flux


def flux_direction(coefficients):
    """Audit the fitted flux component orthogonal to constant, linear and quadratic terms."""
    points = np.linspace(-1, 1, 129)
    cubic = (5 * points**3 - 3 * points) / 2
    basis = basis_values(1.5 * points)
    projection = np.trapezoid(basis * cubic[:, None], points, axis=0)
    components = np.zeros(2)
    weights = coefficients[5:].reshape(-1, len(KNOTS))
    components[:len(weights)] = weights @ projection
    return components


def forecast_flux(initial, coefficients, times):
    weights = coefficients[5:]
    if len(weights) == 6 * len(KNOTS):
        modes = spatial_modes(initial.shape[0])
        east, north = np.split(weights, 2)
    else:
        modes = None
        east, north = weights[:len(KNOTS)], weights[len(KNOTS):]
    return nonlinear_forecast(
        initial, coefficients[:4], times, transverse_strength=coefficients[4],
        extra_flux=flux_function(east, modes=modes),
        extra_transverse_flux=flux_function(north, modes=modes),
    )


def fit_flux(support, *, bidirectional=False, spatial=False, endpoint_mask=None,
             base_fit=None, prefix_fit=None):
    """Fit the base and one regularized, support-conditioned correction."""
    if len(support) < 2:
        raise ValueError("At least two transitions are needed to validate a correction")
    if spatial and not bidirectional:
        raise ValueError("Spatial fitting requires both flux directions")
    masks = endpoint_masks(support, endpoint_mask)
    basis_count = len(KNOTS) * (2 if bidirectional else 1) * (3 if spatial else 1)
    scales = np.r_[PARAMETER_SCALE, 1.0, np.full(basis_count, 0.1)]
    if base_fit is None:
        base_fit = fit_burgers(support, PRIOR, transverse_prior=0.0, endpoint_mask=masks)
    base = np.r_[base_fit.coefficients, np.zeros(basis_count)]
    if base_fit.support_rmse < 1e-7:
        return FluxFit(base, base.copy(), False, base_fit.support_rmse)

    def endpoints(coefficients, transitions, selected_masks):
        errors = []
        for transition, mask in zip(transitions, selected_masks, strict=True):
            predicted = forecast_flux(transition.initial, coefficients, [transition.elapsed])[0]
            errors.append(endpoint_error(predicted, transition.final, mask))
        return np.concatenate(errors)

    # Reserve the final endpoint for selection. Refit the baseline on the same
    # prefix so neither candidate sees that endpoint during fitting.
    fit_support = support[:-1]
    if prefix_fit is None:
        prefix_fit = fit_burgers(
            fit_support, PRIOR, transverse_prior=0.0, endpoint_mask=masks[:-1],
        )
    prefix = np.r_[prefix_fit.coefficients, np.zeros(basis_count)]
    errors = endpoints(prefix, fit_support, masks[:-1])
    jacobian = np.column_stack([
        (endpoints(prefix + offset, fit_support, masks[:-1]) - errors) / DIFFERENCE_STEP
        for offset in np.diag(scales * DIFFERENCE_STEP)
    ])
    # Penalize the function coefficients without shrinking the known PDE terms.
    penalty = np.eye(len(base))[5:] * np.sqrt(RIDGE * errors.size)
    update = np.linalg.lstsq(
        np.vstack((jacobian, penalty)), np.r_[-errors, np.zeros(basis_count)], rcond=1e-6,
    )[0]
    candidate = prefix + update * scales
    candidate[2] = max(candidate[2], 0.0)
    held_out = support[-1:]
    old_error = np.mean(endpoints(prefix, held_out, masks[-1:])**2)
    new_error = np.mean(endpoints(candidate, held_out, masks[-1:])**2)
    if new_error < 0.9 * old_error:
        return FluxFit(base, candidate, True, float(np.sqrt(new_error)))
    return FluxFit(base, base.copy(), False, base_fit.support_rmse)


def evaluate_pair(seed, index, noise, cubic):
    strength = np.random.default_rng(np.random.SeedSequence([seed, index, 4])).uniform(0.2, 0.4)
    cases = [make_case(seed, index, t, cubic_strength=cubic) for t in (-strength, strength)]
    started = perf_counter()
    fits = []
    for case in cases:
        support = case.support
        if noise:
            support = add_noise(support, np.random.default_rng([seed, index, 9]), noise)
        fits.append(fit_flux(support))
    elapsed = perf_counter() - started
    for method, coefficients in (
        ("joint", [fit.base for fit in fits]),
        ("learned_flux", [fit.coefficients for fit in fits]),
        ("swapped", [fit.coefficients for fit in reversed(fits)]),
    ):
        predictions = [
            forecast_flux(case.query, coefficient, FORECAST_TIMES)
            for case, coefficient in zip(cases, coefficients, strict=True)
        ]
        yield method, {
            **score_pair(predictions, [case.target for case in cases]),
            "gate_rate": float(np.mean([fit.active for fit in fits])) if method != "joint" else 0.0,
            "fit_seconds_per_pair": elapsed,
        }


def run_benchmark(seeds, pairs):
    seeds = list(seeds)
    records = {condition: {} for condition in CONDITIONS}
    for seed in seeds:
        for index in range(pairs):
            for condition, (noise, cubic) in CONDITIONS.items():
                for method, row in evaluate_pair(seed, index, noise, cubic):
                    records[condition].setdefault(method, []).append(
                        {"seed": seed, "pair": index, **row},
                    )
        print(f"Scored seed {seed}", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs, "conditions": CONDITIONS,
                     "forecast_times": FORECAST_TIMES.tolist()},
        **{name: {"summary": summarize(rows), "records": rows} for name, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or min(args.seeds) < 0:
        parser.error("provide positive pairs and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
