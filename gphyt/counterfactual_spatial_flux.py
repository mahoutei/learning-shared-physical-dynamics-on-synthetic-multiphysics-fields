"""Compare uniform and spatial flux corrections with masked endpoint observations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast, score_pair, summarize
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_joint_flux import NOISE
from gphyt.counterfactual_learned_flux import fit_flux, forecast_flux
from gphyt.counterfactual_missing_flux import PRIOR
from gphyt.counterfactual_transport import FORECAST_TIMES, Transition, initial_field


def observation_mask(shape):
    """Observe three quarters of the periodic rows in each endpoint."""
    height, width = shape
    rows = np.arange(height) >= height // 4
    return np.broadcast_to(rows[:, None], (height, width))


def make_pair(seed, index, amplitude):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 11]))
    query = initial_field(rng)
    support_fields = [initial_field(rng) for _ in range(3)]
    coefficients = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                    rng.uniform(0.008, 0.014), rng.uniform(0.15, 0.35))
    times = rng.uniform(0.025, 0.045, size=len(support_fields))
    theta = rng.uniform(-np.pi, np.pi)
    y = 2 * np.pi * np.arange(query.shape[0])[:, None] / query.shape[0]
    # Ground truth rotates with position, and its phase is hidden from the fitter.
    east, north = np.cos(y + theta), np.sin(y + theta)
    arms = []
    for sign in (-1, 1):
        def simulate(field, times):
            if not amplitude:
                return nonlinear_forecast(field, coefficients, times, transverse_strength=sign * 0.27)
            return nonlinear_forecast(
                field, coefficients, times, transverse_strength=0.27,
                extra_flux=lambda u: sign * amplitude * east * u**3 / 3,
                extra_transverse_flux=lambda u: sign * amplitude * north * u**3 / 3,
            )

        support = tuple(Transition(field, simulate(field, [dt])[0], dt)
                        for field, dt in zip(support_fields, times, strict=True))
        arms.append((support, simulate(query, FORECAST_TIMES)))
    return query, arms


def score_episode(query, arms, mask, *, noise, noise_seed):
    predictions = {name: [] for name in ("quadratic", "uniform", "spatial")}
    active = {name: [] for name in predictions}
    targets = [target for _, target in arms]
    missing_rows = ~observation_mask(query.shape)
    regional_targets = [target[:, missing_rows] for target in targets]
    for support, _ in arms:
        if noise:
            rng = np.random.default_rng(noise_seed)
            support = tuple(Transition(t.initial, t.final + rng.normal(scale=noise, size=t.final.shape),
                                       t.elapsed) for t in support)
        base = fit_burgers(support, PRIOR, transverse_prior=0.0, endpoint_mask=mask)
        prefix = None
        if base.support_rmse >= 1e-7:
            prefix = fit_burgers(support[:-1], PRIOR, transverse_prior=0.0, endpoint_mask=mask)
        uniform = fit_flux(support, bidirectional=True, endpoint_mask=mask,
                           base_fit=base, prefix_fit=prefix)
        spatial = fit_flux(support, bidirectional=True, spatial=True, endpoint_mask=mask,
                           base_fit=base, prefix_fit=prefix)
        for method, coefficients, enabled in (
            ("quadratic", uniform.base, False),
            ("uniform", uniform.coefficients, uniform.active),
            ("spatial", spatial.coefficients, spatial.active),
        ):
            predictions[method].append(forecast_flux(query, coefficients, FORECAST_TIMES))
            active[method].append(enabled)
    predictions["spatial_swapped"] = predictions["spatial"][::-1]
    active["spatial_swapped"] = active["spatial"]
    for method, frames in predictions.items():
        yield method, {
            **score_pair(frames, targets),
            **{f"hidden_{key}": value for key, value in score_pair(
                [frame[:, missing_rows] for frame in frames],
                regional_targets,
            ).items()},
            "gate_rate": float(np.mean(active[method])),
        }


CONDITIONS = {
    "full_clean": (0.16, False, 0.0),
    "partial_clean": (0.16, True, 0.0),
    "partial_noisy": (0.16, True, NOISE),
    "matched_partial_noisy": (0.0, True, NOISE),
}


def run_benchmark(seeds, pairs):
    records = {condition: {} for condition in CONDITIONS}
    for seed in seeds:
        started = perf_counter()
        for index in range(pairs):
            rotating = make_pair(seed, index, 0.16)
            matched = make_pair(seed, index, 0.0)
            for condition, (amplitude, partial, noise) in CONDITIONS.items():
                query, arms = rotating if amplitude else matched
                mask = observation_mask(query.shape) if partial else None
                for method, row in score_episode(query, arms, mask, noise=noise,
                                                 noise_seed=[seed, index, 12]):
                    records[condition].setdefault(method, []).append(
                        {"seed": seed, "pair": index, **row},
                    )
        print(f"Scored seed {seed} in {perf_counter() - started:.1f}s", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs, "conditions": CONDITIONS,
                     "forecast_times": FORECAST_TIMES.tolist(), "masked_row_fraction": 0.25},
        **{name: {"summary": summarize(rows), "records": rows} for name, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or min(args.seeds) < 0:
        parser.error("provide positive pairs and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
