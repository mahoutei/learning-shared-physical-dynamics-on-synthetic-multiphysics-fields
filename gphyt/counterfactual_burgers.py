"""Same-state interventions on periodic scalar Burgers transport."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from gphyt.counterfactual_transport import (
    Episode,
    FORECAST_TIMES,
    GRID,
    HORIZONS,
    SPLITS,
    Transition,
    fit_transport,
    forecast,
    initial_field,
    relative_error,
    wavevectors,
)

MAX_STEP = 0.004


@dataclass(frozen=True)
class BurgersEpisode(Episode):
    intervention: str


def nonlinear_forecast(initial, coefficients, elapsed, *, max_step=MAX_STEP,
                       transverse_strength=0.0, cubic_strength=0.0, extra_flux=None,
                       cubic_transverse_strength=0.0, extra_transverse_flux=None):
    """Solve scalar Burgers transport with optional transverse nonlinear flux.

    Strang splitting applies the linear part exactly and integrates the
    dealiased flux derivatives of u²/2 with RK4. The per-axis coefficients
    are strength and transverse_strength, respectively. The optional cubic
    flux in either direction acts on the inner half-band to avoid cubic aliasing.
    """
    vx, vy, diffusivity, strength = coefficients
    times = np.asarray(elapsed, dtype=float)
    if max_step <= 0 or np.any(np.diff(np.r_[0.0, times]) < 0):
        raise ValueError("Expected a positive max_step and nondecreasing elapsed times")
    has_cubic = cubic_strength != 0 or cubic_transverse_strength != 0
    has_extra = extra_flux is not None or extra_transverse_flux is not None
    if strength == transverse_strength == 0 and not (has_cubic or has_extra):
        return forecast(initial, coefficients[:3], times)
    height, width = initial.shape
    kx, ky = wavevectors(initial.shape)
    radius_squared = kx**2 + ky**2
    keep = (abs(kx) < 2 * np.pi * width / 3) & (abs(ky) < 2 * np.pi * height / 3)
    generator = -diffusivity * radius_squared - 1j * (vx * kx + vy * ky)
    flux_derivative = -0.5j * (strength * kx + transverse_strength * ky) * keep
    cubic_band = (abs(kx) < 2 * np.pi * width / 4) & (abs(ky) < 2 * np.pi * height / 4)
    cubic_derivative = -1j * (cubic_strength * kx + cubic_transverse_strength * ky) * cubic_band / 3
    state = np.fft.fft2(initial) * keep

    def flux_rhs(spectrum):
        field = np.fft.ifft2(spectrum).real
        rhs = flux_derivative * np.fft.fft2(field**2)
        if has_cubic or has_extra:
            filtered = np.fft.ifft2(spectrum * cubic_band).real
            if has_cubic:
                rhs += cubic_derivative * np.fft.fft2(filtered**3)
            if extra_flux is not None:
                rhs -= 1j * kx * cubic_band * np.fft.fft2(extra_flux(filtered))
            if extra_transverse_flux is not None:
                rhs -= 1j * ky * cubic_band * np.fft.fft2(extra_transverse_flux(filtered))
        return rhs

    frames = []
    for duration in np.diff(np.r_[0.0, times]):
        steps = max(1, math.ceil(duration / max_step))
        step = duration / steps
        half = np.exp(generator * (step / 2))
        for _ in range(steps):
            state *= half
            k1 = flux_rhs(state)
            k2 = flux_rhs(state + step * k1 / 2)
            k3 = flux_rhs(state + step * k2 / 2)
            k4 = flux_rhs(state + step * k3)
            state = half * (state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        frames.append(np.fft.ifft2(state).real)
    return np.stack(frames)


def transitions(initial, coefficients, times):
    frames = nonlinear_forecast(initial, coefficients, times)
    return tuple(
        Transition(before, after, elapsed)
        for before, after, elapsed in zip(frames[:-1], frames[1:], np.diff(times), strict=True)
    )


def make_episodes(split, seed, pairs):
    """Vary one hidden coefficient while holding query, cadence, and nuisance factors fixed."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLITS.index(split), 2]))
    for index in range(pairs):
        query = initial_field(rng)
        informative = initial_field(rng)
        support_times = np.r_[0.0, np.cumsum(rng.uniform(0.035, 0.08, size=3))]
        extra_times = np.array([0.0, rng.uniform(0.035, 0.08)])
        vx = rng.uniform(-0.3, 0.3)
        vy = rng.uniform(0.16, 0.36) if split == "test" else rng.uniform(0.08, 0.24)
        diffusivity = rng.uniform(0.008, 0.014)
        strength = rng.uniform(0.25, 0.55) if split == "test" else rng.uniform(0.12, 0.38)
        if index % 2 == 0:
            intervention = "crosswind"
            restricted = initial_field(rng, x_only=True)
            coefficients = ((vx, -vy, diffusivity, strength), (vx, vy, diffusivity, strength))
        else:
            intervention = "nonlinearity"
            profile = initial_field(rng).mean(axis=1, keepdims=True)
            restricted = np.broadcast_to(profile / profile.std(), GRID)
            coefficients = ((vx, vy, diffusivity, 0.5 * strength), (vx, vy, diffusivity, strength))
        support = tuple(transitions(restricted, c, support_times) for c in coefficients)
        extra = tuple(transitions(informative, c, extra_times) for c in coefficients)
        targets = tuple(nonlinear_forecast(query, c, FORECAST_TIMES) for c in coefficients)
        yield BurgersEpisode(query, tuple(map(np.asarray, coefficients)), support, extra,
                             targets, intervention)


def score_pair(predictions, targets):
    scores = {}
    for h in HORIZONS:
        a, b = (frames[h - 1] for frames in predictions)
        truth_a, truth_b = (frames[h - 1] for frames in targets)
        scores[f"nmse_h{h}"] = (relative_error(a, truth_a) + relative_error(b, truth_b)) / 2
        scores[f"pair_nmse_h{h}"] = relative_error(a - b, truth_a - truth_b)
    return scores


def fit_protected_transport(transitions, prior):
    """Use x-averaged observations where the omitted x-flux cancels exactly."""
    averaged = tuple(
        Transition(transition.initial.mean(axis=1, keepdims=True),
                   transition.final.mean(axis=1, keepdims=True), transition.elapsed)
        for transition in transitions
    )
    # FFT roundoff can leave tiny nonconstant modes in a spatially constant mean.
    averaged = tuple(
        transition for transition in averaged
        if np.ptp(transition.initial) > 1e-10 * np.std(transitions[0].initial)
    )
    coefficients = fit_transport(transitions, prior).coefficients
    coefficients[1:] = fit_transport(averaged, prior).coefficients[1:]
    return coefficients


def score_spectral(episode, prior):
    restricted = [fit_transport(support, prior) for support in episode.support]
    restored = [fit_transport((*support, *extra), prior)
                for support, extra in zip(episode.support, episode.additional_support, strict=True)]
    informative = [fit_transport(extra, prior) for extra in episode.additional_support]
    methods = {
        "restricted": [fit.coefficients for fit in restricted],
        "restored": [fit.coefficients for fit in restored],
        "informative_only": [fit.coefficients for fit in informative],
        "swapped": [fit.coefficients for fit in reversed(restored)],
        "protected_restricted": [fit_protected_transport(support, prior)
                                 for support in episode.support],
        "protected_restored": [fit_protected_transport((*support, *extra), prior)
                               for support, extra in zip(episode.support, episode.additional_support,
                                                         strict=True)],
        "oracle_linear": [coefficients[:3] for coefficients in episode.coefficients],
    }
    ranks = {"restricted": min(fit.rank for fit in restricted),
             "restored": min(fit.rank for fit in restored),
             "informative_only": min(fit.rank for fit in informative)}
    for name, coefficients in methods.items():
        predictions = [forecast(episode.query, c, FORECAST_TIMES) for c in coefficients]
        scores = score_pair(predictions, episode.targets)
        if name in ranks:
            scores["rank"] = ranks[name]
        yield name, scores


def summarize(records):
    return {
        name: {f"median_{key}": float(np.median([row[key] for row in rows]))
               for key in rows[0] if key not in ("seed", "pair")}
        for name, rows in records.items()
    }


def run_benchmark(seeds, pairs):
    seeds = list(seeds)
    report = {"protocol": {"grid": GRID, "forecast_times": FORECAST_TIMES.tolist(),
                           "seeds": seeds, "pairs_per_split_seed": pairs, "max_step": MAX_STEP}}
    for split in ("validation", "test"):
        records = {}
        for seed in seeds:
            train_coefficients = [c[:3] for episode in make_episodes("train", seed, pairs)
                                  for c in episode.coefficients]
            prior = np.mean(train_coefficients, axis=0)
            for index, episode in enumerate(make_episodes(split, seed, pairs)):
                methods = records.setdefault(episode.intervention, {})
                for name, scores in score_spectral(episode, prior):
                    methods.setdefault(name, []).append({"seed": seed, "pair": index, **scores})
        report[split] = {
            intervention: {"summary": summarize(rows), "pairs": rows}
            for intervention, rows in records.items()
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 2 or min(args.seeds) < 0:
        parser.error("provide at least two pairs and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: result["summary"] for name, result in report["test"].items()},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
