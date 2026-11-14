"""Same-state, different-dynamics benchmark on a periodic unit square."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

GRID = (64, 32)
FORECAST_TIMES = 0.04 * np.arange(1, 25)
HORIZONS = (1, 8, 24)
SPLITS = ("train", "validation", "test")
SUPPORT_NOISE_STD = 0.03


@dataclass(frozen=True)
class Transition:
    initial: np.ndarray
    final: np.ndarray
    elapsed: float


@dataclass(frozen=True)
class Episode:
    query: np.ndarray
    coefficients: tuple[np.ndarray, np.ndarray]
    support: tuple[tuple[Transition, ...], tuple[Transition, ...]]
    additional_support: tuple[tuple[Transition, ...], tuple[Transition, ...]]
    targets: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Fit:
    coefficients: np.ndarray
    rank: int
    singular_values: np.ndarray


def wavevectors(shape):
    height, width = shape
    return np.meshgrid(
        2 * np.pi * np.fft.fftfreq(width, d=1 / width),
        2 * np.pi * np.fft.fftfreq(height, d=1 / height),
    )


def forecast(initial, coefficients, elapsed):
    """Evaluate the exact Fourier solution at physical times after initial."""
    vx, vy, diffusivity = coefficients
    kx, ky = wavevectors(initial.shape)
    generator = -diffusivity * (kx**2 + ky**2) - 1j * (vx * kx + vy * ky)
    if initial.shape[1] % 2 == 0:
        generator[:, initial.shape[1] // 2] += 1j * vx * kx[:, initial.shape[1] // 2]
    if initial.shape[0] % 2 == 0:
        generator[initial.shape[0] // 2] += 1j * vy * ky[initial.shape[0] // 2]
    propagator = np.exp(np.atleast_1d(elapsed)[:, None, None] * generator)
    return np.fft.ifft2(np.fft.fft2(initial) * propagator).real


def fit_transport(transitions, prior, *, noise_std=0.0):
    """Fit phase and decay using observed modes below the phase-wrap limit."""
    matrices, observations = [], []
    for transition in transitions:
        before = np.fft.fft2(transition.initial)
        after = np.fft.fft2(transition.final)
        kx, ky = wavevectors(transition.initial.shape)
        radius_squared = kx**2 + ky**2
        # NumPy's forward FFT is unnormalized. The RMS Fourier noise is
        # sqrt(number of samples) times the per-cell white-noise standard deviation.
        floor = max(1e-8 * abs(before).max(), 5 * noise_std * np.sqrt(before.size))
        excited = (radius_squared > 0) & (abs(before) > floor) & (abs(after) > floor)
        if not excited.any():
            continue
        ratio = after[excited] / before[excited]
        zeros = np.zeros(ratio.size)
        matrices.extend((
            -transition.elapsed * np.column_stack((kx[excited], ky[excited], zeros)),
            -transition.elapsed * np.column_stack((zeros, zeros, radius_squared[excited])),
        ))
        observations.extend((np.angle(ratio), np.log(abs(ratio))))
    prior = np.asarray(prior, dtype=float)
    if not matrices:
        return Fit(prior.copy(), 0, np.zeros(3))
    matrix = np.concatenate(matrices)
    observed = np.concatenate(observations)
    # Fit a minimum-norm change from the prior so null-space components stay fixed.
    correction, _, rank, singular_values = np.linalg.lstsq(
        matrix, observed - matrix @ prior, rcond=1e-6,
    )
    coefficients = prior + correction
    coefficients[2] = max(coefficients[2], 0.0)
    return Fit(coefficients, int(rank), singular_values)


def initial_field(rng, x_only=False):
    height, width = GRID
    shape = (1, width) if x_only else GRID
    kx, ky = wavevectors(shape)
    modes = (abs(kx) <= 8 * np.pi) & (abs(ky) <= 8 * np.pi)
    modes[0, 0] = False
    field = np.fft.ifft2(np.fft.fft2(rng.normal(size=shape)) * modes).real
    return np.broadcast_to(field / field.std(), (height, width))


def observed_transitions(initial, coefficients, times):
    frames = forecast(initial, coefficients, times)
    return tuple(
        Transition(before, after, elapsed)
        for before, after, elapsed in zip(frames[:-1], frames[1:], np.diff(times), strict=True)
    )


def add_noise(transitions, rng, noise_std):
    """Simulate noisy snapshots, sharing the measurement at adjoining endpoints."""
    frames = [transitions[0].initial, *(transition.final for transition in transitions)]
    noisy = [frame + rng.normal(scale=noise_std, size=frame.shape) for frame in frames]
    return tuple(
        Transition(before, after, transition.elapsed)
        for before, after, transition in zip(noisy[:-1], noisy[1:], transitions, strict=True)
    )


def make_episodes(split, seed, pairs):
    """Yield reproducible pairs with disjoint split RNGs and matched nuisance factors."""
    split_index = SPLITS.index(split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_index]))
    for index in range(pairs):
        query = initial_field(rng)
        axis_support = initial_field(rng, x_only=True)
        full_support = initial_field(rng)
        support_times = np.r_[0.0, np.cumsum(rng.uniform(0.035, 0.08, size=3))]
        extra_times = np.array([0.0, rng.uniform(0.035, 0.08)])
        speed = rng.uniform(0.3, 0.5) if split == "test" else rng.uniform(0.08, 0.28)
        vx = speed * rng.choice((-1, 1))
        vy = rng.uniform(0.18, 0.42)
        diffusivity = rng.uniform(0.008, 0.014) if split == "test" else rng.uniform(0.002, 0.007)
        if split == "train":
            # Train-only prior sees pure transport and pure diffusion.
            if index % 2 == 0:
                diffusivity = 0.0
            else:
                vx = vy = 0.0
        regimes = (np.array([vx, -vy, diffusivity]), np.array([vx, vy, diffusivity]))
        support = tuple(observed_transitions(axis_support, c, support_times) for c in regimes)
        extra = tuple(observed_transitions(full_support, c, extra_times) for c in regimes)
        targets = tuple(forecast(query, c, FORECAST_TIMES) for c in regimes)
        yield Episode(query, regimes, support, extra, targets)


def relative_error(prediction, target):
    return float(np.mean((prediction - target)**2) / np.mean(target**2))


def score_episode(episode, prior, noise_seed):
    restricted = [fit_transport(observations, prior) for observations in episode.support]
    restored = [
        fit_transport((*support, *extra), prior)
        for support, extra in zip(episode.support, episode.additional_support, strict=True)
    ]
    methods = {
        "prior": (Fit(prior, 0, np.zeros(3)),) * 2,
        "restricted": restricted,
        "restored": restored,
        "swapped": restored[::-1],
    }
    for support, extra in zip(episode.support, episode.additional_support, strict=True):
        # Keep measurement noise paired across the two coefficient regimes.
        rng = np.random.default_rng(noise_seed)
        noisy = add_noise(support, rng, SUPPORT_NOISE_STD)
        noisy_extra = add_noise(extra, rng, SUPPORT_NOISE_STD)
        for name, observations, floor in (
            ("noisy_unguarded", noisy, 0.0),
            ("noisy_guarded", noisy, SUPPORT_NOISE_STD),
            ("noisy_restored", (*noisy, *noisy_extra), SUPPORT_NOISE_STD),
        ):
            methods.setdefault(name, []).append(fit_transport(observations, prior, noise_std=floor))
    for name, fits in methods.items():
        predictions = [forecast(episode.query, fit.coefficients, FORECAST_TIMES) for fit in fits]
        row = {
            "rank": min(fit.rank for fit in fits),
            "vy_mae": float(np.mean([
                abs(fit.coefficients[1] - truth[1])
                for fit, truth in zip(fits, episode.coefficients, strict=True)
            ])),
        }
        for h in HORIZONS:
            a, b = (prediction[h - 1] for prediction in predictions)
            truth_a, truth_b = (target[h - 1] for target in episode.targets)
            row[f"nmse_h{h}"] = (relative_error(a, truth_a) + relative_error(b, truth_b)) / 2
            row[f"pair_nmse_h{h}"] = relative_error(a - b, truth_a - truth_b)
        yield name, row


def run_benchmark(seeds, pairs):
    report = {"protocol": {"grid": GRID, "forecast_times": FORECAST_TIMES.tolist(),
                           "seeds": list(seeds), "pairs_per_split_seed": pairs,
                           "support_noise_std": SUPPORT_NOISE_STD}}
    for split in ("validation", "test"):
        records = {}
        for seed in report["protocol"]["seeds"]:
            prior = np.mean([
                coefficients
                for episode in make_episodes("train", seed, pairs)
                for coefficients in episode.coefficients
            ], axis=0)
            for index, episode in enumerate(make_episodes(split, seed, pairs)):
                noise_seed = np.random.SeedSequence([seed, SPLITS.index(split), index, 1])
                for name, scores in score_episode(episode, prior, noise_seed):
                    records.setdefault(name, []).append({"seed": seed, "pair": index, **scores})
        report[split] = {
            "summary": {
                name: {f"median_{key}": float(np.median([row[key] for row in rows]))
                       for key in rows[0] if key not in ("seed", "pair")}
                for name, rows in records.items()
            },
            "pairs": records,
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
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({split: report[split]["summary"] for split in ("validation", "test")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
