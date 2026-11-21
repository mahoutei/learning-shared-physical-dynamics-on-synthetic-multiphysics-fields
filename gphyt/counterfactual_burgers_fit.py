"""Fit a known scalar Burgers equation from observed transitions."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import (
    make_episodes,
    nonlinear_forecast,
    score_pair,
    score_spectral,
    summarize,
)
from gphyt.counterfactual_transport import FORECAST_TIMES

# Scale diffusion separately so rank is assessed in comparable parameter units.
PARAMETER_SCALE = np.array([1.0, 1.0, 0.01, 1.0])
DIFFERENCE_STEP = 1e-5
RANK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BurgersFit:
    coefficients: np.ndarray
    rank: int
    support_rmse: float
    singular_values: np.ndarray


def endpoint_masks(transitions, mask):
    """Normalize a shared mask or a sequence of masks to one per transition."""
    if mask is None or (isinstance(mask, np.ndarray) and mask.ndim == 2):
        masks = (mask,) * len(transitions)
    else:
        masks = tuple(mask)
    if len(masks) != len(transitions):
        raise ValueError("Expected one mask per transition")
    for transition, selected in zip(transitions, masks, strict=True):
        if selected is not None and (
            not isinstance(selected, np.ndarray) or selected.dtype != bool
            or selected.shape != transition.final.shape or not selected.any()
        ):
            raise ValueError("Expected a nonempty Boolean endpoint mask matching the grid")
    return masks


def endpoint_error(prediction, target, mask):
    if mask is None:
        return (prediction - target).ravel()
    return prediction[mask] - target[mask]


def fit_burgers(transitions, prior, *, transverse_prior=None, endpoint_mask=None):
    """Fit endpoints in observable directions of the local sensitivity matrix."""
    if not transitions:
        raise ValueError("At least one observed transition is required")
    masks = endpoint_masks(transitions, endpoint_mask)
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (4,) or not np.isfinite(prior).all() or prior[2] < 0:
        raise ValueError("Expected four finite prior coefficients and nonnegative diffusion")
    parameter_scale = PARAMETER_SCALE
    if transverse_prior is not None:
        if not np.isfinite(transverse_prior):
            raise ValueError("Expected a finite transverse flux prior")
        prior = np.append(prior, transverse_prior)
        parameter_scale = np.append(parameter_scale, 1.0)

    def residual(parameters):
        coefficients = parameters * parameter_scale
        transverse = coefficients[4] if transverse_prior is not None else 0.0
        errors = []
        for transition, mask in zip(transitions, masks, strict=True):
            predicted = nonlinear_forecast(
                transition.initial, coefficients[:4], [transition.elapsed],
                transverse_strength=transverse,
            )[0]
            errors.append(endpoint_error(predicted, transition.final, mask))
        return np.concatenate(errors)

    parameters = prior / parameter_scale
    errors = residual(parameters)
    offsets = np.eye(len(prior)) * DIFFERENCE_STEP
    for _ in range(12):
        sensitivity = np.column_stack([
            (residual(parameters + offset) - errors) / DIFFERENCE_STEP
            for offset in offsets
        ])
        update, _, rank, singular_values = np.linalg.lstsq(
            sensitivity, -errors, rcond=RANK_TOLERANCE,
        )
        if np.linalg.norm(update) < 1e-10:
            break
        # Constrain diffusion and backtrack when the linearized update is too large.
        error_norm = np.linalg.norm(errors)
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = parameters + scale * update
            candidate[2] = max(candidate[2], 0.0)
            candidate_errors = residual(candidate)
            if np.linalg.norm(candidate_errors) < error_norm:
                parameters, errors = candidate, candidate_errors
                break
        else:
            break
    return BurgersFit(parameters * parameter_scale, int(rank),
                      float(np.sqrt(np.mean(errors**2))), singular_values)


def score_fits(episode, prior):
    started = perf_counter()
    restricted = [fit_burgers(support, prior) for support in episode.support]
    restricted_seconds = perf_counter() - started
    started = perf_counter()
    restored = [fit_burgers((*support, *extra), prior)
                for support, extra in zip(episode.support, episode.additional_support, strict=True)]
    restored_seconds = perf_counter() - started
    for name, fits, seconds in (
        ("burgers_restricted", restricted, restricted_seconds),
        ("burgers_restored", restored, restored_seconds),
        ("burgers_swapped", restored[::-1], restored_seconds),
    ):
        predictions = [nonlinear_forecast(episode.query, fit.coefficients, FORECAST_TIMES)
                       for fit in fits]
        scores = score_pair(predictions, episode.targets)
        errors = np.abs(np.array([fit.coefficients for fit in fits]) - episode.coefficients)
        yield name, {
            **scores,
            "rank": min(fit.rank for fit in fits),
            "strength_mae": float(errors[:, 3].mean()),
            "vy_mae": float(errors[:, 1].mean()),
            "support_rmse": float(np.mean([fit.support_rmse for fit in fits])),
            "fit_seconds_per_pair": seconds,
        }
    oracle = [nonlinear_forecast(episode.query, coefficients, FORECAST_TIMES)
              for coefficients in episode.coefficients]
    yield "oracle_burgers", score_pair(oracle, episode.targets)


def run_benchmark(seeds, pairs):
    seeds = list(seeds)
    records = {}
    for seed in seeds:
        prior = np.mean([c for episode in make_episodes("train", seed, pairs)
                         for c in episode.coefficients], axis=0)
        for index, episode in enumerate(make_episodes("test", seed, pairs)):
            methods = records.setdefault(episode.intervention, {})
            for name, scores in (*score_spectral(episode, prior[:3]), *score_fits(episode, prior)):
                methods.setdefault(name, []).append({"seed": seed, "pair": index, **scores})
        print(f"Scored seed {seed}", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs,
                     "forecast_times": FORECAST_TIMES.tolist()},
        "test": {intervention: {"summary": summarize(rows), "pairs": rows}
                 for intervention, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2 or min(args.seeds) < 0:
        parser.error("provide at least two pairs and nonnegative seeds")
    report = run_benchmark(args.seeds, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
