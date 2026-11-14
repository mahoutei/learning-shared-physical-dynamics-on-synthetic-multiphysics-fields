"""Learn how to calibrate a candidate flux correction from support observations."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast, score_pair, summarize
from gphyt.counterfactual_burgers_fit import (
    DIFFERENCE_STEP,
    PARAMETER_SCALE,
    RANK_TOLERANCE,
    fit_burgers,
)
from gphyt.counterfactual_joint_flux import CONDITIONS, NOISE
from gphyt.counterfactual_missing_flux import PRIOR, make_case
from gphyt.counterfactual_transport import FORECAST_TIMES, add_noise

CUBIC_SCALE = 0.1
SCALES = np.r_[PARAMETER_SCALE, 1.0, CUBIC_SCALE]


def propagate(initial, coefficients, times):
    return nonlinear_forecast(
        initial, coefficients[:4], times,
        transverse_strength=coefficients[4], cubic_strength=coefficients[5],
    )


def features(support, base):
    """Estimate one joint coefficient update from the observed endpoint residual."""
    def endpoints(coefficients):
        return np.concatenate([
            propagate(t.initial, coefficients, [t.elapsed])[0].ravel() for t in support
        ])

    initial = np.r_[base, 0.0]
    prediction = endpoints(initial)
    observed = np.concatenate([t.final.ravel() for t in support])
    jacobian = np.column_stack([
        (endpoints(initial + perturbation) - prediction) / DIFFERENCE_STEP
        for perturbation in np.diag(SCALES * DIFFERENCE_STEP)
    ])
    adjustment = np.linalg.lstsq(
        jacobian, observed - prediction, rcond=RANK_TOLERANCE,
    )[0]
    return float(adjustment[-1] * CUBIC_SCALE), adjustment


def gated_coefficients(base, adjustment, evidence, threshold):
    coefficients = np.r_[base, 0.0]
    if abs(evidence) > threshold:
        coefficients += adjustment * SCALES
        coefficients[2] = max(coefficients[2], 0.0)
    return coefficients


@dataclass(frozen=True)
class Calibration:
    weights: np.ndarray
    scale: np.ndarray
    threshold: float

    def predict(self, base, adjustment):
        return (np.r_[base, adjustment] / self.scale) @ self.weights


def observed_fit(case, noise, noise_seed):
    support = case.support
    if noise:
        support = add_noise(support, np.random.default_rng(noise_seed), noise)
    base = fit_burgers(support, PRIOR, transverse_prior=0.0).coefficients
    evidence, adjustment = features(support, base)
    return base, evidence, adjustment


def train(seed, pairs):
    started = perf_counter()
    inputs, targets = [], []
    training_seed, validation_seed = np.random.SeedSequence(seed).generate_state(2)
    rng = np.random.default_rng(training_seed)
    for index in range(pairs):
        # Sample both signs and share nuisance factors between arms.
        magnitude = rng.uniform(0.055, 0.19)
        transverse = rng.uniform(0.2, 0.4) * (-1 if index % 2 else 1)
        noise = NOISE if index % 2 else 0.0
        for cubic in (-magnitude, magnitude):
            case = make_case(int(training_seed), index, transverse, cubic_strength=cubic)
            base, _, adjustment = observed_fit(case, noise, [int(training_seed), index])
            inputs.append(np.r_[base, adjustment])
            targets.append((np.r_[case.coefficients, transverse, cubic] - np.r_[base, 0]) / SCALES)
    matrix = np.asarray(inputs)
    scale = np.maximum(np.sqrt(np.mean(matrix**2, axis=0)), 1e-6)
    matrix /= scale
    # Small ridge penalty anchors the supervised map on the limited training sample.
    penalty = np.eye(matrix.shape[1]) * 0.1
    weights = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ targets)
    matched_evidence = []
    for index in range(8):
        case = make_case(int(validation_seed), index, 0.3 * (-1 if index % 2 else 1))
        for noise in (0.0, NOISE):
            _, evidence, _ = observed_fit(case, noise, [int(validation_seed), index])
            matched_evidence.append(abs(evidence))
    threshold = max(1e-8, max(matched_evidence))
    return Calibration(weights, scale, threshold), perf_counter() - started


def score_conditions(seed, index, calibration):
    strength = np.random.default_rng(np.random.SeedSequence([seed, index, 4])).uniform(0.2, 0.4)
    for condition, (noise, cubic) in CONDITIONS.items():
        rows = {name: [] for name in ("joint", "local_fit", "learned", "oracle_quadratic")}
        targets = []
        active = 0
        for transverse in (-strength, strength):
            case = make_case(seed, index, transverse, cubic_strength=cubic)
            base, evidence, adjustment = observed_fit(case, noise, [seed, index, 9])
            learned_update = calibration.predict(base, adjustment)
            coefficients = {
                "joint": np.r_[base, 0.0],
                "local_fit": gated_coefficients(base, adjustment, evidence, calibration.threshold),
                "learned": gated_coefficients(base, learned_update, evidence, calibration.threshold),
                "oracle_quadratic": np.r_[case.coefficients, transverse, 0.0],
            }
            active += abs(evidence) > calibration.threshold
            for method, fitted in coefficients.items():
                rows[method].append(propagate(case.query, fitted, FORECAST_TIMES))
            targets.append(case.target)
        for method, predictions in rows.items():
            yield condition, method, {
                **score_pair(predictions, targets),
                "gate_rate": float(active / 2) if method in ("local_fit", "learned") else 0.0,
            }


def run_benchmark(seeds, train_pairs, test_pairs):
    seeds = list(seeds)
    records = {condition: {} for condition in CONDITIONS}
    calibrations = []
    for seed in seeds:
        print(f"Calibrating seed {seed}", flush=True)
        calibration, fit_seconds = train(seed, train_pairs)
        calibrations.append({"seed": seed, "gate_threshold": calibration.threshold,
                             "calibration_seconds": fit_seconds})
        for index in range(test_pairs):
            for condition, method, row in score_conditions(seed, index, calibration):
                records[condition].setdefault(method, []).append(
                    {"seed": seed, "pair": index, **row},
                )
        print(f"Scored seed {seed}", flush=True)
    return {
        "protocol": {"seeds": seeds, "training_pairs_per_seed": train_pairs,
                     "test_pairs_per_seed": test_pairs, "forecast_times": FORECAST_TIMES.tolist(),
                     "conditions": CONDITIONS},
        "calibrations": calibrations,
        **{condition: {"summary": summarize(rows), "records": rows}
           for condition, rows in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-pairs", type=int, default=12)
    parser.add_argument("--test-pairs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.train_pairs < 2 or args.test_pairs < 1 or min(args.seeds) < 0:
        parser.error("provide at least two training pairs, a test pair, and nonnegative seeds")
    report = run_benchmark(args.seeds, args.train_pairs, args.test_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
