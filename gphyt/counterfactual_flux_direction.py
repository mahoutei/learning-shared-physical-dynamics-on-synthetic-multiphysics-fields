"""Compare fixed-axis and two-axis flux corrections on the same observations."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gphyt.counterfactual_burgers import score_pair, summarize
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_joint_flux import NOISE
from gphyt.counterfactual_learned_flux import fit_flux, flux_direction, forecast_flux
from gphyt.counterfactual_missing_flux import PRIOR, make_case
from gphyt.counterfactual_transport import FORECAST_TIMES, add_noise

CONDITIONS = {
    "oblique_clean": (0.12, 0.0, False),
    "oblique_noisy": (0.12, NOISE, False),
    "transverse_hidden": (0.12, 0.0, True),
    "matched_noisy": (0.0, NOISE, False),
}


def score_episode(seed, index, amplitude, noise, x_only):
    angle = np.random.default_rng([seed, index, 8]).uniform(np.pi / 4, 3 * np.pi / 4)
    direction = np.array([np.cos(angle), np.sin(angle)]) if not x_only else np.array([0., 1.])
    candidates = {method: [] for method in ("joint", "fixed_axis", "two_axis")}
    activations = {method: [] for method in candidates}
    alignments = {method: [] for method in candidates}
    targets, query = [], None
    for sign in (-1, 1):
        x, y = sign * amplitude * direction
        # The matched-equation control varies a known coefficient so its
        # paired difference remains defined. Omitted-flux pairs vary only that flux.
        transverse = 0.27 if amplitude else sign * 0.27
        case = make_case(seed, index, transverse, x_only=x_only, cubic_strength=x,
                         cubic_transverse_strength=y)
        query = case.query
        targets.append(case.target)
        support = case.support
        if noise:
            support = add_noise(support, np.random.default_rng([seed, index, 9]), noise)
        base = fit_burgers(support, PRIOR, transverse_prior=0.0)
        prefix = None
        if base.support_rmse >= 1e-7:
            prefix = fit_burgers(support[:-1], PRIOR, transverse_prior=0.0)
        fixed = fit_flux(support, base_fit=base, prefix_fit=prefix)
        two_axis = fit_flux(support, bidirectional=True, base_fit=base, prefix_fit=prefix)
        for method, coefficients, active in (
            ("joint", fixed.base, False),
            ("fixed_axis", fixed.coefficients, fixed.active),
            ("two_axis", two_axis.coefficients, two_axis.active),
        ):
            candidates[method].append(coefficients)
            activations[method].append(active)
            inferred = flux_direction(coefficients)
            length = np.linalg.norm(inferred)
            alignments[method].append(float(sign * (inferred @ direction) / length)
                                      if amplitude and length > 1e-12 else 0.0)
    for method, coefficients in (*candidates.items(), ("swapped", candidates["two_axis"][::-1])):
        forecasts = [forecast_flux(query, c, FORECAST_TIMES) for c in coefficients]
        row = score_pair(forecasts, targets)
        if method != "swapped":
            row.update(gate_rate=float(np.mean(activations[method])),
                       direction_alignment=float(np.mean(alignments[method])))
        yield method, row


def run_benchmark(seeds, pairs):
    records = {name: {} for name in CONDITIONS}
    for seed in seeds:
        started = perf_counter()
        for index in range(pairs):
            for condition, recipe in CONDITIONS.items():
                for method, scores in score_episode(seed, index, *recipe):
                    records[condition].setdefault(method, []).append(
                        {"seed": seed, "pair": index, **scores},
                    )
        print(f"Scored seed {seed} in {perf_counter() - started:.1f}s", flush=True)
    return {
        "protocol": {"seeds": seeds, "pairs_per_seed": pairs, "conditions": CONDITIONS},
        **{condition: {"summary": summarize(methods), "records": methods}
           for condition, methods in records.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
