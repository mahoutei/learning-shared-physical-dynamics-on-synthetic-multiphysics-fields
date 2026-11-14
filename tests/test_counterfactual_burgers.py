import numpy as np

from gphyt.counterfactual_burgers import (
    fit_protected_transport,
    make_episodes,
    nonlinear_forecast,
    run_benchmark,
)
from gphyt.counterfactual_transport import forecast


def test_zero_nonlinearity_agrees_with_exact_linear_forecast():
    initial = np.random.default_rng(2).normal(size=(17, 17))
    coefficients = (0.2, -0.3, 0.008)
    times = [0.0, 0.04, 0.2]
    np.testing.assert_allclose(
        nonlinear_forecast(initial, (*coefficients, 0.0), times),
        forecast(initial, coefficients, times),
        atol=2e-12,
    )


def test_reference_preserves_mean_and_converges_under_step_halving():
    y, x = np.meshgrid(np.arange(24) / 24, np.arange(32) / 32, indexing="ij")
    initial = 0.2 + np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
    coefficients = (0.3, -0.2, 0.008, 0.55)
    times = [0.1, 0.96]
    coarse = nonlinear_forecast(initial, coefficients, times, max_step=0.004)
    fine = nonlinear_forecast(initial, coefficients, times, max_step=0.002)
    np.testing.assert_allclose(coarse.mean(axis=(1, 2)), initial.mean(), atol=1e-12)
    np.testing.assert_allclose(coarse, fine, atol=3e-5)


def test_one_coefficient_changes_and_restricted_support_is_identical():
    for episode in make_episodes("test", seed=7, pairs=2):
        a, b = episode.coefficients
        assert np.count_nonzero(a != b) == 1
        for left, right in zip(*episode.support, strict=True):
            np.testing.assert_allclose(left.final, right.final, atol=2e-12)
        assert not np.allclose(*episode.targets)


def test_incomplete_library_has_error_even_with_oracle_linear_coefficients():
    report = run_benchmark((0,), pairs=2)
    for experiment in ("crosswind", "nonlinearity"):
        scores = report["test"][experiment]["summary"]
        assert scores["oracle_linear"]["median_nmse_h24"] > 1e-4
        assert scores["restricted"]["median_pair_nmse_h24"] > 0.999999
        assert scores["restored"]["median_rank"] == 3
        assert scores["informative_only"]["median_rank"] == 3


def test_x_average_recovers_linear_crosswind_and_diffusion_despite_nonlinearity():
    prior = np.array([0.0, 0.03, 0.005])
    for episode in make_episodes("test", seed=7, pairs=2):
        for support, extra, truth in zip(
            episode.support, episode.additional_support, episode.coefficients, strict=True
        ):
            fitted = fit_protected_transport((*support, *extra), prior)
            np.testing.assert_allclose(fitted[1:], truth[1:3], atol=1e-9)
            if episode.intervention == "crosswind":
                restricted = fit_protected_transport(support, prior)
                assert restricted[1] == prior[1]
