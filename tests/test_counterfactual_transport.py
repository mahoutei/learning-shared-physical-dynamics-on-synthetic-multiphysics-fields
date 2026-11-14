import numpy as np
import pytest

from gphyt.counterfactual_transport import (
    add_noise,
    fit_transport,
    forecast,
    make_episodes,
    run_benchmark,
)


def test_analytic_forecast_translates_and_decays_a_single_mode():
    shape = (12, 16)
    x = np.arange(shape[1]) / shape[1]
    initial = np.broadcast_to(np.cos(2 * np.pi * x), shape)
    elapsed = np.array([0.0, 0.1])
    velocity_x, diffusivity = 0.35, 0.02
    expected = (
        np.cos(2 * np.pi * (x - velocity_x * elapsed[:, None]))
        * np.exp(-4 * np.pi**2 * diffusivity * elapsed[:, None])
    )
    np.testing.assert_allclose(
        forecast(initial, (velocity_x, 0.0, diffusivity), elapsed),
        np.broadcast_to(expected[:, None], (2, *shape)),
        atol=1e-12,
    )


def test_missing_direction_keeps_prior_and_new_observation_restores_it():
    episode = next(make_episodes("test", seed=0, pairs=1))
    prior = np.array([0.0, 0.13, 0.005])
    for target, support, extra in zip(
        episode.coefficients, episode.support, episode.additional_support
    ):
        restricted = fit_transport(support, prior)
        restored = fit_transport((*support, *extra), prior)
        assert restricted.rank == 2
        assert restricted.coefficients[1] == pytest.approx(prior[1])
        assert restored.rank == 3
        np.testing.assert_allclose(restored.coefficients, target, atol=1e-9)


def test_held_out_support_swap_changes_the_same_query_forecast():
    episode = next(make_episodes("test", seed=0, pairs=1))
    assert not np.array_equal(episode.targets[0], episode.targets[1])
    report = run_benchmark(seeds=(0,), pairs=2)
    result = report["test"]["summary"]
    assert result["restored"]["median_pair_nmse_h24"] < 1e-12
    assert result["restricted"]["median_pair_nmse_h24"] == pytest.approx(1.0)
    assert result["swapped"]["median_pair_nmse_h24"] == pytest.approx(4.0)


def test_noise_floor_prevents_false_evidence_in_missing_direction():
    episode = next(make_episodes("test", seed=0, pairs=1))
    rng = np.random.default_rng(12)
    noise_std = 0.03
    support = add_noise(episode.support[0], rng, noise_std)
    extra = add_noise(episode.additional_support[0], rng, noise_std)
    prior = np.zeros(3)
    assert fit_transport(support, prior).rank == 3
    guarded = fit_transport(support, prior, noise_std=noise_std)
    restored = fit_transport((*support, *extra), prior, noise_std=noise_std)
    assert guarded.rank == 2
    assert guarded.coefficients[1] == pytest.approx(prior[1])
    assert restored.rank == 3
    assert restored.coefficients[1] == pytest.approx(episode.coefficients[0][1], abs=0.02)
