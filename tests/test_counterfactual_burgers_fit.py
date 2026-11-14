import numpy as np

from gphyt.counterfactual_burgers import make_episodes, transitions
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_transport import initial_field


def test_restricted_fit_keeps_hidden_coefficient_at_prior():
    prior = np.array([0.0, 0.03, 0.01, 0.2])
    for episode in make_episodes("test", seed=7, pairs=2):
        hidden = 1 if episode.intervention == "crosswind" else 3
        fits = [fit_burgers(support, prior) for support in episode.support]
        assert all(fit.rank == (3 if hidden == 1 else 2) for fit in fits)
        for fit in fits:
            np.testing.assert_allclose(fit.coefficients[hidden], prior[hidden], atol=1e-12)
        np.testing.assert_allclose(fits[0].coefficients, fits[1].coefficients, atol=1e-10)


def test_informative_transition_restores_all_four_coefficients():
    prior = np.array([0.0, 0.03, 0.01, 0.2])
    for episode in make_episodes("test", seed=7, pairs=2):
        for support, extra, truth in zip(
            episode.support, episode.additional_support, episode.coefficients, strict=True
        ):
            fit = fit_burgers((*support, *extra), prior)
            assert fit.rank == 4
            np.testing.assert_allclose(fit.coefficients, truth, atol=1e-6)


def test_linear_observations_recover_zero_nonlinearity():
    initial = initial_field(np.random.default_rng(4))
    truth = np.array([0.25, -0.2, 0.01, 0.0])
    support = transitions(initial, truth, [0.0, 0.07, 0.11])
    fit = fit_burgers(support, np.array([0.0, 0.0, 0.01, 0.3]))
    assert fit.rank == 4
    np.testing.assert_allclose(fit.coefficients, truth, atol=1e-6)
