import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast
from gphyt.counterfactual_burgers_fit import fit_burgers
from gphyt.counterfactual_missing_flux import PRIOR, make_case


def test_joint_fit_recovers_base_and_transverse_flux_from_support():
    case = make_case(0, 0, transverse_strength=-0.32)
    fit = fit_burgers(case.support, PRIOR, transverse_prior=0.0)
    assert fit.rank == 5
    np.testing.assert_allclose(fit.coefficients, (*case.coefficients, -0.32), atol=1e-6)


def test_joint_fit_preserves_prior_without_transverse_evidence():
    case = make_case(0, 0, transverse_strength=0.32, x_only=True)
    fit = fit_burgers(case.support, PRIOR, transverse_prior=0.0)
    assert fit.rank == 3
    np.testing.assert_allclose(fit.coefficients[[1, 4]], [0.0, 0.0], atol=1e-10)


def test_cubic_mechanism_changes_dynamics_and_preserves_mean():
    first = make_case(0, 0, transverse_strength=0.25, cubic_strength=-0.12)
    second = make_case(0, 0, transverse_strength=0.25, cubic_strength=0.12)
    np.testing.assert_array_equal(first.query, second.query)
    assert first.coefficients == second.coefficients
    assert np.linalg.norm(first.target[-1] - second.target[-1]) > 0.1
    for case in (first, second):
        np.testing.assert_allclose(case.target.mean(axis=(-2, -1)), case.query.mean(), atol=1e-12)
    refined = nonlinear_forecast(
        second.query, second.coefficients, [0.96], transverse_strength=0.25,
        cubic_strength=0.12, max_step=0.002,
    )
    np.testing.assert_allclose(second.target[-1], refined[0], atol=2e-4)
