import numpy as np

from gphyt.counterfactual_learned_flux import fit_flux, forecast_flux, flux_direction
from gphyt.counterfactual_missing_flux import make_case
from gphyt.counterfactual_transport import FORECAST_TIMES, relative_error


def test_learned_flux_improves_an_independent_query():
    case = make_case(0, 0, 0.27, cubic_strength=0.12)
    fit = fit_flux(case.support)
    assert fit.active
    model = forecast_flux(case.query, fit.coefficients, FORECAST_TIMES)
    base = forecast_flux(case.query, fit.base, FORECAST_TIMES)
    assert relative_error(model[-1], case.target[-1]) < 0.5 * relative_error(
        base[-1], case.target[-1],
    )
    np.testing.assert_allclose(model.mean(axis=(1, 2)), case.query.mean(), atol=1e-12)


def test_gate_leaves_an_explained_episode_unchanged():
    case = make_case(0, 0, 0.27)
    fit = fit_flux(case.support)
    assert not fit.active
    np.testing.assert_array_equal(fit.coefficients, fit.base)


def test_two_axis_fit_recovers_an_oblique_flux_on_an_independent_query():
    case = make_case(0, 0, 0.27, cubic_strength=0.08, cubic_transverse_strength=-0.12)
    fixed = fit_flux(case.support)
    found = fit_flux(case.support, bidirectional=True)
    assert found.active
    x, y = flux_direction(found.coefficients)
    assert x > 0 > y
    actual = forecast_flux(case.query, found.coefficients, FORECAST_TIMES)
    control = forecast_flux(case.query, fixed.coefficients, FORECAST_TIMES)
    np.testing.assert_allclose(actual.mean(axis=(1, 2)), case.query.mean(), atol=1e-12)
    assert relative_error(actual[-1], case.target[-1]) < 0.5 * relative_error(
        control[-1], case.target[-1],
    )


def test_unobservable_transverse_flux_keeps_the_two_axis_gate_closed():
    case = make_case(0, 0, 0.27, x_only=True, cubic_transverse_strength=0.12)
    fit = fit_flux(case.support, bidirectional=True)
    assert not fit.active
    np.testing.assert_array_equal(fit.coefficients, fit.base)
