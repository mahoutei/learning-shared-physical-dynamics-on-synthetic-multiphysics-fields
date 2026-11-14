import numpy as np

from gphyt.counterfactual_burgers import nonlinear_forecast
from gphyt.counterfactual_missing_flux import make_case, support_correction
from gphyt.counterfactual_transport import initial_field


def test_missing_flux_is_conservative_and_reference_converges():
    initial = initial_field(np.random.default_rng(42)) + 0.3
    coefficients = (0.12, -0.1, 0.01, 0.25)
    times = [0.04, 0.24]
    coarse = nonlinear_forecast(initial, coefficients, times, transverse_strength=0.3)
    fine = nonlinear_forecast(
        initial, coefficients, times, transverse_strength=0.3, max_step=0.002,
    )
    finer = nonlinear_forecast(
        initial, coefficients, times, transverse_strength=0.3, max_step=0.001,
    )
    base = nonlinear_forecast(initial, coefficients, times)
    assert np.linalg.norm(base - fine) > 0.1
    assert np.linalg.norm(fine - finer) < 0.3 * np.linalg.norm(coarse - fine)
    np.testing.assert_allclose(fine, finer, atol=1.2e-4)
    np.testing.assert_allclose(fine.mean(axis=(1, 2)), initial.mean(), atol=1e-12)


def test_support_gate_is_inactive_for_known_equation_or_unobservable_flux():
    exact = make_case(seed=0, index=0, transverse_strength=0.0)
    hidden = make_case(seed=0, index=0, transverse_strength=0.3, x_only=True)
    assert support_correction(exact.support, exact.coefficients).strength == 0.0
    assert support_correction(hidden.support, hidden.coefficients).strength == 0.0


def test_informative_support_gives_opposite_signed_corrections():
    for strength in (-0.3, 0.3):
        case = make_case(seed=0, index=0, transverse_strength=strength)
        correction = support_correction(case.support, case.coefficients)
        assert abs(correction.strength - strength) < 0.04
