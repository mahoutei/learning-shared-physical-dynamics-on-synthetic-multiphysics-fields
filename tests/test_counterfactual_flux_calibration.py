import numpy as np

from gphyt.counterfactual_flux_calibration import (
    features,
    gated_coefficients,
    CUBIC_SCALE,
)
from gphyt.counterfactual_missing_flux import make_case


def test_support_feature_detects_missing_flux_sign():
    for strength in (-0.12, 0.12):
        case = make_case(0, 0, 0.27, cubic_strength=strength)
        base = np.array([*case.coefficients, 0.27])
        evidence, adjustment = features(case.support, base)
        assert np.sign(evidence) == np.sign(strength)
        assert abs(adjustment[-1] * CUBIC_SCALE - strength) < 0.04


def test_gate_preserves_baseline_and_enforces_nonnegative_diffusion():
    base = np.array([0.1, -0.2, 0.01, 0.2, 0.3])
    update = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.5])
    np.testing.assert_array_equal(gated_coefficients(base, update, 0.01, 0.02),
                                  np.r_[base, 0.0])
    corrected = gated_coefficients(base, update, 0.03, 0.02)
    assert corrected[2] == 0.0
    assert corrected[-1] == 0.5 * CUBIC_SCALE
