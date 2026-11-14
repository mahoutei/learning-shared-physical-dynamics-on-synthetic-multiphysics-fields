from dataclasses import replace

import numpy as np

from gphyt.counterfactual_flux_coverage import coverage_masks
from gphyt.counterfactual_learned_flux import fit_flux, forecast_flux
from gphyt.counterfactual_spatial_flux import make_pair, observation_mask
from gphyt.counterfactual_transport import FORECAST_TIMES, relative_error


def test_spatial_fit_improves_an_independent_query_with_partial_support():
    query, arms = make_pair(0, 0, amplitude=0.16)
    support, target = arms[0]
    mask = observation_mask(query.shape)
    uniform = fit_flux(support, bidirectional=True, endpoint_mask=mask)
    spatial = fit_flux(support, bidirectional=True, spatial=True, endpoint_mask=mask)
    assert spatial.active
    actual = forecast_flux(query, spatial.coefficients, FORECAST_TIMES)
    control = forecast_flux(query, uniform.coefficients, FORECAST_TIMES)
    assert relative_error(actual[-1], target[-1]) < relative_error(control[-1], target[-1])
    np.testing.assert_allclose(actual.mean(axis=(1, 2)), query.mean(), atol=1e-12)


def test_mask_discards_hidden_endpoint_values():
    query, arms = make_pair(1, 0, amplitude=0.0)
    support, _ = arms[0]
    mask = observation_mask(query.shape)
    corrupted = tuple(replace(t, final=np.where(mask, t.final, 1000)) for t in support)
    original = fit_flux(support, bidirectional=True, spatial=True, endpoint_mask=mask)
    changed = fit_flux(corrupted, bidirectional=True, spatial=True, endpoint_mask=mask)
    assert not original.active and not changed.active
    np.testing.assert_array_equal(original.coefficients, changed.coefficients)


def test_coverage_changes_only_the_second_fitting_mask():
    fixed, varied = coverage_masks((64, 32))
    assert all(mask.sum() == 48 * 32 for mask in (*fixed, *varied))
    np.testing.assert_array_equal(np.stack(fixed), np.stack([fixed[0]] * 3))
    np.testing.assert_array_equal(fixed[0], varied[0])
    np.testing.assert_array_equal(fixed[2], varied[2])
    assert np.all(varied[0] | varied[1])


def test_varying_masks_ignore_hidden_endpoint_values_on_an_active_fit():
    query, arms = make_pair(0, 0, amplitude=0.16)
    support, _ = arms[0]
    _, masks = coverage_masks(query.shape)
    corrupted = tuple(replace(t, final=np.where(mask, t.final, np.nan))
                      for t, mask in zip(support, masks, strict=True))
    original = fit_flux(support, bidirectional=True, spatial=True, endpoint_mask=masks)
    changed = fit_flux(corrupted, bidirectional=True, spatial=True, endpoint_mask=masks)
    assert original.active and changed.active
    np.testing.assert_array_equal(original.coefficients, changed.coefficients)
    forecast = forecast_flux(query, changed.coefficients, FORECAST_TIMES)
    np.testing.assert_allclose(forecast.mean(axis=(1, 2)), query.mean(), atol=1e-12)
