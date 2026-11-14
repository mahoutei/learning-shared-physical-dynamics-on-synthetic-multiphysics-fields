import numpy as np
import torch
from torch import nn

from gphyt.counterfactual_burgers_models import (
    SupportStepper,
    pack_support,
)
from gphyt.counterfactual_burgers import make_episodes
from gphyt.counterfactual_transport import FORECAST_TIMES, forecast


class ZeroCorrection(nn.Module):
    def differentiate(self, fields):
        return torch.zeros_like(fields)


def test_hybrid_uses_the_spectral_step_and_matched_control_uses_identity():
    episode = next(make_episodes("test", 0, 1))
    support = (*episode.support[0], *episode.additional_support[0])
    features, fit = pack_support(support, np.zeros(3))
    field = torch.tensor(episode.query, dtype=torch.float32).unsqueeze(0)
    features, fit = features.unsqueeze(0), fit.unsqueeze(0)
    residual = SupportStepper(hybrid=False, backbone=ZeroCorrection())
    hybrid = SupportStepper(hybrid=True, backbone=ZeroCorrection())
    torch.testing.assert_close(residual(field, features, fit), field)
    expected = torch.tensor(forecast(episode.query, fit[0].numpy(), FORECAST_TIMES[:1]),
                            dtype=field.dtype)
    torch.testing.assert_close(hybrid(field, features, fit), expected, atol=2e-6, rtol=1e-5)


def test_missing_support_has_identical_features_for_both_futures():
    episode = next(make_episodes("test", 0, 1))
    features_a, fit_a = pack_support(episode.support[0], np.zeros(3))
    features_b, fit_b = pack_support(episode.support[1], np.zeros(3))
    torch.testing.assert_close(features_a, features_b, atol=1e-7, rtol=0)
    torch.testing.assert_close(fit_a, fit_b)
