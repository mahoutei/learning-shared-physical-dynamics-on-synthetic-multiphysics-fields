import pytest
import torch
from torch import nn

from gphyt.repro.run import DerivativeIntegrator


class RecordingField(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.inputs = []
        self.slopes = []

    def differentiate(self, history):
        self.inputs.append(history.detach().clone())
        slope = history * self.weight
        slope.retain_grad()
        self.slopes.append(slope)
        return slope


@pytest.mark.parametrize("last_frame_only", [False, True])
def test_heun_second_stage_and_gradients(last_frame_only):
    field = RecordingField()
    model = DerivativeIntegrator(field, "Heun", 0.25, last_frame_only=last_frame_only)
    history = torch.arange(1.0, 5.0).reshape(1, 4, 1, 1, 1)
    prediction = model(history)

    expected_stage = 1.5 * history
    if last_frame_only:
        expected_stage[:, :-1] = history[:, :-1]
    torch.testing.assert_close(field.inputs[1], expected_stage)
    torch.testing.assert_close(prediction, 1.625 * history[:, -1:])
    prediction.sum().backward()
    assert all(torch.count_nonzero(slope.grad[:, -1:]) for slope in field.slopes)
