import pytest
import torch

from fireants.losses.cc import (
    LocalNormalizedCrossCorrelationLoss,
    gaussian_1d,
    separable_filtering,
)


pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)


def test_mps_separable_filtering_3d_matches_cpu_forward_and_backward():
    torch.manual_seed(37)
    input_cpu = torch.randn(1, 2, 5, 6, 7, requires_grad=True)
    kernel = gaussian_1d(torch.tensor(1.0))
    kernels = [kernel] * 3

    out_cpu = separable_filtering(input_cpu, kernels)
    out_cpu.square().mean().backward()

    input_mps = input_cpu.detach().to("mps").requires_grad_(True)
    out_mps = separable_filtering(input_mps, kernels)
    out_mps.square().mean().backward()

    assert out_mps.device.type == "mps"
    assert input_mps.grad is not None
    assert input_mps.grad.device.type == "mps"
    assert torch.allclose(out_mps.cpu(), out_cpu.detach(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(input_mps.grad.cpu(), input_cpu.grad, atol=5e-4, rtol=5e-4)


def test_mps_local_normalized_cross_correlation_3d_matches_cpu_backward():
    torch.manual_seed(41)
    pred_cpu = torch.randn(1, 1, 5, 6, 7, requires_grad=True)
    target_cpu = torch.randn(1, 1, 5, 6, 7)
    loss = LocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction="mean")

    loss_cpu = loss(pred_cpu, target_cpu)
    loss_cpu.backward()

    pred_mps = pred_cpu.detach().to("mps").requires_grad_(True)
    target_mps = target_cpu.to("mps")
    loss_mps = loss.to("mps")(pred_mps, target_mps)
    loss_mps.backward()

    assert loss_mps.device.type == "mps"
    assert pred_mps.grad is not None
    assert pred_mps.grad.device.type == "mps"
    assert torch.allclose(loss_mps.cpu(), loss_cpu.detach(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(pred_mps.grad.cpu(), pred_cpu.grad, atol=5e-4, rtol=5e-4)
