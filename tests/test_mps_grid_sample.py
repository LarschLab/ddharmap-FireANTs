import pytest
import torch
from torch.nn import functional as F

from fireants.interpolator.grid_sample import (
    device_aware_interpolate,
    torch_grid_sampler_3d,
    torch_warp_composer_3d,
)


pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)


def test_mps_grid_sampler_3d_matches_cpu_forward_and_grid_grad():
    torch.manual_seed(17)
    input_cpu = torch.randn(1, 2, 4, 5, 6)
    affine_cpu = torch.eye(3, 4).unsqueeze(0)
    affine_cpu[:, :, -1] = torch.tensor([[0.05, -0.1, 0.08]])
    grid_cpu = F.affine_grid(affine_cpu, (1, 2, 3, 4, 5), align_corners=True)
    grid_cpu = (grid_cpu + 0.02 * torch.randn_like(grid_cpu)).requires_grad_(True)

    out_cpu = torch_grid_sampler_3d(
        input_cpu,
        grid=grid_cpu,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        is_displacement=False,
    )
    loss_cpu = out_cpu.square().mean()
    loss_cpu.backward()

    input_mps = input_cpu.to("mps")
    grid_mps = grid_cpu.detach().to("mps").requires_grad_(True)
    out_mps = torch_grid_sampler_3d(
        input_mps,
        grid=grid_mps,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        is_displacement=False,
    )
    loss_mps = out_mps.square().mean()
    loss_mps.backward()

    assert out_mps.device.type == "mps"
    assert grid_mps.grad.device.type == "mps"
    assert torch.allclose(out_mps.cpu(), out_cpu.detach(), atol=2e-5, rtol=2e-5)
    assert torch.allclose(grid_mps.grad.cpu(), grid_cpu.grad, atol=2e-5, rtol=2e-5)


def test_mps_affine_grid_sampler_3d_updates_affine_grad():
    torch.manual_seed(23)
    input_mps = torch.randn(1, 1, 4, 5, 6, device="mps")
    affine = torch.eye(3, 4, device="mps").unsqueeze(0).requires_grad_(True)

    out = torch_grid_sampler_3d(
        input_mps,
        affine=affine,
        out_shape=(1, 1, 3, 4, 5),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    out.square().mean().backward()

    assert out.device.type == "mps"
    assert affine.grad is not None
    assert affine.grad.device.type == "mps"
    assert torch.isfinite(affine.grad).all()


def test_mps_interpolate_3d_matches_cpu_trilinear():
    torch.manual_seed(29)
    input_cpu = torch.randn(1, 2, 4, 5, 6)
    out_cpu = F.interpolate(input_cpu, size=(3, 4, 5), mode="trilinear", align_corners=True)

    out_mps = device_aware_interpolate(
        input_cpu.to("mps"),
        size=(3, 4, 5),
        mode="trilinear",
        align_corners=True,
    )

    assert out_mps.device.type == "mps"
    assert torch.allclose(out_mps.cpu(), out_cpu, atol=2e-5, rtol=2e-5)


def test_mps_warp_composer_3d_matches_cpu_forward_and_v_grad():
    torch.manual_seed(31)
    u_cpu = torch.randn(1, 4, 5, 6, 3) * 0.1
    v_cpu = (torch.randn(1, 3, 4, 5, 3) * 0.02).requires_grad_(True)
    affine_cpu = torch.eye(3, 4).unsqueeze(0)

    out_cpu = torch_warp_composer_3d(u_cpu, affine=affine_cpu, v=v_cpu, align_corners=True)
    out_cpu.square().mean().backward()

    u_mps = u_cpu.to("mps")
    v_mps = v_cpu.detach().to("mps").requires_grad_(True)
    affine_mps = affine_cpu.to("mps")

    out_mps = torch_warp_composer_3d(u_mps, affine=affine_mps, v=v_mps, align_corners=True)
    out_mps.square().mean().backward()

    assert out_mps.device.type == "mps"
    assert v_mps.grad.device.type == "mps"
    assert torch.allclose(out_mps.cpu(), out_cpu.detach(), atol=2e-5, rtol=2e-5)
    assert torch.allclose(v_mps.grad.cpu(), v_cpu.grad, atol=2e-5, rtol=2e-5)
