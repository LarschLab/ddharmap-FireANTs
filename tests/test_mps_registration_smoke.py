import pytest
import torch
import numpy as np
import SimpleITK as sitk

from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration


pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)


def _make_test_image(shift: int = 0):
    array = np.zeros((32, 32, 32), dtype=np.float32)
    array[10:22, 11:23, 12:24] = 1.0
    if shift:
        array = np.roll(array, shift, axis=0)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _make_batches():
    fixed = Image(_make_test_image(0), device="mps")
    moving = Image(_make_test_image(1), device="mps")
    return BatchedImages([fixed]), BatchedImages([moving])


def test_mps_affine_registration_smoke():
    fixed, moving = _make_batches()
    reg = AffineRegistration(
        [1],
        [1],
        fixed,
        moving,
        loss_type="mse",
        optimizer="Adam",
        optimizer_lr=1e-3,
        blur=False,
        progress_bar=False,
    )
    reg.optimize()
    moved = reg.evaluate(fixed, moving)

    assert moved.device.type == "mps"
    assert reg.affine.grad is not None
    assert reg.affine.grad.device.type == "mps"
    assert torch.isfinite(moved).all()


def test_mps_greedy_registration_smoke():
    fixed, moving = _make_batches()
    reg = GreedyRegistration(
        [1],
        [1],
        fixed,
        moving,
        loss_type="mse",
        deformation_type="compositive",
        optimizer="adam",
        optimizer_lr=0.1,
        smooth_grad_sigma=0.0,
        smooth_warp_sigma=0.0,
        blur=False,
        progress_bar=False,
    )
    reg.optimize()
    moved = reg.evaluate(fixed, moving)

    assert moved.device.type == "mps"
    assert reg.warp.warp.grad is not None
    assert reg.warp.warp.grad.device.type == "mps"
    assert torch.isfinite(moved).all()
