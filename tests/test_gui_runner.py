import csv
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from fireants.gui.runner import RegistrationJob, RegistrationSettings, run_batch_registration
from fireants.io.image import BatchedImages, Image
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration


def _write_image(path: Path, offset: int = 0) -> None:
    array = np.zeros((32, 32, 32), dtype=np.float32)
    array[8 + offset : 24 + offset, 9:23, 9:23] = 1.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def test_run_batch_registration_warps_all_selected_inputs(tmp_path):
    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    payload = tmp_path / "moving_payload.mha"
    output = tmp_path / "out"
    _write_image(fixed)
    _write_image(moving, offset=1)
    _write_image(payload, offset=1)

    events = []
    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        progress_bar=False,
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[payload])],
        output,
        settings=settings,
        progress_callback=events.append,
    )

    row_dir = output / "moving_bridge"
    assert (row_dir / "moving_bridge_warped.nii.gz").exists()
    assert (row_dir / "moving_payload_warped.nii.gz").exists()
    assert (row_dir / "moving_bridge_0Warp.nii.gz").exists()
    assert [event["event"] for event in events if event["event"] == "row_complete"]
    progress_events = [event for event in events if event["event"] == "registration_progress"]
    assert progress_events
    assert progress_events[-1]["completed_units"] <= progress_events[-1]["total_units"]


def test_run_batch_registration_writes_metrics(tmp_path):
    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    metrics_dir = output / "_metrics"
    _write_image(fixed)
    _write_image(moving, offset=1)

    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        progress_bar=False,
        metrics_dir=metrics_dir,
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[moving])],
        output,
        settings=settings,
    )

    events_path = metrics_dir / "events.jsonl"
    pairs_path = metrics_dir / "pairs.csv"
    assert events_path.exists()
    assert pairs_path.exists()
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert any(event["event"] == "batch_complete" for event in events)
    assert any(event["event"] == "warp_complete" and "final_ncc" in event for event in events)
    with pairs_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["device"] == "cpu"
    assert rows[0]["warped_bridge_path"].endswith("moving_bridge_warped.nii.gz")
    assert rows[0]["final_mse"] != ""
    assert rows[0]["final_ncc"] != ""


def test_run_batch_registration_records_failed_row_metrics(tmp_path):
    fixed = tmp_path / "fixed_bridge.mha"
    missing_moving = tmp_path / "missing_bridge.mha"
    output = tmp_path / "out"
    metrics_dir = output / "_metrics"
    _write_image(fixed)
    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        progress_bar=False,
        metrics_dir=metrics_dir,
    )

    with pytest.raises(Exception):
        run_batch_registration(
            fixed,
            [RegistrationJob(moving_bridge=missing_moving, warp_files=[])],
            output,
            settings=settings,
        )

    with (metrics_dir / "pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"]


def test_gui_run_output_dir_is_timestamped(tmp_path):
    pytest.importorskip("PySide6")
    from fireants.gui.app import make_run_output_dir

    run_dir = make_run_output_dir(tmp_path)

    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("fireants_gui_run_")
    assert not run_dir.exists()


def test_moments_registration_orientation_runs_on_mps_when_available(tmp_path):
    import torch

    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")
    fixed = tmp_path / "fixed.mha"
    moving = tmp_path / "moving.mha"
    _write_image(fixed)
    _write_image(moving, offset=1)
    fixed_batch = BatchedImages([Image.load_file(str(fixed), device="mps")])
    moving_batch = BatchedImages([Image.load_file(str(moving), device="mps")])

    reg = MomentsRegistration(
        scale=4,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        moments=2,
        orientation="rot",
        loss_type="mse",
        progress_bar=False,
    )
    reg.optimize()

    assert reg.optimized


def test_greedy_transform_save_runs_on_mps_when_available(tmp_path):
    import torch

    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")
    fixed = tmp_path / "fixed.mha"
    moving = tmp_path / "moving.mha"
    transform = tmp_path / "warp.nii.gz"
    _write_image(fixed)
    _write_image(moving, offset=1)
    fixed_batch = BatchedImages([Image.load_file(str(fixed), device="mps")])
    moving_batch = BatchedImages([Image.load_file(str(moving), device="mps")])
    reg = GreedyRegistration(
        scales=[4],
        iterations=[1],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="mse",
        progress_bar=False,
    )
    reg.optimize()

    reg.save_as_ants_transforms(str(transform))

    assert transform.exists()


def test_run_batch_registration_writes_outputs_on_mps_when_available(tmp_path):
    import torch

    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")
    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    _write_image(fixed)
    _write_image(moving, offset=1)
    settings = RegistrationSettings(
        device="mps",
        loss_type="mse",
        moments_scale=4,
        affine_scales=[4],
        affine_iterations=[1],
        greedy_scales=[4],
        greedy_iterations=[1],
        progress_bar=False,
        metrics_dir=output / "_metrics",
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[])],
        output,
        settings=settings,
    )

    row_dir = output / "moving_bridge"
    assert (row_dir / "moving_bridge_0Warp.nii.gz").exists()
    assert (row_dir / "moving_bridge_warped.nii.gz").exists()


def test_mps_grid_sample_3d_chunked_matches_cpu_when_available():
    import torch
    from torch.nn import functional as F
    from fireants.interpolator.grid_sample import _mps_grid_sample_3d

    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")
    input_cpu = torch.arange(1 * 1 * 5 * 6 * 7, dtype=torch.float32).reshape(1, 1, 5, 6, 7) / 100.0
    affine = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    grid_cpu = F.affine_grid(affine, input_cpu.shape, align_corners=True)

    actual = _mps_grid_sample_3d(
        input_cpu.to("mps"),
        grid_cpu.to("mps"),
        align_corners=True,
        max_chunk_voxels=20,
    ).cpu()
    expected = F.grid_sample(input_cpu, grid_cpu, align_corners=True)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_rigid_registration_emits_progress_events(tmp_path):
    fixed = tmp_path / "fixed.mha"
    moving = tmp_path / "moving.mha"
    _write_image(fixed)
    _write_image(moving, offset=1)
    fixed_batch = BatchedImages([Image.load_file(str(fixed), device="cpu")])
    moving_batch = BatchedImages([Image.load_file(str(moving), device="cpu")])
    events = []

    reg = RigidRegistration(
        scales=[1],
        iterations=[1],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="mse",
        progress_bar=False,
        progress_callback=events.append,
    )

    reg.optimize()

    assert events[0]["event"] == "stage_start"
    assert any(event["event"] == "iteration" for event in events)
    assert events[-1]["event"] == "stage_complete"
