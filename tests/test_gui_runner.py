import csv
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from fireants.gui.runner import RegistrationJob, RegistrationSettings, run_batch_registration
from fireants.gui.runner import (
    PIPELINE_ANTS_RIGID_AFFINE_SYN,
    _emit_preview_update,
    _profile_corr,
    _winsorize_tensor,
    parse_float_list,
    registration_settings_for_profile,
    validate_registration_settings,
)
from fireants.io.image import BatchedImages, FakeBatchedImages, Image
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


def test_run_batch_registration_uses_configured_warped_image_extension(tmp_path):
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
        output_image_extension=".nrrd",
        progress_bar=False,
        metrics_dir=metrics_dir,
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[])],
        output,
        settings=settings,
    )

    row_dir = output / "moving_bridge"
    warped_path = row_dir / "moving_bridge_warped.nrrd"
    assert warped_path.exists()
    assert not (row_dir / "moving_bridge_warped.nii.gz").exists()
    assert (row_dir / "moving_bridge_0Warp.nii.gz").exists()
    with (metrics_dir / "pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["warped_bridge_path"].endswith("moving_bridge_warped.nrrd")


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


def test_registration_profile_presets_and_validation():
    current_full = registration_settings_for_profile("Current Full")
    assert current_full.moments_perform_scaling is True
    validate_registration_settings(current_full)

    memory_saver = registration_settings_for_profile("Memory Saver")
    assert memory_saver.loss_type == "mse"
    assert memory_saver.moments_scale == 8
    assert memory_saver.moments_perform_scaling is True
    assert memory_saver.greedy_scales == [4, 2]
    validate_registration_settings(memory_saver)

    ants_like = registration_settings_for_profile("ANTs-like SyN")
    assert ants_like.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN
    assert ants_like.winsorize_enabled
    assert ants_like.rigid_loss_type == "mse"
    assert ants_like.affine_loss_type == "mse"
    assert ants_like.syn_loss_type == "cc"
    assert ants_like.syn_optimizer == "SGD"
    assert ants_like.syn_optimizer_params == {"compose_n": 1}
    validate_registration_settings(ants_like)

    debug = registration_settings_for_profile("Debug")
    assert debug.affine_iterations == [1]
    assert debug.greedy_iterations == [1]
    assert debug.moments_perform_scaling is True
    validate_registration_settings(debug)


def test_validate_registration_settings_rejects_invalid_lists():
    settings = RegistrationSettings(
        device="cpu",
        affine_scales=[1, 2],
        affine_iterations=[1, 1],
        greedy_scales=[1],
        greedy_iterations=[1],
    )
    with pytest.raises(ValueError, match="Affine scales must be strictly decreasing"):
        validate_registration_settings(settings)

    settings.affine_scales = [2, 1]
    settings.affine_iterations = [1]
    with pytest.raises(ValueError, match="same length"):
        validate_registration_settings(settings)

    with pytest.raises(ValueError, match="comma-separated"):
        parse_float_list("4, nope", "Affine scales")


def test_validate_registration_settings_rejects_scaling_without_second_order_moments():
    settings = RegistrationSettings(device="cpu", moments_order=1, moments_perform_scaling=True)

    with pytest.raises(ValueError, match="Moments scaling requires moments order 2"):
        validate_registration_settings(settings)


def test_validate_registration_settings_rejects_unknown_warped_output_extension():
    settings = RegistrationSettings(device="cpu", output_image_extension=".png")

    with pytest.raises(ValueError, match="Warped output extension"):
        validate_registration_settings(settings)


def test_z_profile_correlation_distinguishes_reversed_stack():
    moving_profile = np.array([0.0, 0.2, 0.6, 1.0], dtype=np.float64)
    moved_direct = moving_profile.copy()
    moved_reversed = moving_profile[::-1].copy()

    assert _profile_corr(moved_direct, moving_profile) > _profile_corr(moved_direct, moving_profile[::-1])
    assert _profile_corr(moved_reversed, moving_profile[::-1]) > _profile_corr(moved_reversed, moving_profile)


def test_run_batch_registration_emits_preview_without_metrics_payload(tmp_path):
    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    metrics_dir = output / "_metrics"
    _write_image(fixed)
    _write_image(moving, offset=1)
    events = []

    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        moments_scale=8,
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        preview_enabled=True,
        preview_max_side=16,
        progress_bar=False,
        metrics_dir=metrics_dir,
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[])],
        output,
        settings=settings,
        progress_callback=events.append,
    )

    preview_events = [event for event in events if event["event"] == "preview_update"]
    assert preview_events
    preview = preview_events[0]["preview"]
    assert preview.dtype == np.uint8
    assert preview.ndim == 3
    assert preview.shape[2] == 3

    metric_events = [json.loads(line) for line in (metrics_dir / "events.jsonl").read_text().splitlines()]
    assert not any(event["event"] == "preview_update" for event in metric_events)


def test_run_batch_registration_passes_moments_scaling_setting(tmp_path, monkeypatch):
    from fireants.gui import runner as gui_runner

    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    _write_image(fixed)
    _write_image(moving, offset=1)
    captured = []
    original_init = gui_runner.MomentsRegistration.__init__

    def wrapped_init(self, *args, **kwargs):
        captured.append(kwargs.get("perform_scaling"))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(gui_runner.MomentsRegistration, "__init__", wrapped_init)
    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        moments_scale=8,
        moments_perform_scaling=False,
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        preview_enabled=False,
        progress_bar=False,
    )

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[])],
        output,
        settings=settings,
    )

    assert captured == [False]


def test_emit_preview_update_skips_syn_preview_work():
    class SyNRegistration:
        pass

    events = []
    settings = RegistrationSettings(device="cpu", preview_enabled=True)

    _emit_preview_update(SyNRegistration(), None, None, settings, 0, "SyN", 1, events.append)

    assert events == []


def test_run_batch_registration_ants_like_syn_profile_smoke(tmp_path):
    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    metrics_dir = output / "_metrics"
    _write_image(fixed)
    _write_image(moving, offset=1)
    events = []
    settings = registration_settings_for_profile("ANTs-like SyN")
    settings.device = "cpu"
    settings.rigid_scales = [1]
    settings.rigid_iterations = [1]
    settings.affine_scales = [1]
    settings.affine_iterations = [1]
    settings.syn_scales = [1]
    settings.syn_iterations = [1]
    settings.rigid_loss_type = "mse"
    settings.affine_loss_type = "mse"
    settings.syn_loss_type = "mse"
    settings.metrics_dir = metrics_dir
    settings.preview_enabled = False

    run_batch_registration(
        fixed,
        [RegistrationJob(moving_bridge=moving, warp_files=[])],
        output,
        settings=settings,
        progress_callback=events.append,
    )

    row_dir = output / "moving_bridge"
    warped_path = row_dir / "moving_bridge_warped.nii.gz"
    assert (row_dir / "moving_bridge_0Warp.nii.gz").exists()
    assert warped_path.exists()
    warped = sitk.GetArrayFromImage(sitk.ReadImage(str(warped_path)))
    assert np.isfinite(warped).all()
    assert np.count_nonzero(warped) > 0
    stages = [event.get("stage") for event in events if event.get("event") == "registration_progress"]
    assert "Rigid" in stages
    assert "Affine" in stages
    assert "SyN" in stages
    with (metrics_dir / "pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pipeline"] == PIPELINE_ANTS_RIGID_AFFINE_SYN
    assert rows[0]["rigid_seconds"] != ""
    assert rows[0]["syn_seconds"] != ""
    assert np.isfinite(float(rows[0]["final_mse"]))
    assert np.isfinite(float(rows[0]["final_ncc"]))


def test_run_batch_registration_rejects_invalid_warped_output(tmp_path, monkeypatch):
    from fireants.gui import runner as gui_runner

    fixed = tmp_path / "fixed_bridge.mha"
    moving = tmp_path / "moving_bridge.mha"
    output = tmp_path / "out"
    metrics_dir = output / "_metrics"
    _write_image(fixed)
    _write_image(moving, offset=1)
    events = []

    def zero_evaluate(self, fixed_images, moving_images, *args, **kwargs):
        return torch.zeros_like(fixed_images())

    monkeypatch.setattr(gui_runner.GreedyRegistration, "evaluate", zero_evaluate)
    settings = RegistrationSettings(
        device="cpu",
        loss_type="mse",
        affine_scales=[1],
        affine_iterations=[1],
        greedy_scales=[1],
        greedy_iterations=[1],
        preview_enabled=False,
        progress_bar=False,
        metrics_dir=metrics_dir,
    )

    with pytest.raises(ValueError, match="all zero"):
        run_batch_registration(
            fixed,
            [RegistrationJob(moving_bridge=moving, warp_files=[])],
            output,
            settings=settings,
            progress_callback=events.append,
        )

    warped_path = output / "moving_bridge" / "moving_bridge_warped.nii.gz"
    assert not warped_path.exists()
    assert any(event["event"] == "warp_validation_failed" for event in events)
    with (metrics_dir / "pairs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "failed"
    assert "all zero" in rows[0]["error"]


def test_winsorize_tensor_samples_large_inputs():
    array = torch.arange(2000, dtype=torch.float32)

    actual = _winsorize_tensor(array, 0.05, 0.95, max_samples=1000)

    sample = array[::2]
    lo = torch.quantile(sample, 0.05)
    hi = torch.quantile(sample, 0.95)
    assert torch.isclose(actual.min(), lo)
    assert torch.isclose(actual.max(), hi)
    assert actual.shape == array.shape


def test_fake_batched_images_metadata_follows_tensor_device(tmp_path):
    path = tmp_path / "image.mha"
    _write_image(path)
    image = Image.load_file(str(path), device="cpu")
    batch = BatchedImages([image])
    fake = FakeBatchedImages(batch().clone(), batch)

    assert fake.get_torch2phy().device == fake().device
    assert fake.get_phy2torch().device == fake().device


def test_gui_run_output_dir_is_timestamped(tmp_path):
    pytest.importorskip("PySide6")
    from fireants.gui.app import make_run_output_dir

    run_dir = make_run_output_dir(tmp_path)

    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("fireants_gui_run_")
    assert not run_dir.exists()


def test_gui_profile_fields_build_settings_offscreen(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from fireants.gui.app import MainWindow

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.profile_combo.setCurrentText("Memory Saver")
    window.apply_profile_preset("Memory Saver")
    settings = window._settings_from_profile_fields()

    assert settings.loss_type == "mse"
    assert settings.greedy_scales == [4, 2]
    assert settings.moments_perform_scaling is True
    assert settings.output_image_extension == ".nii.gz"
    assert settings.preview_enabled
    window.output_format_combo.setCurrentText(".nrrd")
    settings = window._settings_from_profile_fields()
    assert settings.output_image_extension == ".nrrd"
    assert window.sidebar_tabs.tabText(0) == "Queue"
    assert window.sidebar_tabs.tabText(1) == "Profile"
    assert window.loss_combo.parent() is not window
    assert "scale differences" in window.moments_scaling_check.toolTip()
    assert "Similarity metric" in window.loss_combo.toolTip()
    assert "Compute device" in window.device_edit.toolTip()
    window.moments_scaling_check.setChecked(False)
    settings = window._settings_from_profile_fields()
    assert settings.moments_perform_scaling is False
    window.sidebar_tabs.setCurrentIndex(0)
    window.advanced_profile_button.click()
    assert window.sidebar_tabs.currentIndex() == 1

    window.profile_combo.setCurrentText("ANTs-like SyN")
    ants_settings = window._settings_from_profile_fields()
    assert ants_settings.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN
    assert not window.rigid_group.isHidden()
    assert not window.syn_group.isHidden()
    assert window.moments_group.isHidden()
    assert window.greedy_group.isHidden()
    window.close()


def test_gui_help_hover_preserves_status_bar_message(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEvent, QSettings
    from PySide6.QtWidgets import QApplication
    from fireants.gui.app import MainWindow

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    message = "Stack 1/2: SyN scale 2 iter 20/50"
    window.statusBar().showMessage(message)

    assert "Compute device" in window.device_edit.toolTip()
    assert window.device_edit.statusTip() == ""
    assert window.add_fixed_action.statusTip() == ""
    QApplication.sendEvent(window.device_edit, QEvent(QEvent.Type.Enter))
    app.processEvents()

    assert window.statusBar().currentMessage() == message
    window.close()


def test_gui_drop_mime_parsing_accepts_supported_local_images(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QMimeData, QUrl
    from fireants.gui.app import image_paths_from_mime

    supported = tmp_path / "moving.nii.gz"
    unsupported = tmp_path / "notes.txt"
    supported.write_text("")
    unsupported.write_text("")
    mime = QMimeData()
    mime.setUrls(
        [
            QUrl.fromLocalFile(str(supported)),
            QUrl.fromLocalFile(str(unsupported)),
            QUrl("https://example.invalid/fixed.nrrd"),
        ]
    )

    assert image_paths_from_mime(mime) == [supported]


def test_gui_drop_handlers_update_fixed_and_moving_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from fireants.gui.app import MainWindow

    fixed = tmp_path / "fixed.mha"
    moving_a = tmp_path / "moving_a.mha"
    moving_b = tmp_path / "moving_b.nrrd"
    extra_fixed = tmp_path / "extra_fixed.mha"
    for path in (fixed, moving_a, moving_b, extra_fixed):
        _write_image(path)

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.fixed_preview.acceptDrops()
    assert window.moving_preview.acceptDrops()
    assert window.job_list.acceptDrops()

    window.set_fixed_bridge_from_drop([fixed, extra_fixed])
    assert window.fixed_path_edit.text() == str(fixed)
    assert window.settings_store.value("fixed_bridge") == str(fixed)
    assert "ignored 1 additional" in window.statusBar().currentMessage()

    window.add_moving_bridge_paths([moving_a, moving_b])
    assert [job.moving_bridge for job in window.jobs] == [moving_a, moving_b]
    assert window.job_list.count() == 2
    assert window.job_list.currentRow() == 0
    assert window.moving_preview.toolTip() == str(moving_a)
    assert not window.moving_preview.pixmap().isNull()

    window._set_running(True)
    assert not window.fixed_preview._drop_enabled()
    assert not window.moving_preview._drop_enabled()
    assert not window.job_list._drop_enabled()
    window._set_running(False)
    window.close()


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


def test_rigid_registration_rejects_nonfinite_loss(tmp_path):
    class NaNLoss(torch.nn.Module):
        def forward(self, pred, target):
            return pred.sum() * torch.tensor(float("nan"), device=pred.device)

    fixed = tmp_path / "fixed.mha"
    moving = tmp_path / "moving.mha"
    _write_image(fixed)
    _write_image(moving, offset=1)
    fixed_batch = BatchedImages([Image.load_file(str(fixed), device="cpu")])
    moving_batch = BatchedImages([Image.load_file(str(moving), device="cpu")])

    reg = RigidRegistration(
        scales=[1],
        iterations=[1],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="custom",
        custom_loss=NaNLoss(),
        progress_bar=False,
    )

    with pytest.raises(FloatingPointError, match="Rigid produced non-finite loss"):
        reg.optimize()
