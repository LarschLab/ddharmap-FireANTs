# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from fireants.io.image import BatchedImages, FakeBatchedImages, Image
from fireants.interpolator.grid_sample import device_aware_interpolate
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration
from fireants.registration.syn import SyNRegistration


RunnerProgressCallback = Optional[Callable[[Dict[str, Any]], None]]

PIPELINE_MOMENTS_AFFINE_GREEDY = "moments_affine_greedy"
PIPELINE_ANTS_RIGID_AFFINE_SYN = "ants_rigid_affine_syn"
REGISTRATION_PROFILE_NAMES = ("Current Full", "Memory Saver", "ANTs-like SyN", "Debug")
SUPPORTED_LOSS_TYPES = ("cc", "mse", "mi", "fusedcc", "fusedmi", "noop")
SUPPORTED_MOMENTS_ORIENTATIONS = ("rot", "antirot", "both")
SUPPORTED_GREEDY_OPTIMIZERS = ("Adam", "SGD")
SUPPORTED_SYN_OPTIMIZERS = ("Adam", "SGD")


@dataclass
class RegistrationSettings:
    device: str = "cuda:0"
    pipeline: str = PIPELINE_MOMENTS_AFFINE_GREEDY
    loss_type: str = "cc"
    cc_kernel_size: int = 5
    winsorize_enabled: bool = False
    winsorize_lower: float = 0.05
    winsorize_upper: float = 0.95
    moments_scale: float = 1.0
    moments_order: int = 2
    moments_orientation: str = "rot"
    rigid_loss_type: str = "mi"
    rigid_mi_bins: int = 32
    rigid_scales: List[float] = field(default_factory=lambda: [12, 8, 4, 2])
    rigid_iterations: List[int] = field(default_factory=lambda: [100, 100, 75, 25])
    rigid_lr: float = 3e-2
    affine_loss_type: str = "cc"
    affine_mi_bins: int = 32
    affine_scales: List[float] = field(default_factory=lambda: [4, 2, 1])
    affine_iterations: List[int] = field(default_factory=lambda: [100, 50, 25])
    affine_lr: float = 3e-3
    greedy_scales: List[float] = field(default_factory=lambda: [4, 2, 1])
    greedy_iterations: List[int] = field(default_factory=lambda: [100, 50, 25])
    greedy_lr: float = 0.5
    greedy_optimizer: str = "Adam"
    greedy_reset: bool = False
    greedy_offload: bool = False
    syn_loss_type: str = "cc"
    syn_cc_kernel_size: int = 7
    syn_scales: List[float] = field(default_factory=lambda: [12, 8, 4, 2, 1])
    syn_iterations: List[int] = field(default_factory=lambda: [100, 100, 75, 50, 10])
    syn_lr: float = 0.1
    syn_optimizer: str = "Adam"
    syn_optimizer_params: Dict[str, Any] = field(default_factory=dict)
    smooth_warp_sigma: float = 0.5
    smooth_grad_sigma: float = 1.0
    preview_enabled: bool = True
    preview_max_side: int = 320
    progress_bar: bool = False
    metrics_dir: Optional[Path] = None

    @classmethod
    def default(cls) -> "RegistrationSettings":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return cls(device=device)

    @property
    def total_registration_iterations(self) -> int:
        if self.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN:
            return sum(self.rigid_iterations) + sum(self.affine_iterations) + sum(self.syn_iterations)
        return 1 + sum(self.affine_iterations) + sum(self.greedy_iterations)


def registration_settings_for_profile(profile_name: str) -> RegistrationSettings:
    settings = RegistrationSettings.default()
    if profile_name == "Current Full":
        return settings
    if profile_name == "Memory Saver":
        settings.loss_type = "mse"
        settings.moments_scale = 8
        settings.affine_scales = [4, 2, 1]
        settings.affine_iterations = [50, 25, 10]
        settings.greedy_scales = [4, 2]
        settings.greedy_iterations = [50, 25]
        return settings
    if profile_name == "ANTs-like SyN":
        settings.pipeline = PIPELINE_ANTS_RIGID_AFFINE_SYN
        settings.winsorize_enabled = True
        settings.rigid_loss_type = "mi"
        settings.rigid_mi_bins = 32
        settings.rigid_scales = [12, 8, 4, 2]
        settings.rigid_iterations = [100, 100, 75, 25]
        settings.rigid_lr = 3e-2
        settings.affine_loss_type = "mi"
        settings.affine_mi_bins = 32
        settings.affine_scales = [12, 8, 4, 2]
        settings.affine_iterations = [100, 100, 75, 25]
        settings.affine_lr = 3e-3
        settings.syn_loss_type = "cc"
        settings.syn_cc_kernel_size = 7
        settings.syn_scales = [12, 8, 4, 2, 1]
        settings.syn_iterations = [100, 100, 75, 50, 10]
        settings.syn_lr = 0.1
        settings.syn_optimizer = "SGD"
        settings.syn_optimizer_params = {"compose_n": 1}
        settings.smooth_warp_sigma = 0.5
        settings.smooth_grad_sigma = 1.0
        return settings
    if profile_name == "Debug":
        settings.loss_type = "mse"
        settings.moments_scale = 8
        settings.affine_scales = [8]
        settings.affine_iterations = [1]
        settings.greedy_scales = [8]
        settings.greedy_iterations = [1]
        return settings
    raise ValueError(f"Unknown registration profile: {profile_name}")


def parse_float_list(raw_value: str, field_name: str) -> List[float]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a comma-separated list of numbers") from exc


def parse_int_list(raw_value: str, field_name: str) -> List[int]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    try:
        parsed = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a comma-separated list of integers") from exc
    return parsed


def validate_registration_settings(settings: RegistrationSettings) -> None:
    if settings.pipeline not in {PIPELINE_MOMENTS_AFFINE_GREEDY, PIPELINE_ANTS_RIGID_AFFINE_SYN}:
        raise ValueError("Registration pipeline is not supported")
    if not _supported_loss_type(settings.loss_type):
        raise ValueError(f"Loss type must be one of {', '.join(SUPPORTED_LOSS_TYPES)}")
    if not _supported_loss_type(settings.rigid_loss_type):
        raise ValueError(f"Rigid loss type must be one of {', '.join(SUPPORTED_LOSS_TYPES)}")
    if not _supported_loss_type(settings.affine_loss_type):
        raise ValueError(f"Affine loss type must be one of {', '.join(SUPPORTED_LOSS_TYPES)}")
    if not _supported_loss_type(settings.syn_loss_type):
        raise ValueError(f"SyN loss type must be one of {', '.join(SUPPORTED_LOSS_TYPES)}")
    if int(settings.cc_kernel_size) <= 0:
        raise ValueError("CC kernel size must be positive")
    if int(settings.rigid_mi_bins) <= 0:
        raise ValueError("Rigid MI bins must be positive")
    if int(settings.affine_mi_bins) <= 0:
        raise ValueError("Affine MI bins must be positive")
    if int(settings.syn_cc_kernel_size) <= 0:
        raise ValueError("SyN CC kernel size must be positive")
    if not 0 <= float(settings.winsorize_lower) < float(settings.winsorize_upper) <= 1:
        raise ValueError("Winsorize quantiles must satisfy 0 <= lower < upper <= 1")
    if float(settings.moments_scale) <= 0:
        raise ValueError("Moments scale must be positive")
    if int(settings.moments_order) <= 0:
        raise ValueError("Moments order must be positive")
    if settings.moments_orientation not in SUPPORTED_MOMENTS_ORIENTATIONS:
        raise ValueError(f"Moments orientation must be one of {', '.join(SUPPORTED_MOMENTS_ORIENTATIONS)}")
    _validate_scales_and_iterations(settings.rigid_scales, settings.rigid_iterations, "Rigid")
    _validate_scales_and_iterations(settings.affine_scales, settings.affine_iterations, "Affine")
    _validate_scales_and_iterations(settings.greedy_scales, settings.greedy_iterations, "Greedy")
    _validate_scales_and_iterations(settings.syn_scales, settings.syn_iterations, "SyN")
    if float(settings.rigid_lr) <= 0:
        raise ValueError("Rigid learning rate must be positive")
    if float(settings.affine_lr) <= 0:
        raise ValueError("Affine learning rate must be positive")
    if float(settings.greedy_lr) <= 0:
        raise ValueError("Greedy learning rate must be positive")
    if float(settings.syn_lr) <= 0:
        raise ValueError("SyN learning rate must be positive")
    if settings.greedy_optimizer.lower() not in {optimizer.lower() for optimizer in SUPPORTED_GREEDY_OPTIMIZERS}:
        raise ValueError(f"Greedy optimizer must be one of {', '.join(SUPPORTED_GREEDY_OPTIMIZERS)}")
    if settings.syn_optimizer.lower() not in {optimizer.lower() for optimizer in SUPPORTED_SYN_OPTIMIZERS}:
        raise ValueError(f"SyN optimizer must be one of {', '.join(SUPPORTED_SYN_OPTIMIZERS)}")
    if float(settings.smooth_warp_sigma) < 0:
        raise ValueError("Smooth warp sigma must be non-negative")
    if float(settings.smooth_grad_sigma) < 0:
        raise ValueError("Smooth grad sigma must be non-negative")
    if int(settings.preview_max_side) <= 0:
        raise ValueError("Preview max side must be positive")


def _validate_scales_and_iterations(scales: List[float], iterations: List[int], stage: str) -> None:
    if len(scales) != len(iterations):
        raise ValueError(f"{stage} scales and iterations must have the same length")
    if not scales:
        raise ValueError(f"{stage} scales must contain at least one value")
    if any(float(scale) <= 0 for scale in scales):
        raise ValueError(f"{stage} scales must be positive")
    if any(scales[index] <= scales[index + 1] for index in range(len(scales) - 1)):
        raise ValueError(f"{stage} scales must be strictly decreasing")
    if any(int(iteration) <= 0 for iteration in iterations):
        raise ValueError(f"{stage} iterations must be positive")


def _supported_loss_type(loss_type: str) -> bool:
    if loss_type in SUPPORTED_LOSS_TYPES:
        return True
    if loss_type.startswith("masked_"):
        return loss_type.replace("masked_", "", 1) in SUPPORTED_LOSS_TYPES
    return False


@dataclass
class RegistrationJob:
    moving_bridge: Path
    warp_files: List[Path]

    def all_warp_files(self) -> List[Path]:
        paths = [self.moving_bridge] + list(self.warp_files)
        deduped = []
        seen = set()
        for path in paths:
            resolved = str(path)
            if resolved not in seen:
                deduped.append(path)
                seen.add(resolved)
        return deduped


def run_batch_registration(
    fixed_bridge: Path,
    jobs: Iterable[RegistrationJob],
    output_dir: Path,
    settings: Optional[RegistrationSettings] = None,
    progress_callback: RunnerProgressCallback = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> None:
    settings = settings or RegistrationSettings.default()
    validate_registration_settings(settings)
    jobs = list(jobs)
    fixed_bridge = Path(fixed_bridge)
    output_dir = Path(output_dir)
    total_units = sum(settings.total_registration_iterations + len(job.all_warp_files()) for job in jobs)
    completed_units = 0
    metrics = _MetricsRecorder(settings.metrics_dir, fixed_bridge, output_dir, settings) if settings.metrics_dir else None

    def emit(record_metrics: bool = True, **event: Any) -> None:
        if metrics is not None and record_metrics:
            metrics.record_event(event)
        if progress_callback is not None:
            progress_callback(event)

    def check_cancelled() -> None:
        if cancel_requested is not None and cancel_requested():
            raise RuntimeError("Registration cancelled")

    try:
        emit(
            event="batch_start",
            total_jobs=len(jobs),
            total_units=total_units,
            output_dir=str(output_dir),
            metrics_dir=str(settings.metrics_dir) if settings.metrics_dir else None,
        )
        fixed_image = Image.load_file(str(fixed_bridge), device=settings.device)
        fixed_batch = BatchedImages([fixed_image])
        emit(event="fixed_loaded", path=str(fixed_bridge), **_image_metadata(fixed_batch))

        for row_index, job in enumerate(jobs):
            check_cancelled()
            moving_bridge = Path(job.moving_bridge)
            row_output_dir = output_dir / _file_stem(moving_bridge)
            row_output_dir.mkdir(parents=True, exist_ok=True)
            row_total_units = settings.total_registration_iterations + len(job.all_warp_files())
            row_completed_units = 0
            stage_totals = _stage_totals(settings)
            stage_completed = {stage: 0 for stage in stage_totals}
            current_registration = None

            emit(
                event="row_start",
                row_index=row_index,
                total_jobs=len(jobs),
                moving_bridge=str(moving_bridge),
                row_total_units=row_total_units,
                output_dir=str(row_output_dir),
            )

            def reg_progress(event: Dict[str, Any]) -> None:
                nonlocal completed_units, row_completed_units, current_registration
                registration_event = event.get("event")
                stage = event.get("stage")
                if registration_event == "iteration" and stage in stage_completed:
                    stage_completed[stage] += 1
                    completed_units += 1
                    row_completed_units += 1
                elif registration_event == "stage_complete" and stage in stage_completed:
                    remaining = max(stage_totals[stage] - stage_completed[stage], 0)
                    stage_completed[stage] += remaining
                    completed_units += remaining
                    row_completed_units += remaining

                forwarded = dict(event)
                forwarded.update(
                    {
                        "event": "registration_progress",
                        "registration_event": registration_event,
                        "row_index": row_index,
                        "total_jobs": len(jobs),
                        "moving_bridge": str(moving_bridge),
                        "completed_units": completed_units,
                        "total_units": total_units,
                        "row_completed_units": row_completed_units,
                        "row_total_units": row_total_units,
                    }
                )
                emit(**forwarded)
                should_preview = (
                    (stage == "Moments" and registration_event == "stage_complete")
                    or (stage in ("Rigid", "Affine", "Greedy", "SyN") and registration_event == "scale_complete")
                )
                if should_preview:
                    _emit_preview_update(
                        current_registration,
                        fixed_batch,
                        moving_batch,
                        settings,
                        row_index,
                        str(stage),
                        event.get("scale"),
                        emit,
                    )
                check_cancelled()

            try:
                moving_original_image = Image.load_file(str(moving_bridge), device=settings.device)
                moving_original_batch = BatchedImages([moving_original_image])
                if settings.winsorize_enabled:
                    fixed_reg_image = _winsorized_image(Image.load_file(str(fixed_bridge), device=settings.device), settings)
                    moving_image = _winsorized_image(Image.load_file(str(moving_bridge), device=settings.device), settings)
                    moving_batch = BatchedImages([moving_image])
                    fixed_reg_batch = BatchedImages([fixed_reg_image])
                else:
                    moving_image = moving_original_image
                    fixed_reg_image = fixed_image
                    moving_batch = moving_original_batch
                    fixed_reg_batch = fixed_batch
                emit(event="moving_loaded", row_index=row_index, path=str(moving_bridge), **_image_metadata(moving_batch))

                if settings.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN:
                    rigid = RigidRegistration(
                        scales=settings.rigid_scales,
                        iterations=settings.rigid_iterations,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        loss_type=settings.rigid_loss_type,
                        loss_params=_loss_params(settings.rigid_loss_type, settings.rigid_mi_bins),
                        optimizer_lr=settings.rigid_lr,
                        init_translation="cof",
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = rigid
                    rigid.optimize()

                    affine = AffineRegistration(
                        scales=settings.affine_scales,
                        iterations=settings.affine_iterations,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        loss_type=settings.affine_loss_type,
                        loss_params=_loss_params(settings.affine_loss_type, settings.affine_mi_bins),
                        cc_kernel_size=settings.cc_kernel_size,
                        optimizer_lr=settings.affine_lr,
                        init_rigid=rigid.get_rigid_matrix(homogenous=False).detach(),
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = affine
                    affine.optimize()

                    reg = SyNRegistration(
                        scales=settings.syn_scales,
                        iterations=settings.syn_iterations,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        loss_type=settings.syn_loss_type,
                        cc_kernel_size=settings.syn_cc_kernel_size,
                        optimizer=settings.syn_optimizer,
                        optimizer_params=settings.syn_optimizer_params,
                        optimizer_lr=settings.syn_lr,
                        smooth_warp_sigma=settings.smooth_warp_sigma,
                        smooth_grad_sigma=settings.smooth_grad_sigma,
                        init_affine=affine.get_affine_matrix().detach(),
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = reg
                    reg.optimize()
                else:
                    moments = MomentsRegistration(
                        scale=settings.moments_scale,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        moments=settings.moments_order,
                        orientation=settings.moments_orientation,
                        loss_type=settings.loss_type,
                        cc_kernel_size=settings.cc_kernel_size,
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = moments
                    moments.optimize()

                    affine = AffineRegistration(
                        scales=settings.affine_scales,
                        iterations=settings.affine_iterations,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        loss_type=settings.loss_type,
                        cc_kernel_size=settings.cc_kernel_size,
                        optimizer_lr=settings.affine_lr,
                        init_rigid=moments.get_affine_init().detach(),
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = affine
                    affine.optimize()

                    greedy_optimizer_params = {}
                    if settings.greedy_optimizer.lower() == "adam":
                        greedy_optimizer_params["reset"] = settings.greedy_reset
                        greedy_optimizer_params["offload"] = settings.greedy_offload
                    reg = GreedyRegistration(
                        scales=settings.greedy_scales,
                        iterations=settings.greedy_iterations,
                        fixed_images=fixed_reg_batch,
                        moving_images=moving_batch,
                        loss_type=settings.loss_type,
                        cc_kernel_size=settings.cc_kernel_size,
                        optimizer=settings.greedy_optimizer,
                        optimizer_params=greedy_optimizer_params,
                        optimizer_lr=settings.greedy_lr,
                        smooth_warp_sigma=settings.smooth_warp_sigma,
                        smooth_grad_sigma=settings.smooth_grad_sigma,
                        init_affine=affine.get_affine_matrix().detach(),
                        progress_bar=settings.progress_bar,
                        progress_callback=reg_progress,
                    )
                    current_registration = reg
                    reg.optimize()

                transform_path = row_output_dir / f"{_file_stem(moving_bridge)}_0Warp.nii.gz"
                reg.save_as_ants_transforms(str(transform_path))
                emit(event="transform_saved", row_index=row_index, path=str(transform_path))

                for source_path in job.all_warp_files():
                    check_cancelled()
                    source_path = Path(source_path)
                    emit(event="warp_start", row_index=row_index, path=str(source_path))
                    source_image = Image.load_file(str(source_path), device=settings.device)
                    source_batch = BatchedImages([source_image])
                    interpolation_device = torch.device("cpu") if settings.device.startswith("mps") else None
                    moved = reg.evaluate(fixed_reg_batch, source_batch, interpolation_device=interpolation_device)
                    moved_batch = FakeBatchedImages(moved, fixed_batch)
                    output_path = row_output_dir / f"{_file_stem(source_path)}_warped.nii.gz"
                    moved_batch.write_image(str(output_path))
                    completed_units += 1
                    row_completed_units += 1
                    quality_metrics = {}
                    if source_path == moving_bridge:
                        quality_metrics = _registration_quality_metrics(fixed_batch, moving_original_batch, moved)
                    emit(
                        event="warp_complete",
                        row_index=row_index,
                        source_path=str(source_path),
                        output_path=str(output_path),
                        completed_units=completed_units,
                        total_units=total_units,
                        row_completed_units=row_completed_units,
                        row_total_units=row_total_units,
                        **quality_metrics,
                    )

                emit(event="row_complete", row_index=row_index, output_dir=str(row_output_dir))
                del moving_batch, moving_image, moving_original_batch, moving_original_image, reg, affine
                if settings.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN:
                    del rigid
                else:
                    del moments
                if settings.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as exc:
                emit(event="row_failed", row_index=row_index, output_dir=str(row_output_dir), error=str(exc))
                raise

        emit(event="batch_complete", completed_units=total_units, total_units=total_units)
    finally:
        if metrics is not None:
            metrics.close()


def _stage_totals(settings: RegistrationSettings) -> Dict[str, int]:
    if settings.pipeline == PIPELINE_ANTS_RIGID_AFFINE_SYN:
        return {
            "Rigid": sum(settings.rigid_iterations),
            "Affine": sum(settings.affine_iterations),
            "SyN": sum(settings.syn_iterations),
        }
    return {
        "Moments": 1,
        "Affine": sum(settings.affine_iterations),
        "Greedy": sum(settings.greedy_iterations),
    }


def _loss_params(loss_type: str, mi_bins: int) -> Dict[str, int]:
    if loss_type in {"mi", "fusedmi"}:
        return {"num_bins": int(mi_bins)}
    return {}


def _winsorized_image(image: Image, settings: RegistrationSettings) -> Image:
    image.array = _winsorize_tensor(image.array, settings.winsorize_lower, settings.winsorize_upper)
    return image


def _winsorize_tensor(array: torch.Tensor, lower: float, upper: float, max_samples: int = 1_000_000) -> torch.Tensor:
    sample = array.detach().float().cpu().flatten()
    if sample.numel() > max_samples:
        step = (sample.numel() + max_samples - 1) // max_samples
        sample = sample[::step]
    lo = torch.quantile(sample, float(lower)).item()
    hi = torch.quantile(sample, float(upper)).item()
    if hi <= lo:
        return array
    return torch.clamp(array, min=lo, max=hi)


def _file_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".tif", ".tiff"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _emit_preview_update(
    registration: Any,
    fixed_batch: BatchedImages,
    moving_batch: BatchedImages,
    settings: RegistrationSettings,
    row_index: int,
    stage: str,
    scale: Any,
    emit: Callable[..., None],
) -> None:
    if not settings.preview_enabled or registration is None:
        return
    if registration.__class__.__name__ == "SyNRegistration":
        return
    try:
        preview = _registration_overlay_preview(registration, fixed_batch, moving_batch, settings)
    except Exception as exc:
        emit(event="preview_failed", row_index=row_index, stage=stage, scale=scale, error=str(exc))
        return
    emit(
        record_metrics=False,
        event="preview_update",
        row_index=row_index,
        stage=stage,
        scale=scale,
        preview=preview,
        height=int(preview.shape[0]),
        width=int(preview.shape[1]),
        channels=3,
        format="rgb888",
    )


def _registration_overlay_preview(
    registration: Any,
    fixed_batch: BatchedImages,
    moving_batch: BatchedImages,
    settings: RegistrationSettings,
) -> np.ndarray:
    preview_shape = _preview_spatial_shape(fixed_batch.shape[2:], settings.preview_max_side)
    interpolation_device = torch.device("cpu") if str(settings.device).startswith("mps") and registration.__class__.__name__ == "GreedyRegistration" else None
    evaluate_shape = preview_shape
    if registration.__class__.__name__ in ("MomentsRegistration", "RigidRegistration", "AffineRegistration"):
        evaluate_shape = [fixed_batch.shape[0], fixed_batch.shape[1], *preview_shape]
    with torch.no_grad():
        moved = registration.evaluate(
            fixed_batch,
            moving_batch,
            shape=evaluate_shape,
            interpolation_device=interpolation_device,
        )
        fixed = fixed_batch()
        if tuple(fixed.shape[2:]) != tuple(preview_shape):
            fixed = device_aware_interpolate(
                fixed,
                size=preview_shape,
                mode=fixed_batch.interpolate_mode,
                align_corners=True,
            )
    return _magenta_green_overlay(fixed, moved)


def _preview_spatial_shape(spatial_shape: Iterable[int], max_side: int) -> List[int]:
    spatial_shape = [int(size) for size in spatial_shape]
    largest = max(spatial_shape)
    if largest <= max_side:
        return spatial_shape
    scale = float(max_side) / float(largest)
    return [max(4, int(round(size * scale))) for size in spatial_shape]


def _magenta_green_overlay(fixed: torch.Tensor, moved: torch.Tensor) -> np.ndarray:
    fixed_slice = _center_slice(_robust_normalize(fixed)[0, 0])
    moved_slice = _center_slice(_robust_normalize(moved)[0, 0])
    fixed_array = (fixed_slice.numpy() * 255).astype(np.uint8)
    moved_array = (moved_slice.numpy() * 255).astype(np.uint8)
    overlay = np.zeros((*fixed_array.shape, 3), dtype=np.uint8)
    overlay[..., 0] = moved_array
    overlay[..., 1] = fixed_array
    overlay[..., 2] = moved_array
    return np.ascontiguousarray(overlay)


def _center_slice(array: torch.Tensor) -> torch.Tensor:
    if array.ndim == 3:
        return array[array.shape[0] // 2]
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported preview dimensions: {tuple(array.shape)}")


class _MetricsRecorder:
    _CSV_FIELDS = [
        "row_index",
        "status",
        "fixed_bridge",
        "moving_bridge",
        "output_dir",
        "device",
        "pipeline",
        "fixed_shape",
        "fixed_spacing",
        "moving_shape",
        "moving_spacing",
        "moments_seconds",
        "rigid_seconds",
        "affine_seconds",
        "greedy_seconds",
        "syn_seconds",
        "total_seconds",
        "transform_path",
        "warped_bridge_path",
        "initial_mse",
        "final_mse",
        "initial_mae",
        "final_mae",
        "initial_ncc",
        "final_ncc",
        "error",
    ]

    def __init__(self, metrics_dir: Path, fixed_bridge: Path, output_dir: Path, settings: RegistrationSettings) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.metrics_dir / "events.jsonl"
        self.pairs_path = self.metrics_dir / "pairs.csv"
        self.fixed_bridge = str(fixed_bridge)
        self.output_dir = str(output_dir)
        self.settings = settings
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self._rows: Dict[int, Dict[str, Any]] = {}
        self._stage_starts: Dict[tuple, float] = {}
        self._fixed_metadata: Dict[str, Any] = {}

    def record_event(self, event: Dict[str, Any]) -> None:
        now = time.perf_counter()
        stamped = {"timestamp": time.time(), **event}
        self._events_file.write(json.dumps(_json_ready(stamped), sort_keys=True) + "\n")
        self._events_file.flush()
        self._record_summary_event(event, now)

    def close(self) -> None:
        self._events_file.close()
        self._write_pairs_csv()

    def _record_summary_event(self, event: Dict[str, Any], now: float) -> None:
        name = event.get("event")
        row_index = event.get("row_index")
        if row_index is None:
            if name == "fixed_loaded":
                self._fixed_metadata = {
                    "fixed_shape": event.get("shape"),
                    "fixed_spacing": event.get("spacing"),
                }
                for row in self._rows.values():
                    row.update(self._fixed_metadata)
            return
        row_index = int(row_index)
        row = self._rows.setdefault(
            row_index,
            {
                "row_index": row_index,
                "status": "running",
                "fixed_bridge": self.fixed_bridge,
                "output_dir": self.output_dir,
                "device": self.settings.device,
                "pipeline": self.settings.pipeline,
                **self._fixed_metadata,
                "moments_seconds": 0.0,
                "rigid_seconds": 0.0,
                "affine_seconds": 0.0,
                "greedy_seconds": 0.0,
                "syn_seconds": 0.0,
                "error": "",
            },
        )
        if name == "row_start":
            row["moving_bridge"] = event.get("moving_bridge")
            row["output_dir"] = event.get("output_dir")
            row["_start"] = now
        elif name == "moving_loaded":
            row["moving_shape"] = event.get("shape")
            row["moving_spacing"] = event.get("spacing")
        elif name == "registration_progress":
            stage = str(event.get("stage", "")).lower()
            registration_event = event.get("registration_event")
            if stage in ("moments", "rigid", "affine", "greedy", "syn") and registration_event == "stage_start":
                self._stage_starts[(row_index, stage)] = now
            elif stage in ("moments", "rigid", "affine", "greedy", "syn") and registration_event == "stage_complete":
                started = self._stage_starts.pop((row_index, stage), now)
                row[f"{stage}_seconds"] = row.get(f"{stage}_seconds", 0.0) + max(now - started, 0.0)
        elif name == "transform_saved":
            row["transform_path"] = event.get("path")
        elif name == "warp_complete":
            if event.get("source_path") == row.get("moving_bridge"):
                row["warped_bridge_path"] = event.get("output_path")
                for key in ("initial_mse", "final_mse", "initial_mae", "final_mae", "initial_ncc", "final_ncc"):
                    row[key] = event.get(key)
        elif name == "row_complete":
            row["status"] = "complete"
            row["total_seconds"] = max(now - row.get("_start", now), 0.0)
        elif name == "row_failed":
            row["status"] = "failed"
            row["error"] = event.get("error", "")
            row["total_seconds"] = max(now - row.get("_start", now), 0.0)

    def _write_pairs_csv(self) -> None:
        with self.pairs_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for row_index in sorted(self._rows):
                row = self._rows[row_index]
                writer.writerow({field: _csv_ready(row.get(field, "")) for field in self._CSV_FIELDS})


def _image_metadata(images: BatchedImages) -> Dict[str, Any]:
    image = images.images[0].itk_image
    return {
        "shape": list(images().shape),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def _registration_quality_metrics(fixed_batch: BatchedImages, moving_batch: BatchedImages, moved: torch.Tensor) -> Dict[str, float]:
    fixed = fixed_batch().detach().cpu()
    moving = moving_batch().detach().cpu()
    moved = moved.detach().cpu()
    if tuple(moving.shape[2:]) != tuple(fixed.shape[2:]):
        moving = device_aware_interpolate(
            moving,
            size=fixed.shape[2:],
            mode=moving_batch.interpolate_mode,
            align_corners=True,
        )
    initial = _pair_metrics(fixed, moving)
    final = _pair_metrics(fixed, moved.detach())
    return {
        "initial_mse": initial["mse"],
        "final_mse": final["mse"],
        "initial_mae": initial["mae"],
        "final_mae": final["mae"],
        "initial_ncc": initial["ncc"],
        "final_ncc": final["ncc"],
    }


def _pair_metrics(fixed: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    fixed_norm = _robust_normalize(fixed)
    candidate_norm = _robust_normalize(candidate)
    diff = candidate_norm - fixed_norm
    mse = torch.mean(diff.square()).item()
    mae = torch.mean(diff.abs()).item()
    fixed_centered = fixed_norm - fixed_norm.mean()
    candidate_centered = candidate_norm - candidate_norm.mean()
    denom = torch.sqrt(torch.sum(fixed_centered.square()) * torch.sum(candidate_centered.square())).item()
    ncc = 0.0 if denom == 0.0 else torch.sum(fixed_centered * candidate_centered).item() / denom
    return {"mse": float(mse), "mae": float(mae), "ncc": float(ncc)}


def _robust_normalize(tensor: torch.Tensor) -> torch.Tensor:
    array = tensor.detach().float().cpu()
    sample = array.flatten()
    max_samples = 1_000_000
    if sample.numel() > max_samples:
        step = (sample.numel() + max_samples - 1) // max_samples
        sample = sample[::step]
    lo = torch.quantile(sample, 0.01)
    hi = torch.quantile(sample, 0.99)
    if (hi - lo).abs().item() < 1e-6:
        hi = lo + 1.0
    return torch.clamp((array - lo) / (hi - lo), 0.0, 1.0)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _csv_ready(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return json.dumps(_json_ready(value))
    return _json_ready(value)
