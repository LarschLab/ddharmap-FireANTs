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

import torch

from fireants.io.image import BatchedImages, FakeBatchedImages, Image
from fireants.interpolator.grid_sample import device_aware_interpolate
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.moments import MomentsRegistration


RunnerProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class RegistrationSettings:
    device: str = "cuda:0"
    loss_type: str = "cc"
    cc_kernel_size: int = 5
    moments_scale: float = 1.0
    moments_order: int = 2
    moments_orientation: str = "rot"
    affine_scales: List[float] = field(default_factory=lambda: [4, 2, 1])
    affine_iterations: List[int] = field(default_factory=lambda: [100, 50, 25])
    affine_lr: float = 3e-3
    greedy_scales: List[float] = field(default_factory=lambda: [4, 2, 1])
    greedy_iterations: List[int] = field(default_factory=lambda: [100, 50, 25])
    greedy_lr: float = 0.5
    smooth_warp_sigma: float = 0.5
    smooth_grad_sigma: float = 1.0
    progress_bar: bool = False
    metrics_dir: Optional[Path] = None

    @classmethod
    def default(cls) -> "RegistrationSettings":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return cls(device=device)

    @property
    def total_registration_iterations(self) -> int:
        return 1 + sum(self.affine_iterations) + sum(self.greedy_iterations)


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
    jobs = list(jobs)
    fixed_bridge = Path(fixed_bridge)
    output_dir = Path(output_dir)
    total_units = sum(settings.total_registration_iterations + len(job.all_warp_files()) for job in jobs)
    completed_units = 0
    metrics = _MetricsRecorder(settings.metrics_dir, fixed_bridge, output_dir, settings) if settings.metrics_dir else None

    def emit(**event: Any) -> None:
        if metrics is not None:
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
            stage_totals = {
                "Moments": 1,
                "Affine": sum(settings.affine_iterations),
                "Greedy": sum(settings.greedy_iterations),
            }
            stage_completed = {stage: 0 for stage in stage_totals}

            emit(
                event="row_start",
                row_index=row_index,
                total_jobs=len(jobs),
                moving_bridge=str(moving_bridge),
                row_total_units=row_total_units,
                output_dir=str(row_output_dir),
            )

            def reg_progress(event: Dict[str, Any]) -> None:
                nonlocal completed_units, row_completed_units
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
                check_cancelled()

            try:
                moving_image = Image.load_file(str(moving_bridge), device=settings.device)
                moving_batch = BatchedImages([moving_image])
                emit(event="moving_loaded", row_index=row_index, path=str(moving_bridge), **_image_metadata(moving_batch))

                moments = MomentsRegistration(
                    scale=settings.moments_scale,
                    fixed_images=fixed_batch,
                    moving_images=moving_batch,
                    moments=settings.moments_order,
                    orientation=settings.moments_orientation,
                    loss_type=settings.loss_type,
                    cc_kernel_size=settings.cc_kernel_size,
                    progress_bar=settings.progress_bar,
                    progress_callback=reg_progress,
                )
                moments.optimize()

                affine = AffineRegistration(
                    scales=settings.affine_scales,
                    iterations=settings.affine_iterations,
                    fixed_images=fixed_batch,
                    moving_images=moving_batch,
                    loss_type=settings.loss_type,
                    cc_kernel_size=settings.cc_kernel_size,
                    optimizer_lr=settings.affine_lr,
                    init_rigid=moments.get_affine_init().detach(),
                    progress_bar=settings.progress_bar,
                    progress_callback=reg_progress,
                )
                affine.optimize()

                reg = GreedyRegistration(
                    scales=settings.greedy_scales,
                    iterations=settings.greedy_iterations,
                    fixed_images=fixed_batch,
                    moving_images=moving_batch,
                    loss_type=settings.loss_type,
                    cc_kernel_size=settings.cc_kernel_size,
                    optimizer_lr=settings.greedy_lr,
                    smooth_warp_sigma=settings.smooth_warp_sigma,
                    smooth_grad_sigma=settings.smooth_grad_sigma,
                    init_affine=affine.get_affine_matrix().detach(),
                    progress_bar=settings.progress_bar,
                    progress_callback=reg_progress,
                )
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
                    moved = reg.evaluate(fixed_batch, source_batch, interpolation_device=interpolation_device)
                    moved_batch = FakeBatchedImages(moved, fixed_batch)
                    output_path = row_output_dir / f"{_file_stem(source_path)}_warped.nii.gz"
                    moved_batch.write_image(str(output_path))
                    completed_units += 1
                    row_completed_units += 1
                    quality_metrics = {}
                    if source_path == moving_bridge:
                        quality_metrics = _registration_quality_metrics(fixed_batch, moving_batch, moved)
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
                del moving_batch, moving_image, reg, affine, moments
                if settings.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as exc:
                emit(event="row_failed", row_index=row_index, output_dir=str(row_output_dir), error=str(exc))
                raise

        emit(event="batch_complete", completed_units=total_units, total_units=total_units)
    finally:
        if metrics is not None:
            metrics.close()


def _file_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".tif", ".tiff"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


class _MetricsRecorder:
    _CSV_FIELDS = [
        "row_index",
        "status",
        "fixed_bridge",
        "moving_bridge",
        "output_dir",
        "device",
        "fixed_shape",
        "fixed_spacing",
        "moving_shape",
        "moving_spacing",
        "moments_seconds",
        "affine_seconds",
        "greedy_seconds",
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
                **self._fixed_metadata,
                "moments_seconds": 0.0,
                "affine_seconds": 0.0,
                "greedy_seconds": 0.0,
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
            if stage in ("moments", "affine", "greedy") and registration_event == "stage_start":
                self._stage_starts[(row_index, stage)] = now
            elif stage in ("moments", "affine", "greedy") and registration_event == "stage_complete":
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
