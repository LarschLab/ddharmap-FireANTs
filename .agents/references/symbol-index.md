# Symbol Index

Purpose: compact index of public or workflow-owning FireANTs symbols that agents are likely to need.

Use this file when a task changes public surface, docs references, imports, or orchestration calls into package code.

## Trusted implementation patterns

- Prefer public package classes and dispatchers over copying wrapper-local helper logic.
- Open the owner module before editing callers.
- Keep docs and tutorials aligned to callable symbols, not private helpers.

## Core images and transforms

- `fireants.io.image.Image`: SimpleITK-backed 2D/3D image representation, tensor conversion, metadata transforms, segmentation encoding, and interpolation metadata.
- `fireants.io.image.BatchedImages`: batch owner for lists of `Image` objects used by registration.
- `fireants.io.image.FakeBatchedImages`: lightweight batch-like wrapper for already-loaded tensors.
- `fireants.io.image.concat`: image batching helper.
- `fireants.io.keypoints.Keypoints`, `BatchedKeypoints`, `compute_keypoint_distance`: keypoint representation and fidelity metrics.
- `fireants.io.imagemask`: mask creation and image-mask concatenation helpers.
- `fireants.io.transform.get_affine_transform_from_file`: affine transform file reader.

## Registration

- `fireants.registration.abstract.AbstractRegistration`: shared lifecycle, loss selection, masked mode, downsampling, convergence, optional progress callbacks, and evaluation contract; `evaluate(..., interpolation_device=...)` may route output interpolation to a safer device.
- `fireants.registration.moments.MomentsRegistration`: physical-space initial alignment.
- `fireants.registration.rigid.RigidRegistration`: quaternion-based rigid registration.
- `fireants.registration.legacyrigid.RigidRegistration`: legacy rigid implementation kept for compatibility.
- `fireants.registration.affine.AffineRegistration`: affine registration.
- `fireants.registration.subspace2daffine.Subspace2DAffineRegistration`: 2D shape/subspace affine registration.
- `fireants.registration.greedy.GreedyRegistration`: greedy deformable registration; supports save/evaluate interpolation-device override for MPS full-resolution output fallbacks.
- `fireants.registration.syn.SyNRegistration`: symmetric deformable registration.
- `fireants.registration.distributedgreedy.DistributedGreedyRegistration`: distributed deformable registration.
- `fireants.registration.optimizers.WarpAdam`, `WarpSGD`, `WarpLevenbergMarquardt`: warp optimizers.

## Losses and interpolation

- `fireants.losses.LocalNormalizedCrossCorrelationLoss`: non-fused local CC.
- `fireants.losses.GlobalMutualInformationLoss`: non-fused global MI.
- `fireants.losses.MeanSquaredError`: MSE loss.
- `fireants.losses.fusedcc.FusedLocalNormalizedCrossCorrelationLoss`: fused CC wrapper.
- `fireants.losses.fusedmi.FusedGlobalMutualInformationLoss`: fused MI wrapper.
- `fireants.interpolator.fireants_interpolator`: singleton `GridSampleDispatcher` for sampling, warp composition, and affine warp.
- `fireants.interpolator.grid_sample.device_aware_interpolate`: PyTorch-compatible resize helper that routes MPS 3D trilinear interpolation through the chunked FireANTs sampler fallback.
- `fireants.utils.device`: small cache, synchronization, and memory helpers for CUDA, MPS, and CPU-safe callers.

## Entrypoints

- `cli/fireantsRegistration`: ANTs-like command-line registration wrapper.
- `fireants.gui.runner.RegistrationSettings`: GUI runner settings for profile-controlled bridge registration, including loss/scales/iterations, Moments scaling, Greedy optimizer state controls, metrics, and live-preview options.
- `fireants.gui.runner.registration_settings_for_profile`: preset factory for GUI registration profiles (`Current Full`, `Memory Saver`, `ANTs-like SyN`, `Debug`).
- `fireants.gui.runner.validate_registration_settings`: shared GUI profile validation for runner and PySide entrypoint controls.
- `fireants.gui.runner.run_batch_registration`: reusable bridge-stack batch runner for GUI workflows; registers moving bridge images to one fixed bridge image, applies the final transform to selected moving-side files, emits JSONL/CSV benchmark metrics through `RegistrationSettings.metrics_dir`, and can emit GUI-only live overlay preview events.
- `fireants.gui.app`: optional PySide6 desktop GUI entrypoint exposed as `fireants-gui`.
- `fireants/scripts/template/build_template.py`: template-building command entrypoint.
- `fireants/scripts/template/registration_pipeline.py::register_batch`: per-batch template registration pipeline.
- `fireants/scripts/template/template_helpers.py`: template output, averaging, shape averaging, and config helpers.
- `fireants/scripts/template/datautils.py`: template input dataset and dataloader helpers.
