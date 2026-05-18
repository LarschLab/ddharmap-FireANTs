# Recent Changes: Core Registration

Purpose: append completed meaningful core registration work.

## Template

```
## YYYY-MM-DD - short label

- Slice goal:
- Passes completed:
- What changed:
- Rerun implications:
- Validation performed:
```

## Log

## 2026-05-18 - Moments Z-direction preservation

- Slice goal: prevent 3D Moments orientation initialization from selecting a moving-Z reversing candidate by default.
- Passes completed: added default Z-direction filtering to 3D Moments candidate scoring with an explicit `preserve_z_direction=False` opt-out.
- What changed: `MomentsRegistration` now prefers the best candidate whose centerline maps to increasing moving Z, falling back to unrestricted scoring only if no candidate preserves Z.
- Rerun implications: downstream Affine/Greedy GUI runs should no longer inherit a Z-reversing Moments initialization unless the caller opts out.
- Validation performed: `python -m pytest -q tests/test_moments_registration.py::TestMomentsRegistration3D::test_orientation_preserves_z_direction_by_default tests/test_moments_registration.py::TestMomentsRegistration3D::test_orientation_can_opt_out_of_z_direction_preservation`; `python -m pytest -q tests/test_gui_runner.py`; MPS real-data rerun under `/Users/ddharmap/dataProcessing/testReg/fireants_gui_run_z_preserve_20260518_153448` with `z_orientation_warning=false` for Moments, Affine, and Greedy.

## 2026-05-17 - MPS moments determinant fallback

- Slice goal: keep GUI runner MPS smoke coverage on stock PyTorch without relying on global CPU fallback.
- Passes completed: routed the small Moments rotation determinant calculation through CPU when tensors are on MPS.
- What changed: `MomentsRegistration` avoids the unsupported stock-MPS `torch.linalg.det` path while preserving the resulting determinant value and device placement.
- Rerun implications: fallback-free MPS runs can progress past the Moments determinant blocker; larger runs may still hit later memory limits.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`.

## 2026-05-17 - full-default GUI gate memory findings

- Slice goal: run the realistic GUI-default profile on full NRRD bridge data and fix owner-layer MPS memory blockers uncovered by the gate.
- Passes completed: routed MPS moments orientation sampling through CPU, chunked MPS 3D grid sampling over Z/Y tiles, and released affine/Greedy per-scale tensors before cache clearing.
- What changed: MPS full-resolution sampling no longer materializes whole-volume trilinear gather tensors; affine/Greedy scale transitions release previous scale tensors explicitly.
- Rerun implications: MPS can progress farther into full-default runs, but full-resolution CC loss still exceeds the 16 GiB machine's MPS memory ceiling; CPU full-default gate completed and exposed registration-quality degradation.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`; CPU one-pair full-default gate under `/Users/ddharmap/dataProcessing/testReg/fireants_gui_full_defaults_cpu_gate_20260517_160057`.

## 2026-05-17 - MPS full-resolution GUI output fallback

- Slice goal: unblock full-size GUI bridge output writing after MPS registration exposes device-memory limits.
- Passes completed: fixed MPS float64 orientation tensors in moments registration and added CPU interpolation fallback hooks for Greedy transform save/evaluate output paths.
- What changed: `MomentsRegistration` creates orientation candidates in the active float dtype; `AbstractRegistration.evaluate` and `GreedyRegistration.get_warp_parameters` accept an optional interpolation device for explicit output-time fallback.
- Rerun implications: MPS registration can still optimize on MPS while full-resolution transform saving and output warping can be routed through CPU by callers that need it.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`; one-pair and two-pair MPS reduced real-data probes with fixed-space NRRD outputs.

## 2026-05-17 - registration progress callbacks

- Slice goal: allow user-facing workflows to report per-stage and per-iteration registration progress without changing optimization semantics.
- Passes completed: added optional callback plumbing through `AbstractRegistration` and emitted stage/scale/iteration events from moments, rigid, affine, Greedy, and SyN registration.
- What changed: registration constructors now accept `progress_callback` via existing kwargs paths while preserving current `progress_bar` behavior.
- Rerun implications: UI and orchestration code can consume callback events instead of parsing `tqdm` text.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`.

## 2026-05-14 - stock PyTorch MPS loss coverage

- Slice goal: keep MPS enablement on stock PyTorch and cover the historical 3D conv blocker through FireANTs loss code.
- Passes completed: added focused MPS tests for 3D separable filtering and LNCC forward/backward.
- What changed: MPS guidance now treats `LalithShiyam/pytorch-mps` as reference-only and records local stock `torch 2.6.0` coverage for 3D conv-backed loss paths.
- Rerun implications: use `PYTORCH_ENABLE_MPS_FALLBACK=0` for MPS validation so unsupported ops fail loudly.
- Validation performed: `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_losses.py -q`; `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_grid_sample.py tests/test_mps_registration_smoke.py -q`.
