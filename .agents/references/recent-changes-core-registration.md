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

## 2026-05-14 - stock PyTorch MPS loss coverage

- Slice goal: keep MPS enablement on stock PyTorch and cover the historical 3D conv blocker through FireANTs loss code.
- Passes completed: added focused MPS tests for 3D separable filtering and LNCC forward/backward.
- What changed: MPS guidance now treats `LalithShiyam/pytorch-mps` as reference-only and records local stock `torch 2.6.0` coverage for 3D conv-backed loss paths.
- Rerun implications: use `PYTORCH_ENABLE_MPS_FALLBACK=0` for MPS validation so unsupported ops fail loudly.
- Validation performed: `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_losses.py -q`; `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_grid_sample.py tests/test_mps_registration_smoke.py -q`.
