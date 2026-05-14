# Recent Changes: Fused Ops

Purpose: append completed meaningful fused ops work.

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

## 2026-05-14 - native MPS 3D fallback MVP

- Slice goal: allow basic 3D affine/deformable registration on Apple MPS without PyTorch CPU fallback for sampling.
- Passes completed: MPS 3D sampler/interpolate fallback, device cache/sync helpers, and targeted MPS tests.
- What changed: `grid_sample.py` now handles MPS 3D trilinear sampling, warp composition, and resize without calling unsupported PyTorch MPS ops; core registration resize call sites use the device-aware helper.
- Rerun implications: run MPS tests with `PYTORCH_ENABLE_MPS_FALLBACK=0` to catch accidental unsupported op usage.
- Validation performed: `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_grid_sample.py tests/test_mps_registration_smoke.py -q`.
