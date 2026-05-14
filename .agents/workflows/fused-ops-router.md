# Fused Ops Router

Purpose: route work for fused CUDA operations, Python dispatch, fused losses, and PyTorch fallback parity.

Use this file when a task names `fused_ops`, `fireants_fused_ops`, `fireants/interpolator`, `USE_FFO`, fused grid sampling, warp composition, fused affine warp, generic-label interpolation, fused CC, or fused MI.

## Read order

1. `coding.md`
2. `.agents/workflows/fireants-router.md`
3. This router
4. `.agents/references/fused-ops-semantics.md`
5. `.agents/references/loss-mask-semantics.md` for fused losses
6. CUDA/Python owner modules
7. Matching fused-op tests

## Task routing table

| Query content | Read first | Primary owner | Validation |
| --- | --- | --- | --- |
| CUDA kernel correctness or memory behavior | `.agents/references/fused-ops-semantics.md` | `fused_ops/include`, `fused_ops/src` | relevant `tests/test_fusedops_*.py` |
| Python wrapper around fused sampler/composer/affine warp | `.agents/references/fused-ops-semantics.md` | `fireants/interpolator/fused_grid_sample.py` | sampler, composer, affine warp tests |
| Dispatcher fallback or `USE_FFO` behavior | `.agents/references/fused-ops-semantics.md` | `fireants/interpolator/__init__.py` | CPU/fallback and fused parity checks |
| Generic-label segmentation interpolation | `.agents/references/image-transform-semantics.md` and `.agents/references/fused-ops-semantics.md` | `fireants/io/image.py`, `fireants/interpolator/fused_grid_sample_genericlabel.py` | generic-label fused tests plus segmentation smoke |
| Fused CC or MI loss behavior | `.agents/references/loss-mask-semantics.md` | `fireants/losses/fusedcc.py`, `fireants/losses/fusedmi.py`, `fused_ops` kernels | fused loss tests and fallback awareness |
| Packaging or extension build issues | `.agents/references/fused-ops-semantics.md` | `fused_ops/setup.py`, `fused_ops/pyproject.toml` | `make build_fused_ops` when toolchain is available |

## Ownership guidance

- CUDA headers and sources own kernel semantics.
- Python autograd wrappers own tensor shape, dtype, contiguity, and backward integration.
- `GridSampleDispatcher` owns backend selection and fallback rules.
- Core registration code should call dispatcher abstractions, not fused kernels directly.
- Use `.agents/references/recent-changes-fused-ops.md` and `.agents/references/remaining-work-fused-ops.md` for handoff.
