# Fused Ops Semantics

Purpose: shared rules for fused CUDA operations and fallback behavior.

Use this file for fused sampler/composer, affine warp, generic-label interpolation, fused CC/MI, extension build, and backend parity tasks.

## Ownership

- `fused_ops/include` and `fused_ops/src` own CUDA kernel semantics.
- `fused_ops/setup.py` and `fused_ops/pyproject.toml` own extension build and packaging.
- `fireants/interpolator/fused_grid_sample.py` owns Python autograd wrappers for fused sampling, warp composition, and affine warp.
- `fireants/interpolator/fused_grid_sample_genericlabel.py` owns generic-label fused sampling.
- `fireants/interpolator/__init__.py` owns backend dispatch, `USE_FFO`, and CPU fallback.
- `fireants/interpolator/grid_sample.py` owns PyTorch fallback sampling, including the native MPS 3D sampler used when PyTorch lacks MPS `grid_sample` support.
- `fireants/losses/fusedcc.py` and `fireants/losses/fusedmi.py` own fused loss modules.

## Semantics to preserve

- Fused ops are optional; FireANTs must still have PyTorch fallbacks where the dispatcher supports them.
- Non-CUDA tensors are routed away from fused kernels by the dispatcher.
- MPS 3D registration paths must avoid PyTorch's unsupported native `grid_sample`/trilinear resize ops by using the fallback helpers in `grid_sample.py`; do not route MPS tensors into fused CUDA kernels.
- Generic-label interpolation requires fused ops; callers should use one-hot segmentation mode when fused ops are unavailable.
- Images are channel-first tensors; displacement/vector fields follow PyTorch grid conventions used by the existing wrappers.
- Fused and non-fused implementations should match shapes, dtype expectations, coordinate conventions, and gradients within test tolerances.

## Validation

- Use the matching `tests/test_fusedops_*.py` file for kernel or wrapper behavior.
- For dispatcher changes, include a fallback scenario when possible.
- For fused loss changes, test both forward value and backward behavior where existing tests do so.
- Run `make build_fused_ops` only when the CUDA toolchain is available and the task touches build/package behavior.
