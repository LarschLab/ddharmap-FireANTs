# Current State

Purpose: active caveats and mixed migration notes that future agents should know.

Use this file before broad refactors or when behavior appears inconsistent across old and new paths.

## Notes

- `setupAgents.md` is currently the source of truth for generated `coding.md`.
- Fused operations are optional and may be unavailable in local environments; many paths must retain non-fused fallback behavior.
- MPS support is stock-PyTorch-first. Treat `LalithShiyam/pytorch-mps` and FireANTs issue #6 as historical context only, not dependency guidance; local stock `torch 2.6.0` passes MPS 3D `conv3d`, grouped `conv3d`, separable CC filtering, and LNCC backward with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Generic-label segmentation interpolation requires fused ops; one-hot mode is the fallback strategy when fused ops are unavailable.
- `legacyrigid.py` remains for compatibility, while `rigid.py` is the current quaternion-based rigid registration owner.
- Paper and dataset scripts contain local-data assumptions and should not be treated as general API authority.
- Template configs under `fireants/scripts/template/configs` are orchestration inputs; template output directories are local artifacts.
