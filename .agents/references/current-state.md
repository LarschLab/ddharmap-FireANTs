# Current State

Purpose: active caveats and mixed migration notes that future agents should know.

Use this file before broad refactors or when behavior appears inconsistent across old and new paths.

## Notes

- `setupAgents.md` is currently the source of truth for generated `coding.md`.
- Fused operations are optional and may be unavailable in local environments; many paths must retain non-fused fallback behavior.
- Generic-label segmentation interpolation requires fused ops; one-hot mode is the fallback strategy when fused ops are unavailable.
- `legacyrigid.py` remains for compatibility, while `rigid.py` is the current quaternion-based rigid registration owner.
- Paper and dataset scripts contain local-data assumptions and should not be treated as general API authority.
- Template configs under `fireants/scripts/template/configs` are orchestration inputs; template output directories are local artifacts.
