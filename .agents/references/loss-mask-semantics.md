# Loss And Mask Semantics

Purpose: route and preserve FireANTs loss selection, fused-loss behavior, and masked-channel conventions.

Use this file before changing `fireants/losses`, loss setup in `AbstractRegistration`, masked losses, or fused loss fallback.

## Loss selection

- `AbstractRegistration` owns translating `loss_type` into a loss module.
- Supported public loss choices include `mi`, `cc`, `mse`, `custom`, `noop`, `fusedcc`, and `fusedmi`.
- Fused loss choices should fall back to non-fused losses when fused ops are unavailable, where current code supports that fallback.
- Loss modules own their mathematical behavior; registration code owns when they are instantiated and how image tensors are prepared.

## Masked mode

- `loss_type` values prefixed with `masked_` enable masked mode and pass `masked=True` through loss params.
- In masked mode, the last channel is treated as mask and should not be Gaussian-smoothed.
- `AbstractRegistration` owns splitting/rejoining image and mask channels during downsampling.
- Do not patch downstream scripts to compensate for incorrect mask semantics in `AbstractRegistration` or loss modules.

## Validation

- Loss changes should use focused loss or registration tests first.
- Masked changes should include `tests/test_masked_greedy_fidelity.py` and 3D masked tests when relevant.
- Fused loss changes should include `tests/test_fusedops_crosscorrelation*.py` or `tests/test_fusedops_mutualinfo.py`.
