# Core Registration Router

Purpose: route work for FireANTs package semantics: image representation, transforms, registration algorithms, losses, masks, keypoints, utilities, and distributed registration.

Use this file when the task names `fireants/io`, `fireants/registration`, `fireants/losses`, `fireants/utils`, keypoints, masks, coordinate spaces, registration stages, or core tests.

## Read order

1. `coding.md`
2. `.agents/workflows/fireants-router.md`
3. This router
4. The smallest relevant reference doc from the table below
5. Owning package module
6. Tests matching the owner

## Task routing table

| Query content | Read first | Primary owner | Validation |
| --- | --- | --- | --- |
| Image load/save, spacing, origin, direction, orientation, segmentation, batching | `.agents/references/image-transform-semantics.md` | `fireants/io` | targeted image/keypoint/registration tests |
| Transform files, affine/warp serialization, coordinate conversion | `.agents/references/image-transform-semantics.md` | `fireants/io/transform.py`, `fireants/utils/util.py` | transform or registration tests |
| Moments, rigid, affine, subspace affine, Greedy, SyN, distributed Greedy | `.agents/references/registration-stage-map.md` | `fireants/registration` | matching `tests/test_*registration*.py`; distributed tests only when touched |
| Optimizers, deformation objects, warp composition, inverse/shape averaging | `.agents/references/registration-stage-map.md` | `fireants/registration/optimizers`, `fireants/registration/deformation`, `fireants/utils/warputils.py` | optimizer, deformable, precision, or distributed tests |
| CC, MI, MSE, custom losses, masked loss behavior | `.agents/references/loss-mask-semantics.md` | `fireants/losses`, `fireants/registration/abstract.py` | loss-specific and masked tests |
| Keypoint fidelity, keypoint spaces, distance reductions | `.agents/references/image-transform-semantics.md` | `fireants/io/keypoints.py` | `tests/test_keypoints.py`, `tests/test_keypoint_fidelity.py` |
| Public symbol or API surface changes | `.agents/references/symbol-index.md` | owning module | targeted tests plus docs/reference update |
| Validation choice | `.agents/references/validation-matrix.md` | owning module | listed check |

## Ownership guidance

- `Image`, `BatchedImages`, `FakeBatchedImages`, masks, keypoints, and metadata conversions own data representation semantics.
- `AbstractRegistration` owns shared registration lifecycle, loss setup, multi-resolution behavior, convergence, dtype, and masked-channel handling.
- Specific registration classes own algorithm-specific parameters and transform state.
- Loss modules own mathematical loss behavior; registration classes own when and how losses are selected.
- Distributed modules own rank/grid partitioning behavior; do not patch single-GPU callers to compensate for distributed state bugs.
- Use `.agents/references/recent-changes-core-registration.md` and `.agents/references/remaining-work-core-registration.md` for handoff.
