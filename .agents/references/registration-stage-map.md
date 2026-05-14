# Registration Stage Map

Purpose: ordered map for core registration work.

Use this file for registration algorithm, transform composition, optimizer, distributed, and stage-order questions.

## End-to-end stages

1. Load images through `Image` and batch with `BatchedImages` or `FakeBatchedImages`.
2. Optionally initialize with `MomentsRegistration` for physical-space alignment.
3. Optionally refine with `RigidRegistration`.
4. Optionally refine with `AffineRegistration`.
5. Optionally run deformable registration with `GreedyRegistration`, `SyNRegistration`, or `DistributedGreedyRegistration`.
6. Evaluate or save transforms and warped images through image/transform utilities or wrapper outputs.

## Key outputs by stage

- Image stage: tensor arrays, metadata matrices, segmentation channel layout, batched image wrappers.
- Moments/rigid/affine stages: initialization matrices and transformed coordinates for later stages.
- Deformable stages: displacement/warp state and warped coordinates.
- Evaluation stage: warped tensors/images and transform files when called by wrappers.

## Concept ownership

- `AbstractRegistration` owns shared scale/iteration validation, loss creation, convergence, dtype, masked mode, and common evaluation behavior.
- Algorithm classes own their transform parameterization and optimization loop.
- Optimizer classes own update rules for warp fields.
- Deformation classes own warp representation and composition semantics.
- Distributed modules own sharding, rank state, ring sampling, padding, and gather/concat behavior.

## Navigation notes

- Read `.agents/references/image-transform-semantics.md` before changing coordinate or metadata behavior.
- Read `.agents/references/loss-mask-semantics.md` before changing loss setup, masking, or fused loss selection.
- Read `.agents/references/validation-matrix.md` before choosing tests.
