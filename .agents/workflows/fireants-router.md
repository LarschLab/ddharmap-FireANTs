# FireANTs Router

Purpose: dispatch future agents to the smallest useful workflow profile for FireANTs work.

Use this file when a task touches FireANTs code, tests, docs, scripts, tutorials, CLI behavior, or fused operations.

## Read this first

1. `coding.md`
2. This router
3. The matching profile router below
4. The smallest relevant reference doc
5. Owning modules, then wrappers only when needed

## Workflow profile dispatch

| Primary target or query content | Profile router |
| --- | --- |
| `fireants/io`, image loading, metadata, batching, keypoints, transforms | `.agents/workflows/core-registration-router.md` |
| `fireants/registration`, moments, rigid, affine, Greedy, SyN, optimizers, distributed registration | `.agents/workflows/core-registration-router.md` |
| `fireants/losses`, CC, MI, MSE, custom losses, masked losses | `.agents/workflows/core-registration-router.md` |
| `fused_ops`, CUDA kernels, fused sampler/composer, fused CC/MI | `.agents/workflows/fused-ops-router.md` |
| `fireants/interpolator`, `USE_FFO`, backend fallback, PyTorch/CUDA parity | `.agents/workflows/fused-ops-router.md` |
| `cli/fireantsRegistration`, ANTs-like argument parsing/output compatibility | `.agents/workflows/entrypoints-router.md` |
| `fireants/scripts/template`, template configs, template outputs, torchrun orchestration | `.agents/workflows/entrypoints-router.md` |
| tutorials, docs examples, paper scripts, benchmark or dataset scripts | `.agents/workflows/entrypoints-router.md` |

## Cross-workflow invariants

- Apply `coding.md` baseline behavior before profile-specific rules.
- Prefer package/module edits over orchestration-layer edits.
- Preserve canonical outputs, file names, public symbols, and stage semantics unless the task explicitly asks for a migration.
- Do not use wrappers, tutorials, or scripts as business-logic authority.
- Fix writer-stage or owner-layer semantic bugs at the owner, not in downstream consumers.
- Run the smallest relevant validation and update workflow logs for meaningful changes.

## Compact scaling rule

Add a new target to an existing profile when it shares the same owners, validation surface, and handoff log. Create a new profile only when it needs its own stage map, semantic references, validation cadence, and recent-changes or remaining-work tracker.
