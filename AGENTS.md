# Agent Entrypoint

FireANTs uses a routed instruction system under `.agents/`. Keep this entrypoint short: load the generic coding baseline, route by workflow profile, then read only the smallest repo-specific reference needed for the task.

## Required startup order

1. Open `coding.md` for baseline coding behavior.
2. Open `.agents/workflows/fireants-router.md`.
3. Follow its dispatch table to the profile router.
4. Read the smallest relevant reference doc.
5. Open stage maps or `.agents/references/symbol-index.md` only if needed.
6. Open owning modules before large scripts, tutorials, or notebooks.
7. Open large script or notebook regions only when owner-module context is insufficient.

## Non-negotiable repo rules

- Prefer edits in package owners over orchestration wrappers.
- Preserve registration stage semantics, coordinate conventions, public names, and canonical outputs unless the task explicitly asks for a migration.
- Treat CLI tools, template scripts, tutorials, and paper scripts as wrappers unless the requested change is wrapper behavior.
- Fix semantics at the writer or owner layer, not in downstream consumers.
- Validate after edits with the smallest relevant test or smoke check; do not claim success from static reasoning alone.
- When public behavior, symbols, or ownership guidance changes, update the relevant `.agents/references/symbol-index.md` or semantic doc and workflow recent-changes log.

## Reference files

- `.agents/workflows/fireants-router.md`: top-level workflow dispatcher.
- `.agents/references/symbol-index.md`: compact public surface and owner index.
- `.agents/references/registration-stage-map.md`: moments, rigid, affine, deformable, distributed registration flow.
- `.agents/references/entrypoint-stage-map.md`: CLI, template builder, tutorials, and paper-script flow.
- `.agents/references/fused-ops-semantics.md`: fused CUDA, dispatcher, fallback, and parity rules.
- `.agents/references/image-transform-semantics.md`: image metadata, coordinate spaces, segmentation, batching, and transforms.
- `.agents/references/loss-mask-semantics.md`: loss selection, fused losses, and masked-channel conventions.
- `.agents/references/validation-matrix.md`: task-to-validation routing.
- `.agents/references/refactor-rules.md`: repo-specific edit ownership rules.
- `.agents/references/refactor-loop-policy.md`: stopping, handoff, and validation cadence.
- `.agents/references/current-state.md`: active caveats and mixed migration state.
- `.agents/references/recent-changes.md`: change-log index.
- `.agents/references/remaining-work.md`: unresolved-work index.

## Scope note

Details live under `.agents/`. Always apply `coding.md` as baseline behavior first, then layer the routed FireANTs workflow guidance on top.
