# Refactor Loop Policy

Purpose: keep agents moving through one coherent ownership slice without stopping after superficial cleanup.

Use this file when work is a refactor, cleanup, migration continuation, or multi-file fix.

## Default working unit

One ownership slice plus its smallest validation surface. Examples: one registration algorithm, one loss family, dispatcher plus fused wrapper tests, or one CLI/template wrapper behavior.

## Required loop

1. Identify the owner and callers.
2. Make the smallest owner-layer change.
3. Remove only dead code made dead by this change.
4. Run targeted validation.
5. Update references/logs when public behavior or handoff context changed.

## Valid stop conditions

- Targeted validation passed.
- A required dependency, dataset, GPU, or CUDA build toolchain is unavailable and the remaining check is recorded.
- The remaining work is outside the current ownership slice.

## Invalid stop conditions

- One helper was cleaned while adjacent broken code in the same owner and validation surface remains.
- A wrapper was patched while the package owner is still semantically wrong.
- Static reasoning was used as the only validation for behavior changes.

## Handoff

- Completed meaningful work goes in the workflow recent-changes log.
- Unresolved actionable work goes in the workflow remaining-work tracker.
