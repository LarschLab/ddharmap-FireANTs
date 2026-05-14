# Entrypoints Router

Purpose: route work for FireANTs wrappers and user-facing orchestration: CLI, template builder, tutorials, paper scripts, and docs examples.

Use this file when a task names `cli/fireantsRegistration`, `fireants/scripts/template`, tutorials, `docs/docs`, paper scripts, dataset scripts, or command-line examples.

## Read order

1. `coding.md`
2. `.agents/workflows/fireants-router.md`
3. This router
4. `.agents/references/entrypoint-stage-map.md`
5. A semantic reference if the issue points to core behavior
6. Wrapper/script/docs region
7. Owning package module only if the wrapper exposes a real library bug

## Task routing table

| Query content | Read first | Primary owner | Validation |
| --- | --- | --- | --- |
| ANTs-like CLI parsing, transform order, argument compatibility, output naming | `.agents/references/entrypoint-stage-map.md` | `cli/fireantsRegistration` | parser/help smoke or minimal CLI run with small images |
| Template config, torchrun setup, template iteration, saved templates, moved images | `.agents/references/entrypoint-stage-map.md` | `fireants/scripts/template` | config parse or small template smoke when data is available |
| Tutorial mismatch or docs example failure | relevant docs/tutorial, then `.agents/references/symbol-index.md` | `docs/docs`, `tutorials` | docs build or example smoke |
| Paper or dataset scripts | `.agents/references/entrypoint-stage-map.md` | `fireants/scripts/pairwise`, `fireants/scripts/hyperparameter_tuning` | syntax/config smoke; avoid full dataset runs unless requested |
| Docs reference/API drift | `.agents/references/symbol-index.md` | docs page and owning public symbol | `mkdocs build -f docs/mkdocs.yml` when feasible |
| Wrapper uncovers core semantic bug | route to core profile | owner module | core validation first, wrapper smoke second |

## Ownership guidance

- CLI scripts own parsing, compatibility flags, command UX, and output naming.
- Template scripts own orchestration, config plumbing, checkpoint cadence, and save directories.
- Tutorials and docs own examples and user narrative only; do not redefine registration, image, loss, or fused-op semantics there.
- Paper/dataset scripts are experiment orchestration and may depend on local paths; avoid treating them as reusable library authority.
- Use `.agents/references/recent-changes-entrypoints.md` and `.agents/references/remaining-work-entrypoints.md` for handoff.
