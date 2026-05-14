# Refactor Rules

Purpose: repo-specific ownership and edit-scope rules for FireANTs.

Use this file before refactors or when a bug could be fixed in multiple layers.

## Rules

- Keep reusable registration, image, loss, and transform logic in `fireants/` package owners.
- Keep CLI, template scripts, tutorials, docs, and paper scripts orchestration-thin.
- Do not add wrapper-local helpers for behavior that belongs in package modules.
- Fix semantic bugs at the narrowest owner.
- Do not patch downstream consumers to compensate for upstream writer-stage bugs.
- Preserve stage order, public symbols, output file names, tensor conventions, and metadata semantics unless explicitly migrating them.
- Preserve compatibility flags in `cli/fireantsRegistration` unless the task intentionally changes CLI compatibility.
- Keep generated or local runtime outputs out of version control unless the repo already intentionally versions them.
- After public surface changes, update `.agents/references/symbol-index.md` and relevant docs references.
