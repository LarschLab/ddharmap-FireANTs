# Validation Matrix

Purpose: choose the smallest credible validation after edits.

Use this file before running tests or claiming work is complete.

## Core registration

- Image, transform, metadata, batching: targeted image/keypoint tests, then a small registration test if behavior reaches registration.
- Moments, rigid, affine: `tests/test_basic_registration.py`, `tests/test_moments_registration.py`, `tests/test_moments_scaling.py`, or `tests/test_legacy_vs_quart_rigid.py`.
- Deformable registration: `tests/test_deformable_registration.py`, masked tests, precision/multiscale tests, or optimizer comparison tests depending on owner.
- Distributed registration: `bash run_tests.sh --distributed-only` only when distributed behavior is touched and hardware supports it.

## Fused ops

- Sampler/composer/affine warp: matching `tests/test_fusedops_*.py`.
- Fused CC/MI: fused loss tests plus a registration smoke if loss selection changes.
- Dispatcher fallback: test a non-fused or CPU path when feasible.

## Entrypoints and docs

- CLI parsing/output: help/parser smoke or minimal CLI run with tiny synthetic images when feasible.
- Template builder: config parse or tiny template smoke when input data is available.
- Docs: `mkdocs build -f docs/mkdocs.yml` when docs structure or API references change.
- Tutorials: inspect or run the smallest relevant notebook/script region, not the whole workflow by default.

## Broad checks

- Default broad basic suite: `python -m pytest -v --log-cli-level=INFO tests/test*.py`.
- Repo wrapper: `bash run_tests.sh`.
- Full suite including distributed: `bash run_tests.sh --all` only when multi-GPU validation is intended.
