# Recent Changes: Entrypoints

Purpose: append completed meaningful entrypoint, docs, tutorial, or script work.

## Template

```
## YYYY-MM-DD - short label

- Slice goal:
- Passes completed:
- What changed:
- Rerun implications:
- Validation performed:
```

## Log

## 2026-05-17 - GUI full-scale findings report

- Slice goal: preserve full-scale GUI pipeline memory and quality findings for future code changes and hardware tests.
- Passes completed: added a how-to report covering data, profiles, MPS limits, CPU gate results, hardware expectations, and recommended rerun strategy.
- What changed: `docs/docs/howto/gui-full-scale-findings.md` is linked from the MkDocs How To navigation.
- Rerun implications: future GUI/default-profile testing should start from the documented gate strategy and compare against recorded artifacts.
- Validation performed: attempted `mkdocs build -f docs/mkdocs.yml`, blocked because `mkdocs` is not installed in the current shell; performed file/nav sanity checks instead.

## 2026-05-17 - full-default GUI gate

- Slice goal: execute a realistic GUI-default gate before committing to a two-pair full run.
- Passes completed: preflight GUI/unit checks, repeated MPS gate attempts to classify memory blockers, and a completed CPU one-pair full-default gate.
- What changed: no GUI API changes beyond benchmark metrics; the two-pair full run was intentionally skipped because the one-pair gate completed but worsened proxy registration metrics.
- Rerun implications: use the gate artifacts and pair CSV to tune registration settings before running both pairs at full defaults.
- Validation performed: `/Users/ddharmap/dataProcessing/testReg/fireants_gui_full_defaults_cpu_gate_20260517_160057`; output geometry valid, metrics worsened (`NCC 0.502887 -> 0.108961`).

## 2026-05-17 - GUI benchmark metrics and real-data probe

- Slice goal: make GUI bridge registration runs benchmarkable and validate the provided real NRRD inputs through the GUI runner.
- Passes completed: added timestamped GUI run folders, JSONL event metrics, CSV pair summaries, and fixed-space proxy MSE/MAE/NCC metrics.
- What changed: `RegistrationSettings.metrics_dir` enables `_metrics/events.jsonl` and `_metrics/pairs.csv`; GUI starts now write into `fireants_gui_run_YYYYMMDD_HHMMSS` children under the selected output root.
- Rerun implications: GUI benchmark runs preserve prior outputs and can be compared by pair CSV without parsing logs.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`; offscreen GUI screenshot `/tmp/fireants_gui_real_inputs_smoke.png`; two-pair MPS reduced real-data run under `/Users/ddharmap/dataProcessing/testReg/fireants_gui_mps_two_pair_probe_20260517_150148`.

## 2026-05-17 - bridge registration GUI

- Slice goal: add a desktop GUI for registering moving bridge stacks to a remembered fixed bridge stack and applying transforms to selected moving-side files.
- Passes completed: added optional PySide6 entrypoint, GUI worker, previews, file pairing list, remembered fixed/output paths, and a reusable GUI runner.
- What changed: `fireants-gui` is exposed through the `gui` optional dependency; `fireants.gui.runner.run_batch_registration` owns bridge/payload batch orchestration.
- Rerun implications: GUI workflows should call the runner instead of duplicating registration sequencing in widgets.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`; offscreen PySide window screenshot `/tmp/fireants_gui_smoke.png` passed nonblank size check.
