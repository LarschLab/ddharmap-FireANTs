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

## 2026-05-18 - ANTs-like SyN GUI profile

- Slice goal: add a GUI registration profile that mirrors the user's prior ANTs stage structure without expanding CLI compatibility.
- Passes completed: added an `ANTs-like SyN` profile with Rigid -> Affine -> SyN execution, per-stage MI/CC settings, winsorization controls, and stage-specific GUI cards; benchmarked the L765_f01 fixed / L765_f03 moving TIFF pair on MPS.
- What changed: `fireants.gui.runner` can branch between the existing Moments/Affine/Greedy workflow and the new ANTs-like SyN workflow; GUI settings persistence, metrics CSV timing fields, and offscreen profile controls understand the new pipeline. Large-tensor winsorization samples quantiles instead of calling full flattened `torch.quantile`; the ANTs-like SyN profile uses SGD with `compose_n=1`; SyN GUI previews are skipped to avoid inverse-warp preview work; SyN export/evaluation can route warp-parameter interpolation through CPU; tensor-backed fake image batches keep metadata on the tensor device.
- Rerun implications: use `ANTs-like SyN` from the GUI Profile dropdown for ANTs-inspired runs; the prior `quantile() input tensor is too large` and MPS full-resolution SyN allocation blockers are fixed. Current L765_f03 benchmark completes but worsens proxy metrics, so tune quality before running additional pairs.
- Validation performed: `python -m compileall -q fireants/gui fireants/io/image.py fireants/registration/syn.py tests/test_gui_runner.py`; `python -m pytest -q tests/test_gui_runner.py`; completed one-pair MPS benchmark under `/Volumes/dataDrive/dataProcessing/FireANTs/testReg/fireants_gui_run_ants_like_syn_mps_l765_f03_20260518_095903` with valid warped/warp geometry and worsened proxy metrics (`NCC 0.169474 -> 0.022831`).

## 2026-05-17 - compact GUI profile layout

- Slice goal: make the profile-enabled GUI usable on smaller displays after full-width parameter fields consumed too much vertical space.
- Passes completed: captured current GUI screenshots, moved advanced profile controls into a scrollable sidebar tab, kept setup controls compact at the top, and visually verified the implemented layout at 1024x768, 1180x760, 1366x768, and 1440x900.
- What changed: `fireants.gui.app` now uses a queue/profile sidebar with the previewer as the main workspace; the Advanced button switches to the profile tab.
- Rerun implications: profile controls remain available without pushing live previews and progress bars off-screen.
- Validation performed: `python -m compileall -q fireants/gui tests/test_gui_runner.py`; `python -m pytest -q tests/test_gui_runner.py`; screenshots under `/tmp/fireants_gui_compact_verified_*.png`.

## 2026-05-17 - GUI profile controls and live overlays

- Slice goal: make GUI registration tuning explicit and add visual scale-by-scale feedback without writing intermediate volumes.
- Passes completed: added runner profile presets, validation, Greedy optimizer state controls, GUI profile persistence, and live magenta/green preview events after Moments and Affine/Greedy scale completion.
- What changed: `RegistrationSettings` now carries preview and Greedy optimizer controls; `run_batch_registration` can emit GUI-only `preview_update` overlays that are excluded from metrics JSONL.
- Rerun implications: the GUI still opens on `Current Full`, but MPS tuning should start from the `Memory Saver` preset and inspect live overlays before full batch runs.
- Validation performed: `python -m pytest -q tests/test_gui_runner.py`.

## 2026-05-17 - GUI memory-reduction strategy notes

- Slice goal: preserve candidate strategies for reducing full-scale GUI MPS memory before the next tuning pass.
- Passes completed: added profile-level and owner-layer memory-reduction ideas to the GUI full-scale findings report.
- What changed: documentation only in `docs/docs/howto/gui-full-scale-findings.md`; no GUI API, runner-default, or public-symbol changes.
- Rerun implications: start with Greedy scales `[4, 2]`, cheaper full-resolution losses, Adam reset/offload, or SGD before deeper MPS interpolation changes.
- Validation performed: documentation diff and whitespace checks.

## 2026-05-17 - 24 GB M4 Pro GUI full-default gate

- Slice goal: rerun the GUI-default full-scale one-pair CRHA gate on the 24 GB M4 Pro data-drive setup.
- Passes completed: preflighted MPS/data geometry, ran plain MPS and fallback-enabled MPS gates, inspected metrics/events, and recorded the failure artifacts.
- What changed: updated `docs/docs/howto/gui-full-scale-findings.md` with the 24 GB result; no GUI API, runner-default, or public-symbol changes.
- Rerun implications: plain stock PyTorch 2.6.0 MPS still needs a Moments determinant fix or fallback; fallback-enabled MPS clears Affine scale `1` but fails entering Greedy scale `1` from MPS memory pressure.
- Validation performed: artifacts under `/Volumes/dataDrive/dataProcessing/FireANTs/testReg/fireants_gui_full_defaults_mps_24gb_gate_20260517_172205` and `/Volumes/dataDrive/dataProcessing/FireANTs/testReg/fireants_gui_full_defaults_mps_24gb_gate_fallback_20260517_172337`; `python -m pytest -q tests/test_gui_runner.py` failed 2 MPS determinant tests without fallback; `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest -q tests/test_gui_runner.py` passed.

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
