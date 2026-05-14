# Entrypoint Stage Map

Purpose: ordered map for FireANTs wrappers, scripts, tutorials, and docs examples.

Use this file when work involves CLI behavior, template construction, docs examples, tutorials, or paper/dataset scripts.

## CLI stages

1. Parse ANTs-like arguments in `cli/fireantsRegistration`.
2. Validate transform, metric, convergence, and shrink-factor lists.
3. Load fixed/moving images and masks.
4. Optionally run initial moment matching.
5. Run each requested transform in order.
6. Save transform outputs and warped image according to CLI output naming.

## Template builder stages

1. Read Hydra/YAML config in `fireants/scripts/template/build_template.py`.
2. Build dataset/dataloader through `datautils.py`.
3. Create or load initial template through `template_helpers.py`.
4. Register batches through `registration_pipeline.py`.
5. Average images and optionally shape-average warps.
6. Save templates, logs, moved images, and optional additional outputs.

## Tutorials, docs, and paper scripts

- Tutorials and docs should demonstrate public APIs and expected workflows.
- Paper and dataset scripts may contain local paths or experiment-specific assumptions.
- Treat these files as orchestration layers unless the task is specifically about their command UX or outputs.

## Canonical outputs

- CLI: output prefix, warped image path, transform files, and compatibility behavior.
- Template builder: `template_*.nii.gz`, optional moved images, logs, and configured save directory.
- Docs/tutorials: rendered documentation pages, notebooks, and example-generated images already intentionally versioned.
