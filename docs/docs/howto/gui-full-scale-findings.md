# GUI Full-Scale Pipeline Findings

Date: 2026-05-17

This note summarizes what we learned while testing the FireANTs GUI bridge-registration pipeline on full-size NRRD data. It is meant as a reference for future code changes, hardware tests, and registration-profile tuning.

## Test Data

Fixed bridge:

```text
/Users/ddharmap/dataProcessing/testOutput/20260311_f01_agrp_488_pomca_546_gnrh3_647_Stitch_preprocessed/20260311_f01_agrp_488_pomca_546_gnrh3_647_Stitch_DAPI_740nm_preprocessed.nrrd
```

Moving bridges:

```text
/Users/ddharmap/dataProcessing/testOutput/20260311_f01_crha_488_calb2(a)_546_crhb_647_Stitch_preprocessed/20260311_f01_crha_488_calb2(a)_546_crhb_647_Stitch_DAPI_740nm_preprocessed.nrrd
/Users/ddharmap/dataProcessing/testOutput/20260311_f01_tph2_488_optb_546_gbx2_647_Stitch_preprocessed/20260311_f01_tph2_488_optb_546_gbx2_647_Stitch_DAPI_740nm_preprocessed.nrrd
```

Image geometry:

| Image | Size | Spacing |
| --- | --- | --- |
| Fixed AGRP/POMCA/GNRH3 DAPI | `(750, 750, 188)` | `(1.186094, 1.186094, 3.0)` |
| Moving CRHA/CALB2/CRHB DAPI | `(750, 750, 172)` | `(1.186094, 1.186094, 3.0)` |
| Moving TPH2/OPTB/GBX2 DAPI | `(750, 750, 187)` | `(1.186094, 1.186094, 3.0)` |

Output root:

```text
/Users/ddharmap/dataProcessing/testReg
```

The 24 GB M4 Pro rerun used the same f01 fixed/moving data mirrored under:

```text
/Volumes/dataDrive/dataProcessing/FireANTs
```

## Profiles Tested

### Reduced Debug Profile

Used to prove the end-to-end GUI runner mechanics on full-size inputs without paying the full optimization cost:

```python
device = "mps"
loss_type = "mse"
moments_scale = 8
affine_scales = [8]
affine_iterations = [1]
greedy_scales = [8]
greedy_iterations = [1]
```

Result:

- Both moving bridges completed on MPS.
- Transform and warped image outputs were written.
- Warped image geometry matched the fixed image.
- Proxy metrics improved for both moving bridges.

Artifact:

```text
/Users/ddharmap/dataProcessing/testReg/fireants_gui_mps_two_pair_probe_20260517_150148
```

Metrics:

| Moving | Initial NCC | Final NCC | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: |
| CRHA/CALB2/CRHB | `0.502887` | `0.574633` | `0.040769` | `0.034709` |
| TPH2/OPTB/GBX2 | `0.356560` | `0.665046` | `0.055585` | `0.029004` |

### GUI Default Full-Scale Profile

The current GUI runner defaults:

```python
loss_type = "cc"
cc_kernel_size = 5
moments_scale = 1.0
moments_order = 2
moments_orientation = "rot"
moments_perform_scaling = True
affine_scales = [4, 2, 1]
affine_iterations = [100, 50, 25]
greedy_scales = [4, 2, 1]
greedy_iterations = [100, 50, 25]
```

This is 351 registration iterations per moving bridge, plus transform save and moving-image warp.

### GUI Profile Controls

The GUI now exposes registration-profile presets plus editable advanced settings. It still opens on `Current Full` for compatibility with earlier behavior, but profile tuning should usually start with `Memory Saver` on MPS hardware:

```python
loss_type = "mse"
moments_scale = 8
affine_scales = [4, 2, 1]
affine_iterations = [50, 25, 10]
greedy_scales = [4, 2]
greedy_iterations = [50, 25]
```

The `Debug` preset keeps the earlier reduced one-iteration coarse profile for quick wiring checks. The GUI also has live magenta/green overlay previews after Moments and after each Affine/Greedy scale, so users can inspect whether the moving bridge is converging toward the fixed bridge before waiting for final output files.

## Memory Findings

The fixed image contains about 106 million voxels:

```text
750 * 750 * 188 ~= 106M voxels
```

A single float32 image at that size is about 424 MB, but registration needs many live tensors at once:

- fixed image
- moving image
- moved/warped image
- affine or deformable grids
- gradients
- CC local sums
- squared images
- convolution buffers
- optimizer state

This is why a single 424 MB volume can turn into many gigabytes of temporary memory.

### 16 GB MPS Limitation

The 16 GB Mac used for testing could not complete the GUI-default full-scale profile on MPS.

Observed MPS failures:

- Full-resolution moments orientation search initially exceeded the MPS memory pool.
- Full-resolution affine scale `1` then exceeded MPS memory in the 3D sampler.
- After chunking the sampler, full-resolution affine scale `1` exceeded MPS memory in the CC loss convolution.

The last MPS failure happened with roughly:

```text
MPS allocated: ~16.6 GiB
Other allocations: ~3.3 GiB
Requested allocation: ~403 MB
```

This means the 16 GB MPS memory pool is too tight for full-resolution `cc` at scale `1` on this data.

### Owner-Layer Fixes Added

Several MPS blockers were fixed in owner layers rather than patched in the GUI:

- Moments orientation candidates now avoid unsupported float64 tensors on MPS.
- MPS moments orientation sampling can route the heavy candidate evaluation through CPU.
- MPS 3D grid sampling is chunked over Z/Y tiles instead of materializing whole-volume gather tensors.
- Affine and Greedy release per-scale tensors before clearing the device cache.
- GUI output save/warp can route full-resolution output interpolation through CPU when the registration device is MPS.

These fixes help MPS progress farther and make reduced/full-size output writing viable, but they do not make full-resolution default `cc` fit on a 16 GB MPS machine.

## CPU Full-Default Gate

Because MPS could not fit the default full-scale profile, the same default settings were run on CPU for the CRHA moving bridge.

Artifact:

```text
/Users/ddharmap/dataProcessing/testReg/fireants_gui_full_defaults_cpu_gate_20260517_160057
```

Result:

- Pipeline completed.
- Transform output was written.
- Warped image output was written.
- Warped image size matched the fixed image.
- Runtime was about 2006 seconds, or roughly 33.4 minutes, for one moving bridge.

Stage times:

| Stage | Seconds |
| --- | ---: |
| Moments | `42.45` |
| Affine | `607.67` |
| Greedy | `1323.40` |
| Total | `2005.59` |

However, the registration-quality proxy metrics worsened:

| Metric | Initial | Final |
| --- | ---: | ---: |
| NCC | `0.502887` | `0.108961` |
| MSE | `0.040769` | `0.065064` |
| MAE | `0.100488` | `0.128981` |

Conclusion: the pipeline mechanics can complete on CPU, but the current GUI-default registration profile is not a good quality profile for this dataset.

## 24 GB M4 Pro Full-Default Gate

The same one-pair CRHA gate was rerun on a 24 GB Apple M4 Pro using `device="mps"` and the GUI-default `cc` profile.

Preflight:

- Machine: Apple M4 Pro, `24 GiB` unified memory.
- PyTorch: `2.6.0`.
- MPS available: `True`.
- CUDA available: `False`.
- Fixed/moving/payload image geometry matched the f01 data recorded above.

### Plain MPS Attempt

Artifact:

```text
/Volumes/dataDrive/dataProcessing/FireANTs/testReg/fireants_gui_full_defaults_mps_24gb_gate_20260517_172205
```

Result:

- Failed during Moments before reaching memory-heavy registration stages.
- `row_complete` and `batch_complete` were not emitted.
- No transform or warped outputs were written.

Failure:

```text
The operator 'aten::_linalg_det.result' is not currently implemented for the MPS device.
```

This is a stock PyTorch MPS backend coverage issue in the moments rotation determinant path, not a full-scale memory result.

### MPS with PyTorch CPU Fallback

To continue the hardware gate past the unsupported determinant op, the same run was repeated with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

Artifact:

```text
/Volumes/dataDrive/dataProcessing/FireANTs/testReg/fireants_gui_full_defaults_mps_24gb_gate_fallback_20260517_172337
```

Result:

- Moments completed.
- Affine completed, including full-resolution scale `1`.
- Greedy completed scales `4` and `2`.
- The run failed entering Greedy full-resolution scale `1`.
- `row_complete` and `batch_complete` were not emitted.
- No transform or warped outputs were written, so no final proxy metrics or output-geometry checks were available.

Stage times:

| Stage | Seconds |
| --- | ---: |
| Moments | `21.71` |
| Affine | `585.27` |
| Greedy | incomplete; failed entering scale `1` |
| Total before failure | `712.46` |

Failure:

```text
MPS backend out of memory (MPS allocated: 5.77 GB, other allocations: 23.50 GB, max allowed: 30.19 GB). Tried to allocate 1.18 GB on private pool.
```

Conclusion: 24 GB unified memory gets farther than the 16 GB machine and clears the previous full-resolution Affine barrier, but it still cannot complete the GUI-default full-scale `cc` profile for this pair on MPS. The next blocker is the Greedy scale `1` transition.

## Hardware Expectations

### 16 GB Mac with MPS

Not enough for GUI-default full-resolution `cc` on this data. Reduced/coarse profiles can run, and output writing can work with CPU fallback, but full-resolution CC at scale `1` exceeds the MPS memory ceiling.

### 24 GB M4 Pro

Tested on the one-pair CRHA gate. With `PYTORCH_ENABLE_MPS_FALLBACK=1`, 24 GB gets past the Affine scale `1` failure seen on the 16 GB machine, but still fails entering Greedy scale `1` with MPS out of memory. The Moments determinant calculation has since been routed through CPU on MPS, so fallback-free reruns can get past that specific unsupported-op blocker.

### 32 GB+ Unified Memory

More plausible for full-default MPS testing, but still should be gated one pair at a time.

### CUDA GPU with 24 GB VRAM

Potentially better suited than MPS for this workload, especially if fused operations are available. It still needs a one-pair gate because registration quality, not just memory, is a blocker for the current defaults.

## Recommendations

Use a gate before any full two-pair run:

1. Run one moving bridge first.
2. Confirm `row_complete` and `batch_complete`.
3. Confirm output geometry.
4. Compare `initial_ncc/final_ncc`, `initial_mse/final_mse`, and `initial_mae/final_mae`.
5. Only run both moving bridges if the one-pair gate improves or at least does not degrade metrics.

Do not use the current GUI-default profile as the final recommended profile for this data. It completed on CPU but degraded proxy metrics. Profile tuning should be the next step.

Likely tuning directions:

- Keep coarse-to-fine registration but avoid or reduce full-resolution `cc` iterations at scale `1`.
- Test `mse` or smaller CC kernels as cheaper alternatives.
- Use `moments_scale > 1` for initialization if full-resolution moments is too costly.
- Compare affine-only versus affine plus Greedy to isolate where quality degrades.
- Add visual slice overlays or downsampled QA images next to numeric metrics.
- Re-run fallback-free MPS gates after profile tuning to confirm the next remaining blocker.

## Memory-Reduction Strategies to Revisit

Recommended first experiments:

1. Start from the GUI `Memory Saver` preset, which keeps full-resolution Affine but skips full-resolution Greedy on MPS by using Greedy scales `[4, 2]`, then still writes final warped outputs in fixed-image space.
2. Try cheaper full-resolution losses: use `mse`, a smaller CC kernel, or a hybrid profile with `cc` at coarse scales and a cheaper final stage.
3. Test Greedy Adam with `optimizer_params={"reset": True}` so optimizer state is reinitialized instead of interpolated between scales.
4. Test Greedy with `optimizer="SGD"` to avoid Adam's two full-size state tensors.
5. Test Greedy Adam with `optimizer_params={"offload": True}` to keep optimizer state on CPU when MPS memory is tight.

Implementation ideas if profile tuning is not enough:

- Reduce the MPS 3D interpolation peak by writing chunked results into a preallocated output tensor instead of collecting chunks and concatenating them.
- Add stronger cleanup before scale transitions: delete previous-scale images, warp temporaries, loss tensors, and optimizer buffers before clearing/synchronizing MPS cache.
- Route optimizer-state resize operations through CPU on MPS when the alternative is exceeding the private MPS memory pool.

Quality caveat: the CPU full-default gate completed but degraded proxy metrics, so lower-memory profiles must be judged by both completion and registration quality.

## 2026-05-18 ANTs-like Empty-Stack Diagnosis

The GUI run below reproduced the apparent empty-stack failure:

```text
/Users/ddharmap/dataProcessing/testReg/fireants_gui_run_20260518_163916
```

Both ANTs-like SyN rows were marked complete, but the warped NIfTI outputs contained only zeros and the final metrics were `NaN`. The event stream showed non-finite losses during Rigid/Affine/SyN before transform saving, so the empty stacks were a numerical divergence symptom rather than a SimpleITK write failure.

Follow-up fix:

- GUI hover help no longer writes to the bottom status bar.
- Rigid, Affine, and SyN now fail immediately on non-finite loss or transform state.
- GUI output writing now rejects non-finite, all-zero, or constant warped outputs for nonzero source images.
- The ANTs-like SyN GUI preset keeps the same stage order but uses `mse` for Rigid and Affine to avoid the known MI-on-MPS TIFF divergence path.

## Useful Commands

Focused GUI runner tests:

```bash
python -m pytest -q tests/test_gui_runner.py
```

Inspect pair metrics:

```bash
cat /Users/ddharmap/dataProcessing/testReg/<run_folder>/_metrics/pairs.csv
```

Inspect final events:

```bash
tail -n 20 /Users/ddharmap/dataProcessing/testReg/<run_folder>/_metrics/events.jsonl
```

Check output geometry with SimpleITK:

```python
import SimpleITK as sitk

fixed = sitk.ReadImage("fixed.nrrd")
warped = sitk.ReadImage("moving_warped.nii.gz")
warp = sitk.ReadImage("moving_0Warp.nii.gz")

print(fixed.GetSize(), warped.GetSize())
print(fixed.GetSpacing(), warped.GetSpacing())
print(warp.GetSize(), warp.GetNumberOfComponentsPerPixel())
```
