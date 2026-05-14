# Image And Transform Semantics

Purpose: preserve FireANTs image, metadata, segmentation, keypoint, and transform conventions.

Use this file before changing `fireants/io`, transform utilities, metadata handling, keypoints, or segmentation interpolation behavior.

## Image representation

- `Image` wraps a SimpleITK image and stores a PyTorch tensor plus metadata transforms.
- Regular images are stored as `[1, C, *spatial]`; single-channel images get an explicit channel dimension.
- Segmentation images can be one-hot or generic-label depending on `is_onehot` and fused-op availability.
- Only 2D and 3D images are supported.
- Spacing, origin, direction, and optional orientation/center adjustments determine coordinate transforms.

## Coordinate conventions

- Preserve conversions between pixel, physical, and torch normalized coordinates.
- Registration code should use package-level transform helpers instead of reconstructing metadata math in scripts.
- If a writer changes transform or metadata semantics, verify the first downstream registration or wrapper consumer.

## Batching and keypoints

- `BatchedImages` owns image batching for registration.
- `FakeBatchedImages` is for tensor-backed batch-like use where full `Image` objects are not needed.
- `Keypoints` and `BatchedKeypoints` own keypoint spaces and distance reductions.

## Masks and segmentation

- Image-mask concatenation helpers live in `fireants/io/imagemask.py`.
- Masked registration treats the last channel as mask when masked loss mode is active.
- Generic-label interpolation depends on fused ops; without fused ops, segmentation loading forces one-hot mode.
