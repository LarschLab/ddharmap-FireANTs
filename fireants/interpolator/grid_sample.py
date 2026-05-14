# Copyright (c) 2026 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


'''
Implementation of the pytorch grid sample interpolator (combining affine and grid_sample)
'''
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Sequence
import logging
logger = logging.getLogger(__name__)


def get_min_coords3d(Z, Y, X, align_corners):
    if not align_corners:
        return -1.0 + 1.0/X, -1.0 + 1.0/Y, -1.0 + 1.0/Z
    return -1.0, -1.0, -1.0

# ZYX order
def get_max_coords3d(Z, Y, X, align_corners):
    if not align_corners:
        return 1.0 - 1.0/X, 1.0 - 1.0/Y, 1.0 - 1.0/Z
    return 1.0, 1.0, 1.0

def get_min_coords2d(Y, X, align_corners):
    if not align_corners:
        return -1.0 + 1.0/X, -1.0 + 1.0/Y
    return -1.0, -1.0

# ZYX order
def get_max_coords2d(Y, X, align_corners):
    if not align_corners:
        return 1.0 - 1.0/X, 1.0 - 1.0/Y
    return 1.0, 1.0


def _is_mps_3d(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "mps" and tensor.ndim == 5


def _normalize_spatial_size(size, input_size: Sequence[int], scale_factor=None):
    if size is not None:
        return tuple(int(s) for s in size[-3:])
    if scale_factor is None:
        return tuple(input_size)
    if isinstance(scale_factor, (int, float)):
        factors = [float(scale_factor)] * 3
    else:
        factors = [float(s) for s in scale_factor]
    return tuple(int(input_size[i] * factors[i]) for i in range(3))


def _mps_grid_sample_3d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    if mode not in ("bilinear", "trilinear", "nearest"):
        raise NotImplementedError(f"MPS 3D grid sampler does not support mode={mode}")
    if padding_mode not in ("zeros", "border"):
        raise NotImplementedError(f"MPS 3D grid sampler does not support padding_mode={padding_mode}")

    B, C, Z, Y, X = input.shape
    gx, gy, gz = grid.unbind(-1)
    if align_corners:
        ix = (gx + 1.0) * 0.5 * (X - 1)
        iy = (gy + 1.0) * 0.5 * (Y - 1)
        iz = (gz + 1.0) * 0.5 * (Z - 1)
    else:
        ix = ((gx + 1.0) * X - 1.0) * 0.5
        iy = ((gy + 1.0) * Y - 1.0) * 0.5
        iz = ((gz + 1.0) * Z - 1.0) * 0.5

    flat_input = input.reshape(B, C, Z * Y * X)
    out_shape = grid.shape[1:-1]

    def gather_at(zidx: torch.Tensor, yidx: torch.Tensor, xidx: torch.Tensor) -> torch.Tensor:
        if padding_mode == "border":
            valid = None
        else:
            valid = (zidx >= 0) & (zidx < Z) & (yidx >= 0) & (yidx < Y) & (xidx >= 0) & (xidx < X)
        zidx = zidx.clamp(0, Z - 1)
        yidx = yidx.clamp(0, Y - 1)
        xidx = xidx.clamp(0, X - 1)
        flat_idx = (zidx * Y * X + yidx * X + xidx).reshape(B, -1)
        values = flat_input.gather(2, flat_idx[:, None].expand(-1, C, -1)).reshape(B, C, *out_shape)
        if valid is not None:
            values = values * valid[:, None].to(input.dtype)
        return values

    if mode == "nearest":
        return gather_at(iz.round().long(), iy.round().long(), ix.round().long())

    x0 = torch.floor(ix).long()
    y0 = torch.floor(iy).long()
    z0 = torch.floor(iz).long()
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    wx1 = (ix - x0.to(ix.dtype)).unsqueeze(1).to(input.dtype)
    wy1 = (iy - y0.to(iy.dtype)).unsqueeze(1).to(input.dtype)
    wz1 = (iz - z0.to(iz.dtype)).unsqueeze(1).to(input.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    c000 = gather_at(z0, y0, x0)
    c001 = gather_at(z0, y0, x1)
    c010 = gather_at(z0, y1, x0)
    c011 = gather_at(z0, y1, x1)
    c100 = gather_at(z1, y0, x0)
    c101 = gather_at(z1, y0, x1)
    c110 = gather_at(z1, y1, x0)
    c111 = gather_at(z1, y1, x1)

    return (
        c000 * wx0 * wy0 * wz0
        + c001 * wx1 * wy0 * wz0
        + c010 * wx0 * wy1 * wz0
        + c011 * wx1 * wy1 * wz0
        + c100 * wx0 * wy0 * wz1
        + c101 * wx1 * wy0 * wz1
        + c110 * wx0 * wy1 * wz1
        + c111 * wx1 * wy1 * wz1
    )


def _grid_sample_3d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    if _is_mps_3d(input):
        return _mps_grid_sample_3d(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def device_aware_interpolate(
    input: torch.Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
):
    if not (_is_mps_3d(input) and mode in ("trilinear", "bilinear")):
        return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    align_corners = False if align_corners is None else align_corners
    output_size = _normalize_spatial_size(size, input.shape[2:], scale_factor)
    if tuple(input.shape[2:]) == output_size:
        return input

    B, C = input.shape[:2]
    affine = torch.eye(3, 4, device=input.device, dtype=input.dtype)[None].expand(B, -1, -1)
    grid = F.affine_grid(affine, (B, C, *output_size), align_corners=align_corners)
    return _mps_grid_sample_3d(input, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)

def torch_grid_sampler_2d(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    out_shape: tuple = None,
    is_displacement: bool = True
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    1. Affine-only transformation (grid=None)
    2. Full warp field (grid provided, is_displacement=False)
    3. Displacement field (grid provided, is_displacement=True)
    
    Args:
        input: Input tensor of shape [B, C, Y, X]
        affine: Affine matrix of shape [B, 2, 3]
        grid: Optional grid tensor of shape [B, Y, X, 2]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Y, X]
    """
    B, C, Y, X = input.shape
    
    # Case 1: Affine-only transformation
    if grid is None:
        if affine is None:
            raise ValueError("Either grid or affine must be provided")
        if out_shape is None:
            logger.warning("out_shape is not provided for affine-only transformation, using input shape")
            out_shape = (Y, X)
        grid = F.affine_grid(affine, (B, C, *out_shape[-2:]), align_corners=align_corners).to(input.dtype)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 2: Full warp field
    if not is_displacement:
        grid = grid.to(input.dtype)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine is None:
        affine = torch.eye(2, 3, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine, (B, C, *out_shape[-2:]), align_corners=align_corners).to(input.dtype)
    # Add displacement
    base_grid = base_grid + grid.to(input.dtype)
    return F.grid_sample(input, base_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def torch_grid_sampler_3d(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    grid_affine: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    out_shape: tuple = None,
    is_displacement: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    1. Affine-only transformation (grid=None)
    2. Full warp field (grid provided, is_displacement=False)
    3. Displacement field (grid provided, is_displacement=True)
    
    Args:
        input: Input tensor of shape [B, C, Z, Y, X]
        affine: Affine matrix of shape [B, 3, 4]
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    B, C, Z, Y, X = input.shape

    # Case 1: Affine-only transformation
    if grid is None:
        if affine is None:
            raise ValueError("Either grid or affine must be provided")
        if out_shape is None:
            logger.warning("out_shape is not provided for affine-only transformation, using input shape")
            out_shape = (Z, Y, X)
        grid = F.affine_grid(affine, (B, C, *out_shape[-3:]), align_corners=align_corners).to(input.dtype)
        ret = _grid_sample_3d(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        if output is not None:
            output.add_(ret)
            return output
        return ret
    
    # see if grid affine
    if grid_affine is not None:
        grid = torch.einsum('bij,b...j->b...i', grid_affine, grid)

    # Case 2: Full warp field
    if not is_displacement:
        grid = grid.to(input.dtype)
        ret = _grid_sample_3d(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        if output is not None:
            output.add_(ret)
            return output
        return ret
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine is None:
        affine = torch.eye(3, 4, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine, (B, C, *out_shape[-3:]), align_corners=align_corners).to(input.dtype)
    # Add displacement
    base_grid = base_grid + grid.to(input.dtype)
    ret = _grid_sample_3d(input, base_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if output is not None:
        output.add_(ret)
        return output
    return ret


def torch_warp_composer_2d(
    u: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    v: torch.Tensor = None,
    align_corners: bool = True,
    grid: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    warp = u \circ (Ax + v)
    
    Args:
        input: Input tensor of shape [B, Yi, Xi, 2]
        affine: Affine matrix of shape [B, 2, 3]
        grid: Optional grid tensor of shape [B, Y, X, 2]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Y, X]
    """
    mode = "bilinear"
    padding_mode = "zeros"
    B = v.shape[0]
    # if grid is not provided, create it from affine
    if grid is None:
        if affine is None:
            affine = torch.eye(2, 3, device=u.device)[None].expand(B, -1, -1)
        grid = F.affine_grid(affine, [B, 1] + list(v.shape[1:-1]), align_corners=align_corners).to(u.dtype)
    sample_grid = (grid + v).to(u.dtype)
    ret = F.grid_sample(u.permute(0, 3, 1, 2), sample_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners).permute(0, 2, 3, 1)
    # add inplace if specified (adam update step)
    if output is not None:
        output.add_(ret)
    else:
        output = ret
    return output

def torch_warp_composer_3d(
    u: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    v: torch.Tensor = None,
    align_corners: bool = True,
    grid: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    
    Args:
        input: Input tensor of shape [B, Z, Y, X, 3]
        affine: Affine matrix of shape [B, 3, 4]
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    mode = "bilinear"
    padding_mode = "zeros"
    B = v.shape[0]
    if grid is None:
        if affine is None:
            affine = torch.eye(3, 4, device=u.device)[None].expand(B, -1, -1)
        grid = F.affine_grid(affine, [B, 1] + list(v.shape[1:-1]), align_corners=align_corners).to(u.dtype)
    sample_grid = (grid + v).to(u.dtype)
    ret = _grid_sample_3d(u.permute(0, 4, 1, 2, 3), sample_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners).permute(0, 2, 3, 4, 1)
    if output is not None:  
        output.add_(ret)
    else:
        output = ret
    return output

def torch_affine_warp_3d(
    affine: Optional[torch.Tensor],
    grid: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    B = grid.shape[0]
    if affine is None:
        affine = torch.eye(3, 4, device=grid.device)[None].expand(B, -1, -1)
    return F.affine_grid(affine, [B, 1] + list(grid.shape[1:-1]), align_corners=align_corners) + grid

def torch_affine_warp_2d(
    affine: Optional[torch.Tensor],
    grid: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    B = grid.shape[0]
    if affine is None:
        affine = torch.eye(2, 3, device=grid.device)[None].expand(B, -1, -1)
    return F.affine_grid(affine, [B, 1] + list(grid.shape[1:-1]), align_corners=align_corners) + grid
