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

import torch


def device_type(device=None) -> str:
    if isinstance(device, torch.Tensor):
        return device.device.type
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return torch.device(device).type


def empty_device_cache(device=None) -> None:
    dtype = device_type(device)
    if dtype == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dtype == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def synchronize_device(device=None) -> None:
    dtype = device_type(device)
    if dtype == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dtype == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def device_memory_allocated(device=None) -> float:
    dtype = device_type(device)
    if dtype == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    if dtype == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0
