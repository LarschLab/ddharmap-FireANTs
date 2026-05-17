# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.

from typing import Any, Callable, Dict, Optional


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def emit_progress(callback: ProgressCallback, **event: Any) -> None:
    """Emit a best-effort progress event."""
    if callback is None:
        return
    callback(event)
