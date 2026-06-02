from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def merged_params(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    params = dict(defaults)
    if overrides:
        params.update(dict(overrides))
    return params


def require_source_to_target_matrix(transformation: Any) -> np.ndarray:
    matrix = np.asarray(transformation, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Transformation must be a 4x4 source -> target matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transformation contains non-finite values.")
    return matrix


def correspondence_count(result: Any) -> int | None:
    correspondence_set = getattr(result, "correspondence_set", None)
    if correspondence_set is None:
        return None
    try:
        return len(correspondence_set)
    except TypeError:
        return None
