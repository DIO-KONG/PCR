from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def motion_prior_error(transform: Any, prior: Mapping[str, Any] | None = None) -> float | None:
    prior = dict(prior or {})
    if not prior.get("enabled", False):
        return None
    matrix = np.asarray(transform, dtype=float)
    error = 0.0
    expected_translation = prior.get("expected_translation")
    if expected_translation is not None:
        expected = np.asarray(expected_translation, dtype=float)
        error += float(np.linalg.norm(matrix[:3, 3] - expected))
    max_translation_norm = prior.get("max_translation_norm")
    if max_translation_norm is not None:
        over = max(0.0, float(np.linalg.norm(matrix[:3, 3])) - float(max_translation_norm))
        error += over
    return error
