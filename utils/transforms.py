from __future__ import annotations

from typing import Any

import numpy as np


def validate_transform(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Transform must be 4x4.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transform contains non-finite values.")
    return matrix


def inverse_transform(matrix: Any) -> np.ndarray:
    return np.linalg.inv(validate_transform(matrix))


def orthogonality_error(matrix: Any) -> float:
    matrix = validate_transform(matrix)
    rotation = matrix[:3, :3]
    return float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))


def compare_inverse(matrix: Any, other: Any) -> float:
    return float(np.linalg.norm(inverse_transform(matrix) - validate_transform(other), ord="fro"))
