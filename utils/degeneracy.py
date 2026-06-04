from __future__ import annotations

import numpy as np


def plane_degeneracy(points: np.ndarray) -> float | None:
    if len(points) < 4:
        return None
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if eig[0] <= 0.0:
        return None
    # High when points mostly lie on a plane.
    return float(max((eig[1] - eig[2]) / eig[0], 0.0))
