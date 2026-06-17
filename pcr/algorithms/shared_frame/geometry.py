from __future__ import annotations

import numpy as np


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用 Kabsch/SVD 估计无尺度刚体变换 `target ~= T @ source`。"""

    if len(source) < 3:
        raise ValueError("At least 3 correspondences are required for rigid transform.")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """应用 4x4 齐次变换到 Nx3 点。"""

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    return (matrix @ homogeneous.T).T[:, :3]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """计算 3x3 旋转矩阵对应的旋转角，单位为度。"""

    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

