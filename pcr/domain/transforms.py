from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class Transform:
    """带坐标系语义的 4x4 刚体变换。

    全项目统一约定：

    ```text
    p_target = T_source_to_target @ p_source
    ```

    以前代码里大量裸 `np.ndarray` 只靠变量名表达方向，后续维护时很容易把
    `source -> target` 和 `target -> source` 写反。本类把 source/target
    放进数据结构，并在 compose 时做运行时校验。
    """

    source: str
    target: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {matrix.shape}")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls, frame: str) -> "Transform":
        """构造 `frame -> frame` 的单位变换。"""

        return cls(source=frame, target=frame, matrix=np.eye(4, dtype=float))

    def inverse(self) -> "Transform":
        """返回反向变换，即 `target -> source`。"""

        return Transform(source=self.target, target=self.source, matrix=np.linalg.inv(self.matrix))

    def then(self, next_transform: "Transform") -> "Transform":
        """右接另一个变换。

        若 `self` 是 `A -> B`，`next_transform` 是 `B -> C`，则结果为
        `A -> C`，矩阵为 `T_B_to_C @ T_A_to_B`。
        """

        if self.target != next_transform.source:
            raise ValueError(
                "Cannot compose transforms: "
                f"{self.source}->{self.target} then "
                f"{next_transform.source}->{next_transform.target}"
            )
        return Transform(
            source=self.source,
            target=next_transform.target,
            matrix=next_transform.matrix @ self.matrix,
        )

    def after(self, previous_transform: "Transform") -> "Transform":
        """可读别名：`previous_transform.then(self)`。

        仅在少量自然语言更适合“在 previous 之后执行当前变换”时使用。
        主流程优先使用 `then()`，减少矩阵方向误读。
        """

        return previous_transform.then(self)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """把 Nx3 点从 source 坐标系变换到 target 坐标系。"""

        points = np.asarray(points, dtype=float)
        homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
        return (self.matrix @ homogeneous.T).T[:, :3]

    def apply_cloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """复制点云并应用变换，避免原地修改调用方持有的点云。"""

        result = o3d.geometry.PointCloud(cloud)
        result.transform(self.matrix)
        return result

    def to_json(self) -> dict[str, object]:
        """转换成 artifact 可序列化结构。"""

        return {
            "source": self.source,
            "target": self.target,
            "matrix": self.matrix.tolist(),
        }
