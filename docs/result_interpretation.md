# Result Interpretation

## 状态
- `success`：算法成功输出 source -> target 变换矩阵。
- `failed`：算法执行失败，查看 `error` 字段。
- `skipped`：缺失依赖或条件不足，未执行算法。

## 指标
- `fitness`：Open3D registration 的内点比例，越高通常越好。
- `inlier_rmse`：Open3D registration 的内点 RMSE，越低通常越好。
- `rmse`：变换后 source 到 target 最近邻距离 RMSE。
- `trimmed_distance`：去除尾部异常距离后的平均距离。
- `orthogonality_error`：旋转矩阵正交误差，接近 0 更好。
- `determinant`：旋转矩阵行列式，应接近 1。
