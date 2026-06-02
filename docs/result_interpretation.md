# Result Interpretation

## 状态
- `success`：算法成功输出 source -> target 变换矩阵。
- `failed`：算法执行失败，查看 `error` 字段。
- `skipped`：缺失依赖或条件不足，未执行算法。

## 指标
- `fitness`：Open3D registration 的内点比例，越高通常越好。
- `inlier_rmse`：Open3D registration 的内点 RMSE，越低通常越好。
- `median_nn_dist`：变换后 source 到 target 最近邻距离中位数。
- `trimmed_mean_nn_dist`：按 `trimmed_ratio` 去除尾部异常距离后的平均最近邻距离。
- `overlap_ratio`：最近邻距离小于 `overlap_threshold` 的 source 点比例。
- `det_R`：旋转矩阵行列式，应接近 1。
- `orthogonality_error`：旋转矩阵正交误差，接近 0 更好。
- `translation_norm`：平移向量长度。

## artifact
- `success`：保存 matrix、registered cloud、overlay cloud、metrics 和 report。
- `failed` / `skipped`：只保存 metrics 和 report，错误写入 `error` / `metric_error`。
