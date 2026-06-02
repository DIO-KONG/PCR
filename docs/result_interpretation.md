# Result Interpretation

## 状态
- `success`：算法成功输出 source -> target 变换矩阵。
- `failed`：算法执行失败，查看 `error` 字段。
- `skipped`：缺失依赖或条件不足，未执行算法。

## 指标
- `algorithm_fitness` / `algorithm_inlier_rmse`：算法内部返回的优化指标，只反映该算法自身定义。
- `eval_fitness` / `eval_inlier_rmse`：统一 evaluator 按公共阈值重新计算的指标，用于跨算法比较。
- `eval_median_nn_dist`：变换后 source 到 target 最近邻距离中位数。
- `eval_trimmed_mean_nn_dist`：按 `trimmed_ratio` 去除尾部异常距离后的平均最近邻距离。
- `eval_overlap_ratio`：最近邻距离小于 `overlap_threshold` 的 source 点比例。
- `eval_det_R`：旋转矩阵行列式，应接近 1。
- `eval_orthogonality_error`：旋转矩阵正交误差，接近 0 更好。
- `eval_translation_norm`：平移向量长度。

## 时间
- `preprocess_time`：预处理和缓存阶段耗时。
- `algorithm_time`：算法本体耗时。
- `evaluation_time`：统一评估阶段耗时。
- `artifact_time`：结果文件保存耗时。
- `total_time`：单条 pairwise 任务总耗时。

## artifact
- `success` 且 `has_valid_transform=true`：保存 matrix、registered cloud、overlay cloud、metrics 和 report。
- `failed` / `skipped`：transformation 为 None，只保存 metrics 和 report，错误写入 `error` / `metric_error`。
