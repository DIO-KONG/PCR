# Algorithm Notes

## ransac_only
- 依赖 Open3D。
- 输入为已经完成下采样、normal、FPFH 的 source/target。
- 使用 FPFH 特征匹配和 RANSAC 输出 source -> target 矩阵。
- 缺失 Open3D 时返回 `skipped`，不伪造成功。
