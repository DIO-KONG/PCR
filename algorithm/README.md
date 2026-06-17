# algorithm

本目录放置点云配准算法实现。每个算法文件都暴露统一入口：

```python
def register(source_path, target_path, config) -> RegistrationResult:
    ...
```

当前方案：

- `fpfh_ransac`：FPFH + RANSAC 粗配准。
- `fpfh_ransac_icp`：FPFH + RANSAC + point-to-plane ICP。
- `fpfh_colored_icp`：FPFH + RANSAC + Colored ICP。
- `teaser_icp`：TEASER++ + point-to-plane ICP，可选依赖 `teaserpp_python`。

算法内部通过 import 调用 `utils/`，不通过 CLI 串流程。
