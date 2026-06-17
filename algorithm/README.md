# algorithm

本目录放置点云配准算法实现。每个算法文件都暴露统一入口：

```python
def register(source_path, target_path, config) -> RegistrationResult:
    ...
```

当前算法：

- `shared_frame_alignment`：利用两个 DA3 batch 的共享 RGB-D 帧，在同一像素上建立 3D 对应关系，并用多阈值 RANSAC + Kabsch/SVD + 逐步收紧 refit 估计无尺度刚体变换；候选选定后可执行有边界的 point-to-plane ICP 刚体微调。

算法内部通过 import 调用 `utils/`，不通过 CLI 串流程。

每个正式算法都应补充一个同名 Markdown 说明文件，解释输入假设、核心步骤、主要参数、输出指标和已知局限。当前共享帧算法说明见 `shared_frame_alignment.md`。
