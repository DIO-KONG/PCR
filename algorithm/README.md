# algorithm

本目录现在只保留算法说明文档，不再承载实际数值实现。

当前 shared-frame 数值代码已经迁移到：

```text
pcr/algorithms/shared_frame/correspondences.py
pcr/algorithms/shared_frame/coarse.py
pcr/algorithms/shared_frame/frame_selection.py
pcr/algorithms/shared_frame/pairwise.py
pcr/algorithms/refinement/icp.py
```

每个正式算法都应补充一个 Markdown 说明文件，解释输入假设、核心步骤、主要参数、输出指标和已知局限。当前共享帧算法说明见 `shared_frame_alignment.md`。
