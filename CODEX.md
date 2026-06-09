# Project Invariants

This project is a Depth Anything 3 processing workspace. Keep these invariants intact:

- All Python packages for this project must be installed into `.env/`; do not install into the system or user Python environment.
- Input RGB images live in `da3/data/raw/image/` and use names like `1-rgb.png`.
- Output predictions and fused point clouds live in `da3/data/raw/pointcloud/`.
- Images must be ordered by the numeric filename prefix, not lexicographically.
- The DA3 batches are fixed:
  - Baseline map batch: frames `1..12`.
  - Sliding batches: frames `9..13`, `10..14`, `11..15`, `12..16`, `13..17`, `14..18`.
- The default DA3 model is `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- CUDA is required for the default Nested model. If CUDA is unavailable, fail clearly instead of silently falling back to CPU.
- Generated `.npz` and `.ply` files may be regenerated from the raw images and runner script.
- Point-cloud registration utilities live under `utils/`.
- Registration algorithms live under `algorithm/`.
- Experiment runners and experiment configs live under `testbench/`.
- Experiment outputs live under `result/` and should be treated as derived artifacts.
- `da3/data/raw/pointcloud/` is read-only input for registration experiments.
