from __future__ import annotations

from pathlib import Path

import numpy as np


def load_da3_batch(path: str | Path) -> dict[str, np.ndarray]:
    """读取 DA3 batch NPZ，并转成普通 dict。

    `np.load` 返回的对象依赖文件句柄生命周期。这里立即复制成普通 dict，
    方便算法模块安全传递。
    """

    data = np.load(Path(path), allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def image_names(batch: dict[str, np.ndarray]) -> list[str]:
    """返回 DA3 batch 中的图像名列表。"""

    return [str(item) for item in batch["image_names"]]
