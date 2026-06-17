from __future__ import annotations

from algorithm import shared_frame_alignment

ALGORITHMS = {
    "shared_frame_alignment": shared_frame_alignment.register,
}


def get_algorithm(name: str):
    """按配置名获取算法入口。

    注册表只保存明确启用的算法，避免实验时误运行已经废弃的方案。
    """

    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registration algorithm: {name}. Available={sorted(ALGORITHMS)}") from exc
