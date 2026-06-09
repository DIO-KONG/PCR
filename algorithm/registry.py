from __future__ import annotations

from algorithm.fpfh_ransac_icp import register as fpfh_ransac_icp
from algorithm.topview_vote import register as topview_vote


ALGORITHMS = {
    # 算法名来自 YAML 配置；value 是对应模块的统一 register(source, target, config) 函数。
    "topview_vote": topview_vote,
    "fpfh_ransac_icp": fpfh_ransac_icp,
}


def get_algorithm(name: str):
    """按配置名获取算法入口。

    统一注册表让 testbench 不需要写 if/else 判断具体算法。
    新增算法时，只需要在这里增加一项映射。
    """

    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registration algorithm: {name}") from exc
