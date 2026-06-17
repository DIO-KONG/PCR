from __future__ import annotations

from algorithm import fpfh_colored_icp, fpfh_ransac, fpfh_ransac_icp, teaser_icp


ALGORITHMS = {
    "fpfh_ransac": fpfh_ransac.register,
    "fpfh_ransac_icp": fpfh_ransac_icp.register,
    "fpfh_colored_icp": fpfh_colored_icp.register,
    "teaser_icp": teaser_icp.register,
}


def get_algorithm(name: str):
    """按配置名获取算法入口。"""

    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registration algorithm: {name}. Available={sorted(ALGORITHMS)}") from exc
