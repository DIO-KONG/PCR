from __future__ import annotations

from algorithm.fpfh_ransac_icp import register as fpfh_ransac_icp
from algorithm.topview_vote import register as topview_vote


ALGORITHMS = {
    "topview_vote": topview_vote,
    "fpfh_ransac_icp": fpfh_ransac_icp,
}


def get_algorithm(name: str):
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registration algorithm: {name}") from exc
