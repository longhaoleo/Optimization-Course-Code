from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def golden_ratio_line_search(
    xk: np.ndarray,
    dk: np.ndarray,
    objective: Objective,
    ak: float = 0.0,
    bk: float = 2.0,
    tol: float = 1e-5,
    max_iter: int = 10_000,
) -> Tuple[float, int]:
    """
    黄金分割精确线搜索。

    多维问题在给定方向 dk 上转化为一维问题：
        phi(alpha) = f(xk + alpha * dk)
    然后在区间 [ak, bk] 内近似求解 argmin phi(alpha)。
    """

    # 区间与精度参数基本合法性。
    assert ak < bk, "ak must satisfy ak < bk."
    assert tol > 0, "tol must be positive."

    # 一维投影函数。
    phi = lambda alpha: objective.value(xk + alpha * dk)
    # 黄金分割比例 (sqrt(5)-1)/2。
    ratio = (np.sqrt(5.0) - 1.0) / 2.0

    # 区间内两个测试点：lam 在左，mu 在右。
    lam = bk - ratio * (bk - ak)
    mu = ak + ratio * (bk - ak)
    # 缓存函数值，避免重复计算。
    phi_lam = phi(lam)
    phi_mu = phi(mu)
    # 内部迭代计数。
    counter = 1

    # 当区间长度未达到容差且未到上限时持续收缩。
    while abs(bk - ak) >= tol and counter < max_iter:
        if phi_lam < phi_mu:
            # 左侧更优 -> 极小值落在 [ak, mu]，收缩右端。
            bk = mu
            mu = lam
            phi_mu = phi_lam
            lam = bk - ratio * (bk - ak)
            phi_lam = phi(lam)
        else:
            # 右侧更优 -> 极小值落在 [lam, bk]，收缩左端。
            ak = lam
            lam = mu
            phi_lam = phi_mu
            mu = ak + ratio * (bk - ak)
            phi_mu = phi(mu)
        # 每次区间收缩后计数 +1。
        counter += 1

    # 返回区间中点作为最终步长估计，并返回迭代计数。
    return (ak + bk) / 2.0, counter
