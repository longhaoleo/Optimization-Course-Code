from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def bb(
    xk: np.ndarray,
    xk_prev: np.ndarray,
    objective: Objective
) -> Tuple[float, int]:
    """
    Barzilai-Borwein (BB) 线搜索。

    该方法通过近似 Hessian 的逆来计算步长，具有较快的收敛速度。
    具体计算方式如下：

    1. 计算 s_k = x_k - x_{k-1} 和 y_k = grad f(x_k) - grad f(x_{k-1})。
    2. 根据 s_k 和 y_k 计算步长 alpha_k：
        alpha_k = (s_k^T s_k) / (s_k^T y_k) 
        或 alpha_k = (s_k^T y_k) / (y_k^T y_k)。
    """

    # 计算 s_k 和 y_k。
    sk = xk - xk_prev
    yk = objective.gradient(xk) - objective.gradient(xk_prev)

    # 计算步长 alpha_k。
    alpha_bb1 = np.dot(sk, sk) / np.dot(sk, yk)
    alpha_bb2 = np.dot(sk, yk) / np.dot(yk, yk)

    # 选择合适的步长，通常选择较小的一个以保证稳定性。
    alpha = min(alpha_bb1, alpha_bb2)

    # 返回计算得到的步长和迭代次数。
    return alpha, 0