from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def armijo_line_search(
    xk: np.ndarray,
    dk: np.ndarray,
    objective: Objective,
    alpha: float = 1.0,
    sigma1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 10_000,
    **_: object,
) -> Tuple[float, int]:
    """
    Armijo 回溯步长搜索。

    下降条件：
        f(xk + alpha*dk) <= f(xk) + sigma1 * alpha * <grad f(xk), dk>
    若不满足，则令 alpha <- rho * alpha 持续回溯。
    """

    # Armijo 参数基本合法性检查。
    assert 0 < sigma1 < 0.5, "sigma1 must satisfy 0 < sigma1 < 0.5."
    assert 0 < rho < 1, "rho must satisfy 0 < rho < 1."

    # 当前点函数值 f(xk)。
    fxk = objective.value(xk)
    # 方向导数 <grad f(xk), dk>，回溯中会重复使用。
    directional_derivative = float(np.dot(objective.gradient(xk), dk))

    def _condition(step: float) -> bool:
        """Armijo 充分下降条件。"""

        return objective.value(xk + step * dk) <= fxk + sigma1 * step * directional_derivative

    # 计数器记录线搜索内部迭代次数。
    counter = 1

    # 不满足 Armijo 条件时，按比例缩短步长。
    while not _condition(alpha):
        # 回溯：alpha <- rho * alpha。
        alpha *= rho
        # 更新迭代计数。
        counter += 1
        # 达到上限后退出，返回当前可用步长。
        if counter >= max_iter:
            break

    # 返回步长与回溯次数。
    return alpha, counter
