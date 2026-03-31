from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def wolfe_powell_line_search(
    xk: np.ndarray,
    dk: np.ndarray,
    objective: Objective,
    alpha: float = 1.0,
    beta: float = 0.5,
    sigma1: float = 1e-4,
    sigma2: float = 0.9,
    rho: float = 0.5,
    max_iter: int = 10_000,
    **_: object,
) -> Tuple[float, int]:
    """
    Wolfe-Powell 条件步长搜索。

    条件 1（Armijo）：
        f(xk + alpha*dk) <= f(xk) + sigma1 * alpha * <grad f(xk), dk>
    条件 2（曲率）：
        <grad f(xk + alpha*dk), dk> >= sigma2 * <grad f(xk), dk>
    """


    # Wolfe-Powell 参数合法性检查。
    assert 0 < sigma1 < sigma2 < 1, "require 0 < sigma1 < sigma2 < 1."
    assert beta > 0, "beta must be positive."
    assert 0 < rho < 1, "rho must satisfy 0 < rho < 1."

    # 预先计算 f(xk)、gk 和方向导数，避免重复计算。
    fxk = objective.value(xk)
    grad_xk = objective.gradient(xk)
    # 方向导数 g_k^T d_k，表示沿当前搜索方向的一阶变化率。
    directional_derivative = float(np.dot(grad_xk, dk))

    def _condition1(step: float) -> bool:
        """Armijo 充分下降条件。"""

        # 该条件要求函数值至少取得“足够下降”。
        return objective.value(xk + step * dk) <= fxk + sigma1 * step * directional_derivative

    def _condition2(step: float) -> bool:
        """Wolfe 曲率条件。"""

        # 该条件要求新点处沿 dk 的斜率已经不再过陡。
        return float(np.dot(objective.gradient(xk + step * dk), dk)) >= sigma2 * directional_derivative

    def _phase1(counter: int) -> Tuple[float, int]:
        """从 beta 起回溯，先找到满足 Armijo 条件的点。"""

        iteration = 1
        step = beta
        while not _condition1(step):
            # 第一阶段只做几何缩小，先找一个满足充分下降的候选步长。
            step = (rho**iteration) * beta
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    def _phase2(alpha0: float, beta0: float, counter: int) -> Tuple[float, int]:
        """在 [alpha0, beta0] 内继续调整，直到满足 Armijo 条件。"""

        iteration = 1
        step = beta0
        while not _condition1(step):
            # 第二阶段在区间 [alpha0, beta0] 内从右向左收缩。
            step = alpha0 + (rho**iteration) * (beta0 - alpha0)
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    # 线搜索内部迭代计数。
    counter = 1
    # 若初始步长已满足两条条件，直接使用。
    if _condition1(alpha) and _condition2(alpha):
        # 初始步长已满足 Wolfe-Powell，直接返回。
        return alpha, counter

    # 否则先执行第一阶段，拿到满足 Armijo 条件的候选步长。
    alpha, counter = _phase1(counter)

    # 再检查曲率条件；若还不满足，就扩大上界后进入第二阶段继续试探。
    while counter < max_iter:
        if _condition2(alpha):
            # 同时满足条件 1 和条件 2，返回当前步长。
            return alpha, counter
        # 放大上界并进入第二阶段继续试探。
        beta = alpha / rho
        alpha, counter = _phase2(alpha, beta, counter)

    # 超出最大迭代时返回当前最优可用值。
    return alpha, counter
