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
) -> Tuple[float, int]:
    """
    Wolfe-Powell 条件线搜索。

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
    directional_derivative = float(np.dot(grad_xk, dk))

    def condition1(step: float) -> bool:
        # Armijo 充分下降条件。
        return objective.value(xk + step * dk) <= fxk + sigma1 * step * directional_derivative

    def condition2(step: float) -> bool:
        # Wolfe 曲率条件。
        return float(np.dot(objective.gradient(xk + step * dk), dk)) >= sigma2 * directional_derivative

    def phase1(counter: int) -> Tuple[float, int]:
        # 第一阶段：从 beta 起回溯，先找到满足 condition1 的点。
        iteration = 1
        step = beta
        while not condition1(step):
            # 指数回缩序列：beta, rho*beta, rho^2*beta, ...
            step = (rho**iteration) * beta
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    def phase2(alpha0: float, beta0: float, counter: int) -> Tuple[float, int]:
        # 第二阶段：在 [alpha0, beta0] 内继续调整，仍以满足 condition1 为目标。
        iteration = 1
        step = beta0
        while not condition1(step):
            # 从右端向左收缩，逐步逼近可行点。
            step = alpha0 + (rho**iteration) * (beta0 - alpha0)
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    # 线搜索内部迭代计数。
    counter = 1
    # 若初始步长已满足两条条件，直接使用。
    if condition1(alpha) and condition2(alpha):
        # 初始步长已满足 Wolfe-Powell，直接返回。
        return alpha, counter

    # 否则先执行第一阶段找满足 condition1 的候选点。
    alpha, counter = phase1(counter)

    # 再检查曲率条件，不满足则扩大区间并继续第二阶段搜索。
    while counter < max_iter:
        if condition2(alpha):
            # 同时满足条件 1 和条件 2，返回当前步长。
            return alpha, counter
        # 放大上界并进入第二阶段继续试探。
        beta = alpha / rho
        alpha, counter = phase2(alpha0=alpha, beta0=beta, counter=counter)

    # 超出最大迭代时返回当前最优可用值。
    return alpha, counter
