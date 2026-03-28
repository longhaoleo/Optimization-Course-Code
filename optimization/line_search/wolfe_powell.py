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
    """Wolfe-Powell 条件线搜索。"""

    assert 0 < sigma1 < sigma2 < 1, "require 0 < sigma1 < sigma2 < 1."
    assert beta > 0, "beta must be positive."
    assert 0 < rho < 1, "rho must satisfy 0 < rho < 1."

    fxk = objective.value(xk)
    grad_xk = objective.gradient(xk)
    directional_derivative = float(np.dot(grad_xk, dk))

    def condition1(step: float) -> bool:
        return objective.value(xk + step * dk) <= fxk + sigma1 * step * directional_derivative

    def condition2(step: float) -> bool:
        return float(np.dot(objective.gradient(xk + step * dk), dk)) >= sigma2 * directional_derivative

    def phase1(counter: int) -> Tuple[float, int]:
        iteration = 1
        step = beta
        while not condition1(step):
            step = (rho**iteration) * beta
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    def phase2(alpha0: float, beta0: float, counter: int) -> Tuple[float, int]:
        iteration = 1
        step = beta0
        while not condition1(step):
            step = alpha0 + (rho**iteration) * (beta0 - alpha0)
            iteration += 1
            if counter + iteration - 1 >= max_iter:
                break
        return step, counter + iteration - 1

    counter = 1
    if condition1(alpha) and condition2(alpha):
        return alpha, counter

    alpha, counter = phase1(counter)

    while counter < max_iter:
        if condition2(alpha):
            return alpha, counter
        beta = alpha / rho
        alpha, counter = phase2(alpha0=alpha, beta0=beta, counter=counter)

    return alpha, counter
