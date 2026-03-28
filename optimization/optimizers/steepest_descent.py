from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..core import Objective

LineSearch = Callable[..., Tuple[float, int]]


def steepest_descent(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch,
    tol: float = 1e-5,
    max_outer_iter: int = 1000,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """最速下降法，线搜索策略通过 line_search_func 插拔。"""

    xk = np.asarray(x0, dtype=float)
    converged = False

    for iteration in range(max_outer_iter):
        gk = objective.gradient(xk)
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < tol:
            converged = True
            break

        dk = -gk
        alpha, _ = line_search_func(xk, dk, objective, **ls_params)
        xk = xk + alpha * dk
    else:
        iteration = max_outer_iter
        grad_norm = float(np.linalg.norm(objective.gradient(xk)))

    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
