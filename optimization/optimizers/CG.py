from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from optimization.line_search import wolfe_powell_line_search

from ..core import Objective

def cg(
    x0: np.ndarray,
    objective: Objective,
    grad_tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    
    converged = False
    xk = x0
    dk = -objective.gradient(xk)

    for interation in range(max_iter):
        if np.linalg.norm(objective.gradient(xk)) < grad_tol:
            converged = True
            break
        # 线搜索
        alpha, _ = wolfe_powell_line_search(xk, dk, objective, max_iter=10_000)
        # 更新位置
        xk_next = xk + alpha * dk

        # 更新方向
        beta = np.dot(objective.gradient(xk_next), objective.gradient(xk_next)) / np.dot(objective.gradient(xk), objective.gradient(xk))
        dk = -objective.gradient(xk_next) + beta * dk

        xk = xk_next
