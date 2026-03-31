from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from optimization.line_search import wolfe_powell_line_search

from ..core import Objective

def update_bk(
    bk: np.ndarray,
    xk_next: np.ndarray,
    xk: np.ndarray,
    objective: Objective
) -> np.ndarray:
    """更新 B_k 矩阵，使用 BFGS 更新公式。
    BFGS 更新公式如下：
    B_{k+1} = B_k - (B_k @ s_k @ s_k^T @ B_k) / (s_k^T @ B_k @ s_k) + (y_k @ y_k^T) / (y_k^T @ s_k)
    其中 s_k = x_{k+1} - x_k，y_k = grad f(x_{k+1}) - grad f(x_k)。
    """
    sk = xk_next - xk
    yk = objective.gradient(xk_next) - objective.gradient(xk)

    bk_next = bk - bk @ sk @ sk.T @ bk / (sk.T @ bk @ sk) + yk @ yk.T / (yk.T @ sk)

    return bk_next

def bfgc(
    x0: np.ndarray,
    objective: Objective,
    max_iter: int = 1000,
    grad_tol: float = 1e-6,
    tol: float = 1e-6
) -> np.ndarray:
    """
    
    """

    converged = False
    xk = x0

    for iteration in range(max_iter):
        gk = objective.gradient(xk_next)
        grad_norm = np.linalg.norm(gk)

        if grad_norm < grad_tol: 
            converged = True
            break
        
        # 确定方向
        dk = np.solve(bk, -objective.gradient(xk))

        # 线搜索
        alpha, _ = wolfe_powell_line_search(
            xk, dk, objective, max_iter=10_000)

        # 更新位置
        xk_next = xk + alpha * dk
        if np.linalg.norm(xk_next - xk) < grad_tol:
            converged = True
            break
        else:
            bk = update_bk(bk, xk_next, xk, objective)
            xk = xk_next
    
    # 返回统一五元组。
    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )