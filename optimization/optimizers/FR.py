from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective
from ..line_search import wolfe_powell_line_search

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def fr(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch = wolfe_powell_line_search,
    grad_tol: float = 1e-6,
    max_outer_iter: int = 1000,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    非线性共轭梯度法（Fletcher-Reeves）。
    属于 CG 方法
    每一轮先沿当前共轭方向做步长搜索，再用
        beta_k = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)
    更新下一轮搜索方向。

    返回值：
        (x_opt, f_opt, iters, converged, grad_norm)
    """

    xk = np.asarray(x0, dtype=float)
    gk = objective.gradient(xk)
    dk = -gk
    converged = False
    eps = 1e-16

    for iteration in range(max_outer_iter):
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < grad_tol:
            converged = True
            break

        # 保护：若方向不下降，则为最速下降方向。
        if float(np.dot(gk, dk)) >= 0.0:
            dk = -gk

        # 用外部步长搜索在当前方向 dk 上选取 alpha_k。
        alpha, _ = line_search_func(xk, dk, objective, **ls_params,)

        x_next = xk + alpha * dk
        g_next = objective.gradient(x_next)

        # Fletcher-Reeves 系数：只用相邻两次梯度范数构造 beta_k。
        denom = float(np.dot(gk, gk))
        if denom <= eps:
            beta = 0.0
        else:
            beta = float(np.dot(g_next, g_next) / denom)
            beta = max(beta, 0.0)

        # d_{k+1} = -g_{k+1} + beta_k d_k
        d_next = -g_next + beta * dk
        if float(np.dot(g_next, d_next)) >= 0.0:
            # 若新方向失去下降性，改为最速下降。
            d_next = -g_next

        if callback is not None:
            callback(
                iteration + 1,
                x_next.copy(),
                objective.value(x_next),
                grad_norm,
                float(alpha),
            )

        xk = x_next
        gk = g_next
        dk = d_next
    else:
        iteration = max_outer_iter
        grad_norm = float(np.linalg.norm(gk))

    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
