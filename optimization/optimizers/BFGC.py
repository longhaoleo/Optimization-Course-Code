from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective
from ..line_search import wolfe_powell_line_search

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def _solve_bfgs_direction(bk: np.ndarray, gk: np.ndarray) -> np.ndarray:
    """求解 Bk * d = -gk，若数值不稳定则回退到伪逆。"""

    n = bk.shape[0]
    if np.linalg.matrix_rank(bk) == n:
        return -np.linalg.solve(bk, gk)
    return -(np.linalg.pinv(bk) @ gk)


def _bfgs_update(
    bk: np.ndarray,
    sk: np.ndarray,
    yk: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """执行一次 BFGS 更新，必要时跳过退化更新。"""

    yTs = float(np.dot(yk, sk))
    if yTs <= eps:
        # 曲率条件过弱时不更新，避免把近似 Hessian 修坏。
        return bk

    bks = bk @ sk
    sTbs = float(np.dot(sk, bks))
    if sTbs <= eps:
        # 分母过小意味着当前 Bk 在 sk 方向上数值不稳定。
        return bk

    # BFGS 公式：
    #   B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)
    term1 = np.outer(bks, bks) / sTbs
    term2 = np.outer(yk, yk) / yTs
    bk_next = bk - term1 + term2

    # 数值上保持对称。
    return 0.5 * (bk_next + bk_next.T)


def bfgc(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch = wolfe_powell_line_search,
    grad_tol: float = 1e-6,
    max_outer_iter: int = 1000,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    BFGS 拟牛顿法。

    这里用 B_k 近似 Hessian，并在每轮解
        B_k d_k = -g_k
    得到搜索方向，再用 BFGS 公式更新 B_k。

    返回值：
        (x_opt, f_opt, iters, converged, grad_norm)
    """

    xk = np.asarray(x0, dtype=float)
    n = xk.size
    bk = np.eye(n, dtype=float)
    gk = objective.gradient(xk)
    converged = False


    for iteration in range(max_outer_iter):
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < grad_tol:
            converged = True
            break

        # 先由当前近似 Hessian Bk 求方向。
        dk = _solve_bfgs_direction(bk, gk)
        if float(np.dot(gk, dk)) >= 0.0:
            # 若求出的方向不是下降方向，则退回最速下降方向。
            dk = -gk

        # 再在该方向上做步长搜索，得到 alpha_k。
        alpha, _ = line_search_func(xk, dk, objective, **ls_params)

        x_next = xk + alpha * dk
        g_next = objective.gradient(x_next)

        # s_k 表示位移，y_k 表示梯度变化量。
        sk = x_next - xk
        yk = g_next - gk
        bk = _bfgs_update(bk, sk, yk)

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
