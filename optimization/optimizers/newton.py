from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def _solve_linear_system(hk: np.ndarray, gk: np.ndarray) -> np.ndarray:
    """求解 hk * d = -gk，若矩阵秩不足则用伪逆。"""

    n = hk.shape[0]
    # 满秩优先 solve，否则再使用伪逆。
    if np.linalg.matrix_rank(hk) == n:
        return -np.linalg.solve(hk, gk)
    return -(np.linalg.pinv(hk) @ gk)


def newton_method(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch,
    grad_tol: float = 1e-5,
    max_outer_iter: int = 200,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    牛顿法。

    要求目标函数提供 Hessian：objective.hessian(x)。
    返回值：
        (x_opt, f_opt, iters, converged, grad_norm)
    """

    if objective.hess is None:
        raise ValueError("Newton method requires objective.hess.")

    # 统一输入格式，确保后续线性代数操作类型稳定。
    xk = np.asarray(x0, dtype=float)
    # 收敛标记。
    converged = False
    # 记录上一步信息，兼容需要历史量的步长策略（如 BB）。
    x_prev: Optional[np.ndarray] = None
    g_prev: Optional[np.ndarray] = None

    # 外层迭代：每轮基于当前点构造二阶局部模型。
    for iteration in range(max_outer_iter):
        # 1) 计算当前梯度并判停。
        gk = objective.gradient(xk)
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < grad_tol:
            converged = True
            break

        # 2) 牛顿方向：解线性方程 Hk * dk = -gk。
        hk = objective.hessian(xk)
        dk = _solve_linear_system(hk, gk)

        # 3) 保险策略：若不是下降方向，退化为最速下降方向。
        if float(np.dot(gk, dk)) >= 0.0:
            dk = -gk

        # 4) 用外部指定的步长搜索方法确定步长。
        alpha, _ = line_search_func(
            xk,
            dk,
            objective,
            x_prev=x_prev,
            g_prev=g_prev,
            gk=gk,
            iteration=iteration,
            **ls_params,
        )
        x_old = xk.copy()
        # 5) 位置更新。
        xk = xk + alpha * dk
        x_prev = x_old
        g_prev = gk.copy()

        # 6) 回调给上层记录实验轨迹（可选）。
        if callback is not None:
            callback(
                iteration + 1,
                xk.copy(),
                objective.value(xk),
                grad_norm,
                alpha,
            )
    else:
        # 达到最大迭代次数仍未满足 grad_tol。
        iteration = max_outer_iter
        grad_norm = float(np.linalg.norm(objective.gradient(xk)))

    # 与其他优化器保持统一返回格式。
    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
